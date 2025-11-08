# Modes vs Polarizations: generate_hplus_hcross vs generate_hplus_hcross_m

## Summary

There are two different waveform generation methods in both `dingo` and `dingo-waveform`:

1. **`generate_hplus_hcross`** - Generates **total** polarizations (h_plus, h_cross)
2. **`generate_hplus_hcross_m`** - Generates **mode-separated** polarizations

## Quick Answer

- **When to use `generate_hplus_hcross`**:
  - Dataset generation
  - Standard waveform generation
  - Most common use case
  - **This is what the dataset package uses**

- **When to use `generate_hplus_hcross_m`**:
  - Phase marginalization in parameter estimation
  - Advanced likelihood calculations
  - When you need individual mode contributions
  - **NOT used in dataset generation**

---

## Detailed Explanation

### 1. generate_hplus_hcross()

**Returns:** `Polarization` (h_plus, h_cross)

**What it does:**
- Generates the **total** gravitational wave polarizations
- These are the **sum of all spherical harmonic modes**
- Standard output: two arrays (h_plus, h_cross)

**Physical meaning:**
```
h_plus  = Σ_{l,m} h_{lm} * Y_{lm}^+ (theta_jn, phi)
h_cross = Σ_{l,m} h_{lm} * Y_{lm}^× (theta_jn, phi)
```

Where:
- `h_{lm}` = Individual mode contributions (e.g., (2,2), (2,1), (3,3), etc.)
- `Y_{lm}` = Spin-weighted spherical harmonics
- The sum is over all modes included in the approximant

**Example output structure:**
```python
polarization = wfg.generate_hplus_hcross(params)
# Returns: Polarization(h_plus=array(...), h_cross=array(...))
# h_plus.shape = (N_freq,)  # Total polarization
# h_cross.shape = (N_freq,)
```

---

### 2. generate_hplus_hcross_m()

**Returns:** `Dict[Mode, Polarization]`

**What it does:**
- Generates polarizations **separated by transformation behavior under phase shifts**
- Each mode `m` contribution transforms as `exp(-1j * m * phase)`
- Returns a dictionary: `{m: {"h_plus": h_plus_m, "h_cross": h_cross_m}}`

**Physical meaning:**

The key insight is **phase marginalization**. Under a phase shift `phase → phase + δφ`:

```python
# Total polarizations:
h_plus(phase + δφ) = complex sum of contributions

# Mode-separated contributions:
h_plus_m(phase + δφ) = exp(-1j * m * δφ) * h_plus_m(phase)
```

This allows **efficient phase marginalization** in Bayesian parameter estimation.

**Example output structure:**
```python
pol_m = wfg.generate_hplus_hcross_m(params)
# Returns: {
#   -2: Polarization(h_plus=array(...), h_cross=array(...)),
#   -1: Polarization(h_plus=array(...), h_cross=array(...)),
#    0: Polarization(h_plus=array(...), h_cross=array(...)),
#   +1: Polarization(h_plus=array(...), h_cross=array(...)),
#   +2: Polarization(h_plus=array(...), h_cross=array(...)),
# }
```

Each `pol_m[m]` contains the contribution that transforms as `exp(-1j * m * phase)`.

**Important note:**
- `pol_m[m]` includes contributions from **both** (l, m) and (l, -m) modes
- This is because positive and negative frequency parts have different phase transformations
- The implementation accounts for this to ensure exact `exp(-1j * m * phase)` transformation

---

## Use Cases

### Dataset Generation: Use `generate_hplus_hcross` ✅

**In dingo-waveform:**
```python
# dingo_waveform/dataset/generate.py
polarization = wfg.generate_hplus_hcross(wf_params)
```

**In original dingo:**
```python
# dingo/gw/dataset/generate_dataset.py
def generate_waveforms_task_func(args, waveform_generator):
    parameters = args[1].to_dict()
    return waveform_generator.generate_hplus_hcross(parameters)
```

**Why?**
- Datasets contain the **observable** waveforms
- These are the total polarizations that detectors measure
- No need for mode separation in training data

### Parameter Estimation: Use `generate_hplus_hcross_m` ✅

**In original dingo:**
```python
# dingo/gw/injection.py (GWSignal class)
def signal_m(self, theta):
    # Generate m-contributions to polarizations
    pol_m = self.waveform_generator.generate_hplus_hcross_m(theta_intrinsic)

    # Project onto detectors (for each m separately)
    # ...allows phase marginalization in likelihood
```

**Why?**
- **Phase marginalization**: Treat phase as a nuisance parameter
- Each `m` contribution has known phase dependence: `exp(-1j * m * phase)`
- Can analytically or efficiently marginalize over phase
- Used in likelihood calculations, not dataset generation

---

## Dataset Package Support

### dingo-waveform

**Current state:**
- ✅ `generate_hplus_hcross` is used
- ❌ Modes NOT supported in dataset generation
- ✅ This matches original dingo

**Files:**
```python
# dingo_waveform/dataset/generate.py
polarization = wfg.generate_hplus_hcross(wf_params)  # Line 52
polarization = waveform_generator.generate_hplus_hcross(wf_params)  # Line 91
```

### Original dingo (dingo-gw)

**Current state:**
- ✅ `generate_hplus_hcross` is used for datasets
- ❌ `generate_hplus_hcross_m` is NOT used for datasets
- ✅ `generate_hplus_hcross_m` IS used for injection/likelihood (phase marginalization)

**Files:**
```python
# dingo/gw/dataset/generate_dataset.py
waveform_generator.generate_hplus_hcross(parameters)

# dingo/gw/injection.py (for parameter estimation)
pol_m = self.waveform_generator.generate_hplus_hcross_m(theta_intrinsic)
```

---

## Should Dataset Support Modes?

### Answer: NO (for standard use cases) ❌

**Reasons:**

1. **Datasets are for training neural networks**
   - Networks learn to predict observable quantities (total h_plus, h_cross)
   - No need for mode decomposition in training data

2. **Storage efficiency**
   - Mode-separated data would be ~5-10x larger
   - Each mode needs its own h_plus and h_cross arrays

3. **Computational efficiency**
   - Generating modes is typically slower
   - Most approximants generate total polarizations directly

4. **Consistency with original dingo**
   - Original dingo datasets use `generate_hplus_hcross`
   - Modes are only used in inference/likelihood calculations

### Exception: Phase Marginalization Training

If you wanted to train a network for **phase-marginalized parameter estimation**, you might consider:

- Storing mode-separated waveforms
- Training the network to output mode contributions
- This would enable phase-free parameter estimation

However, this is an **advanced use case** and NOT the current design of dingo.

---

## Technical Details: Phase Marginalization

### The Problem

In Bayesian parameter estimation:
```
p(θ|d) ∝ L(d|θ) * π(θ)
```

where `θ` includes the **coalescence phase** `φ_c`.

Phase is often a **nuisance parameter** - we don't care about its value, just want to marginalize it out.

### Naive Approach (Slow)

```python
# Evaluate likelihood at many phase values
phases = np.linspace(0, 2*np.pi, 100)
likelihoods = [compute_likelihood(data, waveform(params, phase=p))
               for p in phases]
marginalized_likelihood = np.trapz(likelihoods, phases)
```

This requires generating 100 waveforms! Very slow.

### Efficient Approach with Modes

Since each mode contribution transforms as:
```
h_m(φ + δφ) = exp(-1j * m * δφ) * h_m(φ)
```

The likelihood integral over phase can be computed analytically or with FFTs:

```python
# Generate modes once (with fixed spin_conversion_phase)
pol_m = wfg.generate_hplus_hcross_m(params)

# Project onto detectors for each m
strain_m = {m: project_onto_detector(pol_m[m], ...) for m in pol_m}

# Marginalize over phase using FFT or analytic formula
marginalized_likelihood = compute_phase_marginalized_likelihood(strain_m, data)
```

This is **much faster** - only one waveform generation!

---

## Comparison Table

| Feature | `generate_hplus_hcross` | `generate_hplus_hcross_m` |
|---------|------------------------|--------------------------|
| **Returns** | Polarization | Dict[Mode, Polarization] |
| **Output** | Total h_plus, h_cross | h_plus_m, h_cross_m for each m |
| **Size** | 2 arrays | ~10-20 arrays (5-10 modes × 2) |
| **Speed** | Fast | Typically slower |
| **Dataset generation** | ✅ YES | ❌ NO |
| **Phase marginalization** | ❌ NO | ✅ YES |
| **Injection/Likelihood** | Sometimes | ✅ YES (when marginalizing phase) |
| **dingo-waveform support** | ✅ Full | ✅ Full |
| **dingo dataset support** | ✅ YES | ❌ NO |

---

## Code Examples

### Example 1: Standard Waveform Generation

```python
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import UniformFrequencyDomain

# Setup
domain = UniformFrequencyDomain(delta_f=0.125, f_min=20.0, f_max=1024.0)
wfg = WaveformGenerator(
    approximant=Approximant("IMRPhenomXPHM"),
    domain=domain,
    f_ref=20.0,
)

# Generate total polarizations (standard use case)
params = create_waveform_params(...)
polarization = wfg.generate_hplus_hcross(params)

# Use in dataset
h_plus_array = polarization.h_plus    # shape: (N_freq,)
h_cross_array = polarization.h_cross  # shape: (N_freq,)
```

### Example 2: Mode-Separated Generation

```python
# Generate mode-separated polarizations (advanced use case)
pol_m = wfg.generate_hplus_hcross_m(params)

# pol_m is a dictionary: {m: Polarization}
for m, polarization in pol_m.items():
    print(f"Mode m={m}:")
    print(f"  h_plus shape: {polarization.h_plus.shape}")
    print(f"  h_cross shape: {polarization.h_cross.shape}")

# Reconstruct total (for verification)
h_plus_total = sum(pol.h_plus for pol in pol_m.values())
h_cross_total = sum(pol.h_cross for pol in pol_m.values())

# Should approximately match generate_hplus_hcross output
# (may differ slightly due to spin_conversion_phase handling)
```

---

## Recommendation

### For Dataset Generation: ✅ Use `generate_hplus_hcross`

**Current implementation is correct:**
```python
# dingo_waveform/dataset/generate.py (CORRECT)
polarization = wfg.generate_hplus_hcross(wf_params)
```

**Do NOT change to:**
```python
# DON'T DO THIS for datasets
pol_m = wfg.generate_hplus_hcross_m(wf_params)
```

### For Parameter Estimation: Consider `generate_hplus_hcross_m`

If implementing **phase marginalization** in likelihood calculations:
```python
# For phase-marginalized likelihood
pol_m = wfg.generate_hplus_hcross_m(theta_intrinsic)
# ... then marginalize over phase analytically or with FFT
```

---

## Summary

1. **Two functions serve different purposes:**
   - `generate_hplus_hcross`: Total observable waveforms
   - `generate_hplus_hcross_m`: Mode contributions for phase marginalization

2. **Dataset generation:**
   - Uses `generate_hplus_hcross` (total polarizations)
   - This is correct and matches original dingo
   - Do NOT use modes for datasets

3. **Mode support:**
   - Both dingo and dingo-waveform support modes
   - Modes are for parameter estimation, NOT dataset generation
   - Phase marginalization is the primary use case

4. **Your dataset package is correct:**
   - ✅ Uses `generate_hplus_hcross`
   - ✅ Matches original dingo implementation
   - ✅ No changes needed

---

**Date**: 2025-11-07
**Status**: ✅ Complete Analysis
