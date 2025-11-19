# Waveform Configuration Examples

This directory contains configuration files and Python scripts demonstrating the `dingo-waveform` API for gravitational waveform generation.

## Overview

The examples are organized by use case:
- **Basic waveform generation**: Single waveform with fixed parameters
- **Mode-separated waveforms**: Individual spherical harmonic mode contributions
- **Dataset generation**: Multiple waveforms sampled from priors
- **SVD compression**: Waveform compression using Singular Value Decomposition

## Configuration Files

### Basic Waveform Generation

Configuration files with fixed waveform parameters for generating single waveforms:

| File | Domain | Approximant | Description |
|------|--------|-------------|-------------|
| `basic_uniform_frequency.yaml` | UniformFrequencyDomain | IMRPhenomXPHM | Standard uniform frequency grid |
| `basic_multibanded_frequency.yaml` | MultibandedFrequencyDomain | IMRPhenomXPHM | Non-uniform dyadic frequency spacing |

### Mode-Separated Waveforms

Configuration for generating individual spherical harmonic modes:

| File | Domain | Approximant | Description |
|------|--------|-------------|-------------|
| `modes_imrphenomxphm.yaml` | UniformFrequencyDomain | IMRPhenomXPHM | Demonstrates mode decomposition |

**Note**: Only certain approximants support mode-separated generation:
- IMRPhenomXPHM
- SEOBNRv4PHM
- SEOBNRv5PHM
- SEOBNRv5HM

### Advanced Approximants

Configurations demonstrating specific approximants and scenarios:

| File | Domain | Approximant | Description |
|------|--------|-------------|-------------|
| `advanced_seobnrv4phm.yaml` | UniformFrequencyDomain | SEOBNRv4PHM | Precessing binary with SEOBNRv4 |
| `advanced_seobnrv5hm_multibanded.yaml` | MultibandedFrequencyDomain | SEOBNRv5HM | Higher modes with multibanded domain |

### Dataset Generation

Configurations with priors for generating training datasets:

| File | Domain | Approximant | Description |
|------|--------|-------------|-------------|
| `dataset_quick_imrphenomd.yaml` | UniformFrequencyDomain | IMRPhenomD | Fast testing with simple approximant |
| `dataset_production_imrphenomxphm.yaml` | UniformFrequencyDomain | IMRPhenomXPHM | Production settings with realistic priors |
| `dataset_multibanded_imrphenomxphm.yaml` | MultibandedFrequencyDomain | IMRPhenomXPHM | Multibanded domain with full priors |
| `dataset_multibanded_simple.yaml` | MultibandedFrequencyDomain | IMRPhenomXPHM | Simplified multibanded configuration |

### SVD Compression

Configurations for waveform compression using SVD:

| File | Domain | Approximant | Description |
|------|--------|-------------|-------------|
| `svd_compression_small.yaml` | UniformFrequencyDomain | IMRPhenomXPHM | Small-scale SVD example (50 components) |
| `svd_compression_full.yaml` | UniformFrequencyDomain | IMRPhenomXPHM | Full-scale SVD (200 components) |

## Python Scripts

### `generate_basic_waveform.py`

Generate a single waveform with fixed parameters.

```bash
python generate_basic_waveform.py
```

**Demonstrates:**
- Loading configuration from YAML
- Creating domain and WaveformGenerator
- Generating polarizations (h+, h×)
- Analyzing waveform properties

**Uses:** `basic_uniform_frequency.yaml`

### `generate_modes.py`

Generate mode-separated waveforms.

```bash
python generate_modes.py
```

**Demonstrates:**
- Generating individual spherical harmonic modes
- Accessing specific modes (ℓ, m)
- Analyzing mode amplitudes and contributions
- Finding the dominant mode

**Uses:** `modes_imrphenomxphm.yaml`

### `generate_dataset_example.py`

Generate multiple waveforms by sampling from priors.

```bash
python generate_dataset_example.py
```

**Demonstrates:**
- Creating prior distributions
- Sampling parameters
- Generating multiple waveforms
- Dataset statistics

**Uses:** `dataset_quick_imrphenomd.yaml`

**Note:** For production-scale dataset generation, use the command-line tool:
```bash
dingo_generate_dataset --settings_file dataset_production_imrphenomxphm.yaml --num_processes 8
```

### `generate_with_svd.py`

Demonstrate SVD-based waveform compression.

```bash
python generate_with_svd.py
```

**Demonstrates:**
- Building SVD basis from training waveforms
- Compressing waveforms to reduced dimensions
- Reconstructing waveforms from SVD coefficients
- Measuring reconstruction error (mismatch)

**Uses:** `svd_compression_small.yaml`

## Configuration File Format

All configuration files follow the same YAML structure:

```yaml
domain:
  type: UniformFrequencyDomain  # or MultibandedFrequencyDomain
  delta_f: 0.125                # Frequency spacing (uniform domains)
  f_min: 20.0                   # Minimum frequency (Hz)
  f_max: 1024.0                 # Maximum frequency (Hz)
  # For multibanded domains:
  # nodes: [20.0, 128.0, 512.0, 1024.0]
  # delta_f_initial: 0.125
  # base_delta_f: 0.125

waveform_generator:
  approximant: IMRPhenomXPHM    # LALSimulation approximant
  f_ref: 20.0                   # Reference frequency (Hz)
  f_start: 20.0                 # Starting frequency (Hz)
  spin_conversion_phase: 0.0    # Spin conversion phase

# Option 1: Fixed parameters (for single waveform generation)
waveform_parameters:
  mass_1: 36.0                  # Primary mass (M☉)
  mass_2: 29.0                  # Secondary mass (M☉)
  luminosity_distance: 1000.0   # Distance (Mpc)
  theta_jn: 0.5                 # Inclination angle (rad)
  phase: 0.0                    # Coalescence phase (rad)
  a_1: 0.5                      # Primary spin magnitude
  a_2: 0.3                      # Secondary spin magnitude
  tilt_1: 0.5                   # Primary spin tilt (rad)
  tilt_2: 0.8                   # Secondary spin tilt (rad)
  phi_12: 1.7                   # Azimuthal angle between spins (rad)
  phi_jl: 0.3                   # Azimuthal angle of L (rad)
  geocent_time: 0.0             # Coalescence time (s)

# Option 2: Prior distributions (for dataset generation)
intrinsic_prior:
  chirp_mass: Uniform(minimum=25.0, maximum=35.0)
  mass_ratio: Uniform(minimum=0.8, maximum=1.0)
  luminosity_distance: 1000.0
  theta_jn: Uniform(minimum=0.0, maximum=3.14159)
  phase: 0.0
  a_1: Uniform(minimum=0.0, maximum=0.88)
  a_2: Uniform(minimum=0.0, maximum=0.88)
  tilt_1: Sine(minimum=0.0, maximum=3.14159)
  tilt_2: Sine(minimum=0.0, maximum=3.14159)
  phi_12: Uniform(minimum=0.0, maximum=6.28318)
  phi_jl: Uniform(minimum=0.0, maximum=6.28318)
  geocent_time: 0.0

# Option 3: SVD compression settings
svd:
  n_components: 50              # Number of SVD basis vectors
  num_training: 100             # Training set size
  num_validation: 20            # Validation set size
  method: scipy                 # SVD computation method
```

## Domain Types

### UniformFrequencyDomain

Standard frequency-domain with constant frequency spacing (`delta_f`).

**Parameters:**
- `delta_f`: Frequency resolution (Hz)
- `f_min`: Minimum frequency (Hz)
- `f_max`: Maximum frequency (Hz)

**Use when:** You need standard uniform frequency sampling.

### MultibandedFrequencyDomain

Non-uniform frequency domain with dyadic (powers-of-2) spacing.

**Parameters:**
- `nodes`: Frequency boundaries for bands (Hz)
- `delta_f_initial`: Initial frequency spacing (Hz)
- `base_delta_f`: Base frequency spacing for binning (Hz)

**Use when:** You want efficient representation with higher resolution at low frequencies.

## Supported Approximants

### Precessing Binaries (PHM = Precessing Higher Modes)
- `IMRPhenomXPHM` - Phenomenological, frequency-domain
- `SEOBNRv4PHM` - Effective-one-body, time-domain
- `SEOBNRv5PHM` - Latest EOB with precession

### Higher Modes (HM)
- `SEOBNRv5HM` - EOB with higher modes (non-precessing)
- `SEOBNRv5EHM` - EOB with eccentric orbits

### Aligned-Spin
- `IMRPhenomD` - Fast phenomenological (aligned spins only)
- `IMRPhenomXAS` - Extended IMRPhenomD
- `TaylorF2` - Post-Newtonian inspiral

See [LALSimulation documentation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/) for complete list.

## Parameter Descriptions

### Intrinsic Parameters

- **mass_1, mass_2**: Component masses in solar masses (M☉)
- **chirp_mass**: Combination of masses: ℳ = (m₁m₂)³⁄⁵/(m₁+m₂)¹⁄⁵
- **mass_ratio**: q = m₂/m₁ (convention: q ≤ 1)
- **a_1, a_2**: Dimensionless spin magnitudes (0 ≤ a ≤ 1)
- **tilt_1, tilt_2**: Angle between spin and orbital angular momentum (rad)
- **phi_12**: Azimuthal angle between component spins (rad)
- **phi_jl**: Azimuthal angle of orbital angular momentum (rad)

### Extrinsic Parameters

- **luminosity_distance**: Distance to source (Mpc)
- **theta_jn**: Inclination angle between line-of-sight and total angular momentum (rad)
- **phase**: Coalescence phase (rad)
- **geocent_time**: GPS time at geocenter (s)

## Prior Distributions

Supported prior types (from `bilby`):

- `Uniform(minimum, maximum)` - Uniform distribution
- `Sine(minimum, maximum)` - Sine distribution (for angles)
- `Constraint(minimum, maximum)` - Hard bounds (for bilby constraints)
- `UniformInComponentsChirpMass(minimum, maximum)` - Uniform in chirp mass space
- `UniformInComponentsMassRatio(minimum, maximum)` - Uniform in mass ratio space

## Command-Line Tools

### Verify Waveforms

Compare waveforms against dingo (dingo-gw) reference implementation:

```bash
dingo-verify --config basic_uniform_frequency.yaml --verbose
```

### Generate Datasets

For production-scale parallel dataset generation:

```bash
dingo_generate_dataset \
    --settings_file dataset_production_imrphenomxphm.yaml \
    --num_processes 8 \
    --out_file training_dataset.hdf5
```

### Benchmark Performance

Measure waveform generation performance:

```bash
dingo-benchmark --config basic_uniform_frequency.yaml -n 100
```

## Tips

1. **Start simple**: Begin with `basic_uniform_frequency.yaml` and `generate_basic_waveform.py`

2. **Check approximant support**: Use `dingo-verify` to ensure your approximant works correctly

3. **Mode-separated waveforms**: Only PHM/HM approximants support `generate_hplus_hcross_m()`

4. **Prior specification**: When using priors, ensure mass constraints are satisfied (m₁ ≥ m₂)

5. **Performance**: For large datasets, use `dingo_generate_dataset` with `--num_processes`

6. **SVD compression**: Start with small training sets to test, then scale up

