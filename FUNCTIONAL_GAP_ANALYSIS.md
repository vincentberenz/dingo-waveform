# Functional Gap Analysis: dingo-waveform Refactor

**Purpose**: Identify missing **functionality** (not API differences) needed for porting to dingo-gw
**Date**: 2025-11-10
**Scope**: Waveform generation components only (domains, waveform generators, compression)

---

## Context: Refactor Goals

The dingo-waveform refactor aims to **eliminate dict-based APIs** in favor of type-safe, developer-friendly interfaces using dataclasses and typed objects. API differences (dict vs dataclass) are **intentional improvements**, not gaps.

This analysis focuses only on **missing core functionality** that dingo-gw requires.

---

## Refactor Scope

### ✅ What Has Been Refactored
- **Domains**: UniformFrequencyDomain, MultibandedFrequencyDomain, TimeDomain
- **Waveform Generation**: Polarizations (h_plus, h_cross) for various approximants
- **Waveform Parameters**: Type-safe WaveformParameters dataclass
- **SVD Compression**: Complete and verified (identical to dingo-gw)
- **Prior Sampling**: IntrinsicPriors with type safety

### ❌ What Is Out of Scope (Not Yet Refactored)
These components remain in dingo-gw and are **not** considered gaps:
- Noise/ASD handling (`dingo.gw.noise`)
- Detector response and interferometers (`dingo.gw.detector`, `dingo.gw.interferometers`)
- Data preprocessing transforms (`dingo.gw.transforms`)
- Dataset generation infrastructure (`dingo.gw.dataset`)
- Training infrastructure (`dingo.gw.training`)
- Inference/sampling infrastructure (`dingo.gw.inference`)

---

## Critical Gap: MultibandedFrequencyDomain Base Domain Access

### The Problem

**dingo-gw pattern:**
```python
# Create multibanded domain with explicit base domain
base = UniformFrequencyDomain(f_min=20, f_max=1024, delta_f=0.125)
mfd = MultibandedFrequencyDomain(
    nodes=[20, 128, 1024],
    delta_f_initial=0.125,
    base_domain=base  # Explicit
)

# Later: access the base domain
ufd = mfd.base_domain  # Returns UniformFrequencyDomain object
```

**dingo-waveform pattern:**
```python
# More explicit, but base domain not accessible
mfd = MultibandedFrequencyDomain(
    nodes=[20, 128, 1024],
    delta_f_initial=0.125,
    base_delta_f=0.125  # Stored internally, not exposed
)

# No way to get base domain as a Domain object!
# ufd = mfd.base_domain  # ❌ Doesn't exist
```

### Why It's Needed

The base domain is required for several **functional operations**:

#### 1. **Waveform Generation Workflow**
```python
# Standard workflow in dingo-gw:
# 1. Create waveform generator on BASE domain
wfg_base = WaveformGenerator(domain=mfd.base_domain, ...)

# 2. Generate waveform on uniform grid
pol = wfg_base.generate_hplus_hcross(params)  # Full resolution

# 3. Decimate to multibanded representation
pol_decimated = mfd.decimate(pol)
```

**Current dingo-waveform workaround:**
- Use `waveform_transform()` which implicitly assumes base grid
- But cannot create another WaveformGenerator for the base grid
- Cannot pass base domain to other components that need it

#### 2. **Data Format Checking**
```python
# In transforms: check if data is in base format or decimated
if data.shape[-1] == len(mfd.base_domain):
    # Data is in full resolution, needs decimation
    data = mfd.decimate(data)
elif data.shape[-1] == len(mfd):
    # Already decimated
    pass
```

**Current dingo-waveform:** Cannot perform this check without base domain.

#### 3. **Inference Frequency Bounds Validation**
```python
# In inference: validate frequency bounds against base domain
if isinstance(domain, MultibandedFrequencyDomain):
    base = domain.base_domain
    if f_min < base.f_min or f_max > base.f_max:
        raise ValueError("Frequency bounds exceed base domain")
```

**Current dingo-waveform:** Can check `mfd.f_min`/`mfd.f_max`, but not the underlying uniform grid bounds.

#### 4. **Interpolation and Comparison**
```python
# Comparing multibanded against full resolution
ufd = mfd.base_domain
wf_full = generate_on(ufd)
wf_decimated = generate_on(mfd)
wf_interpolated = interpolate(wf_decimated, ufd())  # Need base frequencies
```

### Impact Assessment

| Use Case | Without base_domain | Impact |
|----------|---------------------|--------|
| Generate on uniform grid then decimate | Cannot create WaveformGenerator for base | **HIGH** |
| Check data format | Cannot determine if data is decimated | **HIGH** |
| Validate frequency bounds | Can only check MFD bounds, not base | **MEDIUM** |
| Interpolation comparisons | Cannot get base frequency array as domain | **MEDIUM** |

### Recommendation

**Add a `base_domain` property that constructs and returns a UniformFrequencyDomain:**

```python
class MultibandedFrequencyDomain(BaseFrequencyDomain):

    @property
    def base_domain(self) -> UniformFrequencyDomain:
        """
        Return the underlying uniform frequency domain.

        The base domain is the uniform grid (f=0 to f_max with spacing base_delta_f)
        on which waveforms are generated before decimation to the multibanded representation.

        Returns
        -------
        UniformFrequencyDomain
            Uniform frequency domain with:
            - f_min = self.nodes[0] (or 0.0 if starting from zero)
            - f_max = self.nodes[-1]
            - delta_f = self.base_delta_f
        """
        # Construct on-demand (could cache if needed)
        return UniformFrequencyDomain(
            f_min=0.0,  # Base grid always starts at 0
            f_max=self.nodes[-1],
            delta_f=self.base_delta_f,
            window_factor=self.window_factor
        )
```

**Effort**: 1-2 hours
**Risk**: Low (well-defined functionality)
**Benefit**: Enables full dingo-gw integration workflows

---

## Other Functional Considerations

### 1. Domain Serialization

**Status**: ✅ Functionality exists, ⚠️ different API

- **dingo-gw**: `domain.domain_dict` → dict → `build_domain(dict)`
- **dingo-waveform**: `domain.get_parameters()` → DomainParameters → `Domain.from_parameters(params)`

**Analysis**: This is an **API difference**, not a functional gap. Both can serialize/deserialize domains. The dingo-waveform approach is more type-safe.

**For porting**: Will need a compatibility layer or adapter, but the core functionality exists.

---

### 2. Domain Updates / Narrowing

**Status**: ✅ Functionality equivalent

- **dingo-gw**: `domain.update({"f_min": 30, "f_max": 512})` - mutates in-place
- **dingo-waveform**: `new_domain = domain.update(f_min=30, f_max=512)` or `new_domain = domain.narrowed(30, 512)` - returns new instance

**Analysis**: **Functional equivalence**. dingo-waveform's immutable approach is safer. This is an intentional API improvement, not a gap.

---

### 3. WaveformGenerator Interface

**Status**: ✅ Functionality equivalent, ⚠️ different API

- **dingo-gw**: Accepts/returns dicts
- **dingo-waveform**: Accepts WaveformParameters, returns Polarization

**Analysis**: Same waveform generation capability (verified by tests). API difference is intentional improvement. Not a functional gap.

---

### 4. SVD Compression

**Status**: ✅ Complete and verified

- Identical numerical results (machine precision)
- Basis interchangeability tested
- API uses dataclasses instead of dicts (intentional improvement)

**Analysis**: No gaps.

---

### 5. Time Domain Support

**Status**: ✅ Implemented

- `TimeDomain` class exists in dingo-waveform
- Provides time-domain properties (duration, sampling_rate)
- Not fully tested for waveform generation, but structure is there

**Analysis**: Basic infrastructure exists. May need more methods for time-domain waveform operations.

---

## Out-of-Scope Components (Integration Points)

These are **not** gaps in the refactor, but will need integration when porting:

### Noise & ASD Infrastructure
**Location in dingo-gw**: `dingo/gw/noise/`

- ASD dataset generation
- ASD estimation from data
- Synthetic noise generation

**Integration need**: These components will need to work with refactored domains. Should be straightforward since domains provide `noise_std` and frequency information.

---

### Detector & Interferometer
**Location in dingo-gw**: `dingo/gw/detector/`, `dingo/gw/gwutils.py`

- Detector response functions
- Antenna patterns
- Projection onto detectors

**Integration need**: Will need to work with Polarization dataclass instead of dicts.

---

### Transforms
**Location in dingo-gw**: `dingo/gw/transforms/`

- WhitenAndScaleStrain
- GetDetectorResponseTorch
- TimeTranslateWaveforms
- MultibandingTransform
- Parameter transforms

**Integration need**: Most transforms depend on:
- ✅ `domain.noise_std` - compatible
- ✅ `domain.time_translate_data()` - compatible
- ⚠️ `mfd.base_domain` - needs the property added
- ⚠️ Polarization dataclass vs dict - minor adaptation needed

**Assessment**: Transforms should be straightforward to adapt once `base_domain` is added.

---

### Dataset Generation
**Location in dingo-gw**: `dingo/gw/dataset/`

- WaveformDataset class
- Dataset generation scripts
- Batch generation infrastructure

**Integration need**: Will need to:
- Use WaveformParameters instead of dicts for parameter generation
- Use Polarization dataclass for waveform storage
- Use DomainParameters for domain serialization

**Assessment**: This is where the dict → dataclass transition will be most visible. But this is the intended refactor outcome, not a gap.

---

### Training & Inference
**Location in dingo-gw**: `dingo/gw/training/`, `dingo/gw/inference/`

- Neural network training loops
- Samplers (GWSampler, GWSamplerGNPE)
- Posterior sampling infrastructure

**Integration need**: Will need to work with new serialization formats and type-safe interfaces.

**Assessment**: Out of scope for current refactor. Will be adapted when integrating.

---

## Summary

### Critical Functional Gap

**Only one critical gap identified:**

| Gap | Severity | Effort | Description |
|-----|----------|--------|-------------|
| MultibandedFrequencyDomain missing `base_domain` property | **HIGH** | 1-2 hours | Cannot access underlying uniform frequency domain needed for generation, data format checking, and validation |

### API Differences (Not Gaps)

These are **intentional improvements**, not missing functionality:

| Component | dingo-gw API | dingo-waveform API | Assessment |
|-----------|--------------|---------------------|------------|
| Domain serialization | `domain_dict` dict | `get_parameters()` DomainParameters | ✅ Equivalent functionality, better types |
| WaveformGenerator | dict params/returns | WaveformParameters/Polarization | ✅ Equivalent functionality, better types |
| Domain updates | Mutable `update(dict)` | Immutable `update(**kwargs)` or `narrowed()` | ✅ Equivalent functionality, safer |
| SVD compression | Dict-based API | Dataclass-based API | ✅ Verified equivalent, better types |

### Out of Scope

Not refactored yet, will need integration when porting:
- ❌ Noise/ASD infrastructure
- ❌ Detector response
- ❌ Transforms (but compatible once `base_domain` added)
- ❌ Dataset generation infrastructure
- ❌ Training/inference infrastructure

---

## Recommendation

### Immediate Action (Before Porting)

**Add `base_domain` property to MultibandedFrequencyDomain**

```python
@property
def base_domain(self) -> UniformFrequencyDomain:
    """Return the underlying uniform frequency domain."""
    return UniformFrequencyDomain(
        f_min=0.0,
        f_max=self.nodes[-1],
        delta_f=self.base_delta_f,
        window_factor=self.window_factor
    )
```

**Estimated time**: 1-2 hours
**Testing**: Verify compatibility with dingo-gw usage patterns

### Integration Strategy

For porting to dingo-gw:

1. **Keep the improved APIs** - Don't regress to dict-based interfaces
2. **Adapt calling code** - Update dingo-gw components to use typed interfaces
3. **Compatibility layers** - Only where absolutely necessary for gradual migration
4. **Test incrementally** - Component by component replacement

### Non-Goals

**Do NOT** add these as they defeat the refactor purpose:
- ❌ Dict-accepting methods just for backward compatibility
- ❌ `domain_dict` property (use `asdict(domain.get_parameters())` if needed)
- ❌ Dict-returning waveform methods

The point is to **improve** dingo-gw, not to replicate its issues.

---

## Conclusion

The dingo-waveform refactor is **nearly complete** for its intended scope (waveform generation):

- ✅ Core functionality fully implemented and verified
- ✅ APIs are type-safe and developer-friendly (intentional improvement)
- ⚠️ One functional gap: `base_domain` property (easy fix)
- ✅ Ready for integration with proper adaptation of calling code

**Next step**: Add `base_domain` property, then begin gradual integration into dingo-gw.
