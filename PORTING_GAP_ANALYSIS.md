# Gap Analysis: dingo-waveform → dingo-gw Porting

**Status**: Waveform generation complete; preparing for integration into dingo-gw
**Date**: 2025-11-10
**Purpose**: Identify missing features in dingo-waveform that are required for porting back to dingo-gw

---

## Executive Summary

This document identifies features present in dingo-gw that are **missing or incompatible** in dingo-waveform. The refactor has successfully implemented waveform generation with verified scientific correctness. However, several features required for the full dingo-gw pipeline (training, inference, dataset generation) are either missing or use incompatible APIs.

**Key Findings:**
1. ✅ **Waveform generation**: Complete and verified
2. ✅ **SVD compression**: Complete and verified
3. ⚠️ **Domain serialization**: Different API (needs compatibility layer)
4. ❌ **MultibandedFrequencyDomain**: Missing `base_domain` property
5. ❌ **Domain building**: No `domain_dict` / `build_domain()` pattern
6. ⚠️ **WaveformGenerator**: Missing properties used by transforms
7. ❌ **Integration points**: No backward compatibility with existing code

---

## 1. Domain Classes

### 1.1 UniformFrequencyDomain

| Feature | dingo-gw | dingo-waveform | Status | Impact |
|---------|----------|----------------|--------|---------|
| **Core properties** | | | |
| `f_min`, `f_max`, `delta_f` | ✅ | ✅ | ✅ Compatible | - |
| `min_idx`, `max_idx` | ✅ | ✅ | ✅ Compatible | - |
| `duration`, `sampling_rate` | ✅ | ✅ | ✅ Compatible | - |
| `sample_frequencies` | ✅ | ✅ | ✅ Compatible | - |
| `frequency_mask` | ✅ | ✅ | ✅ Compatible | - |
| **Noise/windowing** | | | |
| `noise_std` property | ✅ | ✅ | ✅ Compatible | Used by WhitenAndScaleStrain transform |
| `window_factor` | ✅ (optional) | ✅ (optional) | ✅ Compatible | Used for noise calculations |
| **Serialization** | | | |
| `domain_dict` property | ✅ | ❌ | ⚠️ **Incompatible** | **HIGH**: Used everywhere for saving/loading |
| `get_parameters()` method | ❌ | ✅ | ⚠️ **Incompatible** | Different serialization approach |
| `from_parameters()` classmethod | ❌ | ✅ | ⚠️ **Incompatible** | Different deserialization approach |
| **Data manipulation** | | | |
| `update(dict)` method | ✅ | ✅ | ⚠️ **Different API** | dingo-waveform uses kwargs, dingo uses dict |
| `update_data(data, axis, low_value)` | ✅ | ✅ | ✅ Compatible | Truncates/zeros data |
| `time_translate_data(data, dt)` | ✅ | ✅ | ✅ Compatible | Applies phase shift |
| **PyTorch support** | | | |
| `sample_frequencies_torch` | ✅ | ✅ | ✅ Compatible | - |
| `sample_frequencies_torch_cuda` | ✅ | ✅ | ✅ Compatible | - |
| `get_sample_frequencies_astype(data)` | ✅ | ✅ (private) | ⚠️ Different name | dingo-waveform uses `_get_sample_frequencies_astype` |

#### Issues & Recommendations

**CRITICAL: Serialization Incompatibility**

```python
# dingo-gw pattern (used throughout)
domain_dict = domain.domain_dict  # Returns dict
new_domain = build_domain(domain_dict)

# dingo-waveform pattern
domain_params = domain.get_parameters()  # Returns DomainParameters dataclass
new_domain = UniformFrequencyDomain.from_parameters(domain_params)
```

**Recommendation**: Add `domain_dict` property and support `build_domain()` pattern for backward compatibility.

**Method Signature Differences**

```python
# dingo-gw
domain.update({"f_min": 30.0, "f_max": 512.0})

# dingo-waveform
domain.update(f_min=30.0, f_max=512.0)
```

**Recommendation**: Overload `update()` to accept both dict and kwargs.

---

### 1.2 MultibandedFrequencyDomain

| Feature | dingo-gw | dingo-waveform | Status | Impact |
|---------|----------|----------------|--------|---------|
| **Core structure** | | | |
| `nodes` parameter | ✅ | ✅ | ✅ Compatible | - |
| `delta_f_initial` parameter | ✅ | ✅ | ✅ Compatible | - |
| `base_domain` property | ✅ | ❌ | ❌ **MISSING** | **HIGH**: Used extensively |
| `base_delta_f` parameter | ❌ | ✅ | ⚠️ Different API | dingo-waveform more explicit |
| **Serialization** | | | |
| `domain_dict` property | ✅ | ❌ | ❌ **MISSING** | **HIGH**: Required for saving |
| `get_parameters()` method | ❌ | ✅ | ⚠️ **Incompatible** | - |
| **Methods** | | | |
| `decimate(data)` | ✅ | ✅ | ✅ Compatible | Downsamples to banded grid |
| `update(dict)` | ✅ | ❌ | ❌ **MISSING** | Used in inference |
| `update_data(data, axis, low_value)` | ✅ | ❌ | ❌ **MISSING** | Used for data processing |
| **Domain-specific** | | | |
| `waveform_transform()` | ❌ | ✅ | ⚠️ Extra feature | dingo-waveform only |
| `narrowed()` | ❌ | ✅ | ⚠️ Extra feature | dingo-waveform only |

#### Issues & Recommendations

**CRITICAL: Missing `base_domain` Property**

In dingo-gw, `MultibandedFrequencyDomain` requires a `base_domain` (UniformFrequencyDomain) which is used throughout the codebase:

```python
# dingo-gw usage (from inference, transforms, dataset generation)
mfd = MultibandedFrequencyDomain(nodes=[20, 128, 1024],
                                  delta_f_initial=0.125,
                                  base_domain=UniformFrequencyDomain(...))

# Used in:
# - inference: domain.base_domain for data loading
# - transforms: checks if data is in base_domain vs decimated
# - dataset: generates on base_domain, then decimates
```

**dingo-waveform approach:**
```python
# More explicit, but incompatible
mfd = MultibandedFrequencyDomain(nodes=[20, 128, 1024],
                                  delta_f_initial=0.125,
                                  base_delta_f=0.125)  # Constructs internally
```

**Recommendation**: Add `base_domain` property that returns an equivalent `UniformFrequencyDomain`.

---

## 2. Domain Building & Serialization

### 2.1 Missing `build_domain()` Function

**dingo-gw pattern** (used in 40+ places):

```python
from dingo.gw.domains import build_domain

# From settings dict
domain = build_domain({
    "type": "UniformFrequencyDomain",
    "f_min": 20.0,
    "f_max": 1024.0,
    "delta_f": 0.125
})

# From saved model
domain = build_domain(model_metadata["dataset_settings"]["domain"])
```

**Usage locations:**
- `dingo/gw/dataset/generate_dataset.py`
- `dingo/gw/dataset/waveform_dataset.py`
- `dingo/gw/inference/gw_samplers.py`
- `dingo/gw/result.py`
- `dingo/gw/waveform_generator/waveform_generator.py`

**dingo-waveform approach:**
```python
# Must know domain type in advance
from dingo_waveform.domains import UniformFrequencyDomain, DomainParameters

params = DomainParameters(...)
domain = UniformFrequencyDomain.from_parameters(params)
```

**Recommendation**: Add `build_domain()` factory function that wraps `from_parameters()` logic.

---

### 2.2 `domain_dict` vs `get_parameters()`

| Aspect | `domain_dict` (dingo-gw) | `get_parameters()` (dingo-waveform) |
|--------|-------------------------|-------------------------------------|
| Return type | `dict` | `DomainParameters` (dataclass) |
| Usage | Everywhere | New API |
| Reconstruction | `build_domain(domain_dict)` | `Domain.from_parameters(params)` |
| HDF5 saving | Direct dict → HDF5 attrs | Need to convert dataclass |
| JSON compatible | ✅ Yes | ⚠️ Need `asdict()` |

**Used in dingo-gw:**
```python
# Saving model metadata
metadata["domain"] = domain.domain_dict

# Comparing domains
if domain1.domain_dict == domain2.domain_dict:
    ...

# Updating training settings
train_settings["domain_update"] = domain.domain_dict

# Rebuilding after modification
new_domain = build_domain(old_domain.domain_dict)
```

**Recommendation**: Add `domain_dict` property as compatibility layer:
```python
@property
def domain_dict(self) -> dict:
    params = self.get_parameters()
    return asdict(params)
```

---

## 3. WaveformGenerator

### 3.1 Property Comparison

| Feature | dingo-gw | dingo-waveform | Status | Impact |
|---------|----------|----------------|--------|---------|
| **Properties** | | | |
| `domain` property (get/set) | ✅ | ❌ | ⚠️ Different | Stored in `__init__`, not as property |
| `full_domain` property | ✅ | ❌ | ❌ **MISSING** | Used by transforms |
| `spin_conversion_phase` property | ✅ | ❌ | ⚠️ Different | Passed to `__init__` in dingo-waveform |
| **Methods** | | | |
| `generate_hplus_hcross(dict)` | ✅ | ✅ | ⚠️ Different signature | dict vs WaveformParameters |
| `generate_hplus_hcross_m(dict)` | ✅ | ✅ | ⚠️ Different signature | dict vs WaveformParameters |
| `generate_FD_waveform(params)` | ✅ | ❌ | ❌ **MISSING** | Internal method, may not be needed |
| `generate_TD_waveform(params)` | ✅ | ❌ | ❌ **MISSING** | Not yet implemented |

### 3.2 API Differences

**dingo-gw:**
```python
wfg = WaveformGenerator(
    approximant="IMRPhenomXPHM",
    domain=domain,
    f_ref=20.0,
    f_start=20.0,
    spin_conversion_phase=0.0
)

# Parameters as dict
h_plus, h_cross = wfg.generate_hplus_hcross({
    "mass_1": 36.0,
    "mass_2": 29.0,
    ...
})
```

**dingo-waveform:**
```python
from dingo_waveform.approximant import Approximant
from dingo_waveform.waveform_parameters import WaveformParameters

wfg = WaveformGenerator(
    approximant=Approximant("IMRPhenomXPHM"),  # Type-safe
    domain=domain,
    f_ref=20.0,
    f_start=20.0,
    spin_conversion_phase=0.0  # Stored, not property
)

# Parameters as dataclass
params = WaveformParameters(mass_1=36.0, mass_2=29.0, ...)
pol = wfg.generate_hplus_hcross(params)  # Returns Polarization dataclass
```

**Recommendation**: Add overloads to accept both dict and WaveformParameters.

---

## 4. Integration Points

### 4.1 Dataset Generation

**dingo-gw workflow:**
```python
from dingo.gw.domains import build_domain
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.dataset import WaveformDataset

# Build from settings
domain = build_domain(settings["domain"])
wfg = WaveformGenerator(**settings["waveform_generator"], domain=domain)

# Generate dataset (uses dict params)
dataset = WaveformDataset(settings)
dataset.generate_hplus_hcross(parameters_dict)
```

**dingo-waveform:**
```python
# Requires explicit type handling
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.domains import UniformFrequencyDomain, DomainParameters

# Must construct explicitly
params = DomainParameters(...)
domain = UniformFrequencyDomain.from_parameters(params)
wfg = WaveformGenerator(...)

# Generate (uses WaveformParameters)
from dingo_waveform.waveform_parameters import WaveformParameters
wf_params = WaveformParameters(...)
pol = wfg.generate_hplus_hcross(wf_params)
```

**Gap**: dingo-waveform dataset generation not implemented yet.

---

### 4.2 Training Pipeline

**Uses from dingo-gw:**
- `domain.domain_dict` - Saved in model metadata ✅
- `domain.noise_std` - Used by `WhitenAndScaleStrain` transform ✅ (compatible)
- `domain.min_idx` - Used for truncating SVD matrices ✅ (compatible)
- `wfd.domain` - Accessed as property ⚠️ (dingo-waveform: attribute)

---

### 4.3 Inference Pipeline

**Uses from dingo-gw:**
- `domain.f_min`, `domain.f_max` - Frequency bounds checking ✅ (compatible)
- `domain.base_domain` - For MultibandedFrequencyDomain ❌ (**MISSING**)
- `domain.update(dict)` - Runtime domain updates ⚠️ (different API)
- `domain.noise_std` - Whitening ✅ (compatible)
- `build_domain()` - Loading from model ❌ (**MISSING**)

---

### 4.4 Transforms

**Key transforms that depend on domains:**

1. **WhitenAndScaleStrain** - Uses `domain.noise_std` ✅
2. **GetDetectorResponseTorch** - Uses `domain` and `wfg` properties ⚠️
3. **TimeTranslateWaveforms** - Uses `domain.time_translate_data()` ✅
4. **MultibandingTransform** - Needs `domain.base_domain` ❌

---

## 5. Priority Matrix

### Critical (Must Have Before Porting)

1. **Add `domain_dict` property to all domains**
   - Impact: Used in 40+ locations
   - Effort: Medium
   - Implementation: Wrapper around `get_parameters()` + `asdict()`

2. **Add `base_domain` property to MultibandedFrequencyDomain**
   - Impact: Required for inference, transforms, dataset generation
   - Effort: Low
   - Implementation: Construct and cache UniformFrequencyDomain

3. **Implement `build_domain()` factory function**
   - Impact: Core pattern used everywhere
   - Effort: Low
   - Implementation: Parse dict and call `from_parameters()`

4. **Add dict-accepting overloads to WaveformGenerator methods**
   - Impact: Required for existing dataset generation code
   - Effort: Medium
   - Implementation: Convert dict → WaveformParameters internally

### High Priority (Needed for Full Integration)

5. **Add `update(dict)` method compatibility**
   - Impact: Used in inference for runtime updates
   - Effort: Low
   - Implementation: Wrapper that unpacks dict to kwargs

6. **Add `domain` as property (not attribute) in WaveformGenerator**
   - Impact: Expected by some transforms
   - Effort: Low

7. **Implement `update_data()` for MultibandedFrequencyDomain**
   - Impact: Used in data preprocessing
   - Effort: Low

### Medium Priority (Nice to Have)

8. **Standardize method names** (`get_sample_frequencies_astype` → public)
9. **Add TD waveform generation** (not critical for frequency-domain use cases)
10. **Add more comprehensive type conversion utilities**

---

## 6. Compatibility Layer Strategy

### Recommended Approach: Dual API Support

Maintain dingo-waveform's improved API while adding backward compatibility:

```python
class UniformFrequencyDomain:
    # New API (keep)
    def get_parameters(self) -> DomainParameters:
        ...

    @classmethod
    def from_parameters(cls, params: DomainParameters):
        ...

    # Compatibility layer (add)
    @property
    def domain_dict(self) -> dict:
        """Backward compatibility with dingo-gw."""
        params = self.get_parameters()
        d = asdict(params)
        # Remove dingo-waveform-specific fields
        d.pop('type')  # Replace with simpler string
        d['type'] = 'UniformFrequencyDomain'
        return d

    def update(self, settings: Optional[dict] = None, **kwargs):
        """Accept both dict (dingo-gw) and kwargs (dingo-waveform)."""
        if settings is not None:
            # dict-based update (dingo-gw style)
            return self.update(**settings)
        else:
            # kwargs-based update (dingo-waveform style)
            return super().update(**kwargs)
```

### Factory Function

```python
# In domains/__init__.py
def build_domain(settings: dict) -> Domain:
    """
    Factory function for backward compatibility with dingo-gw.

    Wraps the from_parameters() approach.
    """
    domain_type = settings.get("type")

    # Create DomainParameters from dict
    params = DomainParameters(**{k: v for k, v in settings.items() if k != "type"})

    # Dispatch to appropriate class
    if domain_type in ["UniformFrequencyDomain", "FrequencyDomain", "FD"]:
        return UniformFrequencyDomain.from_parameters(params)
    elif domain_type in ["MultibandedFrequencyDomain", "MFD"]:
        return MultibandedFrequencyDomain.from_parameters(params)
    ...
```

---

## 7. Testing Strategy

After implementing compatibility layers:

1. **Unit tests**: Test both APIs work
   ```python
   # Test domain_dict compatibility
   domain_dict = domain.domain_dict
   reconstructed = build_domain(domain_dict)
   assert reconstructed == domain
   ```

2. **Integration tests**: Run existing dingo-gw tests
   - Import dingo-waveform as drop-in replacement
   - Run dingo-gw test suite
   - Check for failures

3. **Verification**: Extend `dingo-verify`
   - Add tests for serialization round-trips
   - Verify transforms work with new domains
   - Check dataset generation compatibility

---

## 8. Summary & Next Steps

### Current State
- ✅ Waveform generation: **Complete and verified**
- ✅ SVD compression: **Complete and verified**
- ⚠️ Domain API: **Incompatible, needs compatibility layer**
- ❌ Integration: **Cannot drop into dingo-gw yet**

### What's Missing
1. **Serialization compatibility** (`domain_dict`, `build_domain()`)
2. **MultibandedFrequencyDomain.base_domain** property
3. **Dict-accepting method overloads**
4. **Domain update API harmonization**

### Recommended Implementation Order
1. Add `domain_dict` property (1 day)
2. Add `build_domain()` function (1 day)
3. Add `base_domain` to MultibandedFrequencyDomain (0.5 days)
4. Add dict overloads to WaveformGenerator (1 day)
5. Add `update(dict)` compatibility (0.5 days)
6. **Testing and verification** (2 days)

**Total estimated effort**: ~6 days

### Migration Path
1. **Phase 1**: Implement compatibility layers (this document)
2. **Phase 2**: Update dingo-gw imports to use dingo-waveform
3. **Phase 3**: Gradually migrate to new APIs
4. **Phase 4**: Deprecate old patterns (optional, long-term)

---

## Appendix: File Locations

### dingo-gw
- Domains: `dingo/gw/domains/`
- WaveformGenerator: `dingo/gw/waveform_generator/waveform_generator.py`
- Domain building: `dingo/gw/domains/build_domain.py`
- Transforms: `dingo/gw/transforms/`

### dingo-waveform
- Domains: `dingo_waveform/domains/`
- WaveformGenerator: `dingo_waveform/waveform_generator.py`
- Domain parameters: `dingo_waveform/domains/domain.py`
- Waveform parameters: `dingo_waveform/waveform_parameters.py`
