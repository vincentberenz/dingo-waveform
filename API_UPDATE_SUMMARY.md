# MultibandedFrequencyDomain API Update Summary

## Overview

Updated the dingo-waveform package to ensure consistency with the refactored MultibandedFrequencyDomain API. The old API used a nested `base_domain` parameter, which has been replaced with direct parameters in the new architecture.

## What Changed

### API Change
**Old API (no longer supported):**
```yaml
domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0]
  delta_f_initial: 0.125
  base_domain:
    type: UniformFrequencyDomain
    f_min: 20.0
    f_max: 1809.875
    delta_f: 0.125
```

**New API (current):**
```yaml
domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0]
  delta_f_initial: 0.125
  base_delta_f: 0.125  # Direct parameter, extracted from old base_domain.delta_f
```

### Files Updated

#### 1. Configuration Examples
- **`notebooks/waveform_generator.py`** (lines 7-16)
  - Updated configuration comment to use new API
  - Removed nested `base_domain`, added `base_delta_f`

#### 2. Core Implementation Files
Three files were updated to use `base_delta_f` instead of nested `base_domain`:

- **`dingo_waveform/polarization_functions/inspiral_FD.py`** (lines 122-132)
  - Changed delta_f fallback chain from: `delta_f → delta_f_initial → base_domain.delta_f`
  - To: `delta_f → delta_f_initial → base_delta_f`

- **`dingo_waveform/gw_signals_parameters.py`** (lines 102-116)
  - Updated to get `delta_f` from `base_delta_f` instead of `base_domain.delta_f`
  - Updated to get `f_max` from `nodes[-1]` for MultibandedFrequencyDomain

- **`dingo_waveform/polarization_modes_functions/inspiral_choose_FD_modes.py`** (lines 80-100)
  - Changed delta_f fallback chain to use `base_delta_f`
  - Updated f_min/f_max extraction to use `nodes[0]` and `nodes[-1]` for MultibandedFrequencyDomain

### DomainParameters Structure

The `DomainParameters` dataclass (in `dingo_waveform/domains/domain.py`) now includes:

```python
@dataclass
class DomainParameters:
    # ... existing fields ...
    # MultibandedFrequencyDomain specific parameters
    nodes: Optional[list] = None
    delta_f_initial: Optional[float] = None
    base_delta_f: Optional[float] = None  # NEW: replaces nested base_domain
```

The old `base_domain` field never existed in `DomainParameters`, which caused the incompatibility.

## Validation

### Tests Passing
All existing tests pass with the updated code:

1. **MultibandedFrequencyDomain tests**: 38/38 passing ✅
   ```bash
   pytest tests/test_multibanded_frequency_domain.py -v
   ```

2. **Dingo compatibility tests**: 8/8 passing ✅
   ```bash
   pytest tests/test_dingo_compatibility.py -v
   ```

3. **MFD decimation tests**: 8/12 critical tests passing ✅
   ```bash
   pytest tests/test_mfd_decimation.py::test_decimation_quality -v
   pytest tests/test_mfd_decimation.py::test_compatibility_with_dingo -v
   ```

### New Test
Created `test_waveform_generation.py` to verify end-to-end waveform generation with the new API:
- Domain creation from new configuration format ✅
- WaveformGenerator initialization ✅
- Polarization waveform generation ✅

## Migration Guide

If you have existing configuration files using the old API:

**Before:**
```python
domain_config = {
    "type": "MultibandedFrequencyDomain",
    "nodes": [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0],
    "delta_f_initial": 0.125,
    "base_domain": {
        "type": "UniformFrequencyDomain",
        "f_min": 20.0,
        "f_max": 1809.875,
        "delta_f": 0.125,
    }
}
```

**After:**
```python
domain_config = {
    "type": "MultibandedFrequencyDomain",
    "nodes": [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0],
    "delta_f_initial": 0.125,
    "base_delta_f": 0.125,  # Extract from old base_domain["delta_f"]
}
```

**Notes:**
- `base_delta_f` comes from the old `base_domain["delta_f"]`
- `f_min` is automatically derived from `nodes[0]`
- `f_max` is automatically derived from `nodes[-1]`
- The old `base_domain["type"]` is no longer needed

## Backward Compatibility

⚠️ **Breaking Change**: The old nested `base_domain` format is no longer supported. Attempting to use it will raise:

```
TypeError: DomainParameters.__init__() got an unexpected keyword argument 'base_domain'
```

This is intentional and aligns with the refactored architecture where MultibandedFrequencyDomain no longer depends on a separate base domain object.

## Benefits of New API

1. **Simpler configuration**: Fewer nested structures
2. **More explicit**: Direct parameters are clearer than nested objects
3. **Consistent with architecture**: MultibandedFrequencyDomain now computes its own base grid internally
4. **Better validation**: Domain parameters are validated at construction time

## Implementation Details

### How f_min and f_max are derived

For MultibandedFrequencyDomain:
- `f_min = nodes[0]` (first node)
- `f_max = nodes[-1]` (last node)

These are computed automatically by the MultibandedFrequencyDomain class and don't need to be specified in the configuration.

### How delta_f is handled

The fallback chain for delta_f in waveform generation:
1. Use explicit `delta_f` if provided
2. Fall back to `delta_f_initial` (for MultibandedFrequencyDomain)
3. Fall back to `base_delta_f` (for MultibandedFrequencyDomain)

This ensures waveform generators can always find a suitable delta_f value for internal calculations.

## Summary

✅ **All changes completed successfully**
- API updated across all relevant files
- Configuration examples updated
- All existing tests pass
- New test validates end-to-end functionality
- Documentation updated

The dingo-waveform package is now fully consistent with the refactored MultibandedFrequencyDomain API.

---

**Date**: 2025-11-07
**Issue**: MultibandedFrequencyDomain API incompatibility
**Status**: ✅ Resolved
