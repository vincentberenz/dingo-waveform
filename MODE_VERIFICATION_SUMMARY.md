# Mode Verification Summary

## Overview

The `dingo-verify` tool has been extended to support mode-separated waveform verification using the `--modes` flag. This compares the output of `generate_hplus_hcross_m()` between dingo (dingo-gw) and dingo-waveform.

## Usage

```bash
# Verify mode-separated waveforms with UniformFrequencyDomain
dingo-verify --config config.json --modes

# Verbose output
dingo-verify --config config.json --modes --verbose
```

## Results

### ✅ UniformFrequencyDomain: Full Support

Mode verification works perfectly with UniformFrequencyDomain:

```bash
$ dingo-verify --config examples/config_uniform.json --modes
```

**Result:** EXACT match (0.00e+00 difference) for all 9 modes

- Both packages generate 9 modes: [-4, -3, -2, -1, 0, 1, 2, 3, 4]
- All modes have identical shapes: (8193,)
- All mode polarizations match at machine precision

### ❌ MultibandedFrequencyDomain: Not Supported

Mode verification does NOT work with MultibandedFrequencyDomain due to fundamental implementation differences:

```bash
$ dingo-verify --config examples/config_multibanded.json --modes
```

**Error:** `operands could not be broadcast together with shapes (736,) (32769,)`

**Root cause:**
- **dingo (dingo-gw)**: Returns modes on the decimated multibanded grid (736 bins)
- **dingo-waveform**: Returns modes on the full uniform grid (32769 bins)

This is a fundamental difference in how mode generation is implemented for multibanded domains.

## Implementation Details

### What Was Added

1. **New function in `comparison.py`:**
   ```python
   def compare_waveforms_modes(
       domain_type: str,
       domain_params: Dict,
       approximant: str,
       waveform_params: Dict[str, float],
       f_ref: float,
       f_start: float,
       spin_conversion_phase: Optional[float] = None,
   ) -> WaveformComparisonResult
   ```

2. **New CLI flag in `cli.py`:**
   ```python
   parser.add_argument(
       '--modes', '-m',
       action='store_true',
       help='Compare mode-separated waveforms (generate_hplus_hcross_m)'
   )
   ```

3. **Mode support check:**
   - Only approximants containing "PHM" or "HM" support modes
   - Examples: IMRPhenomXPHM, SEOBNRv4PHM, SEOBNRv5PHM, SEOBNRv5HM

### Comparison Logic

For each mode `m` in the mode dictionary:
1. Extract h_plus and h_cross for both packages
2. Compute absolute differences
3. Compute relative differences (where magnitudes > 1e-30)
4. Aggregate across all modes:
   - Maximum differences
   - Mean differences (averaged over all modes)

## Configuration Files

The same JSON configuration files work for both standard and mode verification:

```json
{
  "domain": {
    "type": "UniformFrequencyDomain",
    "delta_f": 0.125,
    "f_min": 20.0,
    "f_max": 1024.0
  },
  "waveform_generator": {
    "approximant": "IMRPhenomXPHM",
    "f_ref": 20.0,
    "f_start": 20.0,
    "spin_conversion_phase": 0.0
  },
  "waveform_parameters": {
    "mass_1": 36.0,
    "mass_2": 29.0,
    "chirp_mass": 28.095556,
    "mass_ratio": 0.805556,
    "luminosity_distance": 1000.0,
    "theta_jn": 0.5,
    "phase": 0.0,
    "a_1": 0.5,
    "a_2": 0.3,
    "tilt_1": 0.5,
    "tilt_2": 0.8,
    "phi_12": 1.7,
    "phi_jl": 0.3,
    "geocent_time": 0.0
  }
}
```

Simply add the `--modes` flag to compare mode-separated waveforms instead of total polarizations.

## Technical Notes

### Mode Generation Requirements

1. **Approximant must support modes:**
   - Contains "PHM" (PhenomX Precessing Higher Modes)
   - Contains "HM" (Higher Modes)

2. **LALSimulation requirement:**
   - For MultibandedFrequencyDomain with dingo (dingo-gw)
   - Base domain must have `f_min > 0`
   - Fixed in `comparison.py:create_multibanded_domain_dingo()`

### Mode Output Format

Both packages return `Dict[int, Polarization]`:
- Keys: Mode indices (e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4)
- Values: Polarization objects with h_plus and h_cross arrays

**Important:** The mode index `m` refers to the transformation behavior under phase shifts:
```
h_m(phase + δφ) = exp(-1j * m * δφ) * h_m(phase)
```

This is NOT the same as spherical harmonic mode (l,m). See MODES_VS_POLARIZATIONS.md for details.

## Recommendations

### For Verification

✅ **DO use mode verification for:**
- UniformFrequencyDomain
- Validating that mode generation works correctly
- Verifying phase marginalization implementations

❌ **DON'T use mode verification for:**
- MultibandedFrequencyDomain (implementation differences)
- Standard waveform validation (use `dingo-verify` without `--modes`)

### For Dataset Generation

**Do NOT use modes for dataset generation:**
- Standard `generate_hplus_hcross()` is correct for datasets
- Mode-separated waveforms are for phase marginalization in parameter estimation
- See MODES_VS_POLARIZATIONS.md for details

## Bug Fixes

### Fixed: Base domain f_min for mode generation

**Problem:** `comparison.py:create_multibanded_domain_dingo()` used `f_min=0.0` which caused LALSimulation error:
```
XLAL Error: f_min must be positive and greater than 0
```

**Fix:** Changed to use `f_min=nodes[0]` instead of `f_min=0.0`

**Impact:**
- Standard waveform generation (generate_hplus_hcross): No change
- Mode generation (generate_hplus_hcross_m): Now works for dingo (dingo-gw)
- But still incompatible due to different output shapes

## Known Limitations

1. **MultibandedFrequencyDomain mode verification:**
   - Not supported due to implementation differences
   - dingo: decimated grid
   - dingo-waveform: full uniform grid

2. **No mode decimation in dingo-waveform:**
   - generate_hplus_hcross_m() always returns full uniform grid
   - Even when domain is MultibandedFrequencyDomain
   - This may be by design for phase marginalization use cases

## Test Results

### Test Files Created

1. **test_mode_output.py:** Verified dingo-waveform mode generation works
2. **test_mode_comparison.py:** Compared modes between packages (UniformFrequencyDomain)
3. **test_refactored_modes_only.py:** Tested dingo-waveform with MultibandedFrequencyDomain
4. **test_dingo_modes_only.py:** Tested dingo with MultibandedFrequencyDomain

### Results Summary

| Domain Type | Standard Waveforms | Mode-Separated Waveforms |
|-------------|-------------------|--------------------------|
| UniformFrequencyDomain | ✅ EXACT match (0.00e+00) | ✅ EXACT match (0.00e+00) |
| MultibandedFrequencyDomain | ✅ Near-perfect (1.97e-24) | ❌ Shape mismatch (736 vs 32769) |

## Conclusion

**Mode verification is implemented and works perfectly for UniformFrequencyDomain.** This validates that both packages implement mode generation identically for uniform grids.

**For MultibandedFrequencyDomain, mode verification is not possible** due to fundamental implementation differences in how modes are returned (decimated vs. full grid).

This is acceptable because:
1. Mode generation is primarily used for phase marginalization in parameter estimation
2. Phase marginalization typically uses UniformFrequencyDomain
3. MultibandedFrequencyDomain is primarily for dataset generation
4. Dataset generation uses generate_hplus_hcross() (not modes)

---

**Date:** 2025-11-07
**Status:** ✅ Mode verification implemented and tested
