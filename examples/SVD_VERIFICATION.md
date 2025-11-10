# SVD Compression Verification

The `dingo-verify` tool has been extended to verify that SVD compression in `dingo-waveform` produces identical results to `dingo` (dingo-gw).

## Overview

SVD (Singular Value Decomposition) compression is a critical component of the dingo training pipeline. It compresses waveform datasets before feeding them into neural networks for parameter estimation. This verification ensures that the refactored SVD implementation is a drop-in replacement for the original.

## What is Verified

The SVD verification performs three levels of testing:

1. **Basis Matrix Comparison**: Compares the SVD basis matrices (V) and singular values (s) trained on the same data in both systems
2. **Reconstruction Quality**: Validates that compression/decompression yields the same reconstruction mismatches in both systems
3. **Basis Interchangeability**: Tests that compressed coefficients from one system can be decompressed by the other

## Usage

### Basic Command

```bash
dingo-verify --config config_svd.json --svd
```

### With Options

```bash
# Set random seed for reproducibility
dingo-verify --config config_svd.json --svd --seed 42

# Enable verbose output
dingo-verify --config config_svd.json --svd --verbose

# Both options
dingo-verify --config config_svd.json --svd --seed 42 --verbose
```

## Configuration File Format

SVD verification requires a different configuration format than waveform verification:

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
    "f_start": 20.0
  },
  "prior": {
    "chirp_mass": "Uniform(minimum=25.0, maximum=35.0)",
    "mass_ratio": "Uniform(minimum=0.8, maximum=1.0)",
    "luminosity_distance": 1000.0,
    "theta_jn": "Uniform(minimum=0.0, maximum=3.14159)",
    "phase": 0.0,
    "a_1": "Uniform(minimum=0.0, maximum=0.88)",
    "a_2": "Uniform(minimum=0.0, maximum=0.88)",
    "tilt_1": "Sine(minimum=0.0, maximum=3.14159)",
    "tilt_2": "Sine(minimum=0.0, maximum=3.14159)",
    "phi_12": "Uniform(minimum=0.0, maximum=6.28318)",
    "phi_jl": "Uniform(minimum=0.0, maximum=6.28318)",
    "geocent_time": 0.0
  },
  "svd": {
    "n_components": 200,
    "num_training": 1000,
    "num_validation": 100,
    "method": "scipy"
  }
}
```

### Key Differences from Waveform Verification

- **No `waveform_parameters`**: Instead, use `prior` to define parameter distributions
- **`prior` section**: Defines bilby priors for sampling waveform parameters
  - Fixed values: Use floats (e.g., `"luminosity_distance": 1000.0`)
  - Distributions: Use bilby prior strings (e.g., `"chirp_mass": "Uniform(minimum=25.0, maximum=35.0)"`)
- **`svd` section**: Configures SVD training
  - `n_components`: Number of SVD basis elements to keep
  - `num_training`: Number of waveforms for training SVD basis
  - `num_validation`: Number of waveforms for validation
  - `method`: SVD algorithm to use ("scipy" or "randomized")

## Example Configurations

Two example configurations are provided:

1. **`config_svd_small.json`**: Small test (fast, ~30 seconds)
   - 20 training waveforms
   - 10 validation waveforms
   - 50 components
   - IMRPhenomD approximant

2. **`config_svd.json`**: Production-scale test (~10-15 minutes)
   - 1000 training waveforms
   - 100 validation waveforms
   - 200 components
   - IMRPhenomXPHM approximant

## Interpreting Results

### Success Criteria

The verification passes if:
- SVD basis shapes match
- Basis matrix differences < 1e-12 (machine precision)
- Cross-system compatibility test differences < 1e-12

### Output Sections

1. **SVD Basis Properties**: Shape and size information
2. **Basis Matrix (V) Differences**: Numerical comparison of basis vectors
3. **Singular Values (s) Differences**: Comparison of singular values
4. **Reconstruction Mismatches**: Quality of compression/decompression
5. **Basis Interchangeability Test**: Cross-system compatibility

### Example Output

```
================================================================================
SVD COMPRESSION VERIFICATION RESULTS
================================================================================

SVD Basis:
  n_components:      50
  Dingo shape:       (4098, 20)
  Refactored shape:  (4098, 20)
  Shapes match:      ✅ YES

Basis Matrix (V) Differences:
  Max:               0.00e+00
  Mean:              0.00e+00
  Max relative:      0.00e+00

Reconstruction Mismatches:
  Dingo mean:        3.29e-03
  Refactored mean:   3.29e-03
  Difference:        0.00e+00

Basis Interchangeability Test:
  Max difference:    0.00e+00
  Mean difference:   0.00e+00

✅ VERIFICATION PASSED
✅ dingo-waveform SVD compression is IDENTICAL to dingo (dingo-gw)
✅ Basis matrices are interchangeable between systems
```

## Implementation Details

### What the Tool Does

1. **Sample Parameters**: Generates `num_training + num_validation` waveform parameter sets from the specified priors
2. **Generate Training Data**: Creates waveforms using both dingo and dingo-waveform
3. **Train SVD Basis**: Trains independent SVD bases in both systems using scipy.linalg.svd
4. **Compare Bases**: Numerically compares the V matrices and singular values
5. **Validate Reconstruction**: Compresses and decompresses validation waveforms, computing mismatches
6. **Test Interchangeability**: Verifies that coefficients compressed in one system can be decompressed in the other

### Why Zero Differences?

Both systems use the same underlying scipy.linalg.svd implementation and identical data preprocessing, resulting in bit-for-bit identical results. This confirms the refactor is truly a drop-in replacement.

## When to Use This

Use SVD verification when:
- Validating changes to SVD compression code
- Testing a new domain implementation with compression
- Verifying that file format changes preserve compression compatibility
- Demonstrating scientific correctness to reviewers

This complements waveform verification, which tests waveform generation, and together they provide comprehensive verification of the dingo-waveform refactor.
