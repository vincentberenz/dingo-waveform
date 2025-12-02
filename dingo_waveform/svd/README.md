# Dingo SVD Sub-package

A standalone SVD (Singular Value Decomposition) sub-package for compressing high-dimensional data.

## Features

- **Zero Dependencies on Dingo**: Independent of `dingo.core`, `dingo.gw`, and `dingo.pipe`
- **Type-Safe**: Full type hints throughout using dataclasses
- **Clean API**: High-level `SVDBasis` class for common operations
- **Transform Framework**: Generic preprocessing pipeline abstraction
- **HDF5 I/O**: Compatible with dingo file format
- **Parallel Processing**: Built-in multiprocessing utilities
- **Generic Validation**: Domain-agnostic mismatch computation

## Quick Start

```python
import numpy as np
from dingo_waveform.svd import SVDBasis, SVDGenerationConfig

# Generate training data
training_data = np.random.randn(1000, 500)

# Create and generate SVD basis
basis = SVDBasis()
config = SVDGenerationConfig(n_components=100, method="scipy")
basis.generate(training_data, config)

# Compress and decompress data
data = np.random.randn(10, 500)
compressed = basis.compress(data)
reconstructed = basis.decompress(compressed)

# Save and load
basis.save("svd.h5")
basis2 = SVDBasis()
basis2.load("svd.h5")
```

## API Overview

### Main Classes

- **`SVDBasis`**: High-level interface for all SVD operations
- **`SVDGenerationConfig`**: Configuration for SVD generation
- **`ValidationConfig`**: Configuration for validation
- **`ParallelConfig`**: Configuration for parallel processing
- **`Transform`**: Abstract base class for data transforms
- **`ComposeTransforms`**: Pipeline composition for chaining transforms
- **`ApplySVD`**: Transform for SVD compression/decompression

### Core Functions

- **`generate_svd_basis()`**: Generate SVD basis from training data
- **`compress()` / `decompress()`**: Compress/decompress individual arrays
- **`compress_dict()` / `decompress_dict()`**: Compress/decompress dictionaries
- **`validate_svd()`**: Validate SVD quality on test data
- **`save_svd_to_hdf5()` / `load_svd_from_hdf5()`**: HDF5 I/O

### Utilities

- **`compute_mismatch()`**: Compute reconstruction mismatch
- **`truncate_svd()`**: Truncate SVD to fewer components
- **`estimate_reconstruction_error()`**: Estimate error from singular values
- **`compute_explained_variance_ratio()`**: Compute explained variance
- **`parallel_map()`**: Apply function in parallel with thread limits

## Transform Framework

The transform framework provides a generic abstraction for building preprocessing pipelines. Transforms operate on dictionaries mapping string keys to numpy arrays, making them suitable for multi-field data.

### Basic Usage

```python
from dingo_waveform.svd import SVDBasis, SVDGenerationConfig, ApplySVD, ComposeTransforms
import numpy as np

# Create SVD basis
training_data = np.random.randn(1000, 500)
config = SVDGenerationConfig(n_components=100, method="scipy")
basis = SVDBasis.from_training_data(training_data, config)

# Create transform pipeline
pipeline = ComposeTransforms([
    ApplySVD(basis, inverse=False),  # Compress
])

# Apply to data
data = {"field1": np.random.randn(10, 500), "field2": np.random.randn(10, 500)}
compressed = pipeline(data)
# compressed["field1"].shape == (10, 100)
```

### Custom Transforms

```python
from dingo_waveform.svd import Transform

class NormalizeTransform(Transform):
    """Normalize each array to unit norm."""

    def __call__(self, data):
        return {
            key: value / (np.linalg.norm(value) + 1e-10)
            for key, value in data.items()
        }

# Use in pipeline
pipeline = ComposeTransforms([
    NormalizeTransform(),
    ApplySVD(basis, inverse=False),
])
```

## Methods

### SVD Generation

Two methods are available:

1. **`scipy`** (default): Deterministic, uses `scipy.linalg.svd`
   - Complexity: O(mn²) for m×n matrix
   - Suitable for smaller datasets or when reproducibility is critical

2. **`random`**: Randomized SVD using `sklearn.utils.extmath.randomized_svd`
   - Complexity: O(mnk + k²(m+n)) for k components
   - Much faster for large datasets with small k
   - **Note**: Requires scikit-learn ≤ 1.1.3 for complex-valued arrays

## File Format

HDF5 files produced by this package are compatible with dingo's file format:

- **Datasets**: `V` (right singular vectors), `s` (singular values), `mismatches` (validation)
- **Attributes**: `n_components`, `method`, `dataset_type`, `settings`

## Examples

See `examples/svd_example.py` for a comprehensive demonstration.

## Design Principles

1. **Independence**: No dependencies on other dingo modules
2. **Type Safety**: Dataclasses instead of dictionaries
3. **Clear API**: Explicit return types and configurations
4. **Compatibility**: Maintains compatibility with existing dingo SVD format
5. **Generality**: Domain-agnostic implementation

## Modules

- `basis.py`: Main `SVDBasis` class
- `config.py`: Configuration dataclasses
- `results.py`: Result dataclasses
- `decomposition.py`: Core SVD algorithms
- `compression.py`: Compression/decompression operations
- `validation.py`: Validation utilities
- `io.py`: HDF5 I/O operations
- `parallel.py`: Multiprocessing utilities
- `utils.py`: Helper functions

## Future Work

The next phase will involve:
1. Creating comprehensive unit tests
2. Refactoring existing dingo code to use this package
3. Adding integration tests
4. Performance benchmarking
