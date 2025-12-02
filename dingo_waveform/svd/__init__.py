"""Standalone SVD sub-package for compressing high-dimensional data.

This package provides a clean, type-safe interface for SVD-based compression
that is independent of the main dingo package.

Main Classes
------------
SVDBasis : High-level interface for all SVD operations
SVDGenerationConfig : Configuration for SVD generation
ValidationConfig : Configuration for validation
ParallelConfig : Configuration for parallel processing
SVDMetadata : Metadata associated with SVD basis

Main Functions
--------------
generate_svd_basis : Generate SVD basis from training data
generate_svd_bases_from_dict : Generate SVD bases for multiple data streams
compress : Compress data to SVD coefficients
decompress : Decompress coefficients to data
compress_dict : Compress dictionary of arrays
decompress_dict : Decompress dictionary of coefficient arrays
validate_svd : Validate SVD quality on test data
save_svd_to_hdf5 : Save SVD to HDF5 file
load_svd_from_hdf5 : Load SVD from HDF5 file
"""

from .basis import SVDBasis
from .compression import compress, compress_dict, decompress, decompress_dict
from .config import (
    ParallelConfig,
    SVDGenerationConfig,
    SVDMetadata,
    ValidationConfig,
)
from .decomposition import generate_svd_basis
from .io import load_svd_from_hdf5, save_svd_to_hdf5
from .parallel import generate_svd_bases_from_dict, parallel_map
from .results import (
    CompressionResult,
    DecompressionResult,
    SVDDecompositionResult,
    ValidationResult,
)
from .transforms import ApplySVD, ComposeTransforms, Transform
from .utils import (
    compute_explained_variance_ratio,
    estimate_reconstruction_error,
    truncate_svd,
)
from .validation import compute_mismatch, print_validation_summary, validate_svd

try:
    from .__version__ import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    # Main class
    "SVDBasis",
    # Config classes
    "SVDGenerationConfig",
    "ValidationConfig",
    "ParallelConfig",
    "SVDMetadata",
    # Result classes
    "SVDDecompositionResult",
    "ValidationResult",
    "CompressionResult",
    "DecompressionResult",
    # Transform classes
    "Transform",
    "ComposeTransforms",
    "ApplySVD",
    # Core functions
    "generate_svd_basis",
    "generate_svd_bases_from_dict",
    # Compression functions
    "compress",
    "decompress",
    "compress_dict",
    "decompress_dict",
    # Validation functions
    "validate_svd",
    "print_validation_summary",
    # I/O functions
    "save_svd_to_hdf5",
    "load_svd_from_hdf5",
    # Utility functions
    "compute_mismatch",
    "truncate_svd",
    "estimate_reconstruction_error",
    "compute_explained_variance_ratio",
    "parallel_map",
    # Version
    "__version__",
]
