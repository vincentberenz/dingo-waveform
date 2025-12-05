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


def __getattr__(name):
    """Provide backward compatibility for transform classes.

    Transform classes have been moved to dingo_waveform.transform.
    This provides deprecation warnings when importing from the old location.
    """
    import warnings

    if name == "Transform":
        warnings.warn(
            "Importing Transform from dingo_waveform.svd is deprecated. "
            "Please import DataTransform from dingo_waveform.transform instead. "
            "Note: The class has been renamed from Transform to DataTransform.",
            DeprecationWarning,
            stacklevel=2,
        )
        from dingo_waveform.transform import DataTransform
        return DataTransform
    elif name == "ComposeTransforms":
        warnings.warn(
            "Importing ComposeTransforms from dingo_waveform.svd is deprecated. "
            "Please import ComposeDataTransforms from dingo_waveform.transform instead. "
            "Note: The class has been renamed from ComposeTransforms to ComposeDataTransforms.",
            DeprecationWarning,
            stacklevel=2,
        )
        from dingo_waveform.transform import ComposeDataTransforms
        return ComposeDataTransforms
    elif name == "ApplySVD":
        warnings.warn(
            "Importing ApplySVD from dingo_waveform.svd is deprecated. "
            "Please import from dingo_waveform.transform instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from dingo_waveform.transform import ApplySVD
        return ApplySVD

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
