"""SVD compression and decompression operations."""

from typing import Dict

import numpy as np

from .results import CompressionResult, DecompressionResult


def compress(data: np.ndarray, V: np.ndarray) -> CompressionResult:
    """Compress data using SVD basis.

    Computes coefficients = data @ V, projecting data onto the SVD basis.

    Parameters
    ----------
    data : np.ndarray
        Data to compress. Can be 1D or 2D array.
    V : np.ndarray
        SVD basis (right singular vectors), shape (n_features, n_components).

    Returns
    -------
    CompressionResult
        Contains compressed coefficients and compression ratio.
    """
    coefficients = data @ V

    original_size = data.size
    compressed_size = coefficients.size
    compression_ratio = original_size / compressed_size

    return CompressionResult(coefficients=coefficients, compression_ratio=compression_ratio)


def decompress(coefficients: np.ndarray, Vh: np.ndarray) -> DecompressionResult:
    """Decompress coefficients using SVD basis.

    Computes data = coefficients @ Vh, reconstructing data from SVD coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Compressed coefficients. Can be 1D or 2D array.
    Vh : np.ndarray
        Conjugate transpose of SVD basis, shape (n_components, n_features).

    Returns
    -------
    DecompressionResult
        Contains reconstructed data and its shape.
    """
    data = coefficients @ Vh

    return DecompressionResult(data=data, shape=data.shape)


def compress_dict(data_dict: Dict[str, np.ndarray], V: np.ndarray) -> Dict[str, np.ndarray]:
    """Compress dictionary of arrays (convenience function).

    Useful for compressing multiple related arrays (e.g., waveform polarizations
    for different detectors) using the same SVD basis.

    Parameters
    ----------
    data_dict : dict
        Dictionary with string keys and array values.
    V : np.ndarray
        SVD basis (right singular vectors).

    Returns
    -------
    dict
        Dictionary with same keys but compressed arrays as values.
    """
    return {k: compress(v, V).coefficients for k, v in data_dict.items()}


def decompress_dict(coeff_dict: Dict[str, np.ndarray], Vh: np.ndarray) -> Dict[str, np.ndarray]:
    """Decompress dictionary of coefficient arrays (convenience function).

    Parameters
    ----------
    coeff_dict : dict
        Dictionary with string keys and coefficient array values.
    Vh : np.ndarray
        Conjugate transpose of SVD basis.

    Returns
    -------
    dict
        Dictionary with same keys but decompressed arrays as values.
    """
    return {k: decompress(v, Vh).data for k, v in coeff_dict.items()}
