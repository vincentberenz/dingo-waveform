"""Result dataclasses for SVD operations."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SVDDecompositionResult:
    """Result of SVD decomposition.

    Attributes
    ----------
    V : np.ndarray
        Right singular vectors (m x n), where n = n_components.
    Vh : np.ndarray
        Conjugate transpose of V (n x m).
    s : np.ndarray
        Singular values (n,), sorted in descending order.
    n_components : int
        Number of components kept in the decomposition.
    method : str
        SVD method used ("scipy" or "random").
    """

    V: np.ndarray
    Vh: np.ndarray
    s: np.ndarray
    n_components: int
    method: str


@dataclass
class ValidationResult:
    """Result of SVD validation.

    Attributes
    ----------
    mismatches : pd.DataFrame
        DataFrame containing mismatch values for different truncation levels.
        Columns include parameter labels (if provided) and mismatch_n={size} columns.
    summary : dict
        Summary statistics for each truncation level.
        Keys are truncation sizes, values are dicts with statistics:
        - mean: mean mismatch
        - std: standard deviation
        - max: maximum mismatch
        - median: median mismatch
        - percentiles: dict of percentile values (if computed)
    """

    mismatches: pd.DataFrame
    summary: dict


@dataclass
class CompressionResult:
    """Result of compression operation.

    Attributes
    ----------
    coefficients : np.ndarray
        Compressed SVD coefficients.
    compression_ratio : float
        Ratio of original size to compressed size.
    """

    coefficients: np.ndarray
    compression_ratio: float


@dataclass
class DecompressionResult:
    """Result of decompression operation.

    Attributes
    ----------
    data : np.ndarray
        Reconstructed data from SVD coefficients.
    shape : tuple
        Shape of the reconstructed data array.
    """

    data: np.ndarray
    shape: tuple
