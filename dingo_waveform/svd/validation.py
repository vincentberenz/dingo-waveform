"""Generic validation utilities for SVD basis."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import ValidationConfig
from .results import ValidationResult


def compute_mismatch(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Compute mismatch (1 - overlap) between original and reconstructed data.

    The mismatch is defined as:
        mismatch = 1 - Re(<original, reconstructed>) / (||original|| * ||reconstructed||)

    where <·,·> is the inner product and ||·|| is the L2 norm. This is a
    measure of how much information is lost in the compression/reconstruction.
    A mismatch of 0 indicates perfect reconstruction, while 1 indicates
    complete loss of information.

    Parameters
    ----------
    original : np.ndarray
        Original data vector.
    reconstructed : np.ndarray
        Reconstructed data vector.

    Returns
    -------
    float
        Mismatch value between 0 and 1.
    """
    norm1 = np.sqrt(np.sum(np.abs(original) ** 2))
    norm2 = np.sqrt(np.sum(np.abs(reconstructed) ** 2))
    inner: float = float(np.sum(original.conj() * reconstructed).real)
    return float(1.0 - inner / (norm1 * norm2))


def validate_svd(
    V: np.ndarray,
    Vh: np.ndarray,
    test_data: np.ndarray,
    config: ValidationConfig,
    labels: Optional[pd.DataFrame] = None,
) -> ValidationResult:
    """Validate SVD basis on test data.

    Computes mismatches at different truncation levels to assess how well
    the SVD basis reconstructs test data as a function of the number of
    components used.

    Parameters
    ----------
    V : np.ndarray
        Right singular vectors, shape (n_features, n_components).
    Vh : np.ndarray
        Conjugate transpose, shape (n_components, n_features).
    test_data : np.ndarray
        Test data for validation, shape (n_samples, n_features).
    config : ValidationConfig
        Validation configuration (increment, percentiles, etc.).
    labels : pd.DataFrame, optional
        Labels for test data (e.g., physical parameters). If provided,
        will be included in the output DataFrame.

    Returns
    -------
    ValidationResult
        Contains mismatches DataFrame and summary statistics.

    Raises
    ------
    ValueError
        If test_data and labels have different lengths.
    """
    n_max = V.shape[1]

    # Initialize mismatches DataFrame with labels if provided
    if labels is not None:
        if len(test_data) != len(labels):
            raise ValueError(
                f"Length mismatch: test_data has {len(test_data)} samples, "
                f"labels has {len(labels)} samples"
            )
        mismatches_df = labels.copy()
    else:
        mismatches_df = pd.DataFrame()

    # Compute mismatches at incremental truncations
    truncations = np.append(np.arange(config.increment, n_max, config.increment), n_max)

    for n in truncations:
        mismatches = np.empty(len(test_data))
        for i, data in enumerate(test_data):
            # Compress and decompress using truncated SVD
            compressed = data @ V[:, :n]
            reconstructed = compressed @ Vh[:n]
            mismatches[i] = compute_mismatch(data, reconstructed)
        mismatches_df[f"mismatch_n={n}"] = mismatches

    # Compute summary statistics for each truncation
    summary: Dict[int, Dict[str, Any]] = {}
    for col in mismatches_df.columns:
        if col.startswith("mismatch_n="):
            n = int(col.split("=")[1])
            vals = mismatches_df[col]
            summary[n] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
            }
            if config.compute_percentiles:
                summary[n]["percentiles"] = {
                    p: float(np.percentile(vals, p)) for p in config.percentiles
                }

    return ValidationResult(mismatches=mismatches_df, summary=summary)


def print_validation_summary(result: ValidationResult) -> None:
    """Print formatted validation summary.

    Prints mean, standard deviation, max, median, and percentile statistics
    for each truncation level.

    Parameters
    ----------
    result : ValidationResult
        Validation result to print.
    """
    for n, stats in result.summary.items():
        print(f"n = {n}")
        print(f"  Mean mismatch = {stats['mean']}")
        print(f"  Standard deviation = {stats['std']}")
        print(f"  Max mismatch = {stats['max']}")
        print(f"  Median mismatch = {stats['median']}")
        if "percentiles" in stats:
            print("  Percentiles:")
            for p, val in stats["percentiles"].items():
                print(f"    {p:5.2f}  -> {val}")
