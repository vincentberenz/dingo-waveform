"""Helper utilities for SVD operations."""

from typing import Tuple

import numpy as np


def truncate_svd(
    V: np.ndarray,
    Vh: np.ndarray,
    s: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncate SVD matrices to fewer components.

    Parameters
    ----------
    V : np.ndarray
        Right singular vectors, shape (n_features, n_components_orig).
    Vh : np.ndarray
        Conjugate transpose, shape (n_components_orig, n_features).
    s : np.ndarray
        Singular values, shape (n_components_orig,).
    n_components : int
        Number of components to keep (must be <= n_components_orig).

    Returns
    -------
    V_trunc : np.ndarray
        Truncated V, shape (n_features, n_components).
    Vh_trunc : np.ndarray
        Truncated Vh, shape (n_components, n_features).
    s_trunc : np.ndarray
        Truncated singular values, shape (n_components,).
    """
    return V[:, :n_components], Vh[:n_components, :], s[:n_components]


def estimate_reconstruction_error(
    s: np.ndarray,
    n_components: int,
) -> float:
    """Estimate reconstruction error from singular values.

    The reconstruction error is approximated by the squared norm of the
    truncated singular values relative to the total squared norm:

        error ≈ ||s[n_components:]||^2 / ||s||^2

    This provides an upper bound on the relative L2 error of reconstruction.

    Parameters
    ----------
    s : np.ndarray
        All singular values, shape (n_total,).
    n_components : int
        Number of components to keep.

    Returns
    -------
    float
        Estimated relative reconstruction error (between 0 and 1).
    """
    if n_components >= len(s):
        return 0.0

    s_full_norm = np.linalg.norm(s) ** 2
    s_trunc_norm = np.linalg.norm(s[n_components:]) ** 2

    return float(s_trunc_norm / s_full_norm)


def compute_explained_variance_ratio(s: np.ndarray) -> np.ndarray:
    """Compute explained variance ratio for each component.

    The explained variance ratio indicates the fraction of total variance
    explained by keeping the first k components, for k = 1, 2, ..., n.

    Parameters
    ----------
    s : np.ndarray
        Singular values, shape (n_components,).

    Returns
    -------
    np.ndarray
        Cumulative explained variance ratio, shape (n_components,).
        Values range from 0 to 1, with the last entry equal to 1.
    """
    s_squared: np.ndarray = s**2
    cumsum = np.cumsum(s_squared)
    total: np.ndarray = np.sum(s_squared)

    return cumsum / total  # type: ignore[no-any-return]
