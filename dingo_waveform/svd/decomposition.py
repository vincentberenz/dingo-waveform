"""Core SVD decomposition algorithms."""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg
from sklearn.utils.extmath import randomized_svd

from .config import SVDGenerationConfig
from .results import SVDDecompositionResult


def compute_svd_scipy(
    data: np.ndarray,
    n_components: int,
) -> SVDDecompositionResult:
    """Compute SVD using scipy (deterministic, slower).

    Uses scipy.linalg.svd with full_matrices=False. This method is deterministic
    but has O(mn^2) complexity for an m x n matrix.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (m x n).
    n_components : int
        Number of components to keep. If 0 or greater than available,
        keeps all components.

    Returns
    -------
    SVDDecompositionResult
        Contains V, Vh, s, n_components, and method.
    """
    U, s, Vh = scipy.linalg.svd(data, full_matrices=False)
    V = Vh.T.conj()

    # Truncate if requested
    if n_components == 0 or n_components > V.shape[1]:
        n_components = V.shape[1]
    else:
        V = V[:, :n_components]
        Vh = Vh[:n_components, :]
        s = s[:n_components]

    return SVDDecompositionResult(V=V, Vh=Vh, s=s, n_components=len(Vh), method="scipy")


def compute_svd_random(
    data: np.ndarray,
    n_components: int,
    random_state: Optional[int] = 0,
    power_iteration_normalizer: str = "QR",
) -> SVDDecompositionResult:
    """Compute SVD using randomized algorithm (faster for large data).

    Uses sklearn's randomized_svd which is much faster for small k (n_components)
    compared to full SVD. Complexity is O(mnk + k^2(m+n)) for an m x n matrix.

    Important: Requires scikit-learn <= 1.1.3 for complex-valued arrays, as
    randomized_svd in scikit-learn >= 1.2 does not support complex dtypes.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (m x n).
    n_components : int
        Number of components to keep. If 0, uses min(m, n).
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    power_iteration_normalizer : str
        Normalizer to use in power iteration. Default is "QR" which is more
        stable than "LU" (can cause segfaults with complex data).

    Returns
    -------
    SVDDecompositionResult
        Contains V, Vh, s, n_components, and method.

    Raises
    ------
    ValueError
        If randomized_svd fails, typically due to complex input with
        scikit-learn >= 1.2.
    """
    if n_components == 0:
        n_components = min(data.shape)

    try:
        U, s, Vh = randomized_svd(
            data,
            n_components,
            random_state=random_state,
            power_iteration_normalizer=power_iteration_normalizer,
        )
    except ValueError as e:
        raise ValueError(
            "randomized_svd failed — possibly due to complex-valued input.\n"
            "randomized_svd does not support complex arrays in scikit-learn >=1.2.\n"
            "To proceed, downgrade scikit-learn to version 1.1.3:\n\n"
            "    pip install scikit-learn==1.1.3\n\n"
            f"Original error: {e}"
        )

    # Ensure complex128 dtype (randomized_svd may return real for complex input)
    Vh = Vh.astype(np.complex128)
    V = Vh.T.conj()

    return SVDDecompositionResult(V=V, Vh=Vh, s=s, n_components=n_components, method="random")


def generate_svd_basis(
    training_data: np.ndarray,
    config: SVDGenerationConfig,
) -> SVDDecompositionResult:
    """Generate SVD basis from training data.

    Main entry point for SVD generation. Dispatches to scipy or randomized
    algorithm based on config.

    Parameters
    ----------
    training_data : np.ndarray
        Training data array (n_samples, n_features).
    config : SVDGenerationConfig
        Configuration specifying method, n_components, and other parameters.

    Returns
    -------
    SVDDecompositionResult
        SVD decomposition result.

    Raises
    ------
    ValueError
        If an unknown method is specified.
    """
    if config.method == "scipy":
        return compute_svd_scipy(training_data, config.n_components)
    elif config.method == "random":
        return compute_svd_random(
            training_data,
            config.n_components,
            config.random_state,
            config.power_iteration_normalizer,
        )
    else:
        raise ValueError(f"Unsupported SVD method: {config.method}")
