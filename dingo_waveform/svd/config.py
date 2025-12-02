"""Configuration dataclasses for SVD operations."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class SVDMetadata:
    """Metadata associated with an SVD basis.

    This dataclass stores optional metadata about the SVD generation process,
    training data, or other relevant information. All fields are optional
    and can be None if not applicable.

    Parameters
    ----------
    description : str, optional
        Human-readable description of this SVD basis.
    n_training_samples : int, optional
        Number of samples used to generate the SVD basis.
    n_validation_samples : int, optional
        Number of samples used for validation.
    data_shape : tuple, optional
        Original shape of the training data.
    timestamp : str, optional
        Timestamp of SVD generation (e.g., ISO format string).
    extra : dict, optional
        Additional metadata as key-value pairs. Use sparingly and only
        for truly variable metadata that doesn't fit in defined fields.
    """

    description: Optional[str] = None
    n_training_samples: Optional[int] = None
    n_validation_samples: Optional[int] = None
    data_shape: Optional[tuple] = None
    timestamp: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SVDGenerationConfig:
    """Configuration for SVD generation.

    Parameters
    ----------
    n_components : int
        Number of SVD components to keep. If 0, keeps all components.
    method : {"scipy", "random"}
        SVD method to use. "scipy" is deterministic but slower (O(mn^2)),
        "random" is faster for large data (O(mnk + k^2(m+n))).
    random_state : int, optional
        Random seed for reproducibility (only used with random method).
    power_iteration_normalizer : str
        Normalizer for randomized SVD power iteration. Default is "QR".
    """

    n_components: int
    method: Literal["scipy", "random"] = "scipy"
    random_state: Optional[int] = 0
    power_iteration_normalizer: str = "QR"


@dataclass
class ValidationConfig:
    """Configuration for SVD validation.

    Parameters
    ----------
    increment : int
        Step size for testing SVD truncations. For example, increment=50
        tests truncations at [50, 100, 150, ..., n_components].
    compute_percentiles : bool
        Whether to compute percentile statistics.
    percentiles : tuple of float
        Percentile values to compute (e.g., 99.0, 99.9, 99.99).
    """

    increment: int = 50
    compute_percentiles: bool = True
    percentiles: tuple[float, ...] = (99.0, 99.9, 99.99)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing.

    Parameters
    ----------
    num_workers : int
        Number of parallel workers. If 1, runs sequentially.
    batch_size : int
        Batch size for processing data.
    thread_limits : int
        Maximum number of threads per worker for BLAS operations.
    """

    num_workers: int = 1
    batch_size: int = 1000
    thread_limits: int = 1
