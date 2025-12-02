"""Multiprocessing utilities for SVD operations."""

from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .basis import SVDBasis
from .config import ParallelConfig, SVDGenerationConfig, ValidationConfig


def parallel_map(
    func: Callable,
    items: List[Any],
    config: ParallelConfig,
) -> List[Any]:
    """Apply function to items in parallel.

    Uses multiprocessing.Pool for parallel execution with thread limits
    to avoid nested threading issues with BLAS libraries.

    Parameters
    ----------
    func : callable
        Function to apply to each item. Must be picklable.
    items : list
        Items to process.
    config : ParallelConfig
        Parallel processing configuration.

    Returns
    -------
    list
        Results from applying func to each item.
    """
    if config.num_workers <= 1:
        # Sequential processing
        return [func(item) for item in items]

    # Parallel processing with thread limits to avoid nested threading
    with threadpool_limits(limits=config.thread_limits, user_api="blas"):
        with Pool(processes=config.num_workers) as pool:
            results = pool.map(func, items)

    return results


class ParallelSVDGenerator:
    """Helper for generating SVD basis in parallel from batched data.

    This class is useful when data comes from a DataLoader or needs to be
    processed in batches before generating the SVD basis. It accumulates
    batches in a buffer and then concatenates them for SVD generation.

    Examples
    --------
    >>> from dingo.svd import ParallelSVDGenerator, ParallelConfig
    >>> import numpy as np
    >>>
    >>> # Create generator
    >>> config = ParallelConfig(num_workers=4, batch_size=100)
    >>> generator = ParallelSVDGenerator(config)
    >>>
    >>> # Add batches of data
    >>> for i in range(10):
    ...     batch = np.random.randn(100, 500)
    ...     generator.add_batch(batch)
    >>>
    >>> # Concatenate all batches
    >>> all_data = generator.concatenate()
    >>> print(all_data.shape)  # (1000, 500)
    >>>
    >>> # Clear buffer for next use
    >>> generator.clear()
    """

    def __init__(self, config: ParallelConfig):
        """Initialize parallel SVD generator.

        Parameters
        ----------
        config : ParallelConfig
            Configuration for parallel processing.
        """
        self.config = config
        self.data_buffer: List[np.ndarray] = []

    def add_batch(self, batch: np.ndarray) -> None:
        """Add a batch of data to the buffer.

        Parameters
        ----------
        batch : np.ndarray
            Batch of data to add, shape (batch_size, n_features).
        """
        self.data_buffer.append(batch)

    def concatenate(self) -> np.ndarray:
        """Concatenate all batches into a single array.

        Returns
        -------
        np.ndarray
            Concatenated data, shape (total_samples, n_features).

        Raises
        ------
        ValueError
            If no data has been added to the buffer.
        """
        if not self.data_buffer:
            raise ValueError("No data in buffer")
        return np.vstack(self.data_buffer)

    def clear(self) -> None:
        """Clear the data buffer."""
        self.data_buffer = []

    def __len__(self) -> int:
        """Return number of batches in buffer."""
        return len(self.data_buffer)


def generate_svd_bases_from_dict(
    data_dict: Dict[str, np.ndarray],
    svd_config: SVDGenerationConfig,
    num_training_samples: Optional[int] = None,
    num_validation_samples: Optional[int] = None,
    validation_config: Optional[ValidationConfig] = None,
    labels: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> Dict[str, SVDBasis]:
    """Generate SVD bases for multiple data streams (e.g., interferometers).

    This is a generic function that generates one SVD basis per key in the input
    dictionary. It supports both complex and real-valued data arrays.

    The function can optionally split data into training and validation sets,
    compute validation mismatches, and return SVDBasis objects with validation
    results.

    Parameters
    ----------
    data_dict : Dict[str, np.ndarray]
        Dictionary mapping names (e.g., 'H1', 'L1') to data arrays.
        Each array should have shape (n_samples, n_features).
        Supports complex or real dtypes.
    svd_config : SVDGenerationConfig
        Configuration for SVD generation (method, n_components, etc.).
    num_training_samples : int, optional
        Number of samples to use for training. If None, uses all samples.
        If provided with num_validation_samples, the data will be split.
    num_validation_samples : int, optional
        Number of samples to use for validation. Only used if num_training_samples
        is also provided. These samples come after the training samples.
    validation_config : ValidationConfig, optional
        Configuration for validation. If provided along with num_validation_samples,
        validation will be performed and results stored in the SVDBasis objects.
    labels : pd.DataFrame, optional
        Labels/parameters for the samples. If provided with validation, the
        validation subset will be attached to the validation results.
    verbose : bool
        Whether to print detailed progress information.

    Returns
    -------
    Dict[str, SVDBasis]
        Dictionary mapping names to SVDBasis objects. If validation was requested,
        the SVDBasis objects will contain validation results in their mismatches
        attribute.

    Raises
    ------
    ValueError
        If num_training_samples + num_validation_samples exceeds available data.
        If validation_config is provided but num_validation_samples is not.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.svd import generate_svd_bases_from_dict, SVDGenerationConfig
    >>>
    >>> # Generate complex waveform data for two interferometers
    >>> h1_data = np.random.randn(1000, 500) + 1j * np.random.randn(1000, 500)
    >>> l1_data = np.random.randn(1000, 500) + 1j * np.random.randn(1000, 500)
    >>> data = {'H1': h1_data, 'L1': l1_data}
    >>>
    >>> # Generate SVD bases without validation
    >>> config = SVDGenerationConfig(n_components=100, method='scipy')
    >>> bases = generate_svd_bases_from_dict(data, config)
    >>>
    >>> # With training/validation split
    >>> bases = generate_svd_bases_from_dict(
    ...     data, config,
    ...     num_training_samples=800,
    ...     num_validation_samples=200,
    ...     validation_config=ValidationConfig(increment=25),
    ...     verbose=True
    ... )
    >>> print(bases['H1'].mismatches)  # Validation results

    Notes
    -----
    - This function is domain-agnostic and does not depend on gravitational wave
      specific code.
    - The training/validation split is sequential: first num_training_samples
      are used for training, next num_validation_samples for validation.
    - If validation is performed, it tests reconstruction quality at different
      truncation levels as specified in validation_config.
    """
    # Validate inputs
    if validation_config is not None and num_validation_samples is None:
        raise ValueError(
            "validation_config provided but num_validation_samples is None. "
            "Please specify num_validation_samples to perform validation."
        )

    # Handle empty data dictionary
    if not data_dict:
        if verbose:
            print("Empty data dictionary provided, returning empty result.")
        return {}

    # Check that all arrays have the same number of samples
    names = list(data_dict.keys())
    n_samples_list = [data_dict[name].shape[0] for name in names]
    if len(set(n_samples_list)) > 1:
        raise ValueError(
            f"All arrays must have the same number of samples. "
            f"Got: {dict(zip(names, n_samples_list))}"
        )
    n_samples = n_samples_list[0]

    # Determine training and validation split
    if num_training_samples is None:
        num_training_samples = n_samples
        num_validation_samples = 0
    else:
        if num_validation_samples is None:
            num_validation_samples = 0

    total_samples = num_training_samples + num_validation_samples
    if total_samples > n_samples:
        raise ValueError(
            f"Requested {num_training_samples} training + {num_validation_samples} "
            f"validation samples = {total_samples} total, but only {n_samples} "
            f"samples available."
        )

    if verbose:
        print(f"Generating SVD bases for {len(names)} data streams: {names}")
        print(f"  Training samples: {num_training_samples}")
        if num_validation_samples > 0:
            print(f"  Validation samples: {num_validation_samples}")

    # Generate SVD basis for each data stream
    basis_dict = {}
    for name in names:
        if verbose:
            print(f"\nProcessing '{name}':")

        data = data_dict[name]
        training_data = data[:num_training_samples]

        # Generate SVD basis
        basis = SVDBasis.from_training_data(training_data, svd_config)

        # Perform validation if requested
        if num_validation_samples > 0 and validation_config is not None:
            validation_data = data[
                num_training_samples : num_training_samples + num_validation_samples
            ]

            # Extract validation labels if provided
            validation_labels = None
            if labels is not None:
                validation_labels = labels.iloc[
                    num_training_samples : num_training_samples + num_validation_samples
                ].reset_index(drop=True)

            # Validate and get new basis with validation results
            basis, val_result = basis.validate(
                validation_data,
                validation_config,
                labels=validation_labels,
                verbose=verbose,
            )

        basis_dict[name] = basis

    if verbose:
        print(f"\nDone generating {len(basis_dict)} SVD bases.")

    return basis_dict
