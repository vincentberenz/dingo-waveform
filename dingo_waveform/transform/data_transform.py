"""Generic data transform framework for preprocessing pipelines.

This module provides a generic transform abstraction for building preprocessing
pipelines, similar to torchvision.transforms. Transforms operate on dictionaries
mapping string keys to numpy arrays, making them suitable for multi-field data
like gravitational wave polarizations, multi-detector data, etc.

Note: This is a simple data transform framework separate from the main
Transform class (transform.py), which handles complex inference pipelines.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class DataTransform(ABC):
    """Abstract base class for data transforms.

    Transforms operate on dictionaries mapping string keys to numpy arrays.
    This is a generic pattern useful for any multi-field data (e.g.,
    polarizations, detector channels, etc.).

    Example
    -------
    >>> class MyTransform(DataTransform):
    ...     def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    ...         return {key: value * 2 for key, value in data.items()}
    """

    @abstractmethod
    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply transform to data dictionary.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dictionary mapping field names to arrays.
            Arrays can be 1D (single sample) or 2D (batch of samples).

        Returns
        -------
        Dict[str, np.ndarray]
            Transformed data dictionary with same keys as input.
        """
        pass


class ComposeDataTransforms:
    """Compose multiple transforms into a pipeline.

    Similar to torchvision.transforms.Compose, applies a sequence of
    transforms in order, passing the output of one as input to the next.

    Parameters
    ----------
    transforms : List[DataTransform]
        List of DataTransform instances to apply in sequence.

    Example
    -------
    >>> from dingo_waveform.svd import SVDBasis, SVDGenerationConfig
    >>> from dingo_waveform.transform import ApplySVD, ComposeDataTransforms
    >>> import numpy as np
    >>> training_data = np.random.randn(100, 200)
    >>> from dingo_waveform.svd import SVDGenerationConfig
    >>> config = SVDGenerationConfig(n_components=50)
    >>> basis = SVDBasis.from_training_data(training_data, config)
    >>> pipeline = ComposeDataTransforms([
    ...     ApplySVD(basis, inverse=False),
    ... ])
    >>> data = {"h_plus": np.random.randn(10, 200), "h_cross": np.random.randn(10, 200)}
    >>> compressed = pipeline(data)
    >>> compressed["h_plus"].shape
    (10, 50)
    """

    def __init__(self, transforms: List[DataTransform]):
        """Initialize transform composition.

        Parameters
        ----------
        transforms : List[DataTransform]
            List of transforms to compose. Applied in order.
        """
        self.transforms = transforms

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all transforms in sequence.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Input data dictionary

        Returns
        -------
        Dict[str, np.ndarray]
            Transformed data after applying all transforms
        """
        result = data
        for transform in self.transforms:
            result = transform(result)
        return result

    def __repr__(self) -> str:
        """String representation showing the transform pipeline."""
        transform_strs = [f"  {t.__class__.__name__}" for t in self.transforms]
        return f"ComposeDataTransforms([\n" + ",\n".join(transform_strs) + "\n])"
