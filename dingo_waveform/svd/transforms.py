"""Transform framework for data preprocessing pipelines.

This module provides a generic transform abstraction for building preprocessing
pipelines, similar to torchvision.transforms. Transforms operate on dictionaries
mapping string keys to numpy arrays, making them suitable for multi-field data
like gravitational wave polarizations, multi-detector data, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from .basis import SVDBasis


class Transform(ABC):
    """Abstract base class for data transforms.

    Transforms operate on dictionaries mapping string keys to numpy arrays.
    This is a generic pattern useful for any multi-field data (e.g.,
    polarizations, detector channels, etc.).

    Example
    -------
    >>> class MyTransform(Transform):
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


class ComposeTransforms:
    """Compose multiple transforms into a pipeline.

    Similar to torchvision.transforms.Compose, applies a sequence of
    transforms in order, passing the output of one as input to the next.

    Parameters
    ----------
    transforms : List[Transform]
        List of Transform instances to apply in sequence.

    Example
    -------
    >>> from dingo_svd import SVDBasis, SVDGenerationConfig, ApplySVD, ComposeTransforms
    >>> import numpy as np
    >>> training_data = np.random.randn(100, 200)
    >>> config = SVDGenerationConfig(n_components=50)
    >>> basis = SVDBasis.from_training_data(training_data, config)
    >>> pipeline = ComposeTransforms([
    ...     ApplySVD(basis, inverse=False),
    ... ])
    >>> data = {"h_plus": np.random.randn(10, 200), "h_cross": np.random.randn(10, 200)}
    >>> compressed = pipeline(data)
    >>> compressed["h_plus"].shape
    (10, 50)
    """

    def __init__(self, transforms: List[Transform]):
        """Initialize transform composition.

        Parameters
        ----------
        transforms : List[Transform]
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
        return f"ComposeTransforms([\n" + ",\n".join(transform_strs) + "\n])"


class ApplySVD(Transform):
    """Transform that applies SVD compression or decompression.

    Operates on dictionaries by applying compress() or decompress()
    to each field independently. This is a convenience wrapper around
    SVDBasis.compress_dict() and SVDBasis.decompress_dict().

    Parameters
    ----------
    svd_basis : SVDBasis
        Trained SVD basis to use for compression/decompression
    inverse : bool, optional
        If False, compress (reduce dimensionality). Default.
        If True, decompress (restore dimensionality).

    Raises
    ------
    ValueError
        If svd_basis is not trained (V is None)

    Example
    -------
    >>> from dingo_svd import SVDBasis, SVDGenerationConfig, ApplySVD
    >>> import numpy as np
    >>> training_data = np.random.randn(100, 200)
    >>> config = SVDGenerationConfig(n_components=50)
    >>> basis = SVDBasis.from_training_data(training_data, config)
    >>> compress_transform = ApplySVD(basis, inverse=False)
    >>> decompress_transform = ApplySVD(basis, inverse=True)
    >>> data = {"h_plus": np.random.randn(10, 200), "h_cross": np.random.randn(10, 200)}
    >>> compressed = compress_transform(data)
    >>> compressed["h_plus"].shape
    (10, 50)
    >>> reconstructed = decompress_transform(compressed)
    >>> reconstructed["h_plus"].shape
    (10, 200)
    """

    def __init__(self, svd_basis: SVDBasis, inverse: bool = False):
        """Initialize SVD transform.

        Parameters
        ----------
        svd_basis : SVDBasis
            Trained SVD basis to use
        inverse : bool, optional
            Whether to apply decompression (True) or compression (False).
            Default is False (compression).

        Raises
        ------
        ValueError
            If svd_basis is not trained (V is None)
        """
        if svd_basis.V is None:
            raise ValueError("SVD basis must be trained before use in transforms")

        self.svd_basis = svd_basis
        self.inverse = inverse

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply SVD compression or decompression to data.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dictionary of arrays to transform.
            Each array can be 1D (n_features,) or 2D (n_samples, n_features).

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with same keys, transformed arrays.
            If compressing: (n_features,) -> (n_components,) or
                          (n_samples, n_features) -> (n_samples, n_components)
            If decompressing: (n_components,) -> (n_features,) or
                            (n_samples, n_components) -> (n_samples, n_features)
        """
        if self.inverse:
            # Decompress: coefficients -> full representation
            return self.svd_basis.decompress_dict(data)
        else:
            # Compress: full representation -> coefficients
            return self.svd_basis.compress_dict(data)

    def __repr__(self) -> str:
        """String representation."""
        mode = "decompress" if self.inverse else "compress"
        return f"ApplySVD(n_components={self.svd_basis.n_components}, mode={mode})"
