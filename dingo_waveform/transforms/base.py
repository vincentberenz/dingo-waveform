"""Base classes for waveform transforms."""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from ..polarizations import BatchPolarizations


class Transform(ABC):
    """
    Abstract base class for waveform transforms.

    Transforms operate on polarization dictionaries, applying operations
    like compression, whitening, or other preprocessing steps.
    """

    @abstractmethod
    def __call__(self, polarizations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply the transform to polarizations.

        Parameters
        ----------
        polarizations
            Dictionary with keys like 'h_plus', 'h_cross' containing
            waveform arrays of shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        Transformed polarizations dictionary with the same keys
        """
        pass


class ComposeTransforms:
    """
    Compose multiple transforms into a pipeline.

    Similar to torchvision.transforms.Compose, applies a sequence of
    transforms in order.

    Parameters
    ----------
    transforms
        List of Transform instances to apply in sequence

    Example
    -------
    >>> from dingo_waveform.transforms import ComposeTransforms, ApplySVD
    >>> from dingo_waveform.compression import SVDBasis
    >>> basis = SVDBasis.load("svd_basis.hdf5")
    >>> pipeline = ComposeTransforms([
    ...     ApplySVD(basis, inverse=False),
    ... ])
    >>> compressed = pipeline(polarizations)
    """

    def __init__(self, transforms: List[Transform]):
        """
        Initialize transform composition.

        Parameters
        ----------
        transforms
            List of transforms to compose
        """
        self.transforms = transforms

    def __call__(self, polarizations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply all transforms in sequence.

        Parameters
        ----------
        polarizations
            Input polarizations dictionary

        Returns
        -------
        Transformed polarizations after applying all transforms
        """
        result = polarizations
        for transform in self.transforms:
            result = transform(result)
        return result

    def __repr__(self) -> str:
        """String representation showing the transform pipeline."""
        transform_strs = [f"  {t.__class__.__name__}" for t in self.transforms]
        return f"ComposeTransforms([\n" + ",\n".join(transform_strs) + "\n])"
