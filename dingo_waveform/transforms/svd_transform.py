"""SVD compression/decompression transform."""

from typing import Dict

import numpy as np

from ..compression.svd import SVDBasis
from .base import Transform


class ApplySVD(Transform):
    """
    Transform that applies SVD compression or decompression to waveforms.

    Parameters
    ----------
    svd_basis
        SVDBasis instance with trained basis
    inverse
        If False, applies compression (default).
        If True, applies decompression.

    Example
    -------
    >>> from dingo_waveform.compression import SVDBasis
    >>> from dingo_waveform.transforms import ApplySVD
    >>> basis = SVDBasis.load("svd_basis.hdf5")
    >>> compress_transform = ApplySVD(basis, inverse=False)
    >>> decompress_transform = ApplySVD(basis, inverse=True)
    >>> compressed = compress_transform(polarizations)
    >>> reconstructed = decompress_transform(compressed)
    """

    def __init__(self, svd_basis: SVDBasis, inverse: bool = False):
        """
        Initialize SVD transform.

        Parameters
        ----------
        svd_basis
            Trained SVD basis
        inverse
            Whether to apply decompression (True) or compression (False)
        """
        if svd_basis.V is None:
            raise ValueError("SVD basis has not been trained")

        self.svd_basis = svd_basis
        self.inverse = inverse

    def __call__(self, polarizations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply SVD compression or decompression.

        Parameters
        ----------
        polarizations
            Dictionary with polarization arrays

        Returns
        -------
        Transformed polarizations dictionary
        """
        func = self.svd_basis.decompress if self.inverse else self.svd_basis.compress
        return {key: func(value) for key, value in polarizations.items()}

    def __repr__(self) -> str:
        """String representation."""
        mode = "decompress" if self.inverse else "compress"
        return f"ApplySVD(n_components={self.svd_basis.n_components}, mode={mode})"
