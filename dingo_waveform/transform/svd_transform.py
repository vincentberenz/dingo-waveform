"""SVD-based compression/decompression transform.

This module provides the ApplySVD transform for applying Singular Value
Decomposition (SVD) compression or decompression to waveform data. The transform
operates on dictionaries containing multiple data streams (e.g., h_plus and h_cross
polarizations), compressing or decompressing each stream independently.

SVD compression reduces the dimensionality of waveform data while preserving the
most important features, making it useful for efficient storage and processing
in gravitational wave inference pipelines.
"""

from typing import Dict

import numpy as np

from .data_transform import DataTransform


class ApplySVD(DataTransform):
    """Transform that applies SVD compression or decompression.

    Operates on dictionaries by applying compress() or decompress()
    to each field independently. This is a convenience wrapper around
    SVDBasis.compress_dict() and SVDBasis.decompress_dict().

    The transform can work in two modes:
    - Compression (inverse=False): Reduces dimensionality from full space to latent space
    - Decompression (inverse=True): Restores dimensionality from latent space to full space

    Parameters
    ----------
    svd_basis : SVDBasis
        Trained SVD basis to use for compression/decompression.
        The basis must be trained (V matrix must be computed) before use.
    inverse : bool, optional
        Operation mode:
            - False (default): Apply compression (reduce dimensionality)
            - True: Apply decompression (restore dimensionality)

    Raises
    ------
    ValueError
        If svd_basis is not trained (V is None)

    Notes
    -----
    - Each stream in the data dictionary is processed independently
    - Input arrays can be 1D (single sample) or 2D (batch of samples)
    - Compression: (n_features,) -> (n_components,) or
                  (n_samples, n_features) -> (n_samples, n_components)
    - Decompression: (n_components,) -> (n_features,) or
                    (n_samples, n_components) -> (n_samples, n_features)

    See Also
    --------
    dingo_waveform.svd.SVDBasis : SVD basis class
    dingo_waveform.transform.ComposeDataTransforms : Compose multiple transforms
    dingo_waveform.transform.WhitenUnwhitenTransform : Whitening transform

    Examples
    --------
    Basic compression and decompression:

    >>> from dingo_waveform.svd import SVDBasis, SVDGenerationConfig
    >>> from dingo_waveform.transform import ApplySVD
    >>> import numpy as np
    >>>
    >>> # Train SVD basis
    >>> training_data = np.random.randn(1000, 500)
    >>> config = SVDGenerationConfig(n_components=100)
    >>> basis = SVDBasis.from_training_data(training_data, config)
    >>>
    >>> # Create compression and decompression transforms
    >>> compress = ApplySVD(basis, inverse=False)
    >>> decompress = ApplySVD(basis, inverse=True)
    >>>
    >>> # Apply to polarization data
    >>> data = {"h_plus": np.random.randn(10, 500), "h_cross": np.random.randn(10, 500)}
    >>> compressed = compress(data)
    >>> compressed["h_plus"].shape
    (10, 100)
    >>> reconstructed = decompress(compressed)
    >>> reconstructed["h_plus"].shape
    (10, 500)

    Using in a pipeline with whitening:

    >>> from dingo_waveform.transform import ComposeDataTransforms, WhitenUnwhitenTransform
    >>> from dingo_waveform.domains import UniformFrequencyDomain
    >>>
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>>
    >>> # Compression pipeline: whiten -> compress
    >>> compress_pipeline = ComposeDataTransforms([
    ...     WhitenUnwhitenTransform(domain, "asd.hdf5", inverse=False),
    ...     ApplySVD(basis, inverse=False)
    ... ])
    >>>
    >>> # Decompression pipeline: decompress -> unwhiten
    >>> decompress_pipeline = ComposeDataTransforms([
    ...     ApplySVD(basis, inverse=True),
    ...     WhitenUnwhitenTransform(domain, "asd.hdf5", inverse=True)
    ... ])
    >>>
    >>> compressed = compress_pipeline(polarizations)
    >>> reconstructed = decompress_pipeline(compressed)
    """

    def __init__(self, svd_basis, inverse: bool = False):
        """Initialize SVD transform.

        Parameters
        ----------
        svd_basis : SVDBasis
            Trained SVD basis to use. Must have V matrix computed.
        inverse : bool, optional
            Whether to apply decompression (True) or compression (False).
            Default is False (compression).

        Raises
        ------
        ValueError
            If svd_basis is not trained (V is None)
        """
        # Import here to avoid circular dependency
        from dingo_waveform.svd import SVDBasis

        if not isinstance(svd_basis, SVDBasis):
            raise TypeError(
                f"svd_basis must be an SVDBasis instance, got {type(svd_basis)}"
            )

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
