"""Main SVDBasis class for high-level SVD operations."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .compression import compress, compress_dict, decompress, decompress_dict
from .config import SVDGenerationConfig, SVDMetadata, ValidationConfig
from .decomposition import generate_svd_basis
from .io import load_svd_from_dict, load_svd_from_hdf5, save_svd_to_dict, save_svd_to_hdf5
from .results import SVDDecompositionResult, ValidationResult
from .validation import print_validation_summary, validate_svd


@dataclass(frozen=True)
class SVDBasis:
    """Main class for SVD basis operations.

    This is an immutable value object - instances are always fully initialized
    and cannot be modified. Use factory methods to create instances.

    The SVD basis can be used to compress high-dimensional data by projecting
    it onto a lower-dimensional subspace spanned by the leading singular vectors.

    Factory Methods
    ---------------
    from_training_data : Create SVDBasis by generating from training data
    from_file : Create SVDBasis by loading from HDF5 file
    from_dict : Create SVDBasis from dictionary

    Main Operations
    ---------------
    compress : Compress data to SVD coefficients
    decompress : Decompress coefficients to data
    validate : Validate SVD quality (returns new instance with results)
    truncate : Return truncated SVDBasis (returns new instance)
    save : Save to HDF5 file
    to_dict : Convert to dictionary

    Examples
    --------
    >>> from dingo.svd import SVDBasis, SVDGenerationConfig
    >>> import numpy as np
    >>>
    >>> # Create from training data
    >>> training_data = np.random.randn(1000, 500)
    >>> config = SVDGenerationConfig(n_components=100, method="scipy")
    >>> basis = SVDBasis.from_training_data(training_data, config)
    >>>
    >>> # Compress and decompress
    >>> data = np.random.randn(10, 500)
    >>> compressed = basis.compress(data)
    >>> reconstructed = basis.decompress(compressed)
    >>>
    >>> # Save and load
    >>> basis.save("svd_basis.h5")
    >>> basis2 = SVDBasis.from_file("svd_basis.h5")
    >>>
    >>> # Immutable operations return new instances
    >>> truncated = basis.truncate(50)  # Original unchanged
    >>> validated, result = basis.validate(test_data, config)
    """

    _result: SVDDecompositionResult
    _mismatches: Optional[pd.DataFrame] = None
    _metadata: Optional[SVDMetadata] = None

    @staticmethod
    def from_training_data(
        training_data: np.ndarray,
        config: SVDGenerationConfig,
        metadata: Optional[SVDMetadata] = None,
    ) -> "SVDBasis":
        """Create SVDBasis by generating from training data.

        Parameters
        ----------
        training_data : np.ndarray
            Training data, shape (n_samples, n_features).
        config : SVDGenerationConfig
            Configuration for SVD generation (method, n_components, etc.).
        metadata : SVDMetadata, optional
            Metadata to store with the SVD basis.

        Returns
        -------
        SVDBasis
            Fully initialized SVD basis.
        """
        print(
            f"Generating SVD basis with {config.n_components} components using {config.method} method"
        )
        result = generate_svd_basis(training_data, config)
        print(f"SVD basis generated: {result.n_components} components, shape {result.V.shape}")
        return SVDBasis(_result=result, _metadata=metadata)

    @staticmethod
    def from_file(filename: str) -> "SVDBasis":
        """Create SVDBasis by loading from HDF5 file.

        Parameters
        ----------
        filename : str
            Input file path.

        Returns
        -------
        SVDBasis
            Fully initialized SVD basis.
        """
        result, mismatches, metadata = load_svd_from_hdf5(filename)
        return SVDBasis(_result=result, _mismatches=mismatches, _metadata=metadata)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SVDBasis":
        """Create SVDBasis from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with SVD data.

        Returns
        -------
        SVDBasis
            Fully initialized SVD basis.
        """
        result, mismatches = load_svd_from_dict(d)
        return SVDBasis(_result=result, _mismatches=mismatches)

    @property
    def V(self) -> np.ndarray:
        """Right singular vectors (n_features, n_components)."""
        return self._result.V

    @property
    def Vh(self) -> np.ndarray:
        """Conjugate transpose of right singular vectors (n_components, n_features)."""
        return self._result.Vh

    @property
    def s(self) -> np.ndarray:
        """Singular values (n_components,), in descending order."""
        return self._result.s

    @property
    def n_components(self) -> int:
        """Number of SVD components."""
        return self._result.n_components

    @property
    def method(self) -> str:
        """SVD method used ('scipy' or 'random')."""
        return self._result.method

    @property
    def mismatches(self) -> Optional[pd.DataFrame]:
        """Validation mismatches DataFrame, if validation has been performed."""
        return self._mismatches

    @property
    def metadata(self) -> Optional[SVDMetadata]:
        """Metadata associated with this SVD basis."""
        return self._metadata

    def validate(
        self,
        test_data: np.ndarray,
        config: ValidationConfig,
        labels: Optional[pd.DataFrame] = None,
        verbose: bool = False,
    ) -> tuple["SVDBasis", ValidationResult]:
        """Validate SVD basis on test data.

        Computes reconstruction mismatches at different truncation levels
        to assess SVD quality. Returns a new SVDBasis with validation results.

        Parameters
        ----------
        test_data : np.ndarray
            Test data for validation, shape (n_samples, n_features).
        config : ValidationConfig
            Validation configuration.
        labels : pd.DataFrame, optional
            Labels for test data (e.g., physical parameters).
        verbose : bool
            Whether to print summary statistics.

        Returns
        -------
        tuple[SVDBasis, ValidationResult]
            New SVDBasis with validation results, and ValidationResult.
        """
        print("Validating SVD basis...")
        result = validate_svd(self.V, self.Vh, test_data, config, labels)

        if verbose:
            print_validation_summary(result)

        # Return new instance with validation results
        return (
            SVDBasis(_result=self._result, _mismatches=result.mismatches, _metadata=self._metadata),
            result,
        )

    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress data to SVD coefficients.

        Parameters
        ----------
        data : np.ndarray
            Data to compress, shape (..., n_features).

        Returns
        -------
        np.ndarray
            Compressed coefficients, shape (..., n_components).
        """
        return compress(data, self.V).coefficients

    def decompress(self, coefficients: np.ndarray) -> np.ndarray:
        """Decompress SVD coefficients to data.

        Parameters
        ----------
        coefficients : np.ndarray
            Compressed coefficients, shape (..., n_components).

        Returns
        -------
        np.ndarray
            Decompressed data, shape (..., n_features).
        """
        return decompress(coefficients, self.Vh).data

    def compress_dict(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compress dictionary of arrays.

        Convenience method for compressing multiple arrays using the same
        SVD basis.

        Parameters
        ----------
        data_dict : dict
            Dictionary with array values to compress.

        Returns
        -------
        dict
            Dictionary with compressed arrays.
        """
        return compress_dict(data_dict, self.V)

    def decompress_dict(self, coeff_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Decompress dictionary of coefficient arrays.

        Convenience method for decompressing multiple arrays using the same
        SVD basis.

        Parameters
        ----------
        coeff_dict : dict
            Dictionary with coefficient arrays to decompress.

        Returns
        -------
        dict
            Dictionary with decompressed arrays.
        """
        return decompress_dict(coeff_dict, self.Vh)

    def save(self, filename: str, mode: str = "w") -> None:
        """Save SVD basis to HDF5 file.

        Parameters
        ----------
        filename : str
            Output file path.
        mode : str
            File mode: "w" for write (overwrites), "a" for append.
        """
        save_svd_to_hdf5(filename, self._result, self._mismatches, self._metadata, mode)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SVD basis to dictionary.

        Useful for embedding SVD in larger files or for serialization.

        Returns
        -------
        dict
            Dictionary with SVD data.
        """
        return save_svd_to_dict(self._result, self._mismatches)

    def truncate(self, n_components: int) -> "SVDBasis":
        """Truncate SVD to fewer components.

        Returns a new SVDBasis with fewer components. The original
        instance is not modified.

        Parameters
        ----------
        n_components : int
            New number of components (must be <= current n_components).

        Returns
        -------
        SVDBasis
            New SVDBasis with truncated components.

        Raises
        ------
        ValueError
            If n_components is invalid.
        """
        if n_components > self.n_components or n_components < 1:
            raise ValueError(
                f"Cannot truncate from {self.n_components} to {n_components} components"
            )

        truncated_result = SVDDecompositionResult(
            V=self.V[:, :n_components],
            Vh=self.Vh[:n_components, :],
            s=self.s[:n_components],
            n_components=n_components,
            method=self.method,
        )
        print(f"Truncated SVD to {n_components} components")
        return SVDBasis(
            _result=truncated_result, _mismatches=self._mismatches, _metadata=self._metadata
        )
