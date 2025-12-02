"""HDF5 I/O operations for SVD basis."""

import ast
from dataclasses import asdict
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd

from .config import SVDMetadata
from .results import SVDDecompositionResult


def save_svd_to_hdf5(
    filename: str,
    result: SVDDecompositionResult,
    mismatches: Optional[pd.DataFrame] = None,
    metadata: Optional[SVDMetadata] = None,
    mode: str = "w",
) -> None:
    """Save SVD decomposition result to HDF5 file.

    Creates an HDF5 file with datasets for V and s arrays, and optional
    mismatches DataFrame. Metadata (n_components, method, metadata) are
    stored as attributes for compatibility with dingo's file format.

    Parameters
    ----------
    filename : str
        Output HDF5 file path.
    result : SVDDecompositionResult
        SVD decomposition result to save.
    mismatches : pd.DataFrame, optional
        Validation mismatches DataFrame.
    metadata : SVDMetadata, optional
        Metadata dataclass to save. Will be converted to dict for storage.
    mode : str
        HDF5 file mode. "w" for write (overwrites), "a" for append.
    """
    print(f"Saving SVD to {filename}")

    with h5py.File(filename, mode) as f:
        # Save arrays as datasets
        f.create_dataset("V", data=result.V)
        f.create_dataset("s", data=result.s)

        # Save mismatches if provided (converted to structured array)
        if mismatches is not None and not mismatches.empty:
            f.create_dataset("mismatches", data=mismatches.to_records(index=False))

        # Save metadata as file attributes
        f.attrs["n_components"] = result.n_components
        f.attrs["method"] = result.method
        f.attrs["dataset_type"] = "svd_basis"

        # Save metadata as settings dict (dingo convention for backward compatibility)
        if metadata is not None:
            # Convert dataclass to dict, filtering out None values
            metadata_dict = {k: v for k, v in asdict(metadata).items() if v is not None}
            if metadata_dict:
                f.attrs["settings"] = str(metadata_dict)


def load_svd_from_hdf5(
    filename: str,
) -> tuple[SVDDecompositionResult, Optional[pd.DataFrame], Optional[SVDMetadata]]:
    """Load SVD decomposition result from HDF5 file.

    Loads V and s arrays, computes Vh from V, and reconstructs the
    SVDDecompositionResult. Also loads mismatches and metadata if present.

    Parameters
    ----------
    filename : str
        Input HDF5 file path.

    Returns
    -------
    result : SVDDecompositionResult
        Reconstructed SVD decomposition result.
    mismatches : pd.DataFrame or None
        Validation mismatches if present in file.
    metadata : SVDMetadata or None
        Metadata dataclass reconstructed from settings dict, if present.
    """
    print(f"Loading SVD from {filename}")

    with h5py.File(filename, "r") as f:
        # Load arrays
        V = f["V"][...]
        s = f["s"][...]

        # Compute Vh from V
        Vh = V.T.conj()
        n_components = V.shape[1]

        # Load metadata
        method = f.attrs.get("method", "unknown")

        result = SVDDecompositionResult(V=V, Vh=Vh, s=s, n_components=n_components, method=method)

        # Load mismatches if present
        mismatches = None
        if "mismatches" in f:
            mismatch_data = f["mismatches"][...]
            # Convert structured array to DataFrame if it has column names
            if mismatch_data.dtype.names is not None:
                mismatches = pd.DataFrame(mismatch_data)

        # Load metadata from settings attribute
        metadata = None
        if "settings" in f.attrs:
            try:
                settings_dict = ast.literal_eval(f.attrs["settings"])
                # Known fields of SVDMetadata
                known_fields = [
                    "description",
                    "n_training_samples",
                    "n_validation_samples",
                    "data_shape",
                    "timestamp",
                    "extra",
                ]
                # Extract extra dict if it exists, otherwise collect unknown fields
                extra_dict = settings_dict.get("extra", {})
                # Add any fields not in known_fields to extra
                for k, v in settings_dict.items():
                    if k not in known_fields:
                        extra_dict[k] = v

                # Convert dict to SVDMetadata, using get() for optional fields
                metadata = SVDMetadata(
                    description=settings_dict.get("description"),
                    n_training_samples=settings_dict.get("n_training_samples"),
                    n_validation_samples=settings_dict.get("n_validation_samples"),
                    data_shape=settings_dict.get("data_shape"),
                    timestamp=settings_dict.get("timestamp"),
                    extra=extra_dict,
                )
            except (ValueError, SyntaxError, TypeError):
                # If parsing fails, metadata remains None
                metadata = None

    return result, mismatches, metadata


def save_svd_to_dict(
    result: SVDDecompositionResult,
    mismatches: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Convert SVD result to dictionary.

    Useful for embedding SVD data in larger HDF5 files or for serialization
    to other formats.

    Parameters
    ----------
    result : SVDDecompositionResult
        SVD decomposition result.
    mismatches : pd.DataFrame, optional
        Validation mismatches.

    Returns
    -------
    dict
        Dictionary with SVD data.
    """
    d = {
        "V": result.V,
        "s": result.s,
        "n_components": result.n_components,
        "method": result.method,
    }

    if mismatches is not None:
        d["mismatches"] = mismatches

    return d


def load_svd_from_dict(d: Dict[str, Any]) -> tuple[SVDDecompositionResult, Optional[pd.DataFrame]]:
    """Load SVD from dictionary.

    Reconstructs SVDDecompositionResult from a dictionary, computing Vh
    from V.

    Parameters
    ----------
    d : dict
        Dictionary with SVD data.

    Returns
    -------
    result : SVDDecompositionResult
        Reconstructed SVD decomposition result.
    mismatches : pd.DataFrame or None
        Validation mismatches if present in dictionary.
    """
    V = d["V"]
    s = d["s"]

    # Compute Vh from V
    Vh = V.T.conj()

    # Get metadata with defaults
    n_components = d.get("n_components", V.shape[1])
    method = d.get("method", "unknown")

    result = SVDDecompositionResult(V=V, Vh=Vh, s=s, n_components=n_components, method=method)

    mismatches = d.get("mismatches", None)

    return result, mismatches
