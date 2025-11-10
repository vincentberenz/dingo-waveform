"""Whitening and unwhitening transforms using fixed ASD."""

import logging
from pathlib import Path
from typing import Dict, Union

import h5py
import numpy as np

from ..domains import Domain
from .base import Transform

_logger = logging.getLogger(__name__)


class WhitenAndUnwhiten(Transform):
    """
    Transform that whitens or unwhitens waveforms using a fixed ASD.

    Whitening divides the waveform by the amplitude spectral density (ASD),
    which normalizes the noise across frequencies. This is commonly used
    before SVD compression.

    Parameters
    ----------
    domain
        Frequency domain of the waveforms
    asd_file
        Path to HDF5 file containing the ASD array
    inverse
        If False, applies whitening (default).
        If True, applies unwhitening.

    Example
    -------
    >>> from dingo_waveform.domains import UniformFrequencyDomain
    >>> from dingo_waveform.transforms import WhitenAndUnwhiten
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> whiten = WhitenAndUnwhiten(domain, "asd.hdf5", inverse=False)
    >>> whitened = whiten(polarizations)
    """

    def __init__(
        self,
        domain: Domain,
        asd_file: Union[str, Path],
        inverse: bool = False
    ):
        """
        Initialize whitening transform.

        Parameters
        ----------
        domain
            Frequency domain specification
        asd_file
            Path to file containing ASD data
        inverse
            Whether to apply unwhitening (True) or whitening (False)

        Raises
        ------
        FileNotFoundError
            If ASD file does not exist.
        KeyError
            If ASD file does not contain required data.
        """
        self.domain = domain
        self.asd_file = Path(asd_file)
        self.inverse = inverse

        if not self.asd_file.exists():
            raise FileNotFoundError(f"ASD file not found: {self.asd_file}")

        # Load ASD from file
        self.asd = self._load_asd()

        _logger.info(
            f"Loaded ASD from {self.asd_file} "
            f"(mode: {'unwhiten' if inverse else 'whiten'})"
        )

    def _load_asd(self) -> np.ndarray:
        """
        Load ASD array from HDF5 file.

        Returns
        -------
        ASD array matching the domain

        Raises
        ------
        KeyError
            If file format is not recognized.
        ValueError
            If ASD shape does not match domain.
        """
        with h5py.File(self.asd_file, "r") as f:
            # Try common HDF5 structures for ASD files
            if "asd" in f:
                asd = f["asd"][:]
            elif "ASD" in f:
                asd = f["ASD"][:]
            elif "asds/H1" in f:  # Common format in dingo
                asd = f["asds/H1"][:]
            else:
                raise KeyError(
                    f"Could not find ASD data in {self.asd_file}. "
                    "Expected keys: 'asd', 'ASD', or 'asds/H1'"
                )

        # Validate shape
        expected_length = len(self.domain)
        if len(asd) != expected_length:
            raise ValueError(
                f"ASD length ({len(asd)}) does not match domain length ({expected_length})"
            )

        return asd

    def __call__(self, polarizations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply whitening or unwhitening.

        Parameters
        ----------
        polarizations
            Dictionary with polarization arrays

        Returns
        -------
        Transformed polarizations dictionary
        """
        if self.inverse:
            # Unwhitening: multiply by ASD
            return {key: value * self.asd for key, value in polarizations.items()}
        else:
            # Whitening: divide by ASD
            return {key: value / self.asd for key, value in polarizations.items()}

    def __repr__(self) -> str:
        """String representation."""
        mode = "unwhiten" if self.inverse else "whiten"
        return f"WhitenAndUnwhiten(asd_file={self.asd_file.name}, mode={mode})"
