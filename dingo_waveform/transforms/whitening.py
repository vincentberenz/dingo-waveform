"""
Whitening and unwhitening transforms using fixed ASD.

ASD (Amplitude Spectral Density) Files
---------------------------------------
The ASD represents the noise characteristics of a gravitational wave detector
across different frequencies. It has units of strain/√Hz and describes the
typical noise amplitude at each frequency.

Expected HDF5 File Format
--------------------------
The ASD file must be an HDF5 file containing a 1D array of real-valued ASD
values. The array length must match the number of frequency bins in the domain.

The loader will search for ASD data in the following keys (in order):
1. 'asd' - Generic key for ASD array
2. 'ASD' - Alternative capitalization
3. 'asds/H1' - dingo-gw format for LIGO Hanford detector ASD
4. 'asds/L1' - dingo-gw format for LIGO Livingston detector ASD

Creating ASD Files
------------------
ASD files can be created using:
- dingo-gw's noise generation tools (dingo_generate_ASD_dataset)
- Detector design sensitivity curves (e.g., LIGO O3 sensitivity)
- Measured noise from detector data

Example: Creating a simple ASD file
>>> import h5py
>>> import numpy as np
>>> # Generate example ASD (e.g., flat noise at 1e-23 strain/√Hz)
>>> frequencies = np.linspace(20.0, 1024.0, 8193)
>>> asd_values = np.ones_like(frequencies) * 1e-23
>>> with h5py.File("example_asd.hdf5", "w") as f:
...     f.create_dataset("asd", data=asd_values)

Whitening Process
-----------------
Whitening: waveform_whitened = waveform / ASD
    - Normalizes signal across frequencies
    - Equal noise contribution at all frequencies
    - Commonly applied before SVD compression

Unwhitening: waveform = waveform_whitened * ASD
    - Reverses whitening operation
    - Restores original waveform scaling
    - Applied after decompression
"""

import logging
from pathlib import Path
from typing import Dict, Union

import h5py
import numpy as np

from ..domains import Domain
from dingo_svd import Transform

_logger = logging.getLogger(__name__)


class WhitenAndUnwhiten(Transform):
    """
    Transform that whitens or unwhitens waveforms using a fixed ASD.

    Whitening normalizes the waveform by dividing by the amplitude spectral
    density (ASD), which represents the detector noise characteristics. This
    operation makes the noise contribution equal across all frequencies, which
    is beneficial for SVD compression and matched filtering.

    The mathematical operation is:
        Whitening:   h_whitened(f) = h(f) / ASD(f)
        Unwhitening: h(f) = h_whitened(f) * ASD(f)

    where h(f) is the waveform in frequency domain and ASD(f) is the noise
    amplitude spectral density with units of strain/√Hz.

    Parameters
    ----------
    domain : Domain
        Frequency domain specification. Must match the frequency array in the
        ASD file (same length, frequencies, and spacing).
    asd_file : str or Path
        Path to HDF5 file containing the ASD array. The file must contain
        a 1D real-valued array with length matching the domain. The loader
        searches for keys in this order: 'asd', 'ASD', 'asds/H1', 'asds/L1'.

        File format:
            - Type: HDF5
            - Expected keys: 'asd' (preferred), 'ASD', 'asds/H1', or 'asds/L1'
            - Data shape: (n_frequencies,)
            - Data type: float64
            - Units: strain/√Hz

    inverse : bool, optional
        Operation mode:
            - False (default): Apply whitening (divide by ASD)
            - True: Apply unwhitening (multiply by ASD)

    Raises
    ------
    FileNotFoundError
        If the specified ASD file does not exist
    KeyError
        If the ASD file does not contain any of the expected keys
    ValueError
        If the ASD array length does not match the domain length

    Notes
    -----
    - The ASD must have the same length as the domain's frequency array
    - ASD values should be positive (noise amplitude cannot be negative)
    - Typical ASD values for LIGO are ~10^-23 to 10^-21 strain/√Hz
    - Whitening is typically applied before SVD compression
    - Unwhitening is applied after decompression to restore physical units

    See Also
    --------
    dingo_waveform.transforms.ApplySVD : SVD compression transform
    dingo_waveform.transforms.ComposeTransforms : Compose multiple transforms

    Examples
    --------
    Basic whitening:

    >>> from dingo_waveform.domains import UniformFrequencyDomain
    >>> from dingo_waveform.transforms import WhitenAndUnwhiten
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> whiten = WhitenAndUnwhiten(domain, "asd.hdf5", inverse=False)
    >>> whitened_polarizations = whiten(polarizations)

    Full compression pipeline with whitening and SVD:

    >>> from dingo_waveform.transforms import ComposeTransforms, ApplySVD
    >>> from dingo_svd import SVDBasis
    >>>
    >>> # Load SVD basis
    >>> basis = SVDBasis.from_file("svd_basis.hdf5")
    >>>
    >>> # Create compression pipeline
    >>> compress_pipeline = ComposeTransforms([
    ...     WhitenAndUnwhiten(domain, "asd.hdf5", inverse=False),  # Whiten
    ...     ApplySVD(basis, inverse=False)                          # Compress
    ... ])
    >>>
    >>> # Create decompression pipeline (reverse order, inverse operations)
    >>> decompress_pipeline = ComposeTransforms([
    ...     ApplySVD(basis, inverse=True),                          # Decompress
    ...     WhitenAndUnwhiten(domain, "asd.hdf5", inverse=True)    # Unwhiten
    ... ])
    >>>
    >>> # Apply pipelines
    >>> compressed = compress_pipeline(polarizations)
    >>> reconstructed = decompress_pipeline(compressed)
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

        The loader attempts to find ASD data using common key conventions:
        1. 'asd' - Generic/preferred key
        2. 'ASD' - Alternative capitalization
        3. 'asds/H1' - dingo-gw format for LIGO Hanford detector
        4. 'asds/L1' - dingo-gw format for LIGO Livingston detector

        The first matching key is used.

        Returns
        -------
        np.ndarray
            1D array of ASD values with shape (n_frequencies,).
            Units: strain/√Hz
            Data type: float64

        Raises
        ------
        KeyError
            If none of the expected keys are found in the HDF5 file.
            The error message lists all attempted keys.
        ValueError
            If the loaded ASD array length does not match the domain length.
            This indicates a frequency mismatch between the ASD file and
            the waveform domain.
        """
        with h5py.File(self.asd_file, "r") as f:
            # List of keys to try, in order of preference
            # 'asd' and 'ASD' are generic keys
            # 'asds/H1' and 'asds/L1' are dingo-gw conventions for LIGO detectors
            possible_keys = ["asd", "ASD", "asds/H1", "asds/L1"]

            asd = None
            for key in possible_keys:
                if key in f:
                    asd = f[key][:]
                    _logger.debug(f"Loaded ASD from key '{key}'")
                    break

            if asd is None:
                available_keys = list(f.keys())
                raise KeyError(
                    f"Could not find ASD data in {self.asd_file}.\n"
                    f"Tried keys: {possible_keys}\n"
                    f"Available keys in file: {available_keys}\n"
                    f"Expected HDF5 structure: file['asd'] = array of shape ({len(self.domain)},)"
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
