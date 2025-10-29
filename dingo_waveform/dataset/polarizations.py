"""Dataclass for storing batched gravitational wave polarizations."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Polarizations:
    """
    A dataclass representing batched polarizations of gravitational waves.

    This is the batched version of the Polarization dataclass, used for storing
    multiple waveforms as numpy arrays rather than single FrequencySeries.

    Attributes
    ----------
    h_plus
        Array of plus polarization components with shape (num_waveforms, frequency_bins).
    h_cross
        Array of cross polarization components with shape (num_waveforms, frequency_bins).
    """

    h_plus: np.ndarray
    h_cross: np.ndarray

    def __post_init__(self):
        """Validate that h_plus and h_cross have the same shape."""
        if self.h_plus.shape != self.h_cross.shape:
            raise ValueError(
                f"h_plus and h_cross must have the same shape. "
                f"Got h_plus: {self.h_plus.shape}, h_cross: {self.h_cross.shape}"
            )

    def __len__(self) -> int:
        """Return the number of waveforms (first dimension of arrays)."""
        return self.h_plus.shape[0]

    @property
    def num_waveforms(self) -> int:
        """Return the number of waveforms."""
        return len(self)

    @property
    def num_frequency_bins(self) -> int:
        """Return the number of frequency bins."""
        return self.h_plus.shape[1] if self.h_plus.ndim > 1 else self.h_plus.shape[0]
