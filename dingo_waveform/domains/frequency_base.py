from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import torch
from multipledispatch import dispatch

from .domain import Domain, DomainParameters

if TYPE_CHECKING:
    from ..types import FrequencySeries, Modes
    import lal


class BaseFrequencyDomain(Domain, ABC):
    """
    Abstract base class for frequency-domain-like domains.

    It defines the shared interface between UniformFrequencyDomain and
    MultibandedFrequencyDomain while delegating implementation details
    to concrete subclasses. This class does not add new behavior; it
    only documents and enforces the common API to enable type sharing
    and future refactors.
    """

    # --- Phase utilities (shared across frequency domains) ---
    @staticmethod
    @dispatch(np.ndarray, np.ndarray)
    def add_phase(data: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Add frequency-dependent phase to complex data (NumPy implementation).

        Parameters
        ----------
        data
            Complex-valued frequency series.
        phase
            Phase array to apply.

        Returns
        -------
        np.ndarray
            Data with applied phase shift.

        Raises
        ------
        TypeError
            If data is not complex.
        """
        if not np.iscomplexobj(data):
            raise TypeError("Numpy data must be complex array")
        return data * np.exp(-1j * phase)

    # type ignore: mypy does not catch that add_phase is managed by multipledispatch
    @staticmethod  # type: ignore
    @dispatch(torch.Tensor, torch.Tensor)
    def add_phase(data: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Add frequency-dependent phase to complex data (PyTorch implementation).

        Parameters
        ----------
        data
            Complex-valued frequency series.
        phase
            Phase tensor to apply.

        Returns
        -------
        torch.Tensor
            Data with applied phase shift.
        """
        if torch.is_complex(data):
            while phase.dim() < data.dim():
                phase = phase[..., None, :]
            return data * torch.exp(-1j * phase)
        else:
            while phase.dim() < data.dim() - 1:
                phase = phase[..., None, :]
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            result = torch.empty_like(data)
            result[..., 0, :] = (
                data[..., 0, :] * cos_phase + data[..., 1, :] * sin_phase
            )
            result[..., 1, :] = (
                data[..., 1, :] * cos_phase - data[..., 0, :] * sin_phase
            )
            if data.shape[-2] > 2:
                result[..., 2:, :] = data[..., 2:, :]
            return result

    # --- Time-domain to frequency-domain mode conversion ---
    def convert_td_modes_to_fd(
        self, hlm_td: "Dict[Modes, lal.COMPLEX16FrequencySeries]"
    ) -> "Dict[Modes, FrequencySeries]":
        """
        Convert time-domain modes to frequency-domain modes.

        This method provides domain-specific logic for converting TD modes to FD.
        The default implementation delegates to the existing td_modes_to_fd_modes
        utility function, which works for uniform frequency domains.

        Subclasses (e.g., MultibandedFrequencyDomain) can override this method
        to implement custom conversion strategies (e.g., convert in uniform base
        domain then decimate).

        Parameters
        ----------
        hlm_td : Dict[Modes, lal.COMPLEX16FrequencySeries]
            Dictionary of time-domain modes with (l,m) keys.

        Returns
        -------
        Dict[Modes, FrequencySeries]
            Dictionary of frequency-domain modes.
        """
        # Import here to avoid circular dependency
        from ..polarization_modes_functions.polarization_modes_utils import (
            td_modes_to_fd_modes,
        )

        return td_modes_to_fd_modes(hlm_td, self)

    # --- Core frequency range and resolution ---
    @property
    @abstractmethod
    def f_min(self) -> float:
        """Minimum represented frequency (inclusive)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def f_max(self) -> float:
        """Maximum represented frequency (inclusive or effective upper edge)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def delta_f(self) -> Union[float, np.ndarray]:
        """
        Frequency spacing. For uniform domains this is a scalar; for
        multibanded domains this may be an array matching the number of bins.
        """
        raise NotImplementedError

    # --- Size and indexing helpers ---
    @abstractmethod
    def __len__(self) -> int:
        """Number of frequency bins represented by the domain."""
        raise NotImplementedError

    @abstractmethod
    def min_idx(self) -> int:
        """Index of the first frequency bin included in the domain."""
        raise NotImplementedError

    @abstractmethod
    def max_idx(self) -> int:
        """Index of the last frequency bin included in the domain."""
        raise NotImplementedError

    # --- Frequencies ---
    @abstractmethod
    def sample_frequencies(self) -> np.ndarray:
        """NumPy array of sample frequencies for the domain."""
        raise NotImplementedError

    @abstractmethod
    def sample_frequencies_torch(self) -> torch.Tensor:
        """PyTorch tensor of sample frequencies for the domain (CPU)."""
        raise NotImplementedError

    @abstractmethod
    def sample_frequencies_torch_cuda(self) -> torch.Tensor:
        """PyTorch tensor of sample frequencies for the domain (CUDA)."""
        raise NotImplementedError

    @abstractmethod
    def frequency_mask(self) -> np.ndarray:
        """Boolean mask indicating bins at or above f_min (or valid bins)."""
        raise NotImplementedError

    @abstractmethod
    def frequency_mask_length(self) -> int:
        """Number of true entries in frequency mask."""
        raise NotImplementedError

    # --- Common derived quantities ---
    @abstractmethod
    def duration(self) -> float:
        """Effective time duration corresponding to the frequency resolution."""
        raise NotImplementedError

    @abstractmethod
    def sampling_rate(self) -> float:
        """Effective sampling rate corresponding to the band structure."""
        raise NotImplementedError

    # --- Noise and data utilities ---
    @abstractmethod
    def noise_std(self) -> Union[float, np.ndarray]:
        """Standard deviation of noise per frequency bin."""
        raise NotImplementedError

    @abstractmethod
    def time_translate_data(
        self, data: Union[np.ndarray, torch.Tensor], dt: Union[float, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply a time translation by dt in the frequency domain."""
        raise NotImplementedError

    # --- Parameters IO ---
    @abstractmethod
    def get_parameters(self) -> DomainParameters:
        """Return a DomainParameters instance describing this domain."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_parameters(cls, domain_parameters: DomainParameters) -> "BaseFrequencyDomain":
        """Construct a domain from a DomainParameters instance."""
        raise NotImplementedError
