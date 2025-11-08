"""Multi-banded frequency domain for efficient waveform representation."""

from dataclasses import asdict
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
from typing_extensions import override

from .binning.adaptive_binning import (
    BinningParameters,
    Band,
    compute_adaptive_binning,
    decimate as ab_decimate,
)
from .domain import DomainParameters, build_domain
from .frequency_base import BaseFrequencyDomain
from .frequency_domain import UniformFrequencyDomain
from ..polarizations import Polarization

_module_import_path = "dingo_waveform.domains"


class MultibandedFrequencyDomain(BaseFrequencyDomain):
    """
    Non-uniform frequency domain composed of dyadically spaced uniform bands.

    Immutable-style API:
      - narrowed(f_min, f_max) -> MultibandedFrequencyDomain returns a new instance.
      - slice_from(old_domain, data, axis=-1) slices old-domain-aligned data to this domain.

    Internals:
      - self._binning: BinningParameters (authoritative, immutable band/bin metadata)
      - base_delta_f: Base uniform frequency spacing for binning calculations
      - sample frequencies are the per-bin midpoints of [f_base_lower, f_base_upper]
    """

    def __init__(
        self,
        nodes: Iterable[float],
        delta_f_initial: float,
        base_delta_f: float,
        window_factor: Optional[float] = None,
    ):
        """
        Initialize a MultibandedFrequencyDomain with dyadic spacing.

        Parameters
        ----------
        nodes : Iterable[float]
            Band boundary frequencies. Must have at least 2 elements.
        delta_f_initial : float
            Frequency spacing for the first (lowest frequency) band.
        base_delta_f : float
            Base uniform frequency spacing used for binning calculations.
        window_factor : Optional[float]
            Optional window factor (preserved for narrowing operations).
        """
        self.base_delta_f = float(base_delta_f)
        self.window_factor = window_factor

        # Build binning and caches
        self._binning: BinningParameters = compute_adaptive_binning(
            nodes=list(nodes),
            delta_f_initial=float(delta_f_initial),
            base_delta_f=self.base_delta_f,
        )
        self.num_bands: int = int(self._binning.num_bands)

        # Sample frequencies: mean of inclusive base-frequency bin bounds
        self._sample_frequencies = (
            (self._binning.f_base_lower + self._binning.f_base_upper) / 2.0
        ).astype(np.float32)
        self._sample_frequencies_torch: Optional[torch.Tensor] = None
        self._sample_frequencies_torch_cuda: Optional[torch.Tensor] = None

    # ---------------------------
    # Waveform transformation
    # ---------------------------

    @override
    def waveform_transform(self, polarizations: Polarization) -> Polarization:
        """
        Transform waveform polarizations from base uniform grid to multibanded representation.

        This method decimates waveforms generated on a uniform frequency grid (starting at f=0)
        to the sparse multibanded grid representation of this domain. It uses adaptive binning
        with dyadic spacing to efficiently represent the waveform.

        Parameters
        ----------
        polarizations : Polarization
            Polarization waveforms (h_plus, h_cross) on uniform grid.
            Expected to be generated on a uniform grid starting at f=0 with spacing base_delta_f.

        Returns
        -------
        Polarization
            Decimated polarizations matching this domain's multibanded binning structure.

        Notes
        -----
        The input waveforms should be generated on a uniform frequency grid that:
        - Starts at f=0
        - Has frequency spacing equal to self.base_delta_f
        - Covers at least up to self.f_max

        The decimation uses mode='explicit' assuming data starts at f=0.
        """
        h_plus_decimated = self.decimate(polarizations.h_plus, mode='explicit')
        h_cross_decimated = self.decimate(polarizations.h_cross, mode='explicit')
        return Polarization(h_plus=h_plus_decimated, h_cross=h_cross_decimated)

    # ---------------------------
    # Construction helpers
    # ---------------------------

    def narrowed(
        self,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
    ) -> "MultibandedFrequencyDomain":
        """
        Return a new MultibandedFrequencyDomain narrowed to [f_min, f_max] within this domain.
        The current instance is not modified.

        Notes
        -----
        - f_min/f_max refer to the base grid range.
        - Preserves the dyadic structure by choosing the new delta_f_initial from
          the band that becomes the new first band.
        """
        if self._binning.total_bins == 0:
            raise ValueError("Domain has no bins; cannot narrow.")

        f_min_req = self.f_min if f_min is None else float(f_min)
        f_max_req = self.f_max if f_max is None else float(f_max)
        if f_min_req >= f_max_req:
            raise ValueError("f_min must be strictly smaller than f_max.")

        if not (self.f_min <= f_min_req <= self.f_max):
            raise ValueError(f"Requested f_min={f_min_req} not in [{self.f_min}, {self.f_max}].")
        if not (self.f_min <= f_max_req <= self.f_max):
            raise ValueError(f"Requested f_max={f_max_req} not in [{self.f_min}, {self.f_max}].")

        # Identify surviving bin range in the old domain
        try:
            lower_idx = int(np.where(self._binning.f_base_lower >= f_min_req)[0][0])
            upper_idx = int(np.where(self._binning.f_base_upper <= f_max_req)[0][-1])
        except IndexError:
            raise ValueError("Requested range does not align with bin boundaries.")

        lower_band = int(self._binning.band_assignment[lower_idx])
        upper_band = int(self._binning.band_assignment[upper_idx])

        # Build new nodes covering [lower_band : upper_band], adjust endpoints
        nodes_old = self.nodes  # copy from property
        nodes_new = nodes_old[lower_band : upper_band + 2]
        nodes_new[0] = float(self._binning.f_base_lower[lower_idx])
        nodes_new[-1] = float(self._binning.f_base_upper[upper_idx] + self.base_delta_f)

        # New delta_f_initial derived from the band that becomes the new first band
        new_delta_f_initial = float(self._binning.delta_f_bands[lower_band])

        return MultibandedFrequencyDomain(
            nodes=nodes_new,
            delta_f_initial=new_delta_f_initial,
            base_delta_f=self.base_delta_f,
            window_factor=self.window_factor,
        )


    # ---------------------------
    # Core operations
    # ---------------------------

    def decimate(
        self, data: Union[np.ndarray, torch.Tensor], mode: str = 'auto'
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Decimate data from the base uniform grid to this multi-banded domain.

        Parameters
        ----------
        data : np.ndarray | torch.Tensor
            Input data to decimate along last axis.
        mode : {"auto", "explicit"}, default "auto"
            - "auto": Infer base_offset_idx from data length.
            - "explicit": Use base_offset_idx=0 (assumes data starts at f=0).

        Returns
        -------
        Decimated data matching this domain's binning.
        """
        return self._binning.decimate(data, mode=mode, base_offset_idx=0, policy='pick')


    # ---------------------------
    # Properties and BaseFrequencyDomain interface
    # ---------------------------

    @property
    def fbase(self)->Tuple[float,float]:
        return self._binning.f_base_lower, self._binning.f_base_upper

    @property
    def bands(self) -> Tuple[Band, ...]:
        """Tuple of per-band metadata objects."""
        return self._binning.bands

    @property
    def nodes(self) -> np.ndarray:
        """Copy of the band boundary nodes (shape: num_bands + 1, dtype float32)."""
        return self._binning.nodes.copy()

    @override
    def __len__(self) -> int:
        return int(self._binning.total_bins)

    @override
    def __call__(self) -> np.ndarray:
        return self.sample_frequencies

    @property
    def sample_frequencies(self) -> np.ndarray:
        return self._sample_frequencies

    @property
    def sample_frequencies_torch(self) -> torch.Tensor:
        if self._sample_frequencies_torch is None:
            self._sample_frequencies_torch = torch.tensor(
                self._sample_frequencies, dtype=torch.float32
            )
        return self._sample_frequencies_torch

    @property
    def sample_frequencies_torch_cuda(self) -> torch.Tensor:
        if self._sample_frequencies_torch_cuda is None:
            self._sample_frequencies_torch_cuda = self.sample_frequencies_torch.to("cuda")
        return self._sample_frequencies_torch_cuda

    @property
    def frequency_mask(self) -> np.ndarray:
        # No masking required; domain starts at f_min
        return np.ones_like(self.sample_frequencies, dtype=np.float32)

    @property
    def frequency_mask_length(self) -> int:
        return len(self.frequency_mask)

    @property
    @override
    def min_idx(self) -> int:
        # Multibanded domain starts at its first bin
        return 0

    @property
    @override
    def max_idx(self) -> int:
        return len(self) - 1

    @property
    @override
    def f_max(self) -> float:
        if self._binning.f_base_upper.size == 0:
            return 0.0
        return float(self._binning.f_base_upper[-1])

    @property
    def f_min(self) -> float:
        if self._binning.f_base_lower.size == 0:
            return 0.0
        return float(self._binning.f_base_lower[0])

    @property
    def delta_f(self) -> np.ndarray:
        # Per-bin spacing (varies per band)
        return self._binning.delta_f

    @property
    @override
    def duration(self) -> float:
        raise NotImplementedError("Duration not defined for MultibandedFrequencyDomain")

    @property
    @override
    def sampling_rate(self) -> float:
        raise NotImplementedError("Sampling rate not defined for MultibandedFrequencyDomain")

    @property
    @override
    def noise_std(self) -> np.ndarray:
        # Standard deviation per bin for white noise
        return 1.0 / np.sqrt(4.0 * self.delta_f)

    @override
    def time_translate_data(
        self, data: Union[np.ndarray, torch.Tensor], dt: Union[float, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Time translate frequency-domain data by dt seconds.
        """
        f = self._get_sample_frequencies_astype(data)

        if isinstance(data, np.ndarray):
            phase_shift = 2 * np.pi * np.einsum("...,i", dt, f)
        elif isinstance(data, torch.Tensor):
            phase_shift = 2 * np.pi * torch.einsum("...,i", dt, f)
        else:
            raise NotImplementedError(
                f"Time translation not implemented for data of type {type(data)}"
            )

        return BaseFrequencyDomain.add_phase(data, phase_shift)

    def _get_sample_frequencies_astype(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Return a 1D frequency array compatible with the last index of data array.
        """
        if isinstance(data, np.ndarray):
            f = self.sample_frequencies
        elif isinstance(data, torch.Tensor):
            f = self.sample_frequencies_torch_cuda if data.is_cuda else self.sample_frequencies_torch
        else:
            raise TypeError("Invalid data type. Should be np.array or torch.Tensor.")

        if data.shape[-1] != len(self):
            raise TypeError(
                f"Data with {data.shape[-1]} frequency bins is "
                f"incompatible with domain (length {len(self)})."
            )
        return f

    @override
    def get_parameters(self) -> DomainParameters:
        return DomainParameters(
            f_max=self.f_max,
            f_min=self.f_min,
            nodes=self.nodes.tolist(),
            delta_f_initial=float(self._binning.delta_f_bands[0]) if self.num_bands > 0 else None,
            base_delta_f=self.base_delta_f,
            window_factor=self.window_factor,
            type=f"{_module_import_path}.MultibandedFrequencyDomain",
        )

    @override
    @classmethod
    def from_parameters(
        cls, domain_parameters: DomainParameters
    ) -> "MultibandedFrequencyDomain":
        for attr in ("nodes", "delta_f_initial", "base_delta_f"):
            if getattr(domain_parameters, attr) is None:
                raise ValueError(
                    "Can not construct MultibandedFrequencyDomain from "
                    f"{repr(asdict(domain_parameters))}: {attr} should not be None"
                )

        return cls(
            nodes=domain_parameters.nodes,  # type: ignore[arg-type]
            delta_f_initial=domain_parameters.delta_f_initial,  # type: ignore[arg-type]
            base_delta_f=domain_parameters.base_delta_f,  # type: ignore[arg-type]
            window_factor=domain_parameters.window_factor,
        )


def adapt_data(
        former_domain: MultibandedFrequencyDomain,
        new_domain: MultibandedFrequencyDomain,
        data: Union[np.ndarray, torch.Tensor],
        axis: int = -1,
)-> Union[np.ndarray, torch.Tensor]:
        if data.shape[axis] != len(former_domain):
            raise ValueError(
                f"Data trailing length {data.shape[axis]} does not match source domain length {len(former_domain)}."
            )

        # Map this domain's [f_min, f_max] into indices on the old domain
        try:
            lower_idx = int(np.where(former_domain.fbase[0] >= new_domain.f_min)[0][0])
            upper_idx = int(np.where(former_domain.fbase[1] <= new_domain.f_max)[0][-1])
        except IndexError:
            raise ValueError("New domain range is not a subrange of the old domain.")

        # Safety: ensure number of bins aligns exactly
        if (upper_idx - lower_idx + 1) != len(new_domain):
            raise ValueError(
                "Inconsistent bin mapping between domains; cannot slice data safely."
            )

        sl = [slice(None)] * data.ndim
        sl[axis] = slice(lower_idx, upper_idx + 1)
        return data[tuple(sl)]


######################
### util functions ###
######################

