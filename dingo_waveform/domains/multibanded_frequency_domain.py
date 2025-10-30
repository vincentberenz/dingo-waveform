"""Multi-banded frequency domain for efficient waveform representation."""

from copy import copy
from dataclasses import asdict
from typing import Iterable, Optional, Union

import numpy as np
import torch
from typing_extensions import override

from .domain import DomainParameters, build_domain
from .frequency_domain import FrequencyDomain
from .frequency_base import BaseFrequencyDomain

_module_import_path = "dingo_waveform.domains"


class MultibandedFrequencyDomain(BaseFrequencyDomain):
    r"""
    Defines a non-uniform frequency domain that is made up of a sequence of
    uniform-frequency domain bands. Each subsequent band in the sequence has double the
    bin-width of the previous one, i.e., delta_f is doubled each band as one moves up
    the bands. This is intended to allow for efficient representation of gravitational
    waveforms, which generally have slower oscillations at higher frequencies. Indeed,
    the leading order chirp has phase evolution [see
    https://doi.org/10.1103/PhysRevD.49.2658],
    $$
    \Psi(f) = \frac{3}{4}(8 \pi \mathcal{M} f)^{-5/3},
    $$
    hence a coarser grid can be used at higher f.

    The domain is partitioned into bands via a sequence of nodes that are specified at
    initialization.

    In comparison to the FrequencyDomain, the MultibandedFrequencyDomain has the
    following key differences:

    * The sample frequencies start at the first node, rather than f = 0.0 Hz.

    * Quantities such as delta_f, noise_std, etc., are represented as arrays rather than
    scalars, as they vary depending on f.

    The MultibandedFrequencyDomain furthermore has an attribute base_domain,
    which holds an underlying FrequencyDomain object. The decimate() method
    decimates data in the base_domain to the multi-banded domain.
    """

    def __init__(
        self,
        nodes: Iterable[float],
        delta_f_initial: float,
        base_domain: Union[FrequencyDomain, dict],
    ):
        """
        Parameters
        ----------
        nodes
            Defines the partitioning of the underlying frequency domain into bands. In
            total, there are len(nodes) - 1 frequency bands. Band j consists of
            decimated data from the base domain in the range [nodes[j]:nodes[j+1]).
        delta_f_initial
            delta_f of band 0. The decimation factor doubles between adjacent bands,
            so delta_f is doubled as well.
        base_domain
            Original (uniform frequency) domain of data, which is the starting point
            for the decimation. This determines the decimation details and the noise_std.
            Either provided as dict for build_domain, or as domain_object.
        """
        if isinstance(base_domain, dict):
            base_domain = build_domain(base_domain)

        self.nodes = np.array(nodes, dtype=np.float32)
        self.base_domain: FrequencyDomain = base_domain  # type: ignore[assignment]
        self._initialize_bands(delta_f_initial)

        if not isinstance(self.base_domain, FrequencyDomain):
            raise ValueError(
                f"Expected domain type FrequencyDomain, got {type(base_domain)}."
            )

        # Truncation indices for domain update
        self._range_update_idx_lower: Optional[int] = None
        self._range_update_idx_upper: Optional[int] = None
        self._range_update_initial_length: Optional[int] = None

    def _initialize_bands(self, delta_f_initial: float) -> None:
        """
        Initialize band structure based on nodes and initial delta_f.

        Parameters
        ----------
        delta_f_initial
            Frequency spacing for the first band.
        """
        if len(self.nodes.shape) != 1:
            raise ValueError(
                f"Expected format [num_bands + 1] for nodes, "
                f"got {self.nodes.shape}."
            )
        self.num_bands = len(self.nodes) - 1
        self._nodes_indices = (self.nodes / self.base_domain.delta_f).astype(int)

        self._delta_f_bands = (
            delta_f_initial * (2 ** np.arange(self.num_bands))
        ).astype(np.float32)
        self._decimation_factors_bands = (
            self._delta_f_bands / self.base_domain.delta_f
        ).astype(int)
        self._num_bins_bands = (
            (self._nodes_indices[1:] - self._nodes_indices[:-1])
            / self._decimation_factors_bands
        ).astype(int)

        self._band_assignment = np.concatenate(
            [
                np.ones(num_bins_band, dtype=int) * idx
                for idx, num_bins_band in enumerate(self._num_bins_bands)
            ]
        )
        self._delta_f = self._delta_f_bands[self._band_assignment]

        # For each bin, [self._f_base_lower, self._f_base_upper] describes the
        # frequency range in the base domain which is used for truncation.
        self._f_base_lower = np.concatenate(
            (self.nodes[:1], self.nodes[0] + np.cumsum(self._delta_f[:-1]))
        )
        self._f_base_upper = (
            self.nodes[0] + np.cumsum(self._delta_f) - self.base_domain.delta_f
        )

        # Set sample frequencies as mean of decimation range.
        self._sample_frequencies = (self._f_base_upper + self._f_base_lower) / 2
        self._sample_frequencies_torch: Optional[torch.Tensor] = None
        self._sample_frequencies_torch_cuda: Optional[torch.Tensor] = None

        if self.f_min not in self.base_domain() or self.f_max not in self.base_domain():
            raise ValueError(
                f"Endpoints ({self.f_min}, {self.f_max}) not in base "
                f"domain, {self.get_parameters()}"
            )

    def decimate(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Decimate data from the base_domain to the multi-banded domain.

        Parameters
        ----------
        data
            Decimation is done along the trailing dimension of this array. This
            dimension should therefore be compatible with the base frequency domain,
            i.e., running from 0.0 Hz or f_min up to f_max, with uniform delta_f.

        Returns
        -------
        Decimated array of the same type as the input.
        """
        if data.shape[-1] == len(self.base_domain):
            offset_idx = 0
        elif data.shape[-1] == len(self.base_domain) - self.base_domain.min_idx:
            offset_idx = -self.base_domain.min_idx
        else:
            raise ValueError(
                f"Provided data has {data.shape[-1]} bins, which is incompatible with "
                f"the expected domain of length {len(self.base_domain)}"
            )

        if isinstance(data, np.ndarray):
            data_decimated: Union[np.ndarray, torch.Tensor] = np.empty(
                (*data.shape[:-1], len(self)), dtype=data.dtype
            )
        elif isinstance(data, torch.Tensor):
            data_decimated = torch.empty(
                (*data.shape[:-1], len(self)), dtype=data.dtype, device=data.device
            )
        else:
            raise NotImplementedError(
                f"Decimation not implemented for data of type {type(data)}."
            )

        lower_out = 0  # running index for decimated band data
        for idx_band in range(self.num_bands):
            lower_in = self._nodes_indices[idx_band] + offset_idx
            upper_in = self._nodes_indices[idx_band + 1] + offset_idx
            decimation_factor = self._decimation_factors_bands[idx_band]
            num_bins = self._num_bins_bands[idx_band]

            data_decimated[..., lower_out : lower_out + num_bins] = decimate_uniform(  # type: ignore[assignment,index]
                data[..., lower_in:upper_in], decimation_factor
            )
            lower_out += num_bins

        assert lower_out == len(self)

        return data_decimated

    @override
    def update(self, new_settings: Union[dict, DomainParameters]) -> None:
        """
        Update the domain by truncating the frequency range (by specifying new f_min,
        f_max).

        After calling this function, data from the original domain can be truncated to
        the new domain using self.update_data(). For simplicity, we do not allow for
        multiple updates of the domain.

        Parameters
        ----------
        new_settings
            Settings dictionary or DomainParameters. Keys must either be the keys
            contained in get_parameters(), or a subset of ["f_min", "f_max"].
        """
        if isinstance(new_settings, DomainParameters):
            new_settings = asdict(new_settings)
            # Remove None values
            new_settings = {k: v for k, v in new_settings.items() if v is not None}

        domain_params = asdict(self.get_parameters())
        domain_params = {k: v for k, v in domain_params.items() if v is not None}

        if set(new_settings.keys()).issubset(["f_min", "f_max"]):
            self._set_new_range(**new_settings)
        elif set(new_settings.keys()) == set(domain_params.keys()):
            if new_settings == domain_params:
                return
            # Extract f_min, f_max from base_domain in new_settings
            if "base_domain" in new_settings and isinstance(
                new_settings["base_domain"], dict
            ):
                f_min = new_settings["base_domain"].get("f_min")
                f_max = new_settings["base_domain"].get("f_max")
            else:
                f_min = new_settings.get("f_min")
                f_max = new_settings.get("f_max")
            self._set_new_range(f_min=f_min, f_max=f_max)

            if asdict(self.get_parameters()) != new_settings:
                raise ValueError(
                    f"Update settings {new_settings} are incompatible with "
                    f"domain settings {asdict(self.get_parameters())}."
                )
        else:
            raise ValueError(
                f"Invalid argument for domain update {new_settings}. Must either be "
                f'{list(domain_params.keys())} or a subset of ["f_min", "f_max"].'
            )

    def _set_new_range(
        self, f_min: Optional[float] = None, f_max: Optional[float] = None
    ) -> None:
        """
        Set a new range [f_min, f_max] for the domain. This operation is only allowed
        if the new range is contained within the old one.

        Note: f_min, f_max correspond to the range in the *base_domain*.

        Parameters
        ----------
        f_min
            New minimum frequency (optional).
        f_max
            New maximum frequency (optional).
        """
        if f_min is None and f_max is None:
            return
        if self._range_update_initial_length is not None:
            raise ValueError(f"Can't update domain of type {type(self)} a second time.")
        if f_min is not None and f_max is not None and f_min >= f_max:
            raise ValueError("f_min must not be larger than f_max.")

        lower_bin, upper_bin = 0, len(self) - 1

        if f_min is not None:
            if self._f_base_lower[0] <= f_min <= self._f_base_lower[-1]:
                # find new starting bin (first element with f >= f_min)
                lower_bin = np.where(self._f_base_lower >= f_min)[0][0]
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_base_lower[0]}, {self._f_base_lower[-1]}]."
                )

        if f_max is not None:
            if self._f_base_upper[0] <= f_max <= self._f_base_upper[-1]:
                # find new final bin (last element where f <= f_max)
                upper_bin = np.where(self._f_base_upper <= f_max)[0][-1]
            else:
                raise ValueError(
                    f"f_max = {f_max} is not in expected range "
                    f"[{self._f_base_upper[0]}, {self._f_base_upper[-1]}]."
                )

        lower_band = self._band_assignment[lower_bin]
        upper_band = self._band_assignment[upper_bin]
        # new nodes extend to upper_band + 2: we have +1 from the exclusive end index
        # and +1, as we have num_bands + 1 elements in nodes
        nodes_new = copy(self.nodes)[lower_band : upper_band + 2]
        nodes_new[0] = self._f_base_lower[lower_bin]
        nodes_new[-1] = self._f_base_upper[upper_bin] + self.base_domain.delta_f

        # Update base_domain f_min and f_max. These values might differ slightly from
        # the final values for the multi-banded domain due to edge effects. Note that
        # we do this update *after* all of our validation checks but *before* changing
        # the state of the class.
        self.base_domain.update(f_min=f_min, f_max=f_max)

        self._range_update_initial_length = len(self)
        self._range_update_idx_lower = lower_bin
        self._range_update_idx_upper = upper_bin

        self.nodes = nodes_new
        self._initialize_bands(self._delta_f_bands[lower_band])

        assert self._range_update_idx_upper - self._range_update_idx_lower + 1 == len(
            self
        )
        assert self.base_domain.f_min <= self.f_min
        assert self.base_domain.f_max >= self.f_max

    def update_data(
        self, data: Union[np.ndarray, torch.Tensor], axis: int = -1, **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Truncates the data array to be compatible with the domain. This is used when
        changing f_min or f_max.

        update_data() will only have an effect after updating the domain to have a new
        frequency range using self.update().

        Parameters
        ----------
        data
            Array should be compatible with either the original or updated
            MultibandedFrequencyDomain along the specified axis. In the latter
            case, nothing is done. In the former, data are truncated appropriately.
        axis
            Axis along which to operate.
        **kwargs
            Additional keyword arguments (for compatibility).

        Returns
        -------
        Updated data of the same type as input.
        """
        if data.shape[axis] == len(self):
            return data
        elif (
            self._range_update_initial_length is not None
            and data.shape[axis] == self._range_update_initial_length
        ):
            sl = [slice(None)] * data.ndim
            # First truncate beyond f_max.
            assert self._range_update_idx_lower is not None
            assert self._range_update_idx_upper is not None
            sl[axis] = slice(
                self._range_update_idx_lower, self._range_update_idx_upper + 1
            )
            data = data[tuple(sl)]
            return data
        else:
            raise ValueError(
                f"Data (shape {data.shape}) incompatible with the domain "
                f"(length {len(self)})."
            )

    @override
    def __len__(self) -> int:
        """Return the number of bins in the multi-banded domain."""
        return len(self._sample_frequencies)

    @override
    def __call__(self) -> np.ndarray:
        """Return array of sample frequencies."""
        return self.sample_frequencies

    @property
    def sample_frequencies(self) -> np.ndarray:
        """
        Return the sample frequencies.

        Returns
        -------
        np.ndarray
            Array of sample frequencies.
        """
        return self._sample_frequencies

    @property
    def sample_frequencies_torch(self) -> torch.Tensor:
        """
        Return the sample frequencies as a PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Tensor of sample frequencies.
        """
        if self._sample_frequencies_torch is None:
            self._sample_frequencies_torch = torch.tensor(
                self._sample_frequencies, dtype=torch.float32
            )
        return self._sample_frequencies_torch

    @property
    def sample_frequencies_torch_cuda(self) -> torch.Tensor:
        """
        Return the sample frequencies as a CUDA tensor.

        Returns
        -------
        torch.Tensor
            CUDA tensor of sample frequencies.
        """
        if self._sample_frequencies_torch_cuda is None:
            self._sample_frequencies_torch_cuda = self.sample_frequencies_torch.to(
                "cuda"
            )
        return self._sample_frequencies_torch_cuda

    @property
    def frequency_mask(self) -> np.ndarray:
        """
        Array of len(self) consisting of ones.

        As the MultibandedFrequencyDomain starts from f_min, no masking is generally
        required.

        Returns
        -------
        np.ndarray
            Mask array for frequencies (all ones).
        """
        return np.ones_like(self.sample_frequencies)

    @property
    def frequency_mask_length(self) -> int:
        """
        Return number of samples in the domain.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.frequency_mask)

    @property
    @override
    def min_idx(self) -> int:
        """Return the minimum index (always 0 for multibanded domain)."""
        return 0

    @property
    @override
    def max_idx(self) -> int:
        """Return the maximum index."""
        return len(self) - 1

    @property
    @override
    def f_max(self) -> float:
        """Return the maximum frequency."""
        return float(self._f_base_upper[-1])

    @property
    def f_min(self) -> float:
        """Return the minimum frequency."""
        return float(self._f_base_lower[0])

    @property
    def delta_f(self) -> np.ndarray:
        """
        Return the frequency spacing per bin.

        Returns
        -------
        np.ndarray
            Array of frequency spacings (varies per band).
        """
        return self._delta_f

    @property
    @override
    def duration(self) -> float:
        """
        Duration is not well-defined for MultibandedFrequencyDomain.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Duration not defined for MultibandedFrequencyDomain")

    @property
    @override
    def sampling_rate(self) -> float:
        """
        Sampling rate is not well-defined for MultibandedFrequencyDomain.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Sampling rate not defined for MultibandedFrequencyDomain"
        )

    @property
    @override
    def noise_std(self) -> np.ndarray:
        """
        Standard deviation per bin for white noise.

        For the MultibandedFrequencyDomain, this is an array as delta_f varies
        across bins.

        Returns
        -------
        np.ndarray
            Array of noise standard deviations per bin.
        """
        return 1 / np.sqrt(4.0 * self.delta_f)

    @override
    def time_translate_data(
        self, data: Union[np.ndarray, torch.Tensor], dt: Union[float, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Time translate frequency-domain data by dt seconds.

        Parameters
        ----------
        data
            Input data array/tensor.
        dt
            Time shift amount.

        Returns
        -------
        Translated data.
        """
        f = self._get_sample_frequencies_astype(data)

        if isinstance(data, np.ndarray):
            phase_shift = 2 * np.pi * np.einsum("...,i", dt, f)
        elif isinstance(data, torch.Tensor):
            # Allow for possible multiple "batch" dimensions (e.g., batch + detector)
            phase_shift = 2 * np.pi * torch.einsum("...,i", dt, f)
        else:
            raise NotImplementedError(
                f"Time translation not implemented for data of type {type(data)}"
            )

        return self._add_phase(data, phase_shift)

    def _get_sample_frequencies_astype(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Return a 1D frequency array compatible with the last index of data array.

        Parameters
        ----------
        data
            Input data array/tensor.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Frequency array compatible with data.

        Raises
        ------
        TypeError
            If data type is invalid or incompatible with domain.
        """
        f: Union[np.ndarray, torch.Tensor]
        if isinstance(data, np.ndarray):
            f = self.sample_frequencies
        elif isinstance(data, torch.Tensor):
            if data.is_cuda:
                f = self.sample_frequencies_torch_cuda
            else:
                f = self.sample_frequencies_torch
        else:
            raise TypeError("Invalid data type. Should be np.array or torch.Tensor.")

        if data.shape[-1] != len(self):
            raise TypeError(
                f"Data with {data.shape[-1]} frequency bins is "
                f"incompatible with domain (length {len(self)})."
            )

        return f

    @staticmethod
    def _add_phase(
        data: Union[np.ndarray, torch.Tensor], phase: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add a (frequency-dependent) phase to a frequency series.

        Convention: the phase φ(f) is defined via exp[-i φ(f)].

        Parameters
        ----------
        data
            Complex frequency series or real/imaginary representation.
        phase
            Phase array to apply.

        Returns
        -------
        Data with applied phase shift.
        """
        if isinstance(data, np.ndarray) and np.iscomplexobj(data):
            return data * np.exp(-1j * phase)  # type: ignore[arg-type]

        elif isinstance(data, torch.Tensor):
            assert isinstance(phase, torch.Tensor)
            if torch.is_complex(data):
                # Expand the trailing batch dimensions to allow for broadcasting.
                while phase.dim() < data.dim():
                    phase = phase[..., None, :]
                return data * torch.exp(-1j * phase)
            else:
                # The first two components of the second last index should be the real
                # and imaginary parts of the data.
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
        else:
            raise TypeError(f"Invalid data type {type(data)}.")

    @override
    def get_parameters(self) -> DomainParameters:
        """
        Get the parameters of the multibanded frequency domain.

        Returns
        -------
        DomainParameters
            The parameters of the domain.
        """
        return DomainParameters(
            f_max=self.base_domain.f_max,
            f_min=self.base_domain.f_min,
            nodes=self.nodes.tolist(),
            delta_f_initial=float(self._delta_f_bands[0]),
            base_domain=asdict(self.base_domain.get_parameters()),
            type=f"{_module_import_path}.MultibandedFrequencyDomain",
        )

    @override
    @classmethod
    def from_parameters(
        cls, domain_parameters: DomainParameters
    ) -> "MultibandedFrequencyDomain":
        """
        Create a MultibandedFrequencyDomain instance from given parameters.

        Parameters
        ----------
        domain_parameters
            The parameters to create the domain.

        Returns
        -------
        MultibandedFrequencyDomain
            An instance of the multibanded frequency domain.
        """
        for attr in ("nodes", "delta_f_initial", "base_domain"):
            if getattr(domain_parameters, attr) is None:
                raise ValueError(
                    "Can not construct MultibandedFrequencyDomain from "
                    f"{repr(asdict(domain_parameters))}: {attr} should not be None"
                )

        return cls(
            nodes=domain_parameters.nodes,  # type: ignore
            delta_f_initial=domain_parameters.delta_f_initial,  # type: ignore
            base_domain=domain_parameters.base_domain,  # type: ignore
        )


######################
### util functions ###
######################


def decimate_uniform(
    data: Union[np.ndarray, torch.Tensor], decimation_factor: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reduce dimension of data by decimation_factor along last axis, by uniformly
    averaging sets of decimation_factor neighbouring bins.

    Parameters
    ----------
    data
        Array or tensor to be decimated.
    decimation_factor
        Factor by how much to compress. Needs to divide data.shape[-1].

    Returns
    -------
    Uniformly decimated data, as array or tensor.
    Shape (*data.shape[:-1], data.shape[-1]/decimation_factor).
    """
    if data.shape[-1] % decimation_factor != 0:
        raise ValueError(
            f"data.shape[-1] ({data.shape[-1]}) is not a multiple of decimation_factor "
            f"({decimation_factor})."
        )
    if isinstance(data, np.ndarray):
        return (
            np.sum(np.reshape(data, (*data.shape[:-1], -1, decimation_factor)), axis=-1)
            / decimation_factor
        )
    elif isinstance(data, torch.Tensor):
        return (
            torch.sum(
                torch.reshape(data, (*data.shape[:-1], -1, decimation_factor)), dim=-1
            )
            / decimation_factor
        )
    else:
        raise NotImplementedError(
            f"Decimation not implemented for data of type {type(data)}."
        )
