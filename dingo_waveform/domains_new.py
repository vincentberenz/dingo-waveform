from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from multipledispatch import dispatch
from typing_extensions import override


class Domain(ABC):
    """Base class representing a physical domain for data processing."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of bins or points in the domain."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Generate array of bins in the domain."""
        raise NotImplementedError

    @abstractmethod
    def update(self, *args) -> None:
        """Update domain parameters."""
        raise NotImplementedError

    @abstractmethod
    def time_translate_data(
        self, data: Union[np.ndarray, torch.Tensor], dt: Union[float, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Translate data in time domain by dt seconds.

        Args:
            data: Input data array/tensor
            dt: Time shift amount

        Returns:
            Translated data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution."""
        raise NotImplementedError

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """Sampling rate of the data in Hz."""
        raise NotImplementedError

    @property
    @abstractmethod
    def f_max(self) -> float:
        """Maximum frequency in Hz."""
        raise NotImplementedError

    @property
    @abstractmethod
    def duration(self) -> float:
        """Duration of the waveform in seconds."""
        raise NotImplementedError

    @property
    @abstractmethod
    def min_idx(self) -> int:
        """Minimum index of the domain."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_idx(self) -> int:
        """Maximum index of the domain."""
        raise NotImplementedError


class _SampleFrequencies:
    """Helper class for managing frequency sampling."""

    def __init__(self, f_min: float, f_max: float, delta_f: float) -> None:
        self._len = int((f_max - f_min) / delta_f) + 1
        self._f_min = f_min
        self._f_max = f_max
        self._sample_frequencies = np.linspace(
            f_min, f_max, num=self._len, endpoint=True, dtype=np.float32
        )

    def get(self) -> np.ndarray:
        """Return the sampled frequencies."""
        return self._sample_frequencies

    def __len__(self) -> int:
        """Return the number of frequency samples."""
        return self._len

    @cached_property
    def frequency_mask(self) -> np.ndarray:
        """Create mask for frequencies above f_min."""
        return self._sample_frequencies > self._f_min

    @cached_property
    def _sample_frequencies_torch(self) -> torch.Tensor:
        """Create PyTorch tensor version of frequencies."""
        return torch.linspace(
            self._f_min, self._f_max, steps=len(self), dtype=torch.float32
        )

    @cached_property
    def _sample_frequencies_torch_cuda(self) -> torch.Tensor:
        """Create CUDA version of frequency tensor."""
        return self._sample_frequencies_torch.to("cuda")


class FrequencyDomain(Domain):
    """
    Represents a frequency domain with uniform bins.

    Attributes:
        f_min (float): Minimum frequency in Hz
        f_max (float): Maximum frequency in Hz
        delta_f (float): Frequency spacing in Hz
        window_factor (Optional[float]): Window factor for noise calculations
    """

    def __init__(
        self,
        f_min: float,
        f_max: float,
        delta_f: float,
        window_factor: Optional[float] = None,
    ) -> None:
        self._validate_init_params(f_min, f_max, delta_f)

        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor

        self._sample_frequencies = _SampleFrequencies(f_min, f_max, delta_f)

    def _validate_init_params(self, f_min: float, f_max: float, delta_f: float) -> None:
        """Validate initialization parameters."""
        if f_min < 0:
            raise ValueError("Minimum frequency must be non-negative")
        if f_max <= f_min:
            raise ValueError("Maximum frequency must be greater than minimum frequency")
        if delta_f <= 0:
            raise ValueError("Frequency spacing must be positive")

    @override
    def update(self, f_min: Optional[float], f_max: Optional[float]) -> None:
        """
        Update the domain range while maintaining constraints.

        Args:
            f_min: New minimum frequency (None to keep current value)
            f_max: New maximum frequency (None to keep current value)

        Raises:
            ValueError: If updated values violate domain constraints
        """
        if f_min is not None:
            if f_min < 0:
                raise ValueError("Minimum frequency must be non-negative")
            self._f_min = f_min

        if f_max is not None:
            if f_max <= self._f_min:
                raise ValueError(
                    "Maximum frequency must be greater than minimum frequency"
                )
            self._f_max = f_max

        # Reset cached properties
        self._reset_sample_frequencies()

    def _reset_sample_frequencies(self) -> None:
        """Reset sample frequencies cache after parameter updates."""
        self.__dict__.pop("_sample_frequencies", None)
        self._sample_frequencies = _SampleFrequencies(
            self._f_min, self._f_max, self._delta_f
        )

    @override
    def time_translate_data(
        self, data: Union[np.ndarray, torch.Tensor], dt: Union[float, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Time translate frequency-domain data by dt seconds.

        Args:
            data: Input data array/tensor
            dt: Time shift amount

        Returns:
            Translated data
        """
        f = self._get_sample_frequencies_astype(data)

        if isinstance(data, np.ndarray):
            # Assume numpy arrays un-batched, since they are only used at train time
            phase_shift = 2 * np.pi * dt * f
        elif isinstance(data, torch.Tensor):
            # Allow for possible multiple "batch" dimensions (e.g., batch + detector)
            phase_shift = 2 * np.pi * torch.einsum("...,i", dt, f)
        else:
            raise NotImplementedError(
                f"Time translation not implemented for data of type {data}"
            )

        return self.add_phase(data, phase_shift)

    @override
    def __len__(self) -> int:
        """Return the number of frequency bins in the domain [0, f_max]."""
        return int(self.f_max / self.delta_f) + 1

    @override
    def __call__(self) -> np.ndarray:
        """Return array of uniform frequency bins in the domain [0, f_max]."""
        return self.sample_frequencies

    @property
    @override
    def min_idx(self) -> int:
        """Return the minimum index of the domain."""
        return round(self._f_min / self._delta_f)

    @property
    @override
    def max_idx(self) -> int:
        """Return the maximum index of the domain."""
        return round(self._f_max / self._delta_f)

    @property
    @override
    def noise_std(self) -> float:
        """
        Return the standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal distribution,
        you must divide by this factor. In practice, this means dividing the
        whitened waveforms by this.
        """
        if self._window_factor is None:
            raise ValueError("Window factor needs to be set for noise_std")
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)

    @property
    @override
    def f_max(self) -> float:
        """Return the maximum frequency in Hz."""
        return self._f_max

    @f_max.setter
    def f_max(self, value: float) -> None:
        """Set the maximum frequency."""
        self._f_max = float(value)

    @property
    @override
    def f_min(self) -> float:
        """Return the minimum frequency in Hz."""
        return self._f_min

    @f_min.setter
    def f_min(self, value: float) -> None:
        """Set the minimum frequency."""
        self._f_min = float(value)

    @property
    @override
    def delta_f(self) -> float:
        """Return the frequency spacing in Hz."""
        return self._delta_f

    @delta_f.setter
    def delta_f(self, value: float) -> None:
        """Set the frequency spacing."""
        self._delta_f = float(value)

    @property
    @override
    def duration(self) -> float:
        """Return the duration in seconds."""
        return 1.0 / self.delta_f

    @property
    @override
    def sampling_rate(self) -> float:
        """Return the sampling rate in Hz."""
        return 2.0 * self.f_max

    @property
    def sample_frequencies(self) -> np.ndarray:
        """Return the sample frequencies."""
        return self._sample_frequencies.get()

    @property
    def _sample_frequencies_torch(self) -> torch.Tensor:
        """Return PyTorch tensor version of frequencies."""
        if self._sample_frequencies_torch is None:
            num_bins = len(self)
            self._sample_frequencies_torch = torch.linspace(
                0.0, self.f_max, steps=num_bins, dtype=torch.float32
            )
        return self._sample_frequencies_torch

    @property
    def _sample_frequencies_torch_cuda(self) -> torch.Tensor:
        """Return CUDA version of frequency tensor."""
        if self._sample_frequencies_torch_cuda is None:
            self._sample_frequencies_torch_cuda = self.sample_frequencies_torch.to(
                "cuda"
            )
        return self._sample_frequencies_torch_cuda

    def _get_sample_frequencies_astype(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Return a 1D frequency array compatible with the last index of data array.

        Args:
            data: Input data array/tensor

        Returns:
            Frequency array compatible with data
        """
        # Type
        if isinstance(data, np.ndarray):
            f = self.sample_frequencies
        elif isinstance(data, torch.Tensor):
            if data.is_cuda:
                f = self._sample_frequencies_torch_cuda
            else:
                f = self._sample_frequencies_torch
        else:
            raise TypeError("Invalid data type. Should be np.array or torch.Tensor.")

        # Whether to include zeros below f_min
        if data.shape[-1] == len(self) - self.min_idx:
            f = f[self.min_idx :]
        elif data.shape[-1] != len(self):
            raise TypeError(
                f"Data with {data.shape[-1]} frequency bins is "
                f"incompatible with domain."
            )
        return f

    @property
    def frequency_mask(self) -> np.ndarray:
        """Return mask which selects frequency bins greater than or equal to the starting frequency."""
        return self._sample_frequencies.frequency_mask

    @property
    def frequency_mask_length(self) -> int:
        """Return number of samples in the subdomain domain[frequency_mask]."""
        mask = self.frequency_mask
        return len(np.flatnonzero(np.asarray(mask)))

    def __getitem__(self, idx):
        """Return slice of uniform frequency grid."""
        sample_frequencies = self.__call__()
        return sample_frequencies[idx]

    def update_data(
        self, data: np.ndarray, axis: int = -1, low_value: float = 0.0
    ) -> np.ndarray:
        """
        Adjust data to be compatible with the domain.

        Args:
            data: Data array
            axis: Which data axis to apply the adjustment along
            low_value: Value to set below f_min

        Returns:
            Adjusted data array
        """
        sl = [slice(None)] * data.ndim
        # First truncate beyond f_max
        sl[axis] = slice(0, self.max_idx + 1)
        data = data[tuple(sl)]
        # Set data value below f_min to low_value
        sl[axis] = slice(0, self.min_idx)
        data[tuple(sl)] = low_value
        return data

    @staticmethod
    @dispatch(np.ndarray, np.ndarray)
    def add_phase(data: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Add frequency-dependent phase to complex data (NumPy implementation).

        Args:
            data: Complex-valued frequency series
            phase: Phase array to apply

        Returns:
            Data with applied phase shift

        Raises:
            TypeError: If data is not complex
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

        Args:
            data: Complex-valued frequency series
            phase: Phase tensor to apply

        Returns:
            Data with applied phase shift
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


class TimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.
    window_factor is used to compute noise_std().
    """

    def __init__(self, time_duration: float, sampling_rate: float):
        self._time_duration = time_duration
        self._sampling_rate = sampling_rate

    @override
    def update(self) -> None:
        raise NotImplementedError("TimeDomain does not support update")

    @lru_cache()
    def __len__(self):
        """Number of time bins given duration and sampling rate"""
        return int(self._time_duration * self._sampling_rate)

    @lru_cache()
    def __call__(self) -> np.ndarray:
        """Array of uniform times at which data is sampled"""
        num_bins = self.__len__()
        return np.linspace(
            0.0,
            self._time_duration,
            num=num_bins,
            endpoint=False,
            dtype=np.float32,
        )

    @property
    def delta_t(self) -> float:
        """The size of the time bins"""
        return 1.0 / self._sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t: float):
        self._sampling_rate = 1.0 / delta_t

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        return 1.0 / np.sqrt(2.0 * self.delta_t)

    def time_translate_data(self, data, dt) -> np.ndarray:
        raise NotImplementedError

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._sampling_rate / 2.0

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return self._time_duration

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def min_idx(self) -> int:
        return 0

    @property
    def max_idx(self) -> int:
        return round(self._time_duration * self._sampling_rate)

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "TimeDomain",
            "time_duration": self._time_duration,
            "sampling_rate": self._sampling_rate,
        }


class PCADomain(Domain):
    """TODO"""

    # Not super important right now
    # FIXME: Should this be defined for FD or TD bases or both?
    # Nrb instead of Nf

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        # FIXME
        return np.sqrt(self.window_factor) / np.sqrt(4.0 * self.delta_f)


def build_domain(settings: dict) -> Domain:
    """
    Instantiate a domain class from settings.

    Parameters
    ----------
    settings : dict
        Dicionary with 'type' key denoting the type of domain, and keys corresponding
        to the kwargs needed to construct the Domain.

    Returns
    -------
    A Domain instance of the correct type.
    """
    if "type" not in settings:
        raise ValueError(
            f'Domain settings must include a "type" key. Settings included '
            f"the keys {settings.keys()}."
        )

    # The settings other than 'type' correspond to the kwargs of the Domain constructor.
    kwargs = {k: v for k, v in settings.items() if k != "type"}
    if settings["type"] in ["FrequencyDomain", "FD"]:
        return FrequencyDomain(**kwargs)
    elif settings["type"] == ["TimeDomain", "TD"]:
        return TimeDomain(**kwargs)
    else:
        raise NotImplementedError(f'Domain {settings["name"]} not implemented.')


def build_domain_from_model_metadata(model_metadata) -> Domain:
    """
    Instantiate a domain class from settings of model.

    Parameters
    ----------
    model_metadata: dict
        model metadata containing information to build the domain
        typically obtained from the model.metadata attribute

    Returns
    -------
    A Domain instance of the correct type.
    """
    domain = build_domain(model_metadata["dataset_settings"]["domain"])
    if "domain_update" in model_metadata["train_settings"]["data"]:
        domain.update(model_metadata["train_settings"]["data"]["domain_update"])
    domain.window_factor = get_window_factor(
        model_metadata["train_settings"]["data"]["window"]
    )
    return domain


if __name__ == "__main__":
    kwargs = {"f_min": 20, "f_max": 2048, "delta_f": 0.125}
    domain = FrequencyDomain(**kwargs)

    d1 = domain()
    d2 = domain()
    print("Clearing cache.", end=" ")
    domain.clear_cache_for_all_instances()
    print("Done.")
    d3 = domain()

    print("Changing domain range.", end=" ")
    domain.set_new_range(20, 100)
    print("Done.")

    d4 = domain()
    d5 = domain()

    print(len(d1), len(d4))
