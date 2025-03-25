from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache, wraps
from typing import Dict, Optional, Union, cast

import numpy as np
import torch
from multipledispatch import dispatch
from typing_extensions import override

from .imports import import_entity

_module_import_path = "dingo_waveform.domains"


@dataclass
class DomainParameters:
    delta_f: Optional[float]
    f_min: Optional[float]
    f_max: Optional[float]
    delta_t: Optional[float]
    window_factor: Optional[float]
    type: Optional[str] = None


class Domain(ABC):
    """Base class representing a physical domain for data processing."""

    @abstractmethod
    def get_parameters(self) -> DomainParameters:
        """
        Get the parameters of the domain.

        Returns
        -------
        DomainParameters
            The parameters of the domain.
        """
        raise NotImplementedError(
            "Subclasses of Domain must implement the get_parameters method."
        )

    @classmethod
    @abstractmethod
    def from_parameters(cls, domain_parameters: DomainParameters) -> "Domain":
        """
        Create a domain instance from given parameters.

        Parameters
        ----------
        domain_parameters
            The parameters to create the domain.

        Returns
        -------
        Domain
            A corresponding instance of the domain.
        """
        raise NotImplementedError(
            "Subclasses of Domain must implement the from_parameters class method."
        )

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of bins or points in the domain.

        Returns
        -------
        int
            The number of bins or points in the domain.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Generate array of bins in the domain.

        Returns
        -------
        np.ndarray
            Array of bins in the domain.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args) -> None:
        """
        Update domain parameters.
        """
        raise NotImplementedError(
            "Subclasses of Domain must implement the update method"
        )

    @abstractmethod
    def time_translate_data(
        self, data: Union[np.ndarray, torch.Tensor], dt: Union[float, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Translate data in time domain by dt seconds.

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
        raise NotImplementedError

    @property
    @abstractmethod
    def noise_std(self) -> float:
        """
        Standard deviation of the noise distribution.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """
        Sampling rate of the data.
        """
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


def build_domain(domain_parameters: Union[DomainParameters, Dict]) -> Domain:
    """
    Build an instance of domain based on an instance of DomainParameters,
    or a corresponding dictionary.

    Parameters
    ----------
    domain_parameters
        The parameters to create the domain.

    Returns
    -------
    An instance of the domain.
    """

    # if as dict as been passed as argument, 'casting' is to an instance of
    # DomainParameters
    if not isinstance(domain_parameters, DomainParameters):
        try:
            domain_parameters = DomainParameters(**domain_parameters)
        except Exception as e:
            raise ValueError(
                "Constructing domain: failed to construct from dictionary "
                f"{repr(domain_parameters)}. {type(e)}: {e}"
            ) from e
    domain_parameters = cast(DomainParameters, domain_parameters)

    if domain_parameters.type is None:
        raise ValueError(
            "Constructing domain. Can not construct from "
            f"{repr(asdict(domain_parameters))}: "
            "'type' should not be None."
        )

    # type should be the import path of the domain class to construct
    # (e.g. dingo_waveform.domains.FrequencyDomain).
    # But legacy code may just have used the class name (e.g. FrequencyDomain).
    # Reconstructing the import path.
    if not "." in domain_parameters.type:
        # i.e. dingo_waveform.domains.<DomainClassName>
        domain_parameters.type = f"{_module_import_path}.{domain_parameters.type}"

    # importing the class to build
    class_, _, class_name = import_entity(domain_parameters.type)

    if not issubclass(class_, Domain):
        raise ValueError(
            f"Constructing domain: could import '{domain_parameters.type}', "
            "but this is not a subclass of 'Domain'"
        )

    try:
        instance = class_.from_parameters(domain_parameters)
    except Exception as e:
        raise RuntimeError(
            f"Constructing domain. Failed to construct {class_name} "
            f"from arguments {repr(asdict(domain_parameters))}. {type(e)}: {e}"
        ) from e

    return instance


class _CachedSampleFrequencies:
    # Will be used in the class FrequencyDomain.
    # It will maintain cached values of sample frequencies, so that
    # they do not get computed everytime they are accessed.
    # The FrequencyDomains's attribute instance of _CachedSampleFrequency
    # will be rebuilt everytime f_min, f_max or delta_f is changed, so
    # that the cache get cleared and the sample frequencies get computed
    # again if needed.

    def __init__(self, f_min: float, f_max: float, delta_f: float) -> None:
        # Vincent: f_min does not seem to be used here in the original code ?
        # self._len = int((f_max - f_min) / delta_f) + 1
        self._delta_f = delta_f
        self._f_min = f_min
        self._f_max = f_max

    def __len__(self) -> int:
        """Return the number of frequency samples."""
        return self._len

    def __eq__(self, other) -> bool:
        """
        Check if two _CachedSampleFrequencies instances are equal.

        Parameters
        ----------
        other
            The other instance to compare.

        Returns
        -------
        True if the instances are equal, False otherwise.
        """
        if not isinstance(other, _CachedSampleFrequencies):
            return False
        if any(
            [
                getattr(self, attr) != getattr(other, attr)
                for attr in ("_delta_f", "_f_min", "_f_max")
            ]
        ):
            return False
        return True

    @cached_property
    def _len(self) -> int:
        """
        Return the number of frequency samples.
        """
        return int(self._f_max / self._delta_f) + 1

    @cached_property
    def sample_frequencies(self) -> np.ndarray:
        """
        Return the sample frequencies.
        """
        return np.linspace(
            0.0, self._f_max, num=self._len, endpoint=True, dtype=np.float32
        )

    @cached_property
    def sample_frequencies_torch(self) -> torch.Tensor:
        """Create PyTorch tensor version of frequencies."""
        return torch.linspace(
            0.0, self._f_max, 
            steps=self._len, dtype=torch.float32
        )

    @cached_property
    def sample_frequency_torch_cuda(self) -> torch.Tensor:
        """
        Create CUDA version of frequency tensor.
        """
        return self.sample_frequencies_torch.to("cuda")

    @cached_property
    def frequency_mask(self) -> np.ndarray:
        """Create mask for frequencies above f_min."""
        return self.sample_frequencies >= self._f_min

    @cached_property
    def sample_frequencies_torch_cuda(self) -> torch.Tensor:
        """Create CUDA version of frequency tensor."""
        return self.sample_frequencies_torch.to("cuda")


# decorator to be used in the FrequencyDomain, for managing
# cached sample frequencies. This will decorate the setters for
# the properties f_min, f_max and delta_f. When the value of any
# of these properties change, the attribute '_cached_sample_frequencies'
# is rebuilt, so that the sample frequencies get recomputed if needed.
def _reinit_cached_sample_frequencies(func):
    @wraps(func)
    def wrapper(self, value):
        func(self, value)
        self._cached_sample_frequencies = _CachedSampleFrequencies(
            self._f_min, self._f_max, self._delta_f
        )

    return wrapper


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
        """
        Initialize a FrequencyDomain instance.

        Parameters
        ----------
        f_min
            Minimum frequency in Hz.
        f_max
            Maximum frequency in Hz.
        delta_f
            Frequency spacing in Hz.
        window_factor
            Window factor for noise calculations.
        """
        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor

        self._cached_sample_frequencies = _CachedSampleFrequencies(
            f_min, f_max, delta_f
        )

    def get_parameters(self) -> DomainParameters:
        """
        Get the parameters of the frequency domain.

        Returns
        -------
        DomainParameters
            The parameters of the frequency domain.
            Note: delta_t is computed as `0.5 / f_max`.
        """
        d = DomainParameters(
            f_max=self._f_max,
            f_min=self._f_min,
            delta_t=0.5 / self._f_max,
            delta_f=self._delta_f,
            window_factor=self._window_factor,
            # type will be "dingo_waveform.domains.FrequencyDomain"
            type=f"{_module_import_path}.FrequencyDomain",
        )
        return d

    @classmethod
    def from_parameters(
        cls, domain_parameters: DomainParameters
    ) -> "FrequencyDomain":
        """
        Create a FrequencyDomain instance from given parameters.

        Parameters
        ----------
        domain_parameters
            The parameters to create the frequency domain.

        Returns
        -------
        FrequencyDomain
            An instance of the frequency domain.
        """
        for attr in ("f_min", "f_max", "delta_f"):
            if getattr(domain_parameters, attr) is None:
                raise ValueError(
                    "Can not construct FrequencyDomain from "
                    f"{repr(asdict(domain_parameters))}: {attr} should not be None"
                )
        # type ignore: we know f_min, f_max and delta_f are not None (from the test just above)
        return cls(
            domain_parameters.f_min, domain_parameters.f_max,
            domain_parameters.delta_f  # type: ignore
        )

    @override
    def update(
        self,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        delta_f: Optional[float] = None,
    ) -> None:
        """
        Update the domain range while maintaining constraints.

        Parameters
        ----------
        f_min
            New minimum frequency (None to keep current value).
        f_max
            New maximum frequency (None to keep current value).
        delta_f
            New frequency spacing (None to keep current value).

        Raises
        ------
        ValueError
            If updated values violate domain constraints.
        """
        # note:
        if delta_f is not None and delta_f != self._delta_f:
            raise ValueError(
                "FrequencyDomain.set_new_range: "
                "can not change the value delta_f when "
                "setting a new range. "
                f"Current value: {self._delta_f}, argument: {delta_f}"
            )
        # note: this will clean up the sample frequency cache, so this function does not
        #  need to be directly decorated with @_reinit_cached_sample_frequencies
        self.set_new_range(f_min, f_max)

    def set_new_range(
        self,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        delta_f: Optional[float] = None,
    ) -> None:
        """
        Set a new range for the frequency domain.

        Parameters
        ----------
        f_min
            New minimum frequency.
        f_max
            New maximum frequency.
        delta_f
            New frequency spacing.

        Raises
        ------
        ValueError
            If the new range is not contained within the current range or if
            f_min is not lower than f_max.
        """
        if f_min and f_min < self._f_min:
            raise ValueError(
                f"frequency domain new range: new f_min ({f_min}) should be higher than current "
                f"value ({self._f_min}) - new range should be contained within current one"
            )
        if f_max and f_max > self._f_max:
            raise ValueError(
                f"frequency domain new range: new f_max ({f_max}) should be lower than current "
                f"value ({self._f_max}) - new range should be contained within current one"
            )
        f_min_ = f_min if f_min is not None else self._f_min
        f_max_ = f_max if f_max is not None else self._f_max
        if f_min_ >= f_max_:
            raise ValueError(
                f"frequency domain new range: new f_min ({f_min}) should be lower than "
                f"f_max ({f_max_})"
            )
        if f_max_ <= f_min_:
            raise ValueError(
                f"frequency domain new range: new f_max ({f_min}) should be higher than "
                f"f_min ({f_min_})"
            )
        # note: this will clean up the sample frequency cache, because
        # the f_min and f_max setters are decorated with @_reinit_cached_sample_frequencies
        self.f_min = f_min_
        self.f_max = f_max_

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
        return len(self._cached_sample_frequencies)

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
    def window_factor(self):
        """
        Get the window factor for noise calculations.

        Returns
        -------
        float
            The window factor.
        """
        return self._window_factor

    @window_factor.setter
    def window_factor(self, value: float):
        """
        Set the window factor for noise calculations.

        Parameters
        ----------
        value
            The new window factor.
        """
        self._window_factor = value

    @property
    @override
    def noise_std(self) -> float:
        """
        Return the standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal distribution,
        you must divide by this factor. In practice, this means dividing the
        waveforms by this.

        Returns
        -------
        float
            The standard deviation of the whitened noise distribution.

        Raises
        ------
        ValueError
            If window factor is not set.
        """
        if self._window_factor is None:
            raise ValueError("Window factor needs to be set for noise_std")
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)

    @property
    @override
    def f_max(self) -> float:
        """Return the maximum frequency."""
        return self._f_max

    @f_max.setter
    @_reinit_cached_sample_frequencies
    def f_max(self, value: float) -> None:
        """
        Set the maximum frequency.

        Parameters
        ----------
        value
            The new maximum frequency.
        """
        self._f_max = float(value)

    @property
    @override
    def f_min(self) -> float:
        """Return the minimum frequency."""
        return self._f_min

    @f_min.setter
    @_reinit_cached_sample_frequencies
    def f_min(self, value: float) -> None:
        """
        Set the minimum frequency.

        Parameters
        ----------
        value
            The new minimum frequency.
        """
        self._f_min = float(value)

    @property
    @override
    def delta_f(self) -> float:
        """Return the frequency spacing."""
        return self._delta_f

    @delta_f.setter
    @_reinit_cached_sample_frequencies
    def delta_f(self, value: float) -> None:
        """
        Set the frequency spacing.

        Parameters
        ----------
        value
            The new frequency spacing.
        """
        self._delta_f = float(value)

    @property
    @override
    def duration(self) -> float:
        """Return the duration."""
        return 1.0 / self.delta_f

    @property
    @override
    def sampling_rate(self) -> float:
        """Return the sampling rate."""
        return 2.0 * self.f_max

    @override
    def __call__(self) -> np.ndarray:
        """Return array of uniform frequency bins in the domain [0, f_max]."""
        return self.sample_frequencies()

    def sample_frequencies(self) -> np.ndarray:
        """
        Return the sample frequencies.

        Returns
        -------
        np.ndarray
            Array of sample frequencies.
        """
        return self._cached_sample_frequencies.sample_frequencies

    @property
    def sample_frequencies_torch(self) -> torch.Tensor:
        """
        Return the sample frequencies as a PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Tensor of sample frequencies.
        """
        return self._cached_sample_frequencies.sample_frequencies_torch

    @property
    def sample_frequencies_torch_cuda(self) -> torch.Tensor:
        """
        Return the sample frequencies as a CUDA tensor.

        Returns
        -------
        torch.Tensor
            CUDA tensor of sample frequencies.
        """
        return self._cached_sample_frequencies.sample_frequencies_torch_cuda

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
            f = self._cached_sample_frequencies.sample_frequencies
        elif isinstance(data, torch.Tensor):
            if data.is_cuda:
                f = self._cached_sample_frequencies.sample_frequencies_torch_cuda
            else:
                f = self._cached_sample_frequencies.sample_frequencies_torch
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
        """
        Return mask which selects frequency bins greater than or equal to the starting frequency.

        Returns
        -------
        np.ndarray
            Mask array for frequencies.
        """
        return self._cached_sample_frequencies.frequency_mask

    @property
    def frequency_mask_length(self) -> int:
        """
        Return number of samples in the subdomain domain[frequency_mask].

        Returns
        -------
        int
            Number of samples in the subdomain.
        """
        mask = self.frequency_mask
        return len(np.flatnonzero(np.asarray(mask)))

    def __getitem__(self, idx):
        """
        Return slice of uniform frequency grid.

        Parameters
        ----------
        idx
            Index or slice to retrieve.

        Returns
        -------
        np.ndarray
            Slice of frequency grid.
        """
        return self._cached_sample_frequencies.sample_frequencies[idx]

    def update_data(
        self, data: np.ndarray, axis: int = -1, low_value: float = 0.0
    ) -> np.ndarray:
        """
        Adjust data to be compatible with the domain.

        Parameters
        ----------
        data
            Data array.
        axis
            Which data axis to apply the adjustment along.
        low_value
            Value to set below f_min.

        Returns
        -------
        np.ndarray
            Adjusted data array.
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


class TimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.
    window_factor is used to compute noise_std().

    UNDER CONSTRUCTION.
    """

    def __init__(self, time_duration: float, sampling_rate: float):
        """
        Initialize a TimeDomain instance.

        Parameters
        ----------
        time_duration
            Duration of the time domain in seconds.
        sampling_rate
            Sampling rate in Hz.
        """
        self._time_duration = time_duration
        self._sampling_rate = sampling_rate

    @override
    def update(self) -> None:
        raise NotImplementedError("TimeDomain does not support update")

    @lru_cache()
    def __len__(self):
        """
        Number of time bins given duration and sampling rate.

        Returns
        -------
        int
            Number of time bins.
        """
        return int(self._time_duration * self._sampling_rate)

    @lru_cache()
    def __call__(self) -> np.ndarray:
        """
        Array of uniform times at which data is sampled.

        Returns
        -------
        np.ndarray
            Array of uniform times.
        """
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
        """
        The size of the time bins.

        Returns
        -------
        float
            Size of the time bins.
        """
        return 1.0 / self._sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t: float):
        """
        Set the size of the time bins.

        Parameters
        ----------
        delta_t
            The new size of the time bins.
        """
        self._sampling_rate = 1.0 / delta_t

    @property
    def noise_std(self) -> float:
        """
        Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.

        Returns
        -------
        float
            Standard deviation of the whitened noise distribution.
        """
        return 1.0 / np.sqrt(2.0 * self.delta_t)

    def time_translate_data(self, data, dt) -> np.ndarray:
        raise NotImplementedError

    @property
    def f_max(self) -> float:
        """
        The maximum frequency [Hz] is typically set to half the sampling rate.

        Returns
        -------
        float
            Maximum frequency in Hz.
        """
        return self._sampling_rate / 2.0

    @property
    def duration(self) -> float:
        """
        Waveform duration in seconds.

        Returns
        -------
        float
            Duration of the waveform.
        """
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
