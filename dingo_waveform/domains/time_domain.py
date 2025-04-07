from dataclasses import asdict
from functools import lru_cache

import numpy as np
from typing_extensions import override

from .domain import Domain, DomainParameters

_module_import_path = "dingo_waveform.domains"


class TimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.
    window_factor is used to compute noise_std().
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
        """
        TimeDomain does not support update.

        Raises
        ------
        A NotImplementedError
        """
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
        num_bins = len(self)
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
        """
        Returns the sampling rate.
        """
        return self._sampling_rate

    @property
    def min_idx(self) -> int:
        """Return the minimum index of the domain."""
        return 0

    @property
    def max_idx(self) -> int:
        """Return the maximum index of the domain."""
        return round(self._time_duration * self._sampling_rate)

    @override
    def get_parameters(self) -> DomainParameters:
        """
        Returns the corresponding instance of DomainParameters,
        filling the fields delta_t, f_max, time_duration and
        sampling_rate.
        """
        return DomainParameters(
            delta_t=self.delta_t,
            f_max=self.f_max,
            time_duration=self._time_duration,
            sampling_rate=self._sampling_rate,
            # type will be "dingo_waveform.domains.TimeDomain"
            type=f"{_module_import_path}.TimeDomain",
        )

    @override
    @classmethod
    def from_parameters(cls, domain_parameters: DomainParameters) -> "TimeDomain":
        """
        Construct an instance of TimeDomain from the parameters.
        It uses the fields time_duration and sampling_rate, which should not be
        None

        Raises
        ------
        A ValueError if either the file time_duration or sampling_rate is None.
        """
        for attr in ("time_duration", "sampling_rate"):
            if getattr(domain_parameters, attr) is None:
                raise ValueError(
                    "Can not construct TimeDomain from "
                    f"{repr(asdict(domain_parameters))}: {attr} should not be None"
                )
        # type ignore: we know time_duration and sampling_rate are not None
        return cls(
            domain_parameters.time_duration,  # type: ignore
            domain_parameters.sampling_rate,  # type: ignore
        )
