from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from nptyping import Float32, NDArray, Shape


@dataclass
class DomainParameters:
    delta_f: Optional[float]
    f_min: Optional[float]
    f_max: Optional[float]
    # f_ref: Optional[float]
    delta_t: Optional[float]


class Domain(ABC):

    @abstractmethod
    def get_parameters(self) -> DomainParameters:
        raise NotImplementedError(
            "Subclasses of Domain must implement the get_parameters method."
        )

    @abstractmethod
    def sample_frequencies(self) -> NDArray[Shape["*"], Float32]:
        raise NotImplementedError(
            "Subclasses of Domain must implement the sample_frequencies method."
        )


class TestDomain(Domain):

    def __init__(
        self,
        delta_f: Optional[float] = None,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        # f_ref: Optional[float] = None,
        delta_t: Optional[float] = None,
    ):
        self.delta_f = delta_f
        self.f_min = f_min
        self.f_max = f_max
        # self.f_ref = f_ref
        self.delta_t = delta_t

    def get_parameters(self) -> DomainParameters:
        return DomainParameters(self.delta_f, self.f_min, self.f_max, self.delta_t)


class FrequencyDomain(TestDomain):

    def sample_frequencies(self) -> NDArray[Shape["*"], Float32]:
        if self.f_max is None or self.delta_f is None:
            raise ValueError("can not sample frequencies if f_max or delta_f is None")
        num_bins = int(self.f_max / self.delta_f) + 1
        return np.linspace(
            0.0, self.f_max, num=num_bins, endpoint=True, dtype=np.float32
        )

    def __len__(self):
        """Number of frequency bins in the domain [0, f_max]"""
        return int(self.f_max / self.delta_f) + 1

    def frequency_mask(self):
        return self.sample_frequencies() >= self.f_min


class TimeDomain(TestDomain): ...
