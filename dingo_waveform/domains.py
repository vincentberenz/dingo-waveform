from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


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


class FrequencyDomain(TestDomain): ...


class TimeDomain(TestDomain): ...
