from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DomainParameters:
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    delta_t: float


class Domain(ABC):

    @abstractmethod
    def get_parameters(self) -> DomainParameters:
        raise NotImplementedError(
            "Subclasses of Domain must implement the get_parameters method."
        )


class FrequencyDomain(Domain):

    def get_parameters(self) -> DomainParameters:
        return DomainParameters(0, 0, 0, 0, 0)


class TimeDomain(Domain):

    def get_parameters(self) -> DomainParameters:
        return DomainParameters(0, 0, 0, 0, 0)
