from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union, cast

import numpy as np
import torch

from ..imports import import_entity

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
