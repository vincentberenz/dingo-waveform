import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Union, cast

import numpy as np
import tomli
import torch
from multipledispatch import dispatch

from ..imports import import_entity

_module_import_path = "dingo_waveform.domains"


@dataclass
class DomainParameters:
    """
    Dataclass representation of the domain.
    Domain instances can be created from instances of DomainParameters,
    or "saved" as instances of DomainParameters.

    The type, if not None, is expected to be an import path to a subclass
    of Domain. It will be used by the 'build_domain' function to instantiate
    the proper domain class.
    """

    f_max: Optional[float] = None
    delta_t: Optional[float] = None
    f_min: Optional[float] = None
    delta_f: Optional[float] = None
    window_factor: Optional[float] = None
    time_duration: Optional[float] = None
    sampling_rate: Optional[float] = None
    type: Optional[str] = None
    # MultibandedFrequencyDomain specific parameters
    nodes: Optional[list] = None
    delta_f_initial: Optional[float] = None
    base_domain: Optional[Union[Dict, "DomainParameters"]] = None

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "DomainParameters":
        """
        Create a DomainParameters instance from a TOML or JSON file.

        Parameters
        ----------
        file_path : str
            Path to the TOML or JSON file containing domain parameters.

        Returns
        -------
        DomainParameters
            An instance of DomainParameters with values loaded from the file.

        Raises
        ------
        ValueError
            If the file format is not supported or if the file cannot be parsed.
        """

        if str(file_path).lower().endswith(".json"):
            with open(file_path, "r") as f:
                params = json.load(f)
        elif str(file_path).lower().endswith(".toml"):
            with open(file_path, "rb") as f:
                params = tomli.load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path}. Only .json and .toml files are supported."
            )

        return cls(**params)


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
    def noise_std(self) -> Union[float, np.ndarray]:
        """
        Standard deviation of the noise distribution.

        Returns
        -------
        Union[float, np.ndarray]
            For uniform domains, returns a scalar float.
            For non-uniform domains (e.g., MultibandedFrequencyDomain),
            returns an array with different values per bin.
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


@dispatch(DomainParameters)
def build_domain(domain_parameters: DomainParameters) -> Domain:
    """
    Build an instance of domain based on an instance of DomainParameters.

    Parameters
    ----------
    domain_parameters
        An instance of DomainParameters

    Returns
    -------
    An instance of the domain.
    """
    if domain_parameters.type is None:
        raise ValueError(
            "Constructing domain. Can not construct from "
            f"{repr(asdict(domain_parameters))}: "
            "'type' should not be None."
        )

    if not "." in domain_parameters.type:
        domain_parameters.type = f"{_module_import_path}.{domain_parameters.type}"

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


@dispatch(dict)  # type: ignore[no-redef]
def build_domain(domain_parameters: Dict) -> Domain:
    """
    Build an instance of domain based on a dictionary with domain parameters.

    Parameters
    ----------
    domain_parameters
        A dictionary with domain parameters

    Returns
    -------
    An instance of the domain.
    """
    try:
        domain_parameters = DomainParameters(**domain_parameters)
    except Exception as e:
        raise ValueError(
            "Constructing domain: failed to construct from dictionary "
            f"{repr(domain_parameters)}. {type(e)}: {e}"
        ) from e
    return build_domain(domain_parameters)


@dispatch((str, Path))  # type: ignore[no-redef]
def build_domain(domain_parameters: Union[str, Path]) -> Domain:
    """
    Build an instance of domain based on a path to a TOML/JSON file.
    If the file has a key "domain", then it will build the domain based
    on the related sub-dictionary.

    Parameters
    ----------
    domain_parameters
        A path to a TOML or JSON file containing domain parameters

    Returns
    -------
    An instance of the domain.
    """
    try:
        # Load raw dict from file
        if str(domain_parameters).lower().endswith((".json", ".toml")):
            # Use DomainParameters.from_file to parse, but we need the raw dict to check nested "domain"
            # So re-open the file to get the dict
            dp = DomainParameters.from_file(domain_parameters)
            # Try to detect if the original file contained a nested 'domain' dict by checking attribute presence
            # Since DomainParameters doesn't have a 'domain' field, we re-load the raw file here
            import json as _json
            import tomli as _tomli
            with open(domain_parameters, "rb") as f:
                raw = _tomli.load(f) if str(domain_parameters).lower().endswith(".toml") else _json.load(open(domain_parameters, "r"))
        else:
            dp = DomainParameters.from_file(domain_parameters)
            raw = None
    except Exception as e:
        raise ValueError(
            f"Constructing domain: failed to load parameters from file "
            f"{repr(domain_parameters)}. {type(e)}: {e}"
        ) from e

    # If the file had a top-level 'domain' key, use its value
    params_dict = None
    if isinstance(raw, dict) and "domain" in raw:
        params_dict = raw["domain"]
    if params_dict is None:
        # Fall back to the parsed DomainParameters instance
        return build_domain(dp)
    else:
        return build_domain(params_dict)
