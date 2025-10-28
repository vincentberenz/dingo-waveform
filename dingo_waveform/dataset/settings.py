"""Settings dataclass for waveform dataset generation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from ..domains import DomainParameters


@dataclass
class DatasetSettings:
    """
    Configuration for waveform dataset generation.

    Attributes
    ----------
    domain
        Domain parameters (dict or DomainParameters).
    waveform_generator
        Waveform generator configuration dict.
    intrinsic_prior
        Prior configuration for intrinsic parameters (dict).
    num_samples
        Number of waveforms to generate.
    compression
        Optional compression settings (None for no compression).
    """

    domain: Union[Dict[str, Any], DomainParameters]
    waveform_generator: Dict[str, Any]
    intrinsic_prior: Dict[str, Any]
    num_samples: int
    compression: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary format.

        Returns
        -------
        Dict
            Dictionary representation of settings.
        """
        from dataclasses import asdict

        settings_dict = asdict(self)
        # Convert DomainParameters to dict if needed
        if hasattr(self.domain, "__dict__"):
            settings_dict["domain"] = asdict(self.domain)
        return settings_dict

    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any]) -> "DatasetSettings":
        """
        Create DatasetSettings from dictionary.

        Parameters
        ----------
        settings_dict
            Dictionary with dataset settings.

        Returns
        -------
        DatasetSettings
            Instance created from dictionary.
        """
        return cls(
            domain=settings_dict["domain"],
            waveform_generator=settings_dict["waveform_generator"],
            intrinsic_prior=settings_dict["intrinsic_prior"],
            num_samples=settings_dict["num_samples"],
            compression=settings_dict.get("compression"),
        )

    def validate(self) -> None:
        """
        Validate settings.

        Raises
        ------
        ValueError
            If settings are invalid.
        """
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        required_domain_keys = {"type", "f_max"}
        if isinstance(self.domain, dict):
            if not required_domain_keys.issubset(set(self.domain.keys())):
                raise ValueError(
                    f"Domain dict must contain at least {required_domain_keys}"
                )

        if not isinstance(self.waveform_generator, dict):
            raise ValueError("waveform_generator must be a dictionary")

        if "approximant" not in self.waveform_generator:
            raise ValueError("waveform_generator must specify an approximant")

        if not isinstance(self.intrinsic_prior, dict):
            raise ValueError("intrinsic_prior must be a dictionary")
