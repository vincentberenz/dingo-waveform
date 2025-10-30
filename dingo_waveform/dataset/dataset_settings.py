"""Settings dataclass for waveform dataset generation."""

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Optional, Union

from ..domains import DomainParameters, build_domain
from ..prior import IntrinsicPriors
from .compression_settings import CompressionSettings, SVDSettings
from .waveform_generator_settings import WaveformGeneratorSettings


@dataclass
class DatasetSettings:
    """
    Configuration for waveform dataset generation.

    Attributes
    ----------
    domain
        Domain parameters as DomainParameters dataclass.
    waveform_generator
        Waveform generator configuration.
    intrinsic_prior
        Prior configuration for intrinsic parameters.
    num_samples
        Number of waveforms to generate.
    compression
        Optional compression settings (None for no compression).
    """

    domain: DomainParameters
    waveform_generator: WaveformGeneratorSettings
    intrinsic_prior: IntrinsicPriors
    num_samples: int
    compression: Optional[CompressionSettings] = None

    def __post_init__(self):
        """Validate settings after initialization and coerce types if needed."""
        # Coerce waveform_generator dict to WaveformGeneratorSettings
        if isinstance(self.waveform_generator, dict):
            from .waveform_generator_settings import WaveformGeneratorSettings as _WGS
            self.waveform_generator = _WGS(
                approximant=self.waveform_generator.get("approximant"),
                f_ref=self.waveform_generator.get("f_ref"),
                spin_conversion_phase=self.waveform_generator.get("spin_conversion_phase"),
                f_start=self.waveform_generator.get("f_start"),
            )
        # Coerce intrinsic_prior dict to IntrinsicPriors
        if isinstance(self.intrinsic_prior, dict):
            from ..prior import IntrinsicPriors as _IP
            self.intrinsic_prior = _IP(**self.intrinsic_prior)
        # Domain can be a dict or DomainParameters; both are supported by build_domain
        # Validate num_samples
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary format suitable for serialization.

        Returns
        -------
        Dict
            Dictionary representation of settings with all nested dataclasses converted.
        """
        # Convert domain to dict
        if is_dataclass(self.domain):
            domain_dict = asdict(self.domain)
        else:
            domain_dict = self.domain  # Already a dict (backward compatibility)

        result = {
            "domain": domain_dict,
            "waveform_generator": self.waveform_generator.to_dict(),
            "intrinsic_prior": asdict(self.intrinsic_prior),
            "num_samples": self.num_samples,
        }
        if self.compression is not None:
            result["compression"] = asdict(self.compression)
        return result

    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any]) -> "DatasetSettings":
        """
        Create DatasetSettings from dictionary (e.g., loaded from YAML).

        Parameters
        ----------
        settings_dict
            Dictionary with dataset settings, typically loaded from a YAML config file.

        Returns
        -------
        DatasetSettings
            Instance created from dictionary with all nested structures properly typed.
        """
        # Build domain from dict
        domain = build_domain(settings_dict["domain"])
        domain_params = domain.get_parameters()

        # Build waveform generator settings
        wfg_dict = settings_dict["waveform_generator"]
        wfg_settings = WaveformGeneratorSettings(
            approximant=wfg_dict["approximant"],
            f_ref=wfg_dict["f_ref"],
            spin_conversion_phase=wfg_dict.get("spin_conversion_phase"),
            f_start=wfg_dict.get("f_start"),
        )

        # Build intrinsic prior
        intrinsic_prior = IntrinsicPriors(**settings_dict["intrinsic_prior"])

        # Build compression settings if present
        compression = None
        if "compression" in settings_dict and settings_dict["compression"] is not None:
            comp_dict = settings_dict["compression"]
            svd_settings = None
            if "svd" in comp_dict:
                svd_dict = comp_dict["svd"]
                svd_settings = SVDSettings(
                    size=svd_dict["size"],
                    num_training_samples=svd_dict["num_training_samples"],
                    num_validation_samples=svd_dict.get("num_validation_samples", 0),
                    file=svd_dict.get("file"),
                )
            compression = CompressionSettings(
                svd=svd_settings,
                whitening=comp_dict.get("whitening"),
            )

        return cls(
            domain=domain_params,
            waveform_generator=wfg_settings,
            intrinsic_prior=intrinsic_prior,
            num_samples=settings_dict["num_samples"],
            compression=compression,
        )

    def validate(self) -> None:
        """
        Validate settings (for backward compatibility).

        Validation is now done in __post_init__ and individual dataclass validators.
        This method is kept for backward compatibility but does nothing.
        """
        pass  # Validation happens in __post_init__
