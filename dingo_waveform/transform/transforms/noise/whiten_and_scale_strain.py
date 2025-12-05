"""
WhitenAndScaleStrain transform: Whiten strain and apply scaling factor.

This transform combines whitening with scaling to account for frequency binning.
"""

from dataclasses import dataclass
from typing import Dict, Any
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import NoiseASDSample


@dataclass(frozen=True)
class WhitenAndScaleStrainConfig(WaveformTransformConfig):
    """
    Configuration for WhitenAndScaleStrain transform.

    Attributes
    ----------
    scale_factor : float
        Scale factor for whitening. For uniform frequency domain, this should be
        1 / sqrt(4.0 * delta_f) to account for frequency binning.
    """

    scale_factor: float

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WhitenAndScaleStrainConfig':
        """Create WhitenAndScaleStrainConfig from dictionary."""
        return cls(scale_factor=config_dict['scale_factor'])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {'scale_factor': self.scale_factor}


class WhitenAndScaleStrain(WaveformTransform[WhitenAndScaleStrainConfig]):
    """
    Whiten strain data and scale it with 1/scale_factor.

    This transform divides strain by both the ASD and the scale factor:
        whitened_scaled_strain[ifo] = strain[ifo] / (asd[ifo] * scale_factor)

    The scale factor accounts for frequency binning. In uniform frequency domain,
    use scale_factor = 1 / sqrt(4.0 * delta_f).
    """

    def __init__(self, config: WhitenAndScaleStrainConfig):
        """Initialize WhitenAndScaleStrain transform."""
        super().__init__(config)

    def __call__(self, input_sample: NoiseASDSample) -> NoiseASDSample:  # type: ignore[override]
        """
        Apply whitening and scaling transform.

        Whitens strain with ASD and applies scale factor:
        whitened_scaled_strain = strain / (asd * scale_factor)

        Parameters
        ----------
        input_sample : NoiseASDSample
            Input sample with detector strains and ASDs

        Returns
        -------
        NoiseASDSample
            Sample with whitened and scaled strain data
        """
        sample = input_sample.copy()
        ifos = sample["waveform"].keys()

        if ifos != sample["asds"].keys():
            raise ValueError(
                f"Detectors of strain data, {ifos}, do not match "
                f'those of asds, {sample["asds"].keys()}.'
            )

        whitened_strains = {
            ifo: sample["waveform"][ifo] / (sample["asds"][ifo] * self.config.scale_factor)
            for ifo in ifos
        }
        sample["waveform"] = whitened_strains
        return sample
