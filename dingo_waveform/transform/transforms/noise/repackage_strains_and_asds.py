"""
RepackageStrainsAndASDS transform: Repackage strains and ASDs into tensor format.

This transform converts the dictionary of strains and ASDs into a single
[num_ifos, 3, num_bins] tensor suitable for neural network input.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import InferenceSample, TensorPackedSample


@dataclass(frozen=True)
class RepackageStrainsAndASDSConfig(WaveformTransformConfig):
    """
    Configuration for RepackageStrainsAndASDS transform.

    Attributes
    ----------
    ifos : List[str]
        List of interferometer names in desired order
    first_index : int
        First frequency index to include (for truncating low frequencies)
    """

    ifos: List[str]
    first_index: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.ifos:
            raise ValueError("ifos list cannot be empty")
        if self.first_index < 0:
            raise ValueError(f"first_index must be non-negative, got {self.first_index}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RepackageStrainsAndASDSConfig':
        """Create RepackageStrainsAndASDSConfig from dictionary."""
        return cls(
            ifos=config_dict['ifos'],
            first_index=config_dict.get('first_index', 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'ifos': self.ifos,
            'first_index': self.first_index
        }


class RepackageStrainsAndASDS(WaveformTransform[RepackageStrainsAndASDSConfig]):
    """
    Repackage strains and ASDs into [num_ifos, 3, num_bins] tensor.

    The tensor format is:
        [:, 0, :] = strain.real
        [:, 1, :] = strain.imag
        [:, 2, :] = 1 / (asd * 1e23)

    This format is optimized for neural network input.
    """

    def __init__(self, config: RepackageStrainsAndASDSConfig):
        """Initialize RepackageStrainsAndASDS transform."""
        super().__init__(config)

    def __call__(self, input_sample: InferenceSample) -> TensorPackedSample:  # type: ignore[override]
        """
        Apply repackaging transform.

        Converts nested waveform and ASD dicts into single [num_ifos, 3, num_bins]
        tensor suitable for neural network input.

        Transitions from InferenceSample to TensorPackedSample by repackaging
        detector strain dicts into tensor format.
        """
        sample = input_sample.copy()

        # Determine output shape (handle potential batch dimensions)
        strains = np.empty(
            sample["asds"][self.config.ifos[0]].shape[:-1]  # Possible batch dims
            + (
                len(self.config.ifos),
                3,
                sample["asds"][self.config.ifos[0]].shape[-1] - self.config.first_index,
            ),
            dtype=np.float32,
        )

        # Fill tensor for each interferometer
        for idx_ifo, ifo in enumerate(self.config.ifos):
            strains[..., idx_ifo, 0, :] = sample["waveform"][ifo][
                ..., self.config.first_index :
            ].real
            strains[..., idx_ifo, 1, :] = sample["waveform"][ifo][
                ..., self.config.first_index :
            ].imag
            strains[..., idx_ifo, 2, :] = 1 / (
                sample["asds"][ifo][..., self.config.first_index :] * 1e23
            )

        sample["waveform"] = strains
        return sample
