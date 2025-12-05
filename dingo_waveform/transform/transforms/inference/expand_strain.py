"""
ExpandStrain transform: Expand waveform along batch axis for inference.

This transform adds a batch dimension and copies the waveform num_samples times,
useful for generating multiple samples at inference time.
"""

from dataclasses import dataclass
from typing import Dict, Any
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import TransformSample


@dataclass(frozen=True)
class ExpandStrainConfig(WaveformTransformConfig):
    """
    Configuration for ExpandStrain transform.

    Attributes
    ----------
    num_samples : int
        Number of samples to expand along batch axis

    Examples
    --------
    >>> config = ExpandStrainConfig(num_samples=100)
    >>> config.num_samples
    100
    """

    num_samples: int

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.num_samples, int):
            raise TypeError(f"num_samples must be an int, got {type(self.num_samples)}")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExpandStrainConfig':
        """
        Create ExpandStrainConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'num_samples' key

        Returns
        -------
        ExpandStrainConfig
            Validated configuration instance

        Examples
        --------
        >>> config = ExpandStrainConfig.from_dict({'num_samples': 100})
        >>> config.num_samples
        100
        """
        return cls(num_samples=config_dict['num_samples'])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = ExpandStrainConfig(num_samples=100)
        >>> config.to_dict()
        {'num_samples': 100}
        """
        return {'num_samples': self.num_samples}


class ExpandStrain(WaveformTransform[ExpandStrainConfig]):
    """
    Expand waveform by adding batch axis and copying num_samples times.

    This transform is useful for generating num_samples samples at inference time.
    The waveform is expanded along a new batch dimension:
        waveform.shape: (..., freq_bins) → (num_samples, ..., freq_bins)

    Examples
    --------
    >>> import torch
    >>> from dingo_waveform.transform.transforms.inference import (
    ...     ExpandStrain,
    ...     ExpandStrainConfig
    ... )
    >>>
    >>> config = ExpandStrainConfig(num_samples=100)
    >>> transform = ExpandStrain.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': torch.randn(3, 256),  # 3 detectors, 256 freq bins
    ...     'parameters': {'mass_1': 30.0}
    ... }
    >>> result = transform(sample)
    >>> result['waveform'].shape
    torch.Size([100, 3, 256])

    Notes
    -----
    This transform assumes the waveform is a PyTorch tensor with an .expand()
    method. It is typically used after ToTorch in the inference pipeline.

    See Also
    --------
    ToTorch : Converts numpy arrays to PyTorch tensors
    ResetSample : Resets waveform after GNPE transforms
    """

    def __init__(self, config: ExpandStrainConfig):
        """
        Initialize ExpandStrain transform.

        Parameters
        ----------
        config : ExpandStrainConfig
            Configuration specifying number of samples
        """
        super().__init__(config)

    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Apply waveform expansion.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with waveform (must be torch tensor)

        Returns
        -------
        Dict[str, Any]
            Sample with expanded waveform

        Examples
        --------
        >>> import torch
        >>> config = ExpandStrainConfig(num_samples=100)
        >>> transform = ExpandStrain.from_config(config)
        >>> sample = {'waveform': torch.randn(3, 256)}
        >>> result = transform(sample)
        >>> result['waveform'].shape
        torch.Size([100, 3, 256])
        """
        sample = input_sample.copy()
        waveform = input_sample["waveform"]
        sample["waveform"] = waveform.expand(self.config.num_samples, *waveform.shape)
        return sample
