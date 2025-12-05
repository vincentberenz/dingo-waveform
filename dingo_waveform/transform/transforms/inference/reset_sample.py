"""
ResetSample transform: Reset waveform and filter extrinsic parameters.

This transform resets the waveform (potentially modified by GNPE transforms)
to waveform_ and optionally filters extrinsic_parameters to keep only
required keys.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import TransformSample


@dataclass(frozen=True)
class ResetSampleConfig(WaveformTransformConfig):
    """
    Configuration for ResetSample transform.

    Attributes
    ----------
    extrinsic_parameters_keys : Optional[List[str]]
        If provided, only keep these keys in extrinsic_parameters.
        If None, keep all extrinsic_parameters.

    Examples
    --------
    >>> config = ResetSampleConfig(extrinsic_parameters_keys=['ra', 'dec'])
    >>> config.extrinsic_parameters_keys
    ['ra', 'dec']

    >>> config = ResetSampleConfig(extrinsic_parameters_keys=None)
    >>> config.extrinsic_parameters_keys is None
    True
    """

    extrinsic_parameters_keys: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.extrinsic_parameters_keys is not None:
            if not isinstance(self.extrinsic_parameters_keys, list):
                raise TypeError(
                    f"extrinsic_parameters_keys must be a list or None, "
                    f"got {type(self.extrinsic_parameters_keys)}"
                )
            if not all(isinstance(k, str) for k in self.extrinsic_parameters_keys):
                raise TypeError(
                    "All items in extrinsic_parameters_keys must be strings"
                )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ResetSampleConfig':
        """
        Create ResetSampleConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with optional 'extrinsic_parameters_keys' key

        Returns
        -------
        ResetSampleConfig
            Validated configuration instance

        Examples
        --------
        >>> config = ResetSampleConfig.from_dict({
        ...     'extrinsic_parameters_keys': ['ra', 'dec']
        ... })
        >>> config.extrinsic_parameters_keys
        ['ra', 'dec']

        >>> config = ResetSampleConfig.from_dict({})
        >>> config.extrinsic_parameters_keys is None
        True
        """
        return cls(
            extrinsic_parameters_keys=config_dict.get('extrinsic_parameters_keys', None)
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = ResetSampleConfig(extrinsic_parameters_keys=['ra', 'dec'])
        >>> config.to_dict()
        {'extrinsic_parameters_keys': ['ra', 'dec']}

        >>> config = ResetSampleConfig(extrinsic_parameters_keys=None)
        >>> config.to_dict()
        {'extrinsic_parameters_keys': None}
        """
        return {'extrinsic_parameters_keys': self.extrinsic_parameters_keys}


class ResetSample(WaveformTransform[ResetSampleConfig]):
    """
    Reset waveform and optionally filter extrinsic_parameters.

    This transform performs two operations:
    1. Resets sample["waveform"] to sample["waveform_"] (using .clone())
    2. Optionally filters extrinsic_parameters to keep only specified keys

    The waveform reset is necessary because GNPE transforms may modify the
    waveform, and we need to restore the original for certain operations.

    Examples
    --------
    >>> import torch
    >>> from dingo_waveform.transform.transforms.inference import (
    ...     ResetSample,
    ...     ResetSampleConfig
    ... )
    >>>
    >>> # Filter extrinsic parameters
    >>> config = ResetSampleConfig(extrinsic_parameters_keys=['ra', 'dec'])
    >>> transform = ResetSample.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': torch.randn(3, 256),
    ...     'waveform_': torch.randn(3, 256),
    ...     'extrinsic_parameters': {
    ...         'ra': 1.5,
    ...         'dec': -0.3,
    ...         'geocent_time': 1234567890.0
    ...     }
    ... }
    >>> result = transform(sample)
    >>> # Waveform is reset to waveform_
    >>> torch.allclose(result['waveform'], sample['waveform_'])
    True
    >>> # Only ra and dec kept in extrinsic_parameters
    >>> list(result['extrinsic_parameters'].keys())
    ['ra', 'dec']

    >>> # Keep all extrinsic parameters
    >>> config = ResetSampleConfig(extrinsic_parameters_keys=None)
    >>> transform = ResetSample.from_config(config)
    >>> result = transform(sample)
    >>> list(result['extrinsic_parameters'].keys())
    ['ra', 'dec', 'geocent_time']

    Notes
    -----
    This transform assumes waveform is a PyTorch tensor with .clone() method.
    It is typically used in GNPE inference workflows after GNPE transforms
    have potentially modified the waveform.

    The waveform_ key must exist in the sample, or this transform will fail.

    See Also
    --------
    CopyToExtrinsicParameters : Copies parameters to extrinsic_parameters
    PostCorrectGeocentTime : GNPE geocentric time correction
    ExpandStrain : Expands waveform along batch axis
    """

    def __init__(self, config: ResetSampleConfig):
        """
        Initialize ResetSample transform.

        Parameters
        ----------
        config : ResetSampleConfig
            Configuration specifying which extrinsic parameters to keep
        """
        super().__init__(config)

    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Apply sample reset.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with waveform_, waveform, and extrinsic_parameters

        Returns
        -------
        Dict[str, Any]
            Sample with reset waveform and filtered extrinsic_parameters

        Examples
        --------
        >>> import torch
        >>> config = ResetSampleConfig(extrinsic_parameters_keys=['ra'])
        >>> transform = ResetSample.from_config(config)
        >>> sample = {
        ...     'waveform': torch.randn(3, 256),
        ...     'waveform_': torch.randn(3, 256),
        ...     'extrinsic_parameters': {'ra': 1.5, 'dec': -0.3}
        ... }
        >>> result = transform(sample)
        >>> list(result['extrinsic_parameters'].keys())
        ['ra']
        """
        sample = input_sample.copy()

        # Reset the waveform to waveform_
        sample["waveform"] = sample["waveform_"].clone()

        # Optionally filter extrinsic_parameters
        if self.config.extrinsic_parameters_keys is not None:
            sample["extrinsic_parameters"] = {
                k: sample["extrinsic_parameters"][k]
                for k in self.config.extrinsic_parameters_keys
            }

        return sample
