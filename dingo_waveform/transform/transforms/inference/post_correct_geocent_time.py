"""
PostCorrectGeocentTime transform: Post-correction for geocentric time in GNPE inference.

This transform adjusts geocentric time when exact equivariance is enforced in
GNPE (Group Neural Posterior Estimation). It transfers the GNPE proxy time
offset between parameters and extrinsic_parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import TransformSample


@dataclass(frozen=True)
class PostCorrectGeocentTimeConfig(WaveformTransformConfig):
    """
    Configuration for PostCorrectGeocentTime transform.

    Attributes
    ----------
    inverse : bool
        If True, apply inverse correction (add instead of subtract).
        Default is False.

    Examples
    --------
    >>> config = PostCorrectGeocentTimeConfig(inverse=False)
    >>> config.inverse
    False

    >>> config = PostCorrectGeocentTimeConfig(inverse=True)
    >>> config.inverse
    True
    """

    inverse: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.inverse, bool):
            raise TypeError(f"inverse must be a bool, got {type(self.inverse)}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PostCorrectGeocentTimeConfig':
        """
        Create PostCorrectGeocentTimeConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with optional 'inverse' key

        Returns
        -------
        PostCorrectGeocentTimeConfig
            Validated configuration instance

        Examples
        --------
        >>> config = PostCorrectGeocentTimeConfig.from_dict({'inverse': True})
        >>> config.inverse
        True

        >>> config = PostCorrectGeocentTimeConfig.from_dict({})
        >>> config.inverse
        False
        """
        return cls(inverse=config_dict.get('inverse', False))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = PostCorrectGeocentTimeConfig(inverse=True)
        >>> config.to_dict()
        {'inverse': True}
        """
        return {'inverse': self.inverse}


class PostCorrectGeocentTime(WaveformTransform[PostCorrectGeocentTimeConfig]):
    """
    Post-correction for geocentric time in GNPE inference.

    This transform is only necessary when exact equivariance is enforced in
    GNPE (Group Neural Posterior Estimation). It adjusts the geocentric time
    by transferring the GNPE proxy offset:

    - Forward (inverse=False): parameters['geocent_time'] -= extrinsic_parameters['geocent_time']
    - Inverse (inverse=True): parameters['geocent_time'] += extrinsic_parameters['geocent_time']

    The geocent_time from extrinsic_parameters is removed after the correction.

    Examples
    --------
    >>> from dingo_waveform.transform.transforms.inference import (
    ...     PostCorrectGeocentTime,
    ...     PostCorrectGeocentTimeConfig
    ... )
    >>>
    >>> config = PostCorrectGeocentTimeConfig(inverse=False)
    >>> transform = PostCorrectGeocentTime.from_config(config)
    >>>
    >>> sample = {
    ...     'parameters': {'mass_1': 30.0, 'geocent_time': 1234567890.0},
    ...     'extrinsic_parameters': {'geocent_time': 0.05, 'ra': 1.5}
    ... }
    >>> result = transform(sample)
    >>> result['parameters']['geocent_time']
    1234567889.95
    >>> 'geocent_time' in result['extrinsic_parameters']
    False
    >>> result['extrinsic_parameters']['ra']
    1.5

    Notes
    -----
    This transform is specific to GNPE inference workflows where exact
    time-translation equivariance is enforced. In this case, a proxy
    geocentric time offset is stored in extrinsic_parameters during GNPE
    iterations, and must be corrected in the final parameters.

    The inverse operation is used when computing log probabilities, where
    the correction needs to be undone before evaluating the prior.

    See Also
    --------
    GNPECoalescenceTimes : GNPE transform that creates time proxies
    GetDetectorTimes : Computes detector-specific times from geocentric time
    """

    def __init__(self, config: PostCorrectGeocentTimeConfig):
        """
        Initialize PostCorrectGeocentTime transform.

        Parameters
        ----------
        config : PostCorrectGeocentTimeConfig
            Configuration specifying forward or inverse correction
        """
        super().__init__(config)

    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Apply geocentric time correction.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with structure:
            {
                'parameters': {'geocent_time': float, ...},
                'extrinsic_parameters': {'geocent_time': float, ...},
                ...
            }

        Returns
        -------
        Dict[str, Any]
            Sample with corrected geocentric time:
            {
                'parameters': {'geocent_time': corrected_value, ...},
                'extrinsic_parameters': {...},  # geocent_time removed
                ...
            }

        Notes
        -----
        The correction sign depends on the `inverse` parameter:
        - inverse=False: subtracts extrinsic geocent_time from parameters
        - inverse=True: adds extrinsic geocent_time to parameters

        The extrinsic_parameters['geocent_time'] is always removed (popped)
        after the correction is applied.

        Examples
        --------
        >>> config = PostCorrectGeocentTimeConfig(inverse=False)
        >>> transform = PostCorrectGeocentTime.from_config(config)
        >>> sample = {
        ...     'parameters': {'geocent_time': 10.0},
        ...     'extrinsic_parameters': {'geocent_time': 0.5}
        ... }
        >>> result = transform(sample)
        >>> result['parameters']['geocent_time']
        9.5

        >>> config = PostCorrectGeocentTimeConfig(inverse=True)
        >>> transform = PostCorrectGeocentTime.from_config(config)
        >>> result = transform(sample)
        >>> result['parameters']['geocent_time']
        10.5
        """
        # Determine sign: +1 for forward correction, -1 for inverse
        sign = (1, -1)[self.config.inverse]

        sample = input_sample.copy()
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        # Apply correction and remove geocent_time from extrinsic_parameters
        parameters["geocent_time"] -= extrinsic_parameters.pop("geocent_time") * sign

        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample
