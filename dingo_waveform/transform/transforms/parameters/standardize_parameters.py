"""
StandardizeParameters transform: Standardize parameters with (x - mu) / std.

This transform normalizes parameters according to provided means and standard
deviations, commonly used for neural network input normalization.
"""

from dataclasses import dataclass
from typing import Dict, Any
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import TransformSample


@dataclass(frozen=True)
class StandardizeParametersConfig(WaveformTransformConfig):
    """
    Configuration for StandardizeParameters transform.

    Attributes
    ----------
    mu : Dict[str, float]
        Dictionary of estimated means for each parameter
    std : Dict[str, float]
        Dictionary of estimated standard deviations for each parameter

    Examples
    --------
    >>> config = StandardizeParametersConfig(
    ...     mu={'mass_1': 35.0, 'mass_2': 30.0},
    ...     std={'mass_1': 5.0, 'mass_2': 5.0}
    ... )
    >>> config.mu['mass_1']
    35.0
    """

    mu: Dict[str, float]
    std: Dict[str, float]

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.mu, dict):
            raise TypeError(f"mu must be a dict, got {type(self.mu)}")
        if not isinstance(self.std, dict):
            raise TypeError(f"std must be a dict, got {type(self.std)}")

        if set(self.mu.keys()) != set(self.std.keys()):
            raise ValueError(
                f"Keys in mu and std must match. "
                f"mu: {self.mu.keys()}, std: {self.std.keys()}"
            )

        if len(self.mu) == 0:
            raise ValueError("mu and std cannot be empty")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StandardizeParametersConfig':
        """
        Create StandardizeParametersConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'mu' and 'std' keys

        Returns
        -------
        StandardizeParametersConfig
            Validated configuration instance

        Examples
        --------
        >>> config = StandardizeParametersConfig.from_dict({
        ...     'mu': {'mass_1': 35.0, 'mass_2': 30.0},
        ...     'std': {'mass_1': 5.0, 'mass_2': 5.0}
        ... })
        >>> config.mu['mass_1']
        35.0
        """
        return cls(mu=config_dict['mu'], std=config_dict['std'])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = StandardizeParametersConfig(
        ...     mu={'mass_1': 35.0},
        ...     std={'mass_1': 5.0}
        ... )
        >>> config.to_dict()
        {'mu': {'mass_1': 35.0}, 'std': {'mass_1': 5.0}}
        """
        return {'mu': self.mu, 'std': self.std}


class StandardizeParameters(WaveformTransform[StandardizeParametersConfig]):
    """
    Standardize parameters according to (x - mu) / std transformation.

    This transform applies z-score normalization to specified parameters,
    commonly used to normalize neural network inputs to zero mean and unit
    variance.

    Only parameters present in both the configuration (mu, std) and the
    input sample are transformed. Parameters not in the configuration are
    left unchanged.

    Examples
    --------
    >>> from dingo_waveform.transform.transforms.parameters import (
    ...     StandardizeParameters,
    ...     StandardizeParametersConfig
    ... )
    >>>
    >>> config = StandardizeParametersConfig(
    ...     mu={'mass_1': 35.0, 'mass_2': 30.0},
    ...     std={'mass_1': 5.0, 'mass_2': 5.0}
    ... )
    >>> transform = StandardizeParameters.from_config(config)
    >>>
    >>> sample = {
    ...     'parameters': {
    ...         'mass_1': 40.0,
    ...         'mass_2': 25.0,
    ...         'luminosity_distance': 1000.0  # Not in mu/std, left unchanged
    ...     }
    ... }
    >>> result = transform(sample)
    >>> result['parameters']['mass_1']  # (40 - 35) / 5 = 1.0
    1.0
    >>> result['parameters']['mass_2']  # (25 - 30) / 5 = -1.0
    -1.0
    >>> result['parameters']['luminosity_distance']
    1000.0

    Notes
    -----
    This transform provides both forward (standardize) and inverse
    (de-standardize) operations via the inverse() method.

    The inverse transform is useful for converting normalized neural network
    outputs back to physical parameter values.

    See Also
    --------
    SelectStandardizeRepackageParameters : More complex parameter transform
    """

    def __init__(self, config: StandardizeParametersConfig):
        """
        Initialize StandardizeParameters transform.

        Parameters
        ----------
        config : StandardizeParametersConfig
            Configuration with means and standard deviations
        """
        super().__init__(config)

    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Standardize parameters according to (x - mu) / std.

        Generic transform that works on any pipeline stage with parameters.

        Parameters
        ----------
        input_sample : TransformSample
            Input sample (any stage) with 'parameters' dict

        Returns
        -------
        TransformSample
            Sample with standardized parameters

        Notes
        -----
        Only parameters included in mu, std get transformed.
        Other parameters are left unchanged.

        Examples
        --------
        >>> config = StandardizeParametersConfig(
        ...     mu={'mass_1': 35.0},
        ...     std={'mass_1': 5.0}
        ... )
        >>> transform = StandardizeParameters.from_config(config)
        >>> sample = {'parameters': {'mass_1': 40.0, 'mass_2': 30.0}}
        >>> result = transform(sample)
        >>> result['parameters']['mass_1']
        1.0
        >>> result['parameters']['mass_2']  # Unchanged
        30.0
        """
        x = input_sample["parameters"]
        y = {k: (x[k] - self.config.mu[k]) / self.config.std[k] for k in self.config.mu.keys()}

        samples_out = input_sample.copy()
        samples_out["parameters"] = y
        return samples_out

    def inverse(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        De-standardize parameters: x = mu + y * std.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with standardized 'parameters' dict

        Returns
        -------
        Dict[str, Any]
            Sample with de-standardized parameters

        Notes
        -----
        Only parameters included in mu, std get transformed.

        Examples
        --------
        >>> config = StandardizeParametersConfig(
        ...     mu={'mass_1': 35.0},
        ...     std={'mass_1': 5.0}
        ... )
        >>> transform = StandardizeParameters.from_config(config)
        >>> sample = {'parameters': {'mass_1': 1.0}}  # Standardized value
        >>> result = transform.inverse(sample)
        >>> result['parameters']['mass_1']  # 35.0 + 1.0 * 5.0 = 40.0
        40.0
        """
        y = input_sample["parameters"]
        x = {k: self.config.mu[k] + y[k] * self.config.std[k] for k in self.config.mu.keys()}

        samples_out = input_sample.copy()
        samples_out["parameters"] = x
        return samples_out
