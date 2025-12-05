"""
SampleExtrinsicParameters transform: Sample extrinsic parameters from prior.

This transform samples extrinsic parameters (sky location, orientation, etc.)
from a specified prior distribution and adds them to the sample.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import (
    ExtrinsicPriorDict,
    PolarizationSample,
    ExtrinsicSample,
)


@dataclass(frozen=True)
class SampleExtrinsicParametersConfig(WaveformTransformConfig):
    """
    Configuration for SampleExtrinsicParameters transform.

    Attributes
    ----------
    extrinsic_prior : Any
        Prior object (duck-typed). Must have sample(n) method that returns
        a dictionary mapping parameter names to sampled values.
        Typically dingo.gw.prior.BBHExtrinsicPriorDict or bilby.core.prior.PriorDict.

    Examples
    --------
    >>> from dingo.gw.prior import BBHExtrinsicPriorDict
    >>> prior_dict = {
    ...     'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28},
    ...     'dec': {'type': 'Cosine', 'minimum': -1.57, 'maximum': 1.57},
    ...     'psi': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 3.14}
    ... }
    >>> prior = BBHExtrinsicPriorDict(prior_dict)
    >>> config = SampleExtrinsicParametersConfig(extrinsic_prior=prior)
    >>> hasattr(config.extrinsic_prior, 'sample')
    True

    Notes
    -----
    This configuration cannot be serialized to JSON/YAML since it contains
    an object reference. Use factory functions from transform.factory module
    to build transform chains with explicit object dependencies.
    """

    extrinsic_prior: Any

    def __post_init__(self) -> None:
        """Validate configuration using duck typing."""
        if not hasattr(self.extrinsic_prior, 'sample'):
            raise TypeError(
                f"extrinsic_prior must have sample() method, "
                f"got {type(self.extrinsic_prior)}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SampleExtrinsicParametersConfig':
        """
        Create SampleExtrinsicParametersConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'extrinsic_prior' key containing
            a prior object (NOT a dictionary specification)

        Returns
        -------
        SampleExtrinsicParametersConfig
            Validated configuration instance

        Examples
        --------
        >>> from dingo.gw.prior import BBHExtrinsicPriorDict
        >>> prior = BBHExtrinsicPriorDict({'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28}})
        >>> config = SampleExtrinsicParametersConfig.from_dict({
        ...     'extrinsic_prior': prior
        ... })

        Notes
        -----
        The extrinsic_prior value must be an object (not a dict specification).
        Users are responsible for loading the prior before calling this method.
        """
        return cls(extrinsic_prior=config_dict['extrinsic_prior'])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Notes
        -----
        This returns a dict with the object reference, which cannot be
        serialized to JSON/YAML. This method exists for API compatibility
        but should not be used for persistence.
        """
        return {'extrinsic_prior': self.extrinsic_prior}


class SampleExtrinsicParameters(WaveformTransform[SampleExtrinsicParametersConfig]):
    """
    Sample extrinsic parameters from prior and add to sample.

    This transform samples extrinsic parameters (sky location, polarization angle,
    luminosity distance, etc.) from a specified prior distribution. The sampled
    parameters are added to sample['extrinsic_parameters'].

    The transform detects whether the input is batched (contains multiple samples)
    and samples appropriately:
    - Batched input: samples batch_size parameter sets
    - Single input: samples one parameter set

    Examples
    --------
    >>> from dingo.gw.prior import BBHExtrinsicPriorDict
    >>> from dingo_waveform.transform.transforms.parameters import (
    ...     SampleExtrinsicParameters,
    ...     SampleExtrinsicParametersConfig
    ... )
    >>>
    >>> # Load prior first
    >>> prior_dict = {
    ...     'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28},
    ...     'dec': {'type': 'Cosine', 'minimum': -1.57, 'maximum': 1.57},
    ...     'psi': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 3.14},
    ...     'luminosity_distance': {'type': 'UniformSourceFrame', 'minimum': 100, 'maximum': 5000}
    ... }
    >>> prior = BBHExtrinsicPriorDict(prior_dict)
    >>> config = SampleExtrinsicParametersConfig(extrinsic_prior=prior)
    >>> transform = SampleExtrinsicParameters.from_config(config)
    >>>
    >>> # Single sample
    >>> sample = {
    ...     'parameters': {'mass_1': 30.0, 'mass_2': 25.0},
    ...     'waveform': np.random.randn(1024)
    ... }
    >>> result = transform(sample)
    >>> 'extrinsic_parameters' in result
    True
    >>> 'ra' in result['extrinsic_parameters']
    True

    >>> # Batched sample
    >>> batch_sample = {
    ...     'parameters': {'mass_1': np.array([30.0, 35.0]), 'mass_2': np.array([25.0, 30.0])},
    ...     'waveform': np.random.randn(2, 1024)
    ... }
    >>> result = transform(batch_sample)
    >>> result['extrinsic_parameters']['ra'].shape
    (2,)

    Notes
    -----
    The extrinsic_prior object is passed directly to the config and must be
    loaded before creating the transform. This makes dependencies explicit
    and allows the transform package to be standalone (no dingo imports).

    The prior object must have a sample(n) method that returns a dictionary
    mapping parameter names to sampled values. Typical implementations include:
    - dingo.gw.prior.BBHExtrinsicPriorDict
    - bilby.core.prior.PriorDict

    See Also
    --------
    StandardizeParameters : Normalizes parameters for neural network input
    GetDetectorTimes : Computes detector times from extrinsic parameters
    """

    def __init__(self, prior_or_config):
        """
        Initialize SampleExtrinsicParameters transform.

        Parameters
        ----------
        prior_or_config : Any or SampleExtrinsicParametersConfig
            Either an extrinsic prior object (with sample method)
            or a SampleExtrinsicParametersConfig instance.

        Notes
        -----
        The prior object is validated via duck typing (checking for sample method).

        Examples
        --------
        >>> from dingo.gw.prior import BBHExtrinsicPriorDict
        >>> prior_dict = {'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28}}
        >>> prior = BBHExtrinsicPriorDict(prior_dict)
        >>>
        >>> # Direct instantiation (recommended)
        >>> transform = SampleExtrinsicParameters(prior)
        >>>
        >>> # Or with config object
        >>> config = SampleExtrinsicParametersConfig(extrinsic_prior=prior)
        >>> transform = SampleExtrinsicParameters(config)
        """
        # Handle both direct object and config wrapper
        if isinstance(prior_or_config, SampleExtrinsicParametersConfig):
            config = prior_or_config
        else:
            # Assume it's a prior object - wrap in config
            config = SampleExtrinsicParametersConfig(extrinsic_prior=prior_or_config)

        super().__init__(config)

        # Store reference to prior object
        self.prior = config.extrinsic_prior

    def __call__(self, input_sample: PolarizationSample) -> ExtrinsicSample:  # type: ignore[override]
        """
        Sample extrinsic parameters and add to sample.

        Samples extrinsic parameters from prior and adds them to the sample,
        transitioning from PolarizationSample to ExtrinsicSample.

        Parameters
        ----------
        input_sample : PolarizationSample
            Input sample with parameters and polarization waveforms
            (can be batched or single)

        Returns
        -------
        ExtrinsicSample
            Sample with 'extrinsic_parameters' dict added

        Examples
        --------
        >>> from dingo.gw.prior import BBHExtrinsicPriorDict
        >>> prior = BBHExtrinsicPriorDict({'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28}})
        >>> config = SampleExtrinsicParametersConfig(extrinsic_prior=prior)
        >>> transform = SampleExtrinsicParameters.from_config(config)
        >>> sample = {'parameters': {'mass_1': 30.0}}
        >>> result = transform(sample)
        >>> 'extrinsic_parameters' in result
        True
        >>> 0.0 <= result['extrinsic_parameters']['ra'] <= 6.28
        True
        """
        from dingo_waveform.transform.utils import get_batch_size_of_input_sample

        sample = input_sample.copy()
        batched, batch_size = get_batch_size_of_input_sample(input_sample)

        # Sample from prior (None means single sample)
        extrinsic_parameters = self.prior.sample(batch_size if batched else None)

        # Convert to appropriate precision
        extrinsic_parameters = {
            k: v.astype(np.float32) if batched else float(v)
            for k, v in extrinsic_parameters.items()
        }

        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample
