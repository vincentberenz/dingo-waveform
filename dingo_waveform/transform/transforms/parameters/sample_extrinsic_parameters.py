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
    extrinsic_prior_dict : ExtrinsicPriorDict
        Dictionary specifying prior distributions for extrinsic parameters.
        This is passed to BBHExtrinsicPriorDict from dingo.gw.prior.

    Examples
    --------
    >>> config = SampleExtrinsicParametersConfig(
    ...     extrinsic_prior_dict={
    ...         'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28},
    ...         'dec': {'type': 'Cosine', 'minimum': -1.57, 'maximum': 1.57},
    ...         'psi': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 3.14}
    ...     }
    ... )
    """

    extrinsic_prior_dict: ExtrinsicPriorDict

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.extrinsic_prior_dict, dict):
            raise TypeError(
                f"extrinsic_prior_dict must be a dict, got {type(self.extrinsic_prior_dict)}"
            )
        if len(self.extrinsic_prior_dict) == 0:
            raise ValueError("extrinsic_prior_dict cannot be empty")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SampleExtrinsicParametersConfig':
        """
        Create SampleExtrinsicParametersConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'extrinsic_prior_dict' key

        Returns
        -------
        SampleExtrinsicParametersConfig
            Validated configuration instance

        Examples
        --------
        >>> config = SampleExtrinsicParametersConfig.from_dict({
        ...     'extrinsic_prior_dict': {
        ...         'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28}
        ...     }
        ... })
        """
        return cls(extrinsic_prior_dict=config_dict['extrinsic_prior_dict'])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = SampleExtrinsicParametersConfig(
        ...     extrinsic_prior_dict={'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28}}
        ... )
        >>> config.to_dict()
        {'extrinsic_prior_dict': {'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28}}}
        """
        return {'extrinsic_prior_dict': self.extrinsic_prior_dict}


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
    >>> from dingo_waveform.transform.transforms.parameters import (
    ...     SampleExtrinsicParameters,
    ...     SampleExtrinsicParametersConfig
    ... )
    >>>
    >>> config = SampleExtrinsicParametersConfig(
    ...     extrinsic_prior_dict={
    ...         'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28},
    ...         'dec': {'type': 'Cosine', 'minimum': -1.57, 'maximum': 1.57},
    ...         'psi': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 3.14},
    ...         'luminosity_distance': {'type': 'UniformSourceFrame', 'minimum': 100, 'maximum': 5000}
    ...     }
    ... )
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
    This transform uses dingo.gw.prior.BBHExtrinsicPriorDict to construct
    the prior from the configuration dictionary.

    The extrinsic_prior_dict follows bilby's prior specification format.
    Each parameter can specify:
    - type: Prior distribution type (Uniform, Cosine, etc.)
    - minimum, maximum: Bounds for bounded distributions
    - Additional distribution-specific parameters

    See Also
    --------
    StandardizeParameters : Normalizes parameters for neural network input
    GetDetectorTimes : Computes detector times from extrinsic parameters
    """

    def __init__(self, config: SampleExtrinsicParametersConfig):
        """
        Initialize SampleExtrinsicParameters transform.

        Parameters
        ----------
        config : SampleExtrinsicParametersConfig
            Configuration with extrinsic prior dictionary
        """
        super().__init__(config)

        # Initialize prior from config
        from dingo.gw.prior import BBHExtrinsicPriorDict
        self.prior = BBHExtrinsicPriorDict(config.extrinsic_prior_dict)

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
        >>> config = SampleExtrinsicParametersConfig(
        ...     extrinsic_prior_dict={'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28}}
        ... )
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
