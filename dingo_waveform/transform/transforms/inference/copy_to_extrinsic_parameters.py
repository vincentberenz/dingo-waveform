"""
CopyToExtrinsicParameters transform: Copy parameters to extrinsic_parameters dict.

This transform copies specified parameters from sample["parameters"] to
sample["extrinsic_parameters"], used in GNPE workflows.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import TransformSample


@dataclass(frozen=True)
class CopyToExtrinsicParametersConfig(WaveformTransformConfig):
    """
    Configuration for CopyToExtrinsicParameters transform.

    Attributes
    ----------
    parameter_list : Tuple[str, ...]
        Parameters to copy from parameters to extrinsic_parameters

    Examples
    --------
    >>> config = CopyToExtrinsicParametersConfig(parameter_list=('ra', 'dec'))
    >>> config.parameter_list
    ('ra', 'dec')
    """

    parameter_list: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.parameter_list, tuple):
            raise TypeError(
                f"parameter_list must be a tuple, got {type(self.parameter_list)}"
            )
        if len(self.parameter_list) == 0:
            raise ValueError("parameter_list cannot be empty")
        if not all(isinstance(p, str) for p in self.parameter_list):
            raise TypeError("All items in parameter_list must be strings")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CopyToExtrinsicParametersConfig':
        """
        Create CopyToExtrinsicParametersConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'parameter_list' key

        Returns
        -------
        CopyToExtrinsicParametersConfig
            Validated configuration instance

        Examples
        --------
        >>> config = CopyToExtrinsicParametersConfig.from_dict({
        ...     'parameter_list': ['ra', 'dec', 'geocent_time']
        ... })
        >>> config.parameter_list
        ('ra', 'dec', 'geocent_time')
        """
        parameter_list = config_dict['parameter_list']
        # Convert list to tuple for immutability
        if isinstance(parameter_list, list):
            parameter_list = tuple(parameter_list)
        return cls(parameter_list=parameter_list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = CopyToExtrinsicParametersConfig(parameter_list=('ra', 'dec'))
        >>> config.to_dict()
        {'parameter_list': ['ra', 'dec']}
        """
        return {'parameter_list': list(self.parameter_list)}


class CopyToExtrinsicParameters(WaveformTransform[CopyToExtrinsicParametersConfig]):
    """
    Copy specified parameters from parameters to extrinsic_parameters.

    This transform copies parameters specified in parameter_list from
    sample["parameters"] to sample["extrinsic_parameters"]. Only parameters
    that exist in sample["parameters"] are copied (missing parameters are
    silently skipped).

    This is commonly used in GNPE workflows to separate extrinsic parameters
    (like sky location, time) from intrinsic parameters (like masses, spins).

    Examples
    --------
    >>> from dingo_waveform.transform.transforms.inference import (
    ...     CopyToExtrinsicParameters,
    ...     CopyToExtrinsicParametersConfig
    ... )
    >>>
    >>> config = CopyToExtrinsicParametersConfig(
    ...     parameter_list=('ra', 'dec', 'geocent_time')
    ... )
    >>> transform = CopyToExtrinsicParameters.from_config(config)
    >>>
    >>> sample = {
    ...     'parameters': {
    ...         'mass_1': 30.0,
    ...         'ra': 1.5,
    ...         'dec': -0.3,
    ...         'geocent_time': 1234567890.0
    ...     },
    ...     'extrinsic_parameters': {}
    ... }
    >>> result = transform(sample)
    >>> result['extrinsic_parameters']
    {'ra': 1.5, 'dec': -0.3, 'geocent_time': 1234567890.0}
    >>> result['parameters']['mass_1']  # Intrinsic params remain
    30.0

    Notes
    -----
    This transform does NOT remove parameters from sample["parameters"].
    It only copies them to extrinsic_parameters. If a parameter in
    parameter_list is not found in sample["parameters"], it is silently
    skipped.

    See Also
    --------
    PostCorrectGeocentTime : Corrects geocent_time in GNPE workflows
    ResetSample : Resets waveform and filters extrinsic_parameters
    """

    def __init__(self, config: CopyToExtrinsicParametersConfig):
        """
        Initialize CopyToExtrinsicParameters transform.

        Parameters
        ----------
        config : CopyToExtrinsicParametersConfig
            Configuration specifying parameters to copy
        """
        super().__init__(config)

    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Apply parameter copying.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with parameters and extrinsic_parameters

        Returns
        -------
        Dict[str, Any]
            Sample with copied parameters in extrinsic_parameters

        Examples
        --------
        >>> config = CopyToExtrinsicParametersConfig(parameter_list=('ra', 'dec'))
        >>> transform = CopyToExtrinsicParameters.from_config(config)
        >>> sample = {
        ...     'parameters': {'mass_1': 30.0, 'ra': 1.5},
        ...     'extrinsic_parameters': {}
        ... }
        >>> result = transform(sample)
        >>> result['extrinsic_parameters']['ra']
        1.5
        >>> 'dec' in result['extrinsic_parameters']  # Missing param not copied
        False
        """
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        for par in self.config.parameter_list:
            if par in sample["parameters"]:
                extrinsic_parameters[par] = sample["parameters"][par]

        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample
