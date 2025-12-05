"""
SelectStandardizeRepackageParameters: Select, normalize, and repackage parameters.

This complex transform selects specified parameters, normalizes them with
z-score standardization, and repackages them into numpy arrays or tensors.
Supports both forward and inverse transformations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import pandas as pd
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import (
    StandardizationDict,
    OutputFormat,
    Device,
)


@dataclass(frozen=True)
class SelectStandardizeRepackageParametersConfig(WaveformTransformConfig):
    """
    Configuration for SelectStandardizeRepackageParameters transform.

    Attributes
    ----------
    parameters_dict : Dict[str, List[str]]
        Dictionary mapping output keys to lists of parameter names.
        E.g., {'inference_parameters': ['mass_1', 'mass_2', 'luminosity_distance']}
    standardization_dict : StandardizationDict
        Dictionary with 'mean' and 'std' subdictionaries for normalization
    inverse : bool
        If True, applies inverse transformation (de-standardization). Default False.
    as_type : Optional[OutputFormat]
        Output type for inverse transform: 'dict', 'pandas', or None. Default None.
    device : Device
        Device for torch tensors ('cpu' or 'cuda[:n]'). Default 'cpu'.
    """

    parameters_dict: Dict[str, List[str]]
    standardization_dict: StandardizationDict
    inverse: bool = False
    as_type: Optional[OutputFormat] = None
    device: Device = "cpu"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.parameters_dict, dict):
            raise TypeError(
                f"parameters_dict must be a dict, got {type(self.parameters_dict)}"
            )
        if not isinstance(self.standardization_dict, dict):
            raise TypeError(
                f"standardization_dict must be a dict, got {type(self.standardization_dict)}"
            )

        if 'mean' not in self.standardization_dict or 'std' not in self.standardization_dict:
            raise ValueError(
                "standardization_dict must contain 'mean' and 'std' keys"
            )

        mean_keys = set(self.standardization_dict['mean'].keys())
        std_keys = set(self.standardization_dict['std'].keys())
        if mean_keys != std_keys:
            raise ValueError("Keys of means and stds do not match.")

        if not isinstance(self.inverse, bool):
            raise TypeError(f"inverse must be a bool, got {type(self.inverse)}")

        if self.as_type is not None and self.as_type not in ['dict', 'pandas']:
            raise ValueError(
                f"as_type must be 'dict', 'pandas', or None, got '{self.as_type}'"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SelectStandardizeRepackageParametersConfig':
        """
        Create config from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary

        Returns
        -------
        SelectStandardizeRepackageParametersConfig
            Validated configuration instance
        """
        return cls(
            parameters_dict=config_dict['parameters_dict'],
            standardization_dict=config_dict['standardization_dict'],
            inverse=config_dict.get('inverse', False),
            as_type=config_dict.get('as_type', None),
            device=config_dict.get('device', 'cpu')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'parameters_dict': self.parameters_dict,
            'standardization_dict': self.standardization_dict,
            'inverse': self.inverse,
            'as_type': self.as_type,
            'device': self.device
        }


class SelectStandardizeRepackageParameters(WaveformTransform[SelectStandardizeRepackageParametersConfig]):
    """
    Select, normalize, and repackage parameters into arrays.

    This transform:
    1. Selects parameters specified in parameters_dict
    2. Normalizes using (p - mean) / std
    3. Repackages into numpy arrays or torch tensors

    Can also apply inverse (de-standardization) for converting network
    outputs back to physical parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo_waveform.transform.transforms.parameters import (
    ...     SelectStandardizeRepackageParameters,
    ...     SelectStandardizeRepackageParametersConfig
    ... )
    >>>
    >>> config = SelectStandardizeRepackageParametersConfig(
    ...     parameters_dict={'inference_parameters': ['mass_1', 'mass_2']},
    ...     standardization_dict={
    ...         'mean': {'mass_1': 35.0, 'mass_2': 30.0},
    ...         'std': {'mass_1': 5.0, 'mass_2': 5.0}
    ...     },
    ...     inverse=False
    ... )
    >>> transform = SelectStandardizeRepackageParameters.from_config(config)
    >>>
    >>> sample = {
    ...     'parameters': {'mass_1': 40.0, 'mass_2': 25.0, 'other': 100.0}
    ... }
    >>> result = transform(sample)
    >>> result['inference_parameters']  # Shape: (2,) with standardized values
    array([1.0, -1.0], dtype=float32)

    Notes
    -----
    Forward transform (inverse=False):
    - Looks for parameters in both parameters and extrinsic_parameters dicts
    - extrinsic_parameters supersedes parameters
    - Creates array for each key in parameters_dict

    Inverse transform (inverse=True):
    - De-standardizes: p = mean + p_normalized * std
    - Converts to specified as_type ('dict', 'pandas', or None)
    - Adjusts log_prob if present in sample
    """

    def __init__(self, config: SelectStandardizeRepackageParametersConfig):
        """Initialize transform."""
        super().__init__(config)
        self.mean = config.standardization_dict["mean"]
        self.std = config.standardization_dict["std"]
        self.N = len(self.mean.keys())

    def __call__(
        self, input_sample: Dict[str, Any], as_type: Optional[OutputFormat] = None
    ) -> Dict[str, Any]:
        """
        Apply transformation.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample
        as_type : Optional[str]
            Override for output type (inverse mode only)

        Returns
        -------
        Dict[str, Any]
            Transformed sample
        """
        if not self.config.inverse:
            return self._forward(input_sample)
        else:
            return self._inverse(input_sample, as_type)

    def _forward(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward transform: select, standardize, repackage."""
        # Merge parameters and extrinsic_parameters (extrinsic supersedes)
        if "extrinsic_parameters" in input_sample:
            full_parameters = {
                **input_sample["parameters"],
                **input_sample["extrinsic_parameters"],
            }
        else:
            full_parameters = input_sample["parameters"]

        sample = input_sample.copy()

        for k, v in self.config.parameters_dict.items():
            if len(v) > 0:
                # Determine output type based on first parameter
                first_param = full_parameters[v[0]]

                if isinstance(first_param, torch.Tensor):
                    standardized = torch.empty(
                        (*first_param.shape, len(v)),
                        dtype=torch.float32,
                        device=self.config.device,
                    )
                elif isinstance(first_param, np.ndarray):
                    standardized = np.empty(
                        (*first_param.shape, len(v)), dtype=np.float32
                    )
                else:
                    standardized = np.empty(len(v), dtype=np.float32)

                # Standardize each parameter
                for idx, par in enumerate(v):
                    if self.std[par] == 0:
                        raise ValueError(
                            f"Parameter {par} with standard deviation zero is included "
                            f"in inference parameters. This is not allowed. Please remove "
                            f"it from inference_parameters or create a new dataset where "
                            f"std({par}) is not zero."
                        )
                    standardized[..., idx] = (
                        full_parameters[par] - self.mean[par]
                    ) / self.std[par]

                sample[k] = standardized

        return sample

    def _inverse(
        self, input_sample: Dict[str, Any], as_type: Optional[OutputFormat] = None
    ) -> Dict[str, Any]:
        """Inverse transform: de-standardize and convert to specified type."""
        sample = input_sample.copy()
        inference_parameters = self.config.parameters_dict["inference_parameters"]

        parameters = input_sample["parameters"][:]
        assert parameters.shape[-1] == len(inference_parameters), (
            f"Expected {len(inference_parameters)} parameters "
            f"({inference_parameters}), but got {parameters.shape[-1]}."
        )

        # De-normalize parameters
        for idx, par in enumerate(inference_parameters):
            parameters[..., idx] = (
                parameters[..., idx] * self.std[par] + self.mean[par]
            )

        # Determine output type
        output_type = as_type if as_type is not None else self.config.as_type

        # Return normalized parameters as desired type
        if output_type is None:
            sample["parameters"] = parameters

        elif output_type == "dict":
            sample["parameters"] = {}
            for idx, par in enumerate(inference_parameters):
                sample["parameters"][par] = parameters[..., idx]

        elif output_type == "pandas":
            sample["parameters"] = pd.DataFrame(
                np.array(parameters), columns=inference_parameters
            )

        else:
            raise NotImplementedError(
                f"Unexpected type {output_type}, "
                f"expected one of [None, 'pandas', 'dict']."
            )

        # Adjust log_prob if present
        if "log_prob" in sample:
            log_std = np.sum(np.log([self.std[p] for p in inference_parameters]))
            sample["log_prob"] -= log_std

        return sample
