"""
Base protocol and classes for transform classes.
"""

from typing import Protocol, Dict, Any, TypeVar, Generic, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import new types from types module
from dingo_waveform.transform.types import NestedDict


class TransformProtocol(Protocol):
    """
    Protocol that all transforms must implement.
    Transforms operate on nested dictionaries containing parameters, waveforms, ASDs.
    """

    def __call__(self, input_sample: Dict[str, Any]) -> Union[Dict[str, Any], List[Any]]:
        """
        Apply transform to input sample.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Nested dictionary with structure:
            {
                "parameters": {param_name: value, ...},
                "waveform": {pol_name: array, ...},
                "extrinsic_parameters": {...},  # added by transforms
                "asds": {...},  # added by transforms
            }

        Returns
        -------
        Union[Dict[str, Any], List[Any]]
            Transformed sample (usually a copy to avoid in-place modification).
            Most transforms return Dict[str, Any].
            Exception: UnpackDict returns List[Any] for DataLoader compatibility.
        """
        ...


# Type variable for configuration generic typing
ConfigT = TypeVar('ConfigT', bound='WaveformTransformConfig')


@dataclass(frozen=True)
class WaveformTransformConfig(ABC):
    """
    Abstract base class for transform configuration dataclasses.

    All transform configs must:
    1. Be frozen dataclasses (immutable)
    2. Implement from_dict() classmethod for deserialization
    3. Implement to_dict() method for serialization
    4. Provide validation in __post_init__ if needed

    This base class ensures all transform configurations are serializable
    and can be reconstructed from dictionaries (e.g., from JSON/YAML configs).
    """

    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WaveformTransformConfig':
        """
        Create configuration from dictionary.

        This method enables deserialization from JSON/YAML configuration files
        and ensures proper type conversion of all fields.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with keys matching dataclass fields

        Returns
        -------
        WaveformTransformConfig
            Validated configuration instance

        Raises
        ------
        KeyError
            If required configuration keys are missing
        ValueError
            If validation fails in __post_init__
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        This method enables serialization to JSON/YAML format and ensures
        complex objects (e.g., Domain, InterferometerList) are properly
        converted to serializable types.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for JSON serialization
        """
        pass


class WaveformTransform(ABC, Generic[ConfigT]):
    """
    Abstract base class for all waveform transforms.

    This class provides:
    1. Type-safe configuration storage via Generic[ConfigT]
    2. Configuration-based instantiation via from_config() factory method
    3. Common interface compatible with TransformProtocol
    4. Immutable-style API (transforms return new samples)

    All subclasses must:
    1. Define a ConfigT type (config dataclass inheriting WaveformTransformConfig)
    2. Implement __init__ accepting the config
    3. Implement __call__ for transform logic
    4. Optionally override from_config for custom instantiation logic

    Pipeline Stage Typing
    ---------------------
    Transforms operate on dictionaries that evolve through distinct pipeline stages.
    Each stage has specific keys and data structures:

    1. **PolarizationSample**: waveform={'h_plus', 'h_cross'} (polarizations)
       - Initial stage after waveform generation
       - Contains intrinsic parameters and polarizations

    2. **ExtrinsicSample**: adds extrinsic_parameters
       - After SampleExtrinsicParameters
       - Adds sky location, distance, time parameters

    3. **DetectorStrainSample**: waveform={'H1', 'L1', 'V1'} (detector strains)
       - **Critical transition**: After ProjectOntoDetectors
       - Waveform keys change from polarizations to detector names
       - Extrinsic params consolidated into parameters dict

    4. **NoiseASDSample**: adds asds dict
       - After SampleNoiseASD or similar
       - ASDs for each detector (keys match waveform keys)

    5. **InferenceSample**: adds inference_parameters array
       - After SelectStandardizeRepackageParameters
       - Standardized parameters for neural network input

    6. **TensorPackedSample**: waveform as [detectors, channels, bins] tensor
       - After RepackageStrainsAndASDS
       - Flattened structure for efficient processing

    7. **TorchSample**: numpy arrays → torch tensors
       - After ToTorch
       - Final stage before neural network inference

    Subclasses should specify input/output stage types in __call__ signature
    for better type safety and IDE support. See dingo_waveform.transform.types
    for TypedDict definitions of each stage.

    Example with Pipeline Stage Types
    ----------------------------------
    >>> from dingo_waveform.transform.types import NoiseASDSample
    >>>
    >>> class WhitenStrain(WaveformTransform[WhitenStrainConfig]):
    ...     def __call__(self, input_sample: NoiseASDSample) -> NoiseASDSample:
    ...         # Mypy knows: input has waveform, asds dicts
    ...         # IDE autocomplete: input_sample['waveform']['H1']
    ...         return whitened_sample

    Example
    -------
    >>> from dataclasses import dataclass
    >>> from typing import List
    >>>
    >>> @dataclass(frozen=True)
    >>> class MyTransformConfig(WaveformTransformConfig):
    ...     selected_keys: List[str]
    ...
    ...     @classmethod
    ...     def from_dict(cls, config_dict):
    ...         return cls(selected_keys=config_dict['selected_keys'])
    ...
    ...     def to_dict(self):
    ...         return {'selected_keys': self.selected_keys}
    ...
    >>> class MyTransform(WaveformTransform[MyTransformConfig]):
    ...     def __init__(self, config: MyTransformConfig):
    ...         super().__init__(config)
    ...
    ...     def __call__(self, sample):
    ...         # Transform logic here
    ...         return sample
    ...
    >>> config = MyTransformConfig(selected_keys=['key1', 'key2'])
    >>> transform = MyTransform.from_config(config)
    >>> result = transform({'key1': 1, 'key2': 2, 'key3': 3})
    """

    def __init__(self, config: ConfigT):
        """
        Initialize transform with configuration.

        Parameters
        ----------
        config : ConfigT
            Frozen configuration dataclass instance
        """
        self.config = config

    @classmethod
    def from_config(cls, config: ConfigT) -> 'WaveformTransform[ConfigT]':
        """
        Factory method to create transform from configuration.

        This is the recommended way to instantiate transforms, providing
        a uniform interface across all transform classes.

        Parameters
        ----------
        config : ConfigT
            Configuration dataclass instance

        Returns
        -------
        WaveformTransform[ConfigT]
            Transform instance initialized with configuration

        Examples
        --------
        >>> config = MyTransformConfig(selected_keys=['a', 'b'])
        >>> transform = MyTransform.from_config(config)
        >>> isinstance(transform, MyTransform)
        True
        """
        return cls(config)

    @abstractmethod
    def __call__(self, input_sample: Dict[str, Any]) -> Union[Dict[str, Any], List[Any]]:
        """
        Apply transform to input sample.

        Subclasses must implement the transform logic. Transforms should
        generally not modify the input sample in-place but return a new
        dictionary with the transformed data.

        Subclasses can use more specific pipeline stage types (e.g., NoiseASDSample,
        InferenceSample) for input_sample and return types. These are compatible
        with Dict[str, Any] at runtime.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Nested dictionary with structure:
            {
                "parameters": {param_name: value, ...},
                "waveform": {detector: array, ...},
                "extrinsic_parameters": {param: value, ...},
                "asds": {detector: array, ...},
            }

        Returns
        -------
        Union[Dict[str, Any], List[Any]]
            Transformed sample (new dict, input preserved).
            Most subclasses should return Dict[str, Any].
            Exception: UnpackDict returns List[Any] for DataLoader compatibility.

        Notes
        -----
        The exact structure depends on which transforms have been applied
        in the pipeline. Early transforms may only have "parameters" and
        "waveform" keys, while later transforms add "extrinsic_parameters"
        and "asds" keys.
        """
        pass
