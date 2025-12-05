"""
UnpackDict transform: Unpack dictionary to list of selected values.

This transform is typically used at the end of the transform pipeline to
extract only the required keys for neural network input.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import TransformSample


@dataclass(frozen=True)
class UnpackDictConfig(WaveformTransformConfig):
    """
    Configuration for UnpackDict transform.

    Attributes
    ----------
    selected_keys : List[str]
        List of keys to extract from the sample dictionary.
        The output will be a list with values in the order of these keys.

    Examples
    --------
    >>> config = UnpackDictConfig(selected_keys=['inference_parameters', 'waveform'])
    >>> config.selected_keys
    ['inference_parameters', 'waveform']
    """

    selected_keys: List[str]

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.selected_keys:
            raise ValueError("selected_keys cannot be empty")

        if not isinstance(self.selected_keys, list):
            raise TypeError(f"selected_keys must be a list, got {type(self.selected_keys)}")

        # Check for duplicates
        if len(self.selected_keys) != len(set(self.selected_keys)):
            raise ValueError("selected_keys contains duplicate entries")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnpackDictConfig':
        """
        Create UnpackDictConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'selected_keys' key

        Returns
        -------
        UnpackDictConfig
            Validated configuration instance

        Raises
        ------
        KeyError
            If 'selected_keys' key is missing
        ValueError
            If selected_keys is empty or contains duplicates

        Examples
        --------
        >>> config_dict = {'selected_keys': ['key1', 'key2']}
        >>> config = UnpackDictConfig.from_dict(config_dict)
        >>> config.selected_keys
        ['key1', 'key2']
        """
        return cls(selected_keys=config_dict['selected_keys'])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = UnpackDictConfig(selected_keys=['a', 'b'])
        >>> config.to_dict()
        {'selected_keys': ['a', 'b']}
        """
        return {'selected_keys': self.selected_keys}


class UnpackDict(WaveformTransform[UnpackDictConfig]):
    """
    Unpacks the dictionary to prepare it for final output of the dataloader.

    This transform extracts only the specified keys from the input dictionary
    and returns them as a list. It is typically the last transform in the
    pipeline, converting the nested dictionary structure into a format
    suitable for DataLoader output.

    The transform returns a **list** (not a dict), unlike most other transforms.
    This is intentional for compatibility with PyTorch DataLoader batching.

    Examples
    --------
    >>> from dingo_waveform.transform.transforms.general import UnpackDict, UnpackDictConfig
    >>> config = UnpackDictConfig(selected_keys=['inference_parameters', 'waveform'])
    >>> transform = UnpackDict.from_config(config)
    >>> sample = {
    ...     'parameters': {'mass_1': 30.0},
    ...     'inference_parameters': [1.0, 2.0],
    ...     'waveform': [0.1, 0.2, 0.3],
    ...     'extra_data': 'ignored'
    ... }
    >>> result = transform(sample)
    >>> result
    [[1.0, 2.0], [0.1, 0.2, 0.3]]

    Notes
    -----
    This transform is unique in that it returns a List[Any] rather than
    Dict[str, Any]. This breaks the usual pattern but is necessary for
    DataLoader compatibility.

    See Also
    --------
    SelectStandardizeRepackageParameters : Prepares parameters before unpacking
    RepackageStrainsAndASDS : Prepares waveform data before unpacking
    """

    def __init__(self, config: UnpackDictConfig):
        """
        Initialize UnpackDict transform.

        Parameters
        ----------
        config : UnpackDictConfig
            Configuration specifying which keys to extract
        """
        super().__init__(config)

    def __call__(self, input_sample: TransformSample) -> List[Any]:  # type: ignore[override]
        """
        Apply transform to unpack dictionary to list.

        Can work on any pipeline stage (typically TorchSample, but accepts any stage).
        This is the only transform that returns a List instead of a dict.

        Parameters
        ----------
        input_sample : TransformSample
            Input sample dictionary from any pipeline stage, containing at least
            the keys specified in config.selected_keys

        Returns
        -------
        List[Any]
            List of values corresponding to selected_keys, in order.
            NOTE: Returns List, not Dict (unique among transforms)

        Raises
        ------
        KeyError
            If any selected_key is not present in input_sample

        Examples
        --------
        >>> config = UnpackDictConfig(selected_keys=['a', 'b'])
        >>> transform = UnpackDict.from_config(config)
        >>> result = transform({'a': 1, 'b': 2, 'c': 3})
        >>> result
        [1, 2]
        """
        try:
            return [input_sample[k] for k in self.config.selected_keys]
        except KeyError as e:
            available_keys = list(input_sample.keys())
            raise KeyError(
                f"Key {e} from selected_keys not found in input_sample. "
                f"Available keys: {available_keys}"
            ) from e
