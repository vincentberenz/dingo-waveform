"""
ToTorch transform: Convert numpy arrays to PyTorch tensors.

This transform converts all numpy arrays in the sample dictionary to PyTorch
tensors and moves them to the specified device (CPU or CUDA).
"""

from dataclasses import dataclass
from typing import Dict, Any, Union
import numpy as np
import torch
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import (
    Device,
    InferenceSample,
    TensorPackedSample,
    TorchSample,
)


@dataclass(frozen=True)
class ToTorchConfig(WaveformTransformConfig):
    """
    Configuration for ToTorch transform.

    Attributes
    ----------
    device : Device
        Device to move tensors to. Valid values are "cpu" or "cuda[:n]"
        where n is the GPU index (default: "cpu").

    Examples
    --------
    >>> config = ToTorchConfig(device="cpu")
    >>> config.device
    'cpu'

    >>> config = ToTorchConfig(device="cuda:0")
    >>> config.device
    'cuda:0'
    """

    device: Device = "cpu"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.device, str):
            raise TypeError(f"device must be a string, got {type(self.device)}")

        # Basic validation of device string
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError(
                f"device must be 'cpu' or 'cuda[:n]', got '{self.device}'"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ToTorchConfig':
        """
        Create ToTorchConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with optional 'device' key

        Returns
        -------
        ToTorchConfig
            Validated configuration instance

        Examples
        --------
        >>> config = ToTorchConfig.from_dict({'device': 'cuda:0'})
        >>> config.device
        'cuda:0'

        >>> config = ToTorchConfig.from_dict({})  # Uses default
        >>> config.device
        'cpu'
        """
        return cls(device=config_dict.get('device', 'cpu'))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = ToTorchConfig(device="cuda:0")
        >>> config.to_dict()
        {'device': 'cuda:0'}
        """
        return {'device': self.device}


class ToTorch(WaveformTransform[ToTorchConfig]):
    """
    Convert all numpy arrays in sample to torch tensors and push to device.

    This transform iterates through the sample dictionary and converts any
    numpy arrays to PyTorch tensors. The tensors are moved to the specified
    device (CPU or CUDA). Non-array items (e.g., nested dicts, scalars) are
    left unchanged.

    The conversion uses non-blocking transfers for efficiency when moving
    to CUDA devices.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo_waveform.transform.transforms.inference import ToTorch, ToTorchConfig
    >>>
    >>> config = ToTorchConfig(device="cpu")
    >>> transform = ToTorch.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': np.array([1.0, 2.0, 3.0]),
    ...     'parameters': {'mass': 30.0},  # Dict left unchanged
    ...     'data': np.array([[1, 2], [3, 4]])
    ... }
    >>> result = transform(sample)
    >>> isinstance(result['waveform'], torch.Tensor)
    True
    >>> result['waveform'].device
    device(type='cpu')
    >>> isinstance(result['parameters'], dict)
    True

    Notes
    -----
    Only top-level numpy arrays are converted. Arrays nested within
    dictionaries (e.g., sample['waveform']['H1']) are not automatically
    converted. For such cases, use RepackageStrainsAndASDS first to
    flatten the structure.

    The transform uses `non_blocking=True` for CUDA transfers, which
    allows asynchronous data transfer while computation continues.

    See Also
    --------
    RepackageStrainsAndASDS : Flattens nested detector dictionaries
    """

    def __init__(self, config: ToTorchConfig):
        """
        Initialize ToTorch transform.

        Parameters
        ----------
        config : ToTorchConfig
            Configuration specifying target device
        """
        super().__init__(config)

    def __call__(
        self, input_sample: Union[InferenceSample, TensorPackedSample]
    ) -> TorchSample:  # type: ignore[override]
        """
        Apply numpy to torch conversion.

        Transitions from InferenceSample or TensorPackedSample to TorchSample
        by converting numpy arrays to PyTorch tensors on the specified device.

        Parameters
        ----------
        input_sample : Union[InferenceSample, TensorPackedSample]
            Input sample with numpy arrays and other data

        Returns
        -------
        TorchSample
            Sample with numpy arrays converted to torch tensors on device.
            Non-array items are left unchanged.

        Examples
        --------
        >>> config = ToTorchConfig(device="cpu")
        >>> transform = ToTorch.from_config(config)
        >>> sample = {
        ...     'array': np.array([1, 2, 3]),
        ...     'scalar': 5.0,
        ...     'dict': {'nested': 'value'}
        ... }
        >>> result = transform(sample)
        >>> type(result['array'])
        <class 'torch.Tensor'>
        >>> type(result['scalar'])
        <class 'float'>
        >>> type(result['dict'])
        <class 'dict'>
        """
        sample = input_sample.copy()

        for k, v in sample.items():
            # Only convert numpy arrays, leave other types unchanged
            if type(v) == np.ndarray:
                sample[k] = torch.from_numpy(v).to(
                    self.config.device, non_blocking=True
                )

        return sample
