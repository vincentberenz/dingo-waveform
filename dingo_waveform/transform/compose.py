"""Simple composition of transforms for nested dict data structure.

This module provides TransformCompose, a replacement for torchvision.transforms.Compose
that works with gravitational wave waveform samples stored as nested dictionaries.
"""

from typing import Any, Callable, Dict, List, Union


class TransformCompose:
    """
    Composes transforms for gravitational wave nested dict samples.

    Similar to torchvision.transforms.Compose but works with nested dictionary
    structure used for gravitational wave samples:

    {
        'parameters': {...},
        'waveform': {...},
        'extrinsic_parameters': {...},
        'asds': {...},
        'inference_parameters': array,
    }

    The final transform in the chain may return a List instead of a Dict
    (e.g., UnpackDict transform).

    Parameters
    ----------
    transforms : List[Callable]
        List of transform callables to apply sequentially. Each transform
        should accept a Dict[str, Any] and return either Dict[str, Any] or
        List[Any] (if it's the final transform).

    Examples
    --------
    >>> from dingo_waveform.transform import TransformCompose
    >>> transform_chain = TransformCompose([
    ...     SampleExtrinsicParameters(config1),
    ...     ProjectOntoDetectors(config2),
    ...     WhitenAndScaleStrain(config3),
    ... ])
    >>> output_sample = transform_chain(input_sample)
    """

    def __init__(self, transforms: List[Callable[[Dict[str, Any]], Any]]):
        """
        Initialize transform composition.

        Parameters
        ----------
        transforms : List[Callable]
            List of transform callables to apply in sequence.
        """
        self.transforms = transforms

    def __call__(
        self, sample: Dict[str, Any]
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Apply all transforms in sequence.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample dictionary with nested structure.

        Returns
        -------
        Union[Dict[str, Any], List[Any]]
            Transformed sample. Usually a Dict, but may be a List if the
            final transform unpacks the dictionary (e.g., UnpackDict).
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        """
        Return string representation of the transform chain.

        Returns
        -------
        str
            Formatted string showing all transforms in the chain.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n    {0}".format(t)
        format_string += "\n)"
        return format_string
