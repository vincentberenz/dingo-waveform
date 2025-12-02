"""
Base protocol for transform classes.
"""

from typing import Protocol, Dict, Any


class TransformProtocol(Protocol):
    """
    Protocol that all transforms must implement.
    Transforms operate on nested dictionaries containing parameters, waveforms, ASDs.
    """

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
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
        Dict[str, Any]
            Transformed sample (usually a copy to avoid in-place modification)
        """
        ...
