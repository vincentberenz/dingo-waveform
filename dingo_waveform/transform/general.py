"""
General utility transform classes.
"""

from typing import Dict, Any, List


class UnpackDict:
    """
    Unpacks the dictionary to prepare it for final output of the dataloader.
    Only returns elements specified in selected_keys.
    """

    def __init__(self, selected_keys: List[str]) -> None:
        """
        Parameters
        ----------
        selected_keys : List[str]
            List of keys to extract from the sample dictionary
        """
        self.selected_keys = selected_keys

    def __call__(self, input_sample: Dict[str, Any]) -> List[Any]:
        """
        Apply transform to unpack dictionary to list.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample dictionary

        Returns
        -------
        List[Any]
            List of values corresponding to selected_keys
        """
        return [input_sample[k] for k in self.selected_keys]
