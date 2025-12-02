"""
Inference-specific transform classes.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import torch


class PostCorrectGeocentTime:
    """
    Post correction for geocent time: add GNPE proxy (only necessary if exact
    equivariance is enforced)
    """

    def __init__(self, inverse: bool = False) -> None:
        """
        Parameters
        ----------
        inverse : bool
            If True, apply inverse correction
        """
        self.inverse = inverse

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply geocent time correction.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with parameters and extrinsic_parameters

        Returns
        -------
        Dict[str, Any]
            Sample with corrected geocent_time
        """
        sign = (1, -1)[self.inverse]
        sample = input_sample.copy()
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        parameters["geocent_time"] -= extrinsic_parameters.pop("geocent_time") * sign
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class CopyToExtrinsicParameters:
    """
    Copy parameters specified in self.parameter_list from sample["parameters"] to
    sample["extrinsic_parameters"].
    """

    def __init__(self, *parameter_list: str) -> None:
        """
        Parameters
        ----------
        *parameter_list : str
            Parameters to copy to extrinsic_parameters
        """
        self.parameter_list = parameter_list

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter copying.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with parameters

        Returns
        -------
        Dict[str, Any]
            Sample with copied parameters in extrinsic_parameters
        """
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        for par in self.parameter_list:
            if par in sample["parameters"]:
                extrinsic_parameters[par] = sample["parameters"][par]
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class ExpandStrain:
    """
    Expand the waveform of sample by adding a batch axis and copying the waveform
    num_samples times along this new axis. This is useful for generating num_samples
    samples at inference time.
    """

    def __init__(self, num_samples: int) -> None:
        """
        Parameters
        ----------
        num_samples : int
            Number of samples to expand along batch axis
        """
        self.num_samples = num_samples

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply waveform expansion.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with waveform

        Returns
        -------
        Dict[str, Any]
            Sample with expanded waveform
        """
        sample = input_sample.copy()
        waveform = input_sample["waveform"]
        sample["waveform"] = waveform.expand(self.num_samples, *waveform.shape)
        return sample


class ToTorch:
    """
    Convert all numpy arrays in sample to torch tensors and push them to the specified
    device. All items of sample that are not numpy arrays (e.g., dicts of arrays)
    remain unchanged.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        device : str
            Device to move tensors to ("cpu" or "cuda")
        """
        self.device = device

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply numpy to torch conversion.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with numpy arrays

        Returns
        -------
        Dict[str, Any]
            Sample with torch tensors
        """
        sample = input_sample.copy()
        for k, v in sample.items():
            if type(v) == np.ndarray:
                sample[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
        return sample


class ResetSample:
    """
    Resets sample:
        * waveform was potentially modified by gnpe transforms, so reset to waveform_
        * optionally remove all non-required extrinsic parameters
    """

    def __init__(
        self, extrinsic_parameters_keys: Optional[List[str]] = None
    ) -> None:
        """
        Parameters
        ----------
        extrinsic_parameters_keys : Optional[List[str]]
            If provided, only keep these keys in extrinsic_parameters
        """
        self.extrinsic_parameters_keys = extrinsic_parameters_keys

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply sample reset.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with waveform_ and extrinsic_parameters

        Returns
        -------
        Dict[str, Any]
            Sample with reset waveform and filtered extrinsic_parameters
        """
        sample = input_sample.copy()
        # reset the waveform
        sample["waveform"] = sample["waveform_"].clone()
        # optionally remove all non-required extrinsic parameters
        if self.extrinsic_parameters_keys is not None:
            sample["extrinsic_parameters"] = {
                k: sample["extrinsic_parameters"][k]
                for k in self.extrinsic_parameters_keys
            }
        return sample
