"""
Noise-related transform classes.
"""

from typing import Dict, Any
import numpy as np
import torch
from bilby.gw.detector import PowerSpectralDensity
from scipy.interpolate import interp1d

from dingo.gw.domains import UniformFrequencyDomain
from .utils import get_batch_size_of_input_sample


class SampleNoiseASD:
    """
    Sample a batch of random ASDs for each detector and place them in sample['asds'].
    """

    def __init__(self, asd_dataset: Any) -> None:
        """
        Parameters
        ----------
        asd_dataset : ASDDataset
            Dataset of amplitude spectral densities
        """
        self.asd_dataset = asd_dataset

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply ASD sampling transform.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample

        Returns
        -------
        Dict[str, Any]
            Sample with sampled ASDs
        """
        sample = input_sample.copy()
        batched, batch_size = get_batch_size_of_input_sample(input_sample)
        sample["asds"] = self.asd_dataset.sample_random_asds(n=batch_size)
        if not batched:
            sample["asds"] = {k: v[0] for k, v in sample["asds"].items()}

        return sample


class WhitenStrain:
    """
    Whiten the strain data by dividing w.r.t. the corresponding asds.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply whitening transform.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with waveform and asds

        Returns
        -------
        Dict[str, Any]
            Sample with whitened waveform
        """
        sample = input_sample.copy()
        ifos = sample["waveform"].keys()
        if ifos != sample["asds"].keys():
            raise ValueError(
                f"Detectors of strain data, {ifos}, do not match "
                f'those of asds, {sample["asds"].keys()}.'
            )
        whitened_strains = {
            ifo: sample["waveform"][ifo] / sample["asds"][ifo] for ifo in ifos
        }
        sample["waveform"] = whitened_strains
        return sample


class WhitenFixedASD:
    """
    Whiten frequency-series data according to an ASD specified in a file. This uses the
    ASD files contained in Bilby.
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain,
        asd_file: str = None,
        inverse: bool = False,
        precision: str = None,
    ) -> None:
        """
        Parameters
        ----------
        domain : UniformFrequencyDomain
            ASD is interpolated to the associated frequency grid.
        asd_file : str
            Name of the ASD file. If None, use the aligo ASD.
        inverse : bool
            Whether to apply the inverse whitening transform, to un-whiten data.
        precision : str
            If not None, sets precision of ASD to specified precision ("single" or "double").
        """
        if asd_file is not None:
            psd = PowerSpectralDensity(asd_file=asd_file)
        else:
            psd = PowerSpectralDensity.from_aligo()

        if psd.frequency_array[-1] < domain.f_max:
            raise ValueError(
                f"ASD in {asd_file} has f_max={psd.frequency_array[-1]}, "
                f"which is lower than domain f_max={domain.f_max}."
            )
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        self.asd_array = asd_interp(domain.sample_frequencies)
        self.asd_array = domain.update_data(self.asd_array, low_value=1e-22)

        if precision is not None:
            if precision == "single":
                self.asd_array = self.asd_array.astype(np.float32)
            elif precision == "double":
                self.asd_array = self.asd_array.astype(np.float64)
            else:
                raise TypeError(
                    'precision can only be changed to "single" or "double".'
                )

        self.inverse = inverse

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply fixed ASD whitening.

        Parameters
        ----------
        sample : Dict[str, Any]
            Dictionary of arrays to whiten

        Returns
        -------
        Dict[str, Any]
            Dictionary with whitened/unwhitened data
        """
        result = {}
        for k, v in sample.items():
            if self.inverse:
                result[k] = v * self.asd_array
            else:
                result[k] = v / self.asd_array
        return result


class WhitenAndScaleStrain:
    """
    Whiten the strain data by dividing w.r.t. the corresponding asds,
    and scale it with 1/scale_factor.

    In uniform frequency domain the scale factor should be
    1 / np.sqrt(4.0 * delta_f).
    This accounts for frequency binning.
    """

    def __init__(self, scale_factor: float) -> None:
        """
        Parameters
        ----------
        scale_factor : float
            Scale factor for whitening
        """
        self.scale_factor = scale_factor

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply whitening and scaling transform.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with waveform and asds

        Returns
        -------
        Dict[str, Any]
            Sample with whitened and scaled waveform
        """
        sample = input_sample.copy()
        ifos = sample["waveform"].keys()
        if ifos != sample["asds"].keys():
            raise ValueError(
                f"Detectors of strain data, {ifos}, do not match "
                f'those of asds, {sample["asds"].keys()}.'
            )
        whitened_strains = {
            ifo: sample["waveform"][ifo] / (sample["asds"][ifo] * self.scale_factor)
            for ifo in ifos
        }
        sample["waveform"] = whitened_strains
        return sample


class AddWhiteNoiseComplex:
    """
    Adds white noise with a standard deviation determined by self.scale to the
    complex strain data.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply white noise addition.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with waveform

        Returns
        -------
        Dict[str, Any]
            Sample with noisy waveform
        """
        sample = input_sample.copy()
        noisy_strains = {}
        for ifo, pure_strain in sample["waveform"].items():
            # Use torch rng and convert to numpy, which is slightly faster than using
            # numpy directly. Using torch.randn gives single-precision floats by default
            # (which we want) whereas np.random.random gives double precision (and
            # must subsequently be cast to single precision).
            noise = (
                torch.randn(pure_strain.shape, device=torch.device("cpu"))
                + torch.randn(pure_strain.shape, device=torch.device("cpu")) * 1j
            )
            noise = noise.numpy()
            noisy_strains[ifo] = pure_strain + noise
        sample["waveform"] = noisy_strains
        return sample


class RepackageStrainsAndASDS:
    """
    Repackage the strains and the asds into an [num_ifos, 3, num_bins]
    dimensional tensor. Order of ifos is provided by self.ifos. By
    convention, [:,i,:] is used for:
        i = 0: strain.real
        i = 1: strain.imag
        i = 2: 1 / (asd * 1e23)
    """

    def __init__(self, ifos: list, first_index: int = 0) -> None:
        """
        Parameters
        ----------
        ifos : list
            List of interferometer names
        first_index : int
            First frequency index to include
        """
        self.ifos = ifos
        self.first_index = first_index

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply repackaging transform.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with waveform and asds

        Returns
        -------
        Dict[str, Any]
            Sample with repackaged waveform tensor
        """
        sample = input_sample.copy()
        strains = np.empty(
            sample["asds"][self.ifos[0]].shape[:-1]  # Possible batch dims
            + (
                len(self.ifos),
                3,
                sample["asds"][self.ifos[0]].shape[-1] - self.first_index,
            ),
            dtype=np.float32,
        )
        for idx_ifo, ifo in enumerate(self.ifos):
            strains[..., idx_ifo, 0, :] = sample["waveform"][ifo][
                ..., self.first_index :
            ].real
            strains[..., idx_ifo, 1, :] = sample["waveform"][ifo][
                ..., self.first_index :
            ].imag
            strains[..., idx_ifo, 2, :] = 1 / (
                sample["asds"][ifo][..., self.first_index :] * 1e23
            )
        sample["waveform"] = strains
        return sample
