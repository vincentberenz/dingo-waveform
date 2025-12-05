"""
AddWhiteNoiseComplex transform: Add complex white noise to strain data.

This transform adds complex Gaussian white noise to the strain data in each detector.
The noise has unit variance and is added to both real and imaginary components.
"""

from dataclasses import dataclass
from typing import Dict, Any, Union
import torch
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DetectorStrainSample, NoiseASDSample


@dataclass(frozen=True)
class AddWhiteNoiseComplexConfig(WaveformTransformConfig):
    """
    Configuration for AddWhiteNoiseComplex transform.

    This transform has no parameters - it adds standard complex white noise
    (unit variance) to strain data.

    Examples
    --------
    >>> config = AddWhiteNoiseComplexConfig()
    >>> config.to_dict()
    {}
    """

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AddWhiteNoiseComplexConfig':
        """
        Create AddWhiteNoiseComplexConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary (can be empty for this transform)

        Returns
        -------
        AddWhiteNoiseComplexConfig
            Configuration instance

        Examples
        --------
        >>> config = AddWhiteNoiseComplexConfig.from_dict({})
        >>> isinstance(config, AddWhiteNoiseComplexConfig)
        True
        """
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Empty dictionary (this transform has no parameters)

        Examples
        --------
        >>> config = AddWhiteNoiseComplexConfig()
        >>> config.to_dict()
        {}
        """
        return {}


class AddWhiteNoiseComplex(WaveformTransform[AddWhiteNoiseComplexConfig]):
    """
    Adds white noise with unit standard deviation to complex strain data.

    The noise is complex Gaussian with independent real and imaginary components,
    each drawn from N(0, 1). This results in total power of 2 per frequency bin.

    The implementation uses PyTorch's random number generator for efficiency
    (single-precision floats by default) and converts to numpy arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo_waveform.transform.transforms.noise import (
    ...     AddWhiteNoiseComplex,
    ...     AddWhiteNoiseComplexConfig
    ... )
    >>>
    >>> config = AddWhiteNoiseComplexConfig()
    >>> transform = AddWhiteNoiseComplex.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': {
    ...         'H1': np.array([1.0+0j, 2.0+0j, 3.0+0j]),
    ...         'L1': np.array([4.0+0j, 5.0+0j, 6.0+0j])
    ...     }
    ... }
    >>> result = transform(sample)
    >>> # Result has noise added (not shown due to randomness)
    >>> result['waveform']['H1'].shape
    (3,)
    >>> result['waveform']['H1'].dtype
    dtype('complex128')

    Notes
    -----
    This transform is typically applied after whitening and scaling the strain.
    The unit variance noise assumption relies on proper whitening having been
    performed earlier in the pipeline.

    The noise generation uses torch.randn which produces single-precision
    floats by default. This is faster than numpy and automatically provides
    the correct precision for neural network training.

    See Also
    --------
    WhitenAndScaleStrain : Whitening that prepares data for noise addition
    SampleNoiseASD : Samples ASDs for whitening
    """

    def __init__(self, config: AddWhiteNoiseComplexConfig):
        """
        Initialize AddWhiteNoiseComplex transform.

        Parameters
        ----------
        config : AddWhiteNoiseComplexConfig
            Configuration (has no parameters for this transform)
        """
        super().__init__(config)

    def __call__(
        self,
        input_sample: Union[DetectorStrainSample, NoiseASDSample]
    ) -> Union[DetectorStrainSample, NoiseASDSample]:  # type: ignore[override]
        """
        Apply white noise addition to detector strains.

        Adds complex Gaussian white noise (unit variance) to each detector's strain.

        Parameters
        ----------
        input_sample : Union[DetectorStrainSample, NoiseASDSample]
            Input sample with waveform dict containing detector strains

        Returns
        -------
        Union[DetectorStrainSample, NoiseASDSample]
            Sample with noisy waveform (detector strains with added noise)

        Notes
        -----
        The noise is generated using torch.randn for efficiency:
            noise = randn(shape) + 1j * randn(shape)

        This gives complex noise with unit variance per component,
        resulting in total power of 2 per frequency bin.

        Examples
        --------
        >>> config = AddWhiteNoiseComplexConfig()
        >>> transform = AddWhiteNoiseComplex.from_config(config)
        >>> sample = {'waveform': {'H1': np.array([1.0+0j, 2.0+0j])}}
        >>> result = transform(sample)
        >>> # Noise has been added (exact values random)
        >>> result['waveform']['H1'].shape
        (2,)
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
