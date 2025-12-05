"""
SampleNoiseASD transform: Sample random ASDs from dataset for detectors.

This transform samples amplitude spectral densities (ASDs) from an ASD dataset
and places them in sample['asds'] for use in whitening and noise addition.
"""

from dataclasses import dataclass
from typing import Dict, Any
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DetectorStrainSample, NoiseASDSample


@dataclass(frozen=True)
class SampleNoiseASDConfig(WaveformTransformConfig):
    """
    Configuration for SampleNoiseASD transform.

    Attributes
    ----------
    asd_dataset : Any
        ASD dataset object (duck-typed). Must have sample_random_asds(n) method
        that returns a dictionary mapping detector names to ASD arrays.

    Examples
    --------
    >>> from dingo.gw.noise.asd_dataset import ASDDataset
    >>> asd_dataset = ASDDataset('/path/to/asd_dataset.hdf5')
    >>> config = SampleNoiseASDConfig(asd_dataset=asd_dataset)
    >>> hasattr(config.asd_dataset, 'sample_random_asds')
    True

    Notes
    -----
    This configuration cannot be serialized to JSON/YAML since it contains
    an object reference. Use factory functions from transform.factory module
    to build transform chains with explicit object dependencies.
    """

    asd_dataset: Any

    def __post_init__(self) -> None:
        """Validate configuration using duck typing."""
        if not hasattr(self.asd_dataset, 'sample_random_asds'):
            raise TypeError(
                f"asd_dataset must have sample_random_asds() method, "
                f"got {type(self.asd_dataset)}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SampleNoiseASDConfig':
        """
        Create SampleNoiseASDConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'asd_dataset' key containing
            an ASD dataset object (NOT a path string)

        Returns
        -------
        SampleNoiseASDConfig
            Validated configuration instance

        Examples
        --------
        >>> from dingo.gw.noise.asd_dataset import ASDDataset
        >>> asd_dataset = ASDDataset('/path/to/asd.hdf5')
        >>> config = SampleNoiseASDConfig.from_dict({
        ...     'asd_dataset': asd_dataset
        ... })

        Notes
        -----
        The asd_dataset value must be an object (not a path string).
        Users are responsible for loading the dataset before calling this method.
        """
        return cls(asd_dataset=config_dict['asd_dataset'])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Notes
        -----
        This returns a dict with the object reference, which cannot be
        serialized to JSON/YAML. This method exists for API compatibility
        but should not be used for persistence.
        """
        return {'asd_dataset': self.asd_dataset}


class SampleNoiseASD(WaveformTransform[SampleNoiseASDConfig]):
    """
    Sample random ASDs for each detector and place in sample['asds'].

    This transform loads an ASD dataset and samples random amplitude spectral
    densities for use in whitening gravitational wave strain data. The sampled
    ASDs represent realistic detector noise characteristics.

    The transform automatically detects whether the input is batched:
    - Batched input: samples batch_size ASDs
    - Single input: samples one ASD and removes batch dimension

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.noise.asd_dataset import ASDDataset
    >>> from dingo_waveform.transform.transforms.noise import (
    ...     SampleNoiseASD,
    ...     SampleNoiseASDConfig
    ... )
    >>>
    >>> # Load ASD dataset first
    >>> asd_dataset = ASDDataset('/path/to/asd_dataset.hdf5')
    >>> config = SampleNoiseASDConfig(asd_dataset=asd_dataset)
    >>> transform = SampleNoiseASD.from_config(config)
    >>>
    >>> # Single sample
    >>> sample = {
    ...     'parameters': {'mass_1': 30.0},
    ...     'waveform': {'H1': np.random.randn(1024), 'L1': np.random.randn(1024)}
    ... }
    >>> result = transform(sample)
    >>> 'asds' in result
    True
    >>> list(result['asds'].keys())
    ['H1', 'L1']

    >>> # Batched sample
    >>> batch_sample = {
    ...     'parameters': {'mass_1': np.array([30.0, 35.0])},
    ...     'waveform': {'H1': np.random.randn(2, 1024), 'L1': np.random.randn(2, 1024)}
    ... }
    >>> result = transform(batch_sample)
    >>> result['asds']['H1'].shape
    (2, 1024)

    Notes
    -----
    The asd_dataset object is passed directly to the config and must be
    loaded before creating the transform. This makes dependencies explicit
    and allows the transform package to be standalone (no dingo imports).

    The asd_dataset object must have a sample_random_asds(n) method that
    returns a dictionary mapping detector names to ASD arrays.

    See Also
    --------
    WhitenAndScaleStrain : Uses ASDs to whiten strain data
    WhitenFixedASD : Whitens with a fixed ASD from file
    AddWhiteNoiseComplex : Adds white noise after whitening
    """

    def __init__(self, asd_dataset_or_config):
        """
        Initialize SampleNoiseASD transform.

        Parameters
        ----------
        asd_dataset_or_config : Any or SampleNoiseASDConfig
            Either an ASD dataset object (with sample_random_asds method)
            or a SampleNoiseASDConfig instance.

        Notes
        -----
        The asd_dataset object is validated via duck typing (checking for
        sample_random_asds method).

        Examples
        --------
        >>> from dingo.gw.noise.asd_dataset import ASDDataset
        >>> asd_dataset = ASDDataset('/path/to/asd.hdf5')
        >>>
        >>> # Direct instantiation (recommended)
        >>> transform = SampleNoiseASD(asd_dataset)
        >>>
        >>> # Or with config object
        >>> config = SampleNoiseASDConfig(asd_dataset=asd_dataset)
        >>> transform = SampleNoiseASD(config)
        """
        # Handle both direct object and config wrapper
        if isinstance(asd_dataset_or_config, SampleNoiseASDConfig):
            config = asd_dataset_or_config
        else:
            # Assume it's an asd_dataset object - wrap in config
            config = SampleNoiseASDConfig(asd_dataset=asd_dataset_or_config)

        super().__init__(config)

        # Store reference to ASD dataset object
        self.asd_dataset = config.asd_dataset

    def __call__(self, input_sample: DetectorStrainSample) -> NoiseASDSample:  # type: ignore[override]
        """
        Sample random ASDs and add to sample.

        Transitions from DetectorStrainSample to NoiseASDSample by adding
        amplitude spectral densities for each detector.

        Parameters
        ----------
        input_sample : DetectorStrainSample
            Input sample with detector strains (can be batched or single)

        Returns
        -------
        NoiseASDSample
            Sample with 'asds' dict added containing sampled ASDs for each detector

        Examples
        --------
        >>> from dingo.gw.noise.asd_dataset import ASDDataset
        >>> asd_dataset = ASDDataset('/path/to/asd.hdf5')
        >>> config = SampleNoiseASDConfig(asd_dataset=asd_dataset)
        >>> transform = SampleNoiseASD.from_config(config)
        >>> sample = {
        ...     'parameters': {'mass_1': 30.0},
        ...     'waveform': {'H1': np.random.randn(1024)}
        ... }
        >>> result = transform(sample)
        >>> 'asds' in result
        True
        >>> 'H1' in result['asds']
        True
        """
        from dingo_waveform.transform.utils import get_batch_size_of_input_sample

        sample = input_sample.copy()
        batched, batch_size = get_batch_size_of_input_sample(input_sample)

        # Sample ASDs from dataset
        sample["asds"] = self.asd_dataset.sample_random_asds(n=batch_size)

        # If not batched, remove batch dimension
        if not batched:
            sample["asds"] = {k: v[0] for k, v in sample["asds"].items()}

        return sample
