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
    asd_dataset_path : str
        Path to HDF5 file containing ASD dataset.
        The dataset is loaded lazily in __init__ to avoid serialization issues.

    Examples
    --------
    >>> config = SampleNoiseASDConfig(
    ...     asd_dataset_path='/path/to/asd_dataset.hdf5'
    ... )
    >>> config.asd_dataset_path
    '/path/to/asd_dataset.hdf5'
    """

    asd_dataset_path: str

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.asd_dataset_path, str):
            raise TypeError(
                f"asd_dataset_path must be a str, got {type(self.asd_dataset_path)}"
            )
        if len(self.asd_dataset_path) == 0:
            raise ValueError("asd_dataset_path cannot be empty")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SampleNoiseASDConfig':
        """
        Create SampleNoiseASDConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'asd_dataset_path' key

        Returns
        -------
        SampleNoiseASDConfig
            Validated configuration instance

        Examples
        --------
        >>> config = SampleNoiseASDConfig.from_dict({
        ...     'asd_dataset_path': '/path/to/asd_dataset.hdf5'
        ... })
        >>> config.asd_dataset_path
        '/path/to/asd_dataset.hdf5'
        """
        return cls(asd_dataset_path=config_dict['asd_dataset_path'])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = SampleNoiseASDConfig(asd_dataset_path='/path/to/asd.hdf5')
        >>> config.to_dict()
        {'asd_dataset_path': '/path/to/asd.hdf5'}
        """
        return {'asd_dataset_path': self.asd_dataset_path}


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
    >>> from dingo_waveform.transform.transforms.noise import (
    ...     SampleNoiseASD,
    ...     SampleNoiseASDConfig
    ... )
    >>>
    >>> config = SampleNoiseASDConfig(
    ...     asd_dataset_path='/path/to/asd_dataset.hdf5'
    ... )
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
    The ASD dataset is loaded in __init__ from the specified HDF5 file path.
    This approach allows the configuration to be serialized (only storing the
    path) while the actual dataset object is created when needed.

    The asd_dataset object must have a sample_random_asds(n) method that
    returns a dictionary mapping detector names to ASD arrays.

    See Also
    --------
    WhitenAndScaleStrain : Uses ASDs to whiten strain data
    WhitenFixedASD : Whitens with a fixed ASD from file
    AddWhiteNoiseComplex : Adds white noise after whitening
    """

    def __init__(self, config: SampleNoiseASDConfig):
        """
        Initialize SampleNoiseASD transform.

        Parameters
        ----------
        config : SampleNoiseASDConfig
            Configuration with ASD dataset path
        """
        super().__init__(config)

        # Load ASD dataset from path
        from dingo.gw.noise.asd_dataset import ASDDataset
        self.asd_dataset = ASDDataset(config.asd_dataset_path)

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
        >>> config = SampleNoiseASDConfig(asd_dataset_path='/path/to/asd.hdf5')
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
