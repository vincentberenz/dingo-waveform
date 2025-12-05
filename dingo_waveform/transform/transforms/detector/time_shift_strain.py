"""
TimeShiftStrain transform: Time-shift detector strains by detector times.

This transform applies time shifts to strains in individual detectors
according to the detector-specific times in extrinsic_parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union
import torch
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DomainProtocol, DetectorStrainSample, NoiseASDSample


@dataclass(frozen=True)
class TimeShiftStrainConfig(WaveformTransformConfig):
    """
    Configuration for TimeShiftStrain transform.

    Attributes
    ----------
    ifo_list : List[str]
        List of interferometer names
    domain : DomainProtocol
        Domain object with time_translate_data() method.
        Can be UniformFrequencyDomain, MultibandedFrequencyDomain, etc.

    Examples
    --------
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> domain = UniformFrequencyDomain(...)
    >>> config = TimeShiftStrainConfig(
    ...     ifo_list=['H1', 'L1'],
    ...     domain=domain
    ... )
    """

    ifo_list: List[str]
    domain: DomainProtocol

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.ifo_list, list):
            raise TypeError(f"ifo_list must be a list, got {type(self.ifo_list)}")
        if len(self.ifo_list) == 0:
            raise ValueError("ifo_list cannot be empty")
        if not all(isinstance(ifo, str) for ifo in self.ifo_list):
            raise TypeError("All items in ifo_list must be strings")
        if self.domain is None:
            raise ValueError("domain cannot be None")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TimeShiftStrainConfig':
        """
        Create TimeShiftStrainConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with keys:
            - 'ifo_list': List of IFO names or InterferometerList
            - 'domain': Domain object or dict representation

        Returns
        -------
        TimeShiftStrainConfig
            Validated configuration instance

        Examples
        --------
        >>> from dingo.gw.domains import UniformFrequencyDomain
        >>> domain = UniformFrequencyDomain(...)
        >>> config = TimeShiftStrainConfig.from_dict({
        ...     'ifo_list': ['H1', 'L1'],
        ...     'domain': domain
        ... })

        >>> # Also works with dict representation
        >>> config = TimeShiftStrainConfig.from_dict({
        ...     'ifo_list': ['H1', 'L1'],
        ...     'domain': {'type': 'UniformFrequencyDomain', ...}
        ... })
        """
        from dingo.gw.domains import build_domain

        ifo_list = config_dict['ifo_list']
        domain = config_dict['domain']

        # Convert InterferometerList to list of names
        if not isinstance(ifo_list, list):
            ifo_list = [ifo.name for ifo in ifo_list]

        # Convert dict to Domain object if needed
        if isinstance(domain, dict):
            domain = build_domain(domain)

        return cls(ifo_list=ifo_list, domain=domain)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> domain = UniformFrequencyDomain(...)
        >>> config = TimeShiftStrainConfig(
        ...     ifo_list=['H1', 'L1'],
        ...     domain=domain
        ... )
        >>> config_dict = config.to_dict()
        >>> config_dict['ifo_list']
        ['H1', 'L1']
        """
        return {
            'ifo_list': self.ifo_list,
            'domain': self.domain.domain_dict
        }


class TimeShiftStrain(WaveformTransform[TimeShiftStrainConfig]):
    """
    Time-shift detector strains by detector-specific times.

    This transform applies time shifts to strains in individual detectors
    according to the times '{ifo_name}_time' provided in extrinsic_parameters.
    The detector times are removed from extrinsic_parameters after use.

    The time shift is performed using domain.time_translate_data(), which
    applies a phase shift in the frequency domain corresponding to a time
    translation in the time domain.

    Supports two input formats:
    1. Dict[str, array]: waveform = {'H1': array, 'L1': array, ...}
    2. Tensor: waveform = tensor of shape (batch, detectors, frequency_bins)

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> from dingo_waveform.transform.transforms.detector import (
    ...     TimeShiftStrain,
    ...     TimeShiftStrainConfig
    ... )
    >>>
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> config = TimeShiftStrainConfig(
    ...     ifo_list=['H1', 'L1'],
    ...     domain=domain
    ... )
    >>> transform = TimeShiftStrain.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': {
    ...         'H1': np.random.randn(len(domain)),
    ...         'L1': np.random.randn(len(domain))
    ...     },
    ...     'extrinsic_parameters': {
    ...         'H1_time': 1234567890.0,
    ...         'L1_time': 1234567890.01
    ...     }
    ... }
    >>> result = transform(sample)
    >>> # Waveforms are time-shifted, detector times removed from extrinsic_parameters
    >>> 'H1_time' in result['extrinsic_parameters']
    False

    Notes
    -----
    The time shift is performed in the frequency domain using:
        shifted_data[f] = data[f] * exp(-2πi f dt)

    For tensor input (after ToTorch), the time shifts are stacked and applied
    to all detectors simultaneously.

    See Also
    --------
    GetDetectorTimes : Computes detector times from sky position
    ProjectOntoDetectors : Projects polarizations onto detectors
    """

    def __init__(self, config: TimeShiftStrainConfig):
        """
        Initialize TimeShiftStrain transform.

        Parameters
        ----------
        config : TimeShiftStrainConfig
            Configuration with IFO list and domain
        """
        super().__init__(config)

        # Load InterferometerList from names
        from bilby.gw.detector import InterferometerList
        self.ifo_list = InterferometerList(config.ifo_list)

    def __call__(
        self,
        input_sample: Union[DetectorStrainSample, NoiseASDSample]
    ) -> Union[DetectorStrainSample, NoiseASDSample]:  # type: ignore[override]
        """
        Apply time shift transform.

        Can operate on DetectorStrainSample or NoiseASDSample - any stage that has
        detector waveforms and extrinsic_parameters with detector times.

        Parameters
        ----------
        input_sample : Union[DetectorStrainSample, NoiseASDSample]
            Input sample with waveform dict (or tensor) and extrinsic_parameters
            containing '{ifo_name}_time' for each detector

        Returns
        -------
        Union[DetectorStrainSample, NoiseASDSample]
            Sample with time-shifted strains and detector times removed
            from extrinsic_parameters

        Examples
        --------
        >>> domain = UniformFrequencyDomain(...)
        >>> config = TimeShiftStrainConfig(ifo_list=['H1'], domain=domain)
        >>> transform = TimeShiftStrain.from_config(config)
        >>> sample = {
        ...     'waveform': {'H1': np.random.randn(len(domain))},
        ...     'extrinsic_parameters': {'H1_time': 1234567890.0}
        ... }
        >>> result = transform(sample)
        >>> 'H1_time' in result['extrinsic_parameters']
        False
        """
        sample = input_sample.copy()
        extrinsic_parameters = input_sample["extrinsic_parameters"].copy()

        strains = {}

        if isinstance(input_sample["waveform"], dict):
            # Dict format: {'H1': array, 'L1': array, ...}
            for ifo in self.ifo_list:
                strain = input_sample["waveform"][ifo.name]
                dt = extrinsic_parameters.pop(f"{ifo.name}_time")
                strains[ifo.name] = self.config.domain.time_translate_data(strain, dt)

        elif isinstance(input_sample["waveform"], torch.Tensor):
            # Tensor format: (batch, detectors, frequency_bins)
            strains = input_sample["waveform"]
            dt = [
                extrinsic_parameters.pop(f"{ifo.name}_time")
                for ifo in self.ifo_list
            ]
            dt = torch.stack(dt, 1)
            strains = self.config.domain.time_translate_data(strains, dt)

        else:
            raise NotImplementedError(
                f"Unexpected type {type(input_sample['waveform'])}, expected dict or "
                f"torch.Tensor"
            )

        sample["waveform"] = strains
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample
