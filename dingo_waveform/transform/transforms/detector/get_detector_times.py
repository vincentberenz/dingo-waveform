"""
GetDetectorTimes transform: Compute detector-specific times from sky position.

This transform calculates time shifts for individual detectors based on
the sky position (ra, dec), geocent_time, and a reference GPS time.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import torch
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import ExtrinsicSample


@dataclass(frozen=True)
class GetDetectorTimesConfig(WaveformTransformConfig):
    """
    Configuration for GetDetectorTimes transform.

    Attributes
    ----------
    ifo_list : List[str]
        List of interferometer names (e.g., ['H1', 'L1', 'V1'])
    ref_time : float
        Reference GPS time for detector time calculations

    Examples
    --------
    >>> config = GetDetectorTimesConfig(
    ...     ifo_list=['H1', 'L1'],
    ...     ref_time=1234567890.0
    ... )
    >>> config.ifo_list
    ['H1', 'L1']
    """

    ifo_list: List[str]
    ref_time: float

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.ifo_list, list):
            raise TypeError(f"ifo_list must be a list, got {type(self.ifo_list)}")
        if len(self.ifo_list) == 0:
            raise ValueError("ifo_list cannot be empty")
        if not all(isinstance(ifo, str) for ifo in self.ifo_list):
            raise TypeError("All items in ifo_list must be strings")
        if not isinstance(self.ref_time, (int, float)):
            raise TypeError(f"ref_time must be a number, got {type(self.ref_time)}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GetDetectorTimesConfig':
        """
        Create GetDetectorTimesConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with keys:
            - 'ifo_list': List of IFO names or InterferometerList
            - 'ref_time': Reference GPS time

        Returns
        -------
        GetDetectorTimesConfig
            Validated configuration instance

        Examples
        --------
        >>> config = GetDetectorTimesConfig.from_dict({
        ...     'ifo_list': ['H1', 'L1', 'V1'],
        ...     'ref_time': 1234567890.0
        ... })

        >>> # Also works with InterferometerList
        >>> from bilby.gw.detector import InterferometerList
        >>> ifo_list = InterferometerList(['H1', 'L1'])
        >>> config = GetDetectorTimesConfig.from_dict({
        ...     'ifo_list': ifo_list,
        ...     'ref_time': 1234567890.0
        ... })
        >>> config.ifo_list
        ['H1', 'L1']
        """
        ifo_list = config_dict['ifo_list']

        # Convert InterferometerList to list of names
        if not isinstance(ifo_list, list):
            # Assume it's InterferometerList with .name attributes
            ifo_list = [ifo.name for ifo in ifo_list]

        return cls(
            ifo_list=ifo_list,
            ref_time=config_dict['ref_time']
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation

        Examples
        --------
        >>> config = GetDetectorTimesConfig(
        ...     ifo_list=['H1', 'L1'],
        ...     ref_time=1234567890.0
        ... )
        >>> config.to_dict()
        {'ifo_list': ['H1', 'L1'], 'ref_time': 1234567890.0}
        """
        return {
            'ifo_list': self.ifo_list,
            'ref_time': self.ref_time
        }


class GetDetectorTimes(WaveformTransform[GetDetectorTimesConfig]):
    """
    Compute detector-specific times from sky position and geocentric time.

    This transform calculates the time at which a gravitational wave signal
    arrives at each detector based on:
    - Sky position (ra, dec)
    - Geocentric time
    - Reference GPS time
    - Detector locations

    For each detector, computes:
        detector_time = geocent_time + time_delay_from_geocenter(detector, ra, dec, ref_time)

    The detector times are added to extrinsic_parameters as '{ifo_name}_time'.

    Examples
    --------
    >>> from bilby.gw.detector import InterferometerList
    >>> from dingo_waveform.transform.transforms.detector import (
    ...     GetDetectorTimes,
    ...     GetDetectorTimesConfig
    ... )
    >>>
    >>> config = GetDetectorTimesConfig(
    ...     ifo_list=['H1', 'L1'],
    ...     ref_time=1234567890.0
    ... )
    >>> transform = GetDetectorTimes.from_config(config)
    >>>
    >>> sample = {
    ...     'extrinsic_parameters': {
    ...         'ra': 1.5,
    ...         'dec': -0.3,
    ...         'geocent_time': 1234567890.0
    ...     }
    ... }
    >>> result = transform(sample)
    >>> 'H1_time' in result['extrinsic_parameters']
    True
    >>> 'L1_time' in result['extrinsic_parameters']
    True

    Notes
    -----
    This transform uses the time_delay_from_geocenter function which supports
    batched computation for numpy arrays and PyTorch tensors.

    For GPU tensors, computation is temporarily moved to CPU for compatibility
    with LIGO time delay functions.

    See Also
    --------
    ProjectOntoDetectors : Projects polarizations onto detectors
    TimeShiftStrain : Time-shifts detector strains
    """

    def __init__(self, config: GetDetectorTimesConfig):
        """
        Initialize GetDetectorTimes transform.

        Parameters
        ----------
        config : GetDetectorTimesConfig
            Configuration with IFO list and reference time
        """
        super().__init__(config)

        # Load InterferometerList from names
        from bilby.gw.detector import InterferometerList
        self.ifo_list = InterferometerList(config.ifo_list)

    def __call__(self, input_sample: ExtrinsicSample) -> ExtrinsicSample:  # type: ignore[override]
        """
        Apply detector time calculation.

        Adds detector-specific times to extrinsic_parameters based on sky position.

        Parameters
        ----------
        input_sample : ExtrinsicSample
            Input sample with extrinsic_parameters containing:
            - 'ra': Right ascension
            - 'dec': Declination
            - 'geocent_time': Geocentric time

        Returns
        -------
        ExtrinsicSample
            Sample with detector times added to extrinsic_parameters as
            '{ifo_name}_time' for each detector

        Examples
        --------
        >>> config = GetDetectorTimesConfig(
        ...     ifo_list=['H1', 'L1'],
        ...     ref_time=1234567890.0
        ... )
        >>> transform = GetDetectorTimes.from_config(config)
        >>> sample = {
        ...     'extrinsic_parameters': {
        ...         'ra': 1.5,
        ...         'dec': -0.3,
        ...         'geocent_time': 1234567890.0
        ...     }
        ... }
        >>> result = transform(sample)
        >>> 'H1_time' in result['extrinsic_parameters']
        True
        """
        from dingo_waveform.transform.detector import time_delay_from_geocenter

        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        ra = extrinsic_parameters["ra"]
        dec = extrinsic_parameters["dec"]
        geocent_time = extrinsic_parameters["geocent_time"]

        for ifo in self.ifo_list:
            if type(ra) == torch.Tensor:
                # Computation does not work on GPU, so do it on CPU
                ra_cpu = ra.cpu()
                dec_cpu = dec.cpu()
            else:
                ra_cpu = ra
                dec_cpu = dec

            dt = time_delay_from_geocenter(ifo, ra_cpu, dec_cpu, self.config.ref_time)

            if type(dt) == torch.Tensor:
                dt = dt.to(geocent_time.device)

            ifo_time = geocent_time + dt
            extrinsic_parameters[f"{ifo.name}_time"] = ifo_time

        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample
