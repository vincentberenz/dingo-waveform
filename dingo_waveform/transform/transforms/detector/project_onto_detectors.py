"""
ProjectOntoDetectors transform: Project GW polarizations onto detector network.

This transform projects polarizations (h_plus, h_cross) onto a network of
detectors, accounting for luminosity distance, antenna patterns, and time delays.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union
import numpy as np
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import (
    DomainProtocol,
    PolarizationSample,
    ExtrinsicSample,
    DetectorStrainSample,
)


@dataclass(frozen=True)
class ProjectOntoDetectorsConfig(WaveformTransformConfig):
    """
    Configuration for ProjectOntoDetectors transform.

    Attributes
    ----------
    ifo_list : List[str]
        List of interferometer names
    domain : DomainProtocol
        Domain object with time_translate_data() method
    ref_time : float
        Reference GPS time for antenna pattern calculations

    Examples
    --------
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> domain = UniformFrequencyDomain(...)
    >>> config = ProjectOntoDetectorsConfig(
    ...     ifo_list=['H1', 'L1'],
    ...     domain=domain,
    ...     ref_time=1234567890.0
    ... )
    """

    ifo_list: List[str]
    domain: DomainProtocol
    ref_time: float

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
        if not isinstance(self.ref_time, (int, float)):
            raise TypeError(f"ref_time must be a number, got {type(self.ref_time)}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProjectOntoDetectorsConfig':
        """
        Create ProjectOntoDetectorsConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with keys:
            - 'ifo_list': List of IFO names or InterferometerList
            - 'domain': Domain object or dict representation
            - 'ref_time': Reference GPS time

        Returns
        -------
        ProjectOntoDetectorsConfig
            Validated configuration instance

        Examples
        --------
        >>> from dingo.gw.domains import UniformFrequencyDomain
        >>> domain = UniformFrequencyDomain(...)
        >>> config = ProjectOntoDetectorsConfig.from_dict({
        ...     'ifo_list': ['H1', 'L1'],
        ...     'domain': domain,
        ...     'ref_time': 1234567890.0
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

        return cls(
            ifo_list=ifo_list,
            domain=domain,
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
        >>> domain = UniformFrequencyDomain(...)
        >>> config = ProjectOntoDetectorsConfig(
        ...     ifo_list=['H1', 'L1'],
        ...     domain=domain,
        ...     ref_time=1234567890.0
        ... )
        >>> config_dict = config.to_dict()
        >>> config_dict['ref_time']
        1234567890.0
        """
        return {
            'ifo_list': self.ifo_list,
            'domain': self.domain.domain_dict,
            'ref_time': self.ref_time
        }


class ProjectOntoDetectors(WaveformTransform[ProjectOntoDetectorsConfig]):
    """
    Project GW polarizations onto detector network.

    This transform performs three key operations:

    1. **Rescale polarizations** to account for sampled luminosity distance:
           h_scaled = h * (d_ref / d_new)

    2. **Project onto antenna patterns** using extrinsic parameters (ra, dec, psi):
           strain[ifo] = F_plus[ifo] * h_plus + F_cross[ifo] * h_cross

    3. **Apply time shifts** according to detector-specific times:
           strain_shifted[ifo] = time_translate(strain[ifo], dt[ifo])

    The transform consumes extrinsic parameters (luminosity_distance, ra, dec, psi,
    geocent_time, detector times) and moves them to the parameters dict.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> from bilby.gw.detector import InterferometerList
    >>> from dingo_waveform.transform.transforms.detector import (
    ...     ProjectOntoDetectors,
    ...     ProjectOntoDetectorsConfig
    ... )
    >>>
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> config = ProjectOntoDetectorsConfig(
    ...     ifo_list=['H1', 'L1'],
    ...     domain=domain,
    ...     ref_time=1234567890.0
    ... )
    >>> transform = ProjectOntoDetectors.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': {
    ...         'h_plus': np.random.randn(len(domain)),
    ...         'h_cross': np.random.randn(len(domain))
    ...     },
    ...     'parameters': {
    ...         'mass_1': 30.0,
    ...         'luminosity_distance': 1000.0,
    ...         'geocent_time': 0.0
    ...     },
    ...     'extrinsic_parameters': {
    ...         'luminosity_distance': 500.0,
    ...         'ra': 1.5,
    ...         'dec': -0.3,
    ...         'psi': 0.5,
    ...         'geocent_time': 0.01,
    ...         'H1_time': 0.01,
    ...         'L1_time': 0.015
    ...     }
    ... }
    >>> result = transform(sample)
    >>> # Waveform is now detector-projected strains
    >>> list(result['waveform'].keys())
    ['H1', 'L1']
    >>> # Extrinsic parameters moved to parameters
    >>> result['parameters']['ra']
    1.5

    Notes
    -----
    This transform assumes that the reference polarizations have geocent_time = 0.0.
    If this is not the case, remove the assertion in the code.

    The luminosity distance rescaling uses:
        scale_factor = d_ref / d_new

    Antenna patterns are computed using bilby.gw.detector.Interferometer methods
    at the reference time.

    See Also
    --------
    GetDetectorTimes : Computes detector times from sky position
    TimeShiftStrain : Time-shifts detector strains
    """

    def __init__(self, config: ProjectOntoDetectorsConfig):
        """
        Initialize ProjectOntoDetectors transform.

        Parameters
        ----------
        config : ProjectOntoDetectorsConfig
            Configuration with IFO list, domain, and reference time
        """
        super().__init__(config)

        # Load InterferometerList from names
        from bilby.gw.detector import InterferometerList
        self.ifo_list = InterferometerList(config.ifo_list)

    def __call__(
        self,
        input_sample: Union[PolarizationSample, ExtrinsicSample]
    ) -> DetectorStrainSample:  # type: ignore[override]
        """
        Apply projection transform.

        This is a critical pipeline transition: waveform dict changes from
        polarizations (h_plus, h_cross) to detector strains (H1, L1, V1).

        Parameters
        ----------
        input_sample : Union[PolarizationSample, ExtrinsicSample]
            Input sample with structure:
            {
                'waveform': {'h_plus': array, 'h_cross': array},
                'parameters': {'luminosity_distance': float, 'geocent_time': 0.0, ...},
                'extrinsic_parameters': {
                    'luminosity_distance': float,
                    'ra': float, 'dec': float, 'psi': float,
                    'geocent_time': float,
                    'H1_time': float, 'L1_time': float, ...
                }
            }

        Returns
        -------
        DetectorStrainSample
            Sample with structure:
            {
                'waveform': {detector_name: projected_strain, ...},
                'parameters': {..., 'ra': float, 'dec': float, ...},
                'extrinsic_parameters': {remaining_params}
            }

        Notes
        -----
        - **Critical transition**: Waveform keys change from polarizations
          (h_plus, h_cross) to detector names (H1, L1, V1)
        - Extrinsic parameters (luminosity_distance, ra, dec, psi, geocent_time,
          detector times) are popped from extrinsic_parameters and added to
          parameters for consolidation

        Raises
        ------
        ValueError
            If required parameters are missing
        AssertionError
            If reference geocent_time is not 0.0
        """
        sample = input_sample.copy()
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        try:
            d_ref = parameters["luminosity_distance"]
            d_new = extrinsic_parameters.pop("luminosity_distance")
            ra = extrinsic_parameters.pop("ra")
            dec = extrinsic_parameters.pop("dec")
            psi = extrinsic_parameters.pop("psi")
            tc_ref = parameters["geocent_time"]
            assert np.allclose(tc_ref, 0.0), (
                "This should always be 0. If for some reason "
                "you want to save time shifted polarizations, "
                "then remove this assert statement."
            )
            tc_new = extrinsic_parameters.pop("geocent_time")
        except:
            raise ValueError("Missing parameters.")

        # (1) Rescale polarizations and set distance parameter to sampled value
        if np.isscalar(d_ref) or np.isscalar(d_new):
            d_ratio = d_ref / d_new
        elif isinstance(d_ref, np.ndarray) and isinstance(d_new, np.ndarray):
            d_ratio = (d_ref / d_new)[:, np.newaxis]
        else:
            raise ValueError("luminosity_distance should be a float or a numpy array.")

        hc = sample["waveform"]["h_cross"] * d_ratio
        hp = sample["waveform"]["h_plus"] * d_ratio
        parameters["luminosity_distance"] = d_new

        strains = {}
        for ifo in self.ifo_list:
            # (2) Project strains onto the different detectors
            if any(np.isscalar(x) for x in [ra, dec, psi]):
                fp = ifo.antenna_response(ra, dec, self.config.ref_time, psi, mode="plus")
                fc = ifo.antenna_response(ra, dec, self.config.ref_time, psi, mode="cross")
            else:
                fp = np.array(
                    [
                        ifo.antenna_response(r, d, self.config.ref_time, p, mode="plus")
                        for r, d, p in zip(ra, dec, psi)
                    ],
                    dtype=np.float32,
                )
                fc = np.array(
                    [
                        ifo.antenna_response(r, d, self.config.ref_time, p, mode="cross")
                        for r, d, p in zip(ra, dec, psi)
                    ],
                    dtype=np.float32,
                )
                fp = fp[..., np.newaxis]
                fc = fc[..., np.newaxis]

            strain = fp * hp + fc * hc

            # (3) Time shift the strain. If polarizations are timeshifted by
            #     tc_ref != 0, undo this here by subtracting it from dt.
            dt = extrinsic_parameters[f"{ifo.name}_time"] - tc_ref
            strains[ifo.name] = self.config.domain.time_translate_data(strain, dt)

        # Add extrinsic parameters corresponding to the transformations
        # applied above to parameters. These have all been popped off of
        # extrinsic_parameters, so they only live in one place.
        parameters["ra"] = ra
        parameters["dec"] = dec
        parameters["psi"] = psi
        parameters["geocent_time"] = tc_new
        for ifo in self.ifo_list:
            param_name = f"{ifo.name}_time"
            parameters[param_name] = extrinsic_parameters.pop(param_name)

        sample["waveform"] = strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample
