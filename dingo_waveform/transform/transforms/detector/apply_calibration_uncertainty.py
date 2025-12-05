"""
ApplyCalibrationUncertainty: Apply detector calibration uncertainty to waveforms.

This transform samples calibration curves from priors and applies them to
detector waveforms, modeling calibration uncertainty in gravitational wave data.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DomainProtocol, DetectorStrainSample


# Fallback calibration correction type lookup
try:
    from bilby_pipe.utils import CALIBRATION_CORRECTION_TYPE_LOOKUP as _LOOKUP
    DEFAULT_CORRECTION_TYPE_LOOKUP = _LOOKUP
except ImportError:
    DEFAULT_CORRECTION_TYPE_LOOKUP = {
        "H1": "template",
        "L1": "template",
        "V1": "template",
    }


@dataclass(frozen=True)
class ApplyCalibrationUncertaintyConfig(WaveformTransformConfig):
    """
    Configuration for ApplyCalibrationUncertainty transform.

    Attributes
    ----------
    ifo_list : List[str]
        List of interferometer names
    data_domain : Any
        Domain object defining frequency grid for calibration
    calibration_envelope : Dict[str, str]
        Mapping detector names to calibration envelope file paths (.txt/.dat)
    num_calibration_curves : int
        Number of calibration curves to sample per detector
    num_calibration_nodes : int
        Number of frequency nodes for calibration spline
    correction_type : Optional[Union[str, Dict[str, str]]]
        Correction type: 'data', 'template', or dict mapping detectors to types
    """

    ifo_list: List[str]
    data_domain: DomainProtocol
    calibration_envelope: Dict[str, str]
    num_calibration_curves: int
    num_calibration_nodes: int
    correction_type: Optional[Union[str, Dict[str, str]]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.ifo_list, list):
            raise TypeError(f"ifo_list must be a list, got {type(self.ifo_list)}")
        if len(self.ifo_list) == 0:
            raise ValueError("ifo_list cannot be empty")

        if self.data_domain is None:
            raise ValueError("data_domain cannot be None")

        if not isinstance(self.calibration_envelope, dict):
            raise TypeError(
                f"calibration_envelope must be a dict, got {type(self.calibration_envelope)}"
            )

        if not isinstance(self.num_calibration_curves, int):
            raise TypeError(
                f"num_calibration_curves must be an int, got {type(self.num_calibration_curves)}"
            )
        if self.num_calibration_curves <= 0:
            raise ValueError(
                f"num_calibration_curves must be positive, got {self.num_calibration_curves}"
            )

        if not isinstance(self.num_calibration_nodes, int):
            raise TypeError(
                f"num_calibration_nodes must be an int, got {type(self.num_calibration_nodes)}"
            )
        if self.num_calibration_nodes <= 0:
            raise ValueError(
                f"num_calibration_nodes must be positive, got {self.num_calibration_nodes}"
            )

        # Validate calibration envelope files are .txt or .dat
        for ifo, path in self.calibration_envelope.items():
            if not path.endswith(('.txt', '.dat')):
                raise ValueError(
                    f"Calibration envelope for {ifo} must be .txt or .dat file, got {path}"
                )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ApplyCalibrationUncertaintyConfig':
        """
        Create config from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'data_domain' key containing a Domain object
            (NOT a dict - must already be constructed).

        Returns
        -------
        ApplyCalibrationUncertaintyConfig
            Validated configuration instance

        Notes
        -----
        The domain must have sample_frequencies attribute.
        Users are responsible for building the domain before calling this method.
        """
        ifo_list = config_dict['ifo_list']
        data_domain = config_dict['data_domain']

        # Convert InterferometerList to list of names
        if not isinstance(ifo_list, list):
            ifo_list = [ifo.name for ifo in ifo_list]

        # Validate domain using duck typing
        if not hasattr(data_domain, 'sample_frequencies'):
            raise TypeError(
                f"data_domain must have sample_frequencies attribute "
                f"(expected Domain-like object), got {type(data_domain)}"
            )

        return cls(
            ifo_list=ifo_list,
            data_domain=data_domain,
            calibration_envelope=config_dict['calibration_envelope'],
            num_calibration_curves=config_dict['num_calibration_curves'],
            num_calibration_nodes=config_dict['num_calibration_nodes'],
            correction_type=config_dict.get('correction_type', None)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'ifo_list': self.ifo_list,
            'data_domain': self.data_domain.domain_dict,
            'calibration_envelope': self.calibration_envelope,
            'num_calibration_curves': self.num_calibration_curves,
            'num_calibration_nodes': self.num_calibration_nodes,
            'correction_type': self.correction_type
        }


class ApplyCalibrationUncertainty(WaveformTransform[ApplyCalibrationUncertaintyConfig]):
    """
    Apply detector calibration uncertainty to waveforms.

    This transform models calibration uncertainty by:
    1. Loading calibration envelopes from files
    2. Sampling calibration parameters from priors
    3. Computing calibration correction factors using cubic splines
    4. Multiplying waveforms by calibration factors

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> from bilby.gw.detector import InterferometerList
    >>> from dingo_waveform.transform.transforms.detector import (
    ...     ApplyCalibrationUncertainty,
    ...     ApplyCalibrationUncertaintyConfig
    ... )
    >>>
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> config = ApplyCalibrationUncertaintyConfig(
    ...     ifo_list=['H1', 'L1'],
    ...     data_domain=domain,
    ...     calibration_envelope={
    ...         'H1': '/path/to/H1_calibration.txt',
    ...         'L1': '/path/to/L1_calibration.txt'
    ...     },
    ...     num_calibration_curves=10,
    ...     num_calibration_nodes=5,
    ...     correction_type='template'
    ... )
    >>> transform = ApplyCalibrationUncertainty.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': {
    ...         'H1': np.random.randn(len(domain)) + 1j * np.random.randn(len(domain)),
    ...         'L1': np.random.randn(len(domain)) + 1j * np.random.randn(len(domain))
    ...     }
    ... }
    >>> result = transform(sample)
    >>> # Waveforms now have calibration uncertainty applied

    Notes
    -----
    Calibration uncertainty accounts for systematic errors in detector response.
    The uncertainty is modeled using cubic spline interpolation across frequency.

    Calibration priors are loaded from envelope files using bilby's
    CalibrationPriorDict.from_envelope_file().

    The num_calibration_curves parameter determines how many calibration
    realizations are sampled (expanding the batch dimension).

    See Also
    --------
    GetDetectorTimes : Computes detector times
    ProjectOntoDetectors : Projects polarizations onto detectors
    """

    def __init__(self, config: ApplyCalibrationUncertaintyConfig):
        """Initialize transform."""
        super().__init__(config)

        from bilby.gw.detector import InterferometerList, calibration
        from bilby.gw.prior import CalibrationPriorDict

        # Load interferometers
        self.ifo_list = InterferometerList(config.ifo_list)

        # Determine correction type for each detector
        if config.correction_type is None:
            correction_type_dict = {
                ifo: DEFAULT_CORRECTION_TYPE_LOOKUP[ifo]
                for ifo in config.ifo_list
            }
        elif config.correction_type in ["data", "template"]:
            correction_type_dict = {
                ifo: config.correction_type for ifo in config.ifo_list
            }
        elif isinstance(config.correction_type, dict):
            correction_type_dict = config.correction_type
        else:
            raise ValueError(
                f"correction_type must be None, 'data', 'template', or dict, "
                f"got {config.correction_type}"
            )

        # Initialize calibration models and priors
        self.calibration_prior = {}
        for ifo in self.ifo_list:
            # Set calibration model to cubic spline
            ifo.calibration_model = calibration.CubicSpline(
                f"recalib_{ifo.name}_",
                minimum_frequency=config.data_domain.f_min,
                maximum_frequency=config.data_domain.f_max,
                n_points=config.num_calibration_nodes,
            )

            # Load priors from envelope file
            self.calibration_prior[ifo.name] = CalibrationPriorDict.from_envelope_file(
                config.calibration_envelope[ifo.name],
                config.data_domain.f_min,
                config.data_domain.f_max,
                config.num_calibration_nodes,
                ifo.name,
                correction_type=correction_type_dict[ifo.name],
            )

    def __call__(self, input_sample: DetectorStrainSample) -> DetectorStrainSample:  # type: ignore[override]
        """
        Apply calibration uncertainty to detector strains.

        Samples calibration parameters and applies calibration factors to each
        detector's waveform data.

        Parameters
        ----------
        input_sample : DetectorStrainSample
            Input sample with detector waveforms (H1, L1, V1 keys)

        Returns
        -------
        DetectorStrainSample
            Sample with calibration uncertainty applied to waveforms
        """
        sample = input_sample.copy()

        for ifo in self.ifo_list:
            # Sample calibration parameters
            calibration_parameter_draws = pd.DataFrame(
                self.calibration_prior[ifo.name].sample(
                    self.config.num_calibration_curves
                )
            )

            # Compute calibration factors
            calibration_draws = np.zeros(
                (
                    self.config.num_calibration_curves,
                    len(self.config.data_domain.sample_frequencies),
                ),
                dtype=complex,
            )

            for i in range(self.config.num_calibration_curves):
                calibration_draws[
                    i, self.config.data_domain.frequency_mask
                ] = ifo.calibration_model.get_calibration_factor(
                    self.config.data_domain.sample_frequencies[
                        self.config.data_domain.frequency_mask
                    ],
                    prefix=f"recalib_{ifo.name}_",
                    **calibration_parameter_draws.iloc[i],
                )

            # Apply calibration to waveform
            sample["waveform"][ifo.name] = (
                sample["waveform"][ifo.name] * calibration_draws
            )

        return sample
