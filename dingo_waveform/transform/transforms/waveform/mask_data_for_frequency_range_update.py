"""
MaskDataForFrequencyRangeUpdate: Mask waveform and ASD outside frequency range.

This transform sets waveform to zero and ASD to one outside specified
frequency ranges, allowing different frequency bounds per detector.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import numpy as np
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DomainProtocol, NoiseASDSample


@dataclass(frozen=True)
class MaskDataForFrequencyRangeUpdateConfig(WaveformTransformConfig):
    """
    Configuration for MaskDataForFrequencyRangeUpdate transform.

    Attributes
    ----------
    domain : Any
        Domain object (UniformFrequencyDomain or MultibandedFrequencyDomain)
        providing sample_frequencies
    minimum_frequency : Optional[Union[float, Dict[str, float]]]
        New f_min. If float, same for all detectors. If dict, per-detector.
    maximum_frequency : Optional[Union[float, Dict[str, float]]]
        New f_max. If float, same for all detectors. If dict, per-detector.
    print_output : bool
        Whether to print configuration on initialization. Default False.
    """

    domain: DomainProtocol
    minimum_frequency: Optional[Union[float, Dict[str, float]]] = None
    maximum_frequency: Optional[Union[float, Dict[str, float]]] = None
    print_output: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.domain is None:
            raise ValueError("domain cannot be None")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MaskDataForFrequencyRangeUpdateConfig':
        """
        Create config from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'domain' key containing a Domain object
            (NOT a dict - must already be constructed).

        Returns
        -------
        MaskDataForFrequencyRangeUpdateConfig
            Validated configuration instance

        Notes
        -----
        The domain must have sample_frequencies attribute.
        Users are responsible for building the domain before calling this method.
        """
        domain = config_dict['domain']

        # Validate domain using duck typing
        if not hasattr(domain, 'sample_frequencies'):
            raise TypeError(
                f"domain must have sample_frequencies attribute "
                f"(expected Domain-like object), got {type(domain)}"
            )

        return cls(
            domain=domain,
            minimum_frequency=config_dict.get('minimum_frequency', None),
            maximum_frequency=config_dict.get('maximum_frequency', None),
            print_output=config_dict.get('print_output', False)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'domain': self.domain.domain_dict,
            'minimum_frequency': self.minimum_frequency,
            'maximum_frequency': self.maximum_frequency,
            'print_output': self.print_output
        }


class MaskDataForFrequencyRangeUpdate(WaveformTransform[MaskDataForFrequencyRangeUpdateConfig]):
    """
    Mask waveform and ASD outside specified frequency ranges.

    Sets waveform to zero and ASD to one for frequencies outside the
    specified minimum and maximum bounds. Supports per-detector bounds.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> from dingo_waveform.transform.transforms.waveform import (
    ...     MaskDataForFrequencyRangeUpdate,
    ...     MaskDataForFrequencyRangeUpdateConfig
    ... )
    >>>
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> config = MaskDataForFrequencyRangeUpdateConfig(
    ...     domain=domain,
    ...     minimum_frequency=30.0,  # Mask below 30 Hz
    ...     maximum_frequency=512.0  # Mask above 512 Hz
    ... )
    >>> transform = MaskDataForFrequencyRangeUpdate.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': {
    ...         'H1': np.random.randn(len(domain)),
    ...         'L1': np.random.randn(len(domain))
    ...     },
    ...     'asds': {
    ...         'H1': np.random.randn(len(domain)),
    ...         'L1': np.random.randn(len(domain))
    ...     }
    ... }
    >>> result = transform(sample)
    >>> # Waveforms are zero outside [30, 512] Hz
    >>> # ASDs are one outside [30, 512] Hz

    >>> # Per-detector bounds
    >>> config_per_det = MaskDataForFrequencyRangeUpdateConfig(
    ...     domain=domain,
    ...     minimum_frequency={'H1': 30.0, 'L1': 25.0},
    ...     maximum_frequency={'H1': 512.0, 'L1': 600.0}
    ... )

    Notes
    -----
    Masking operation:
    - waveform[f < f_min or f > f_max] = 0.0
    - asd[f < f_min or f > f_max] = 1.0

    This allows the neural network to learn that certain frequency ranges
    should be ignored (zero signal, unit noise variance).
    """

    def __init__(self, config: MaskDataForFrequencyRangeUpdateConfig):
        """Initialize transform."""
        super().__init__(config)
        self.sample_frequencies = config.domain.sample_frequencies

        if config.print_output:
            print(
                f"Transform MaskDataForFrequencyRangeUpdate activated:"
                f"  Settings: \n"
                f"    - Minimum_frequency update: {config.minimum_frequency}\n"
                f"    - Maximum_frequency update: {config.maximum_frequency}\n"
            )

    def __call__(self, input_sample: NoiseASDSample) -> NoiseASDSample:  # type: ignore[override]
        """
        Apply frequency range masking.

        Masks waveform and ASDs outside specified frequency ranges for each detector.

        Parameters
        ----------
        input_sample : NoiseASDSample
            Input sample with waveform and asds dicts

        Returns
        -------
        NoiseASDSample
            Sample with masked waveform and asds
        """
        sample = input_sample.copy()

        ifos = list(sample["waveform"].keys())
        frequency_masks = self._create_masks(ifos)

        # Apply masks without modifying dicts in-place
        sample["waveform"] = {
            ifo: np.where(frequency_masks[ifo], sample["waveform"][ifo], 0.0)
            for ifo in ifos
        }
        sample["asds"] = {
            ifo: np.where(frequency_masks[ifo], sample["asds"][ifo], 1.0)
            for ifo in ifos
        }

        return sample

    def _create_masks(self, detectors: list) -> Dict[str, np.ndarray]:
        """
        Create frequency masks for each detector.

        Parameters
        ----------
        detectors : list
            List of detector names

        Returns
        -------
        Dict[str, np.ndarray]
            Boolean masks for each detector
        """
        frequency_masks = {
            ifo: np.ones_like(self.sample_frequencies, dtype=bool)
            for ifo in detectors
        }

        for d in detectors:
            # Apply minimum_frequency mask
            if self.config.minimum_frequency is not None:
                if isinstance(self.config.minimum_frequency, (float, int)):
                    mask_min = self.sample_frequencies >= self.config.minimum_frequency
                elif isinstance(self.config.minimum_frequency, dict):
                    mask_min = self.sample_frequencies >= self.config.minimum_frequency[d]
                else:
                    raise ValueError(
                        f"minimum_frequency must be dict, int, or float, "
                        f"not {type(self.config.minimum_frequency)}."
                    )
                frequency_masks[d] = frequency_masks[d] & mask_min

            # Apply maximum_frequency mask
            if self.config.maximum_frequency is not None:
                if isinstance(self.config.maximum_frequency, (float, int)):
                    mask_max = self.sample_frequencies <= self.config.maximum_frequency
                elif isinstance(self.config.maximum_frequency, dict):
                    mask_max = self.sample_frequencies <= self.config.maximum_frequency[d]
                else:
                    raise ValueError(
                        f"maximum_frequency must be dict, int, or float, "
                        f"not {type(self.config.maximum_frequency)}."
                    )
                frequency_masks[d] = frequency_masks[d] & mask_max

        return frequency_masks
