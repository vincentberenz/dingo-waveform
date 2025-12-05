"""
WhitenFixedASD transform: Whiten data with fixed ASD from file.

This transform whitens frequency-domain data according to an ASD specified
in a file, typically using Bilby's built-in ASD files (e.g., aLIGO design curve).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import numpy as np
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DomainProtocol, DetectorStrainSample, NoiseASDSample


@dataclass(frozen=True)
class WhitenFixedASDConfig(WaveformTransformConfig):
    """
    Configuration for WhitenFixedASD transform.

    Attributes
    ----------
    domain : Any
        UniformFrequencyDomain object. ASD is interpolated to this frequency grid.
    asd_file : Optional[str]
        Path to ASD file. If None, uses Bilby's aLIGO ASD.
    inverse : bool
        If True, applies inverse whitening (un-whitening). Default is False.
    precision : Optional[str]
        If specified, sets ASD precision to "single" or "double". Default is None.

    Examples
    --------
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> config = WhitenFixedASDConfig(
    ...     domain=domain,
    ...     asd_file=None,  # Use aLIGO default
    ...     inverse=False,
    ...     precision='single'
    ... )
    """

    domain: DomainProtocol
    asd_file: Optional[str] = None
    inverse: bool = False
    precision: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.domain is None:
            raise ValueError("domain cannot be None")

        if self.precision is not None:
            if self.precision not in ["single", "double"]:
                raise ValueError(
                    f"precision must be 'single' or 'double', got '{self.precision}'"
                )

        if not isinstance(self.inverse, bool):
            raise TypeError(f"inverse must be a bool, got {type(self.inverse)}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WhitenFixedASDConfig':
        """
        Create WhitenFixedASDConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with keys:
            - 'domain': Domain object or dict representation
            - 'asd_file': Optional path to ASD file
            - 'inverse': Optional bool (default False)
            - 'precision': Optional str ('single' or 'double')

        Returns
        -------
        WhitenFixedASDConfig
            Validated configuration instance

        Examples
        --------
        >>> from dingo.gw.domains import UniformFrequencyDomain
        >>> domain = UniformFrequencyDomain(...)
        >>> config = WhitenFixedASDConfig.from_dict({
        ...     'domain': domain,
        ...     'asd_file': None,
        ...     'inverse': False,
        ...     'precision': 'single'
        ... })
        """
        from dingo.gw.domains import build_domain

        domain = config_dict['domain']

        # Convert dict to Domain object if needed
        if isinstance(domain, dict):
            domain = build_domain(domain)

        return cls(
            domain=domain,
            asd_file=config_dict.get('asd_file', None),
            inverse=config_dict.get('inverse', False),
            precision=config_dict.get('precision', None)
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
        >>> config = WhitenFixedASDConfig(
        ...     domain=domain,
        ...     asd_file='/path/to/asd.txt',
        ...     inverse=False,
        ...     precision='single'
        ... )
        >>> config_dict = config.to_dict()
        >>> config_dict['asd_file']
        '/path/to/asd.txt'
        """
        return {
            'domain': self.domain.domain_dict,
            'asd_file': self.asd_file,
            'inverse': self.inverse,
            'precision': self.precision
        }


class WhitenFixedASD(WaveformTransform[WhitenFixedASDConfig]):
    """
    Whiten frequency-domain data with fixed ASD from file.

    This transform loads an ASD from a file (or uses Bilby's default aLIGO curve),
    interpolates it to the domain's frequency grid, and whitens data by dividing
    by the ASD.

    Can also apply inverse whitening (un-whitening) by multiplying by the ASD.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> from dingo_waveform.transform.transforms.noise import (
    ...     WhitenFixedASD,
    ...     WhitenFixedASDConfig
    ... )
    >>>
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> config = WhitenFixedASDConfig(
    ...     domain=domain,
    ...     asd_file=None,  # Use aLIGO default
    ...     inverse=False,
    ...     precision='single'
    ... )
    >>> transform = WhitenFixedASD.from_config(config)
    >>>
    >>> # Whiten data
    >>> sample = {
    ...     'H1': np.random.randn(len(domain)) + 1j * np.random.randn(len(domain)),
    ...     'L1': np.random.randn(len(domain)) + 1j * np.random.randn(len(domain))
    ... }
    >>> result = transform(sample)
    >>> # Data is now whitened

    >>> # Un-whiten data
    >>> unwhiten_config = WhitenFixedASDConfig(
    ...     domain=domain,
    ...     asd_file=None,
    ...     inverse=True
    ... )
    >>> unwhiten_transform = WhitenFixedASD.from_config(unwhiten_config)
    >>> original = unwhiten_transform(result)

    Notes
    -----
    The ASD is loaded and interpolated once during initialization. The interpolated
    ASD is stored in self.asd_array for efficient repeated application.

    If the ASD file's f_max is lower than the domain's f_max, a ValueError is raised.

    The ASD is interpolated using scipy.interpolate.interp1d with bounds_error=False
    and fill_value=np.inf (outside the ASD range, whitening divides by infinity,
    effectively zeroing those frequencies).

    Low-frequency bins are updated using domain.update_data() to avoid division
    by very small values.

    See Also
    --------
    WhitenAndScaleStrain : Whitens with sampled ASDs
    SampleNoiseASD : Samples ASDs from dataset
    """

    def __init__(self, config: WhitenFixedASDConfig):
        """
        Initialize WhitenFixedASD transform.

        Parameters
        ----------
        config : WhitenFixedASDConfig
            Configuration with domain, ASD file path, and options
        """
        super().__init__(config)

        from bilby.gw.detector import PowerSpectralDensity
        from scipy.interpolate import interp1d
        from dingo.gw.domains import UniformFrequencyDomain

        # Load PSD from file or use default
        if config.asd_file is not None:
            psd = PowerSpectralDensity(asd_file=config.asd_file)
        else:
            psd = PowerSpectralDensity.from_aligo()

        # Validate frequency range
        if psd.frequency_array[-1] < config.domain.f_max:
            raise ValueError(
                f"ASD in {config.asd_file} has f_max={psd.frequency_array[-1]}, "
                f"which is lower than domain f_max={config.domain.f_max}."
            )

        # Interpolate ASD to domain frequency grid
        asd_interp = interp1d(
            psd.frequency_array,
            psd.asd_array,
            bounds_error=False,
            fill_value=np.inf
        )
        self.asd_array = asd_interp(config.domain.sample_frequencies)
        self.asd_array = config.domain.update_data(self.asd_array, low_value=1e-22)

        # Set precision if specified
        if config.precision is not None:
            if config.precision == "single":
                self.asd_array = self.asd_array.astype(np.float32)
            elif config.precision == "double":
                self.asd_array = self.asd_array.astype(np.float64)

    def __call__(
        self,
        input_sample: Union[DetectorStrainSample, NoiseASDSample]
    ) -> Union[DetectorStrainSample, NoiseASDSample]:  # type: ignore[override]
        """
        Apply fixed ASD whitening or un-whitening to detector strains.

        Parameters
        ----------
        input_sample : Union[DetectorStrainSample, NoiseASDSample]
            Sample with detector waveforms to whiten/un-whiten

        Returns
        -------
        Union[DetectorStrainSample, NoiseASDSample]
            Sample with whitened/un-whitened detector strains

        Examples
        --------
        >>> domain = UniformFrequencyDomain(...)
        >>> config = WhitenFixedASDConfig(domain=domain, inverse=False)
        >>> transform = WhitenFixedASD.from_config(config)
        >>> sample = {'H1': np.random.randn(len(domain))}
        >>> result = transform(sample)
        >>> # result['H1'] is whitened

        >>> # Inverse (un-whiten)
        >>> unwhiten_config = WhitenFixedASDConfig(domain=domain, inverse=True)
        >>> unwhiten = WhitenFixedASD.from_config(unwhiten_config)
        >>> original = unwhiten(result)
        """
        result = {}
        for k, v in input_sample.items():
            if self.config.inverse:
                result[k] = v * self.asd_array
            else:
                result[k] = v / self.asd_array
        return result
