"""
DecimateWaveformsAndASDS transform: Specialized decimation for waveforms and ASDs.

This transform decimates unwhitened waveforms and corresponding ASDs to
multibanded frequency domain, with two modes: 'whitened' and 'unwhitened'.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DecimationMode, DomainProtocol, NoiseASDSample


@dataclass(frozen=True)
class DecimateWaveformsAndASDSConfig(WaveformTransformConfig):
    """
    Configuration for DecimateWaveformsAndASDS transform.

    Attributes
    ----------
    multibanded_frequency_domain : DomainProtocol
        MultibandedFrequencyDomain object for decimation.
        Original data must be in multibanded_frequency_domain.base_domain.
    decimation_mode : DecimationMode
        One of ["whitened", "unwhitened"]. Determines whether decimation is
        performed on whitened data or unwhitened data.

    Notes
    -----
    Decimation modes:

    1) decimation_mode = "whitened"
       Data is whitened first (dw = d / ASD), then decimated (dw_mfd = decimate(dw)).
       Effective ASD: ASD_mfd = 1 / decimate(1 / ASD).
       Better signal preservation.

    2) decimation_mode = "unwhitened"
       Data is decimated first (d_mfd = decimate(d)), then whitened.
       Effective ASD: ASD_mfd = decimate(ASD ** 2) ** 0.5.
       (Decimates PSD, not ASD).

    See: https://github.com/dingo-gw/dingo/blob/fede5c01524f3e205acf5750c0a0f101ff17e331/binary_neutron_stars/prototyping/psd_decimation.ipynb

    Examples
    --------
    >>> from dingo.gw.domains import MultibandedFrequencyDomain
    >>> mfd = MultibandedFrequencyDomain(...)
    >>> config = DecimateWaveformsAndASDSConfig(
    ...     multibanded_frequency_domain=mfd,
    ...     decimation_mode='whitened'
    ... )
    >>> config.decimation_mode
    'whitened'
    """

    multibanded_frequency_domain: DomainProtocol  # MultibandedFrequencyDomain
    decimation_mode: DecimationMode

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.multibanded_frequency_domain is None:
            raise ValueError("multibanded_frequency_domain cannot be None")

        if self.decimation_mode not in ["whitened", "unwhitened"]:
            raise ValueError(
                f"Unsupported decimation mode {self.decimation_mode}, "
                f"needs to be one of ['whitened', 'unwhitened']."
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DecimateWaveformsAndASDSConfig':
        """
        Create DecimateWaveformsAndASDSConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with keys:
            - 'multibanded_frequency_domain': MultibandedFrequencyDomain object
              (NOT a dict - must already be constructed)
            - 'decimation_mode': str ('whitened' or 'unwhitened')

        Returns
        -------
        DecimateWaveformsAndASDSConfig
            Validated configuration instance

        Examples
        --------
        >>> from dingo_waveform.domains import MultibandedFrequencyDomain
        >>> mfd = MultibandedFrequencyDomain(...)
        >>> config = DecimateWaveformsAndASDSConfig.from_dict({
        ...     'multibanded_frequency_domain': mfd,
        ...     'decimation_mode': 'whitened'
        ... })

        Notes
        -----
        The domain must have decimate() method (MultibandedFrequencyDomain).
        Users are responsible for building the domain before calling this method.
        """
        mfd = config_dict['multibanded_frequency_domain']

        # Validate domain using duck typing
        if not hasattr(mfd, 'decimate'):
            raise TypeError(
                f"multibanded_frequency_domain must have decimate() method "
                f"(expected MultibandedFrequencyDomain-like object), got {type(mfd)}"
            )

        return cls(
            multibanded_frequency_domain=mfd,
            decimation_mode=config_dict['decimation_mode']
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
        >>> mfd = MultibandedFrequencyDomain(...)
        >>> config = DecimateWaveformsAndASDSConfig(
        ...     multibanded_frequency_domain=mfd,
        ...     decimation_mode='whitened'
        ... )
        >>> config_dict = config.to_dict()
        >>> config_dict['decimation_mode']
        'whitened'
        """
        return {
            'multibanded_frequency_domain':
                self.multibanded_frequency_domain.domain_dict,
            'decimation_mode': self.decimation_mode
        }


class DecimateWaveformsAndASDS(WaveformTransform[DecimateWaveformsAndASDSConfig]):
    """
    Transform operator for decimation of waveforms and ASDs to multibanded domain.

    This transform handles two decimation strategies:

    1. **Whitened mode** (decimation_mode="whitened"):
       - Whiten data first: dw = d / ASD
       - Decimate whitened data: dw_mfd = decimate(dw)
       - Compute effective ASD: ASD_mfd = 1 / decimate(1 / ASD)
       - Re-color: waveform_mfd = dw_mfd * ASD_mfd
       - **Better signal preservation**

    2. **Unwhitened mode** (decimation_mode="unwhitened"):
       - Decimate data first: d_mfd = decimate(d)
       - Decimate PSD: ASD_mfd = decimate(ASD ** 2) ** 0.5
       - Whitening: dw_mfd = d_mfd / ASD_mfd

    The transform only decimates if data is in base_domain; already-decimated
    data is left unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.domains import MultibandedFrequencyDomain
    >>> from dingo_waveform.transform.transforms.waveform import (
    ...     DecimateWaveformsAndASDS,
    ...     DecimateWaveformsAndASDSConfig
    ... )
    >>>
    >>> mfd = MultibandedFrequencyDomain(...)
    >>> config = DecimateWaveformsAndASDSConfig(
    ...     multibanded_frequency_domain=mfd,
    ...     decimation_mode='whitened'
    ... )
    >>> transform = DecimateWaveformsAndASDS.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': {
    ...         'H1': np.random.randn(len(mfd.base_domain)),
    ...         'L1': np.random.randn(len(mfd.base_domain))
    ...     },
    ...     'asds': {
    ...         'H1': np.random.randn(len(mfd.base_domain)),
    ...         'L1': np.random.randn(len(mfd.base_domain))
    ...     }
    ... }
    >>> result = transform(sample)
    >>> result['waveform']['H1'].shape
    (len(mfd),)

    Notes
    -----
    The original data must be in multibanded_frequency_domain.base_domain.
    If data has already been decimated, it is left unchanged.

    Reference: https://github.com/dingo-gw/dingo/blob/fede5c01/binary_neutron_stars/prototyping/psd_decimation.ipynb

    See Also
    --------
    DecimateAll : General decimation for all nested arrays
    MultibandedFrequencyDomain : Domain class with decimate() method
    """

    def __init__(self, config: DecimateWaveformsAndASDSConfig):
        """
        Initialize DecimateWaveformsAndASDS transform.

        Parameters
        ----------
        config : DecimateWaveformsAndASDSConfig
            Configuration with multibanded domain and decimation mode
        """
        super().__init__(config)

    def __call__(self, input_sample: NoiseASDSample) -> NoiseASDSample:  # type: ignore[override]
        """
        Apply waveform and ASD decimation to multibanded domain.

        Decimates both waveforms and ASDs from uniform frequency domain to
        multibanded frequency domain with dyadic spacing.

        Parameters
        ----------
        input_sample : NoiseASDSample
            Input sample with waveform and asds dicts in uniform frequency domain

        Returns
        -------
        NoiseASDSample
            Sample with decimated waveforms and asds in multibanded domain

        Examples
        --------
        >>> mfd = MultibandedFrequencyDomain(...)
        >>> config = DecimateWaveformsAndASDSConfig(
        ...     multibanded_frequency_domain=mfd,
        ...     decimation_mode='whitened'
        ... )
        >>> transform = DecimateWaveformsAndASDS.from_config(config)
        >>> sample = {
        ...     'waveform': {'H1': np.random.randn(len(mfd.base_domain))},
        ...     'asds': {'H1': np.random.randn(len(mfd.base_domain))}
        ... }
        >>> result = transform(sample)
        >>> result['waveform']['H1'].shape
        (len(mfd),)
        """
        sample = input_sample.copy()
        mfd = self.config.multibanded_frequency_domain

        # Only decimate if data is in base domain
        if not self._check_sample_in_domain(sample, mfd.base_domain):
            return sample

        if self.config.decimation_mode == "whitened":
            # Whiten, decimate, compute effective ASD, re-color
            whitened_waveforms = {
                k: v / sample["asds"][k]
                for k, v in sample["waveform"].items()
            }
            whitened_waveforms_dec = {
                k: mfd.decimate(v)
                for k, v in whitened_waveforms.items()
            }
            asds_dec = {
                k: 1 / mfd.decimate(1 / v)
                for k, v in sample["asds"].items()
            }
            # Re-color the whitened waveforms with effective ASD
            waveform_dec = {
                k: v * asds_dec[k]
                for k, v in whitened_waveforms_dec.items()
            }
            sample["waveform"] = waveform_dec
            sample["asds"] = asds_dec

        elif self.config.decimation_mode == "unwhitened":
            # Decimate waveforms and PSDs
            sample["waveform"] = {
                k: mfd.decimate(v)
                for k, v in sample["waveform"].items()
            }
            sample["asds"] = {
                k: mfd.decimate(v**2) ** 0.5
                for k, v in sample["asds"].items()
            }

        else:
            raise NotImplementedError()

        return sample

    def _check_sample_in_domain(
        self, sample: Dict[str, Any], domain: Any
    ) -> bool:
        """
        Check if sample data is in the specified domain.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample with waveform and asds
        domain : UniformFrequencyDomain
            Domain to check against

        Returns
        -------
        bool
            True if all arrays match domain length
        """
        lengths = []
        base_domain_length = len(domain)
        for k in ["waveform", "asds"]:
            lengths += [d.shape[-1] for d in sample[k].values()]

        return all(l == base_domain_length for l in lengths)
