"""
DecimateAll transform: Decimate nested dicts/arrays to multibanded frequency domain.

This transform performs recursive decimation of all arrays in the sample
dictionary to a multibanded frequency domain.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DomainProtocol, TransformSample


@dataclass(frozen=True)
class DecimateAllConfig(WaveformTransformConfig):
    """
    Configuration for DecimateAll transform.

    Attributes
    ----------
    multibanded_frequency_domain : Any
        MultibandedFrequencyDomain object for decimation.
        Original data must be in multibanded_frequency_domain.base_domain.

    Notes
    -----
    The multibanded_frequency_domain is stored as-is (not serialized to dict).
    Serialization/deserialization must handle Domain objects appropriately.

    Examples
    --------
    >>> from dingo.gw.domains import MultibandedFrequencyDomain
    >>> mfd = MultibandedFrequencyDomain(...)
    >>> config = DecimateAllConfig(multibanded_frequency_domain=mfd)
    >>> config.multibanded_frequency_domain
    MultibandedFrequencyDomain(...)
    """

    multibanded_frequency_domain: DomainProtocol

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Basic validation - check it's not None
        if self.multibanded_frequency_domain is None:
            raise ValueError("multibanded_frequency_domain cannot be None")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DecimateAllConfig':
        """
        Create DecimateAllConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with 'multibanded_frequency_domain' key.
            The value must be a MultibandedFrequencyDomain object
            (NOT a dict - must already be constructed).

        Returns
        -------
        DecimateAllConfig
            Validated configuration instance

        Examples
        --------
        >>> from dingo_waveform.domains import MultibandedFrequencyDomain
        >>> mfd = MultibandedFrequencyDomain(...)
        >>> config = DecimateAllConfig.from_dict({
        ...     'multibanded_frequency_domain': mfd
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

        return cls(multibanded_frequency_domain=mfd)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation with domain.domain_dict

        Examples
        --------
        >>> mfd = MultibandedFrequencyDomain(...)
        >>> config = DecimateAllConfig(multibanded_frequency_domain=mfd)
        >>> config_dict = config.to_dict()
        >>> config_dict['multibanded_frequency_domain']
        {'type': 'MultibandedFrequencyDomain', ...}
        """
        return {
            'multibanded_frequency_domain':
                self.multibanded_frequency_domain.domain_dict
        }


class DecimateAll(WaveformTransform[DecimateAllConfig]):
    """
    Transform operator for decimation to multibanded frequency domain.

    This transform recursively decimates all numpy arrays in the sample
    dictionary from the base_domain to the multibanded_frequency_domain.
    Arrays are identified by shape matching and decimated in-place.

    The decimation is performed on nested dictionary structures, allowing
    the transform to handle complex sample dictionaries with waveforms,
    ASDs, and other arrays organized by detector.

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.domains import MultibandedFrequencyDomain
    >>> from dingo_waveform.transform.transforms.waveform import (
    ...     DecimateAll,
    ...     DecimateAllConfig
    ... )
    >>>
    >>> # Create multibanded domain
    >>> mfd = MultibandedFrequencyDomain(...)
    >>> config = DecimateAllConfig(multibanded_frequency_domain=mfd)
    >>> transform = DecimateAll.from_config(config)
    >>>
    >>> # Sample with nested structure
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
    >>> # All arrays decimated to multibanded domain
    >>> result['waveform']['H1'].shape
    (len(mfd),)

    Notes
    -----
    The original data must be in multibanded_frequency_domain.base_domain.
    Arrays are decimated if their last dimension matches len(base_domain).

    The decimation is done recursively in-place on copied dictionaries.
    Only numpy arrays are decimated; other types raise ValueError.

    See Also
    --------
    DecimateWaveformsAndASDS : Specialized decimation for waveforms and ASDs
    MultibandedFrequencyDomain : Domain class with decimate() method
    """

    def __init__(self, config: DecimateAllConfig):
        """
        Initialize DecimateAll transform.

        Parameters
        ----------
        config : DecimateAllConfig
            Configuration containing multibanded_frequency_domain
        """
        super().__init__(config)

    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Apply decimation transform to all nested arrays.

        Generic transform that works on any pipeline stage - decimates all arrays
        found in nested dictionary structure.

        Parameters
        ----------
        input_sample : TransformSample
            Input sample (any pipeline stage) with nested dicts/arrays to decimate

        Returns
        -------
        TransformSample
            Sample with decimated data

        Examples
        --------
        >>> mfd = MultibandedFrequencyDomain(...)
        >>> config = DecimateAllConfig(multibanded_frequency_domain=mfd)
        >>> transform = DecimateAll.from_config(config)
        >>> sample = {'waveform': {'H1': np.random.randn(len(mfd.base_domain))}}
        >>> result = transform(sample)
        >>> result['waveform']['H1'].shape
        (len(mfd),)
        """
        sample = input_sample.copy()
        self._decimate_recursive(sample, self.config.multibanded_frequency_domain)
        return sample

    def _decimate_recursive(
        self, d: dict, mfd: Any
    ) -> None:
        """
        In-place decimation of nested dicts of arrays.

        Parameters
        ----------
        d : dict
            Nested dictionary to decimate
        mfd : MultibandedFrequencyDomain
            Multibanded frequency domain for decimation
        """
        for k, v in d.items():
            if isinstance(v, dict):
                self._decimate_recursive(v, mfd)
            elif isinstance(v, np.ndarray):
                if v.shape[-1] == len(mfd.base_domain):
                    d[k] = mfd.decimate(v)
            else:
                raise ValueError(f"Cannot decimate item of type {type(v)}.")
