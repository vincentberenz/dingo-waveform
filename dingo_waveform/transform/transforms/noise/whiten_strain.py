"""
WhitenStrain transform: Whiten strain data by dividing by ASDs.

This transform performs whitening by dividing each detector's strain data
by its corresponding amplitude spectral density (ASD).
"""

from dataclasses import dataclass
from typing import Dict, Any
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import NoiseASDSample


@dataclass(frozen=True)
class WhitenStrainConfig(WaveformTransformConfig):
    """
    Configuration for WhitenStrain transform.

    This transform has no parameters - it simply whitens strain data
    using the ASDs already present in the sample.

    Examples
    --------
    >>> config = WhitenStrainConfig()
    >>> config.to_dict()
    {}
    """

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WhitenStrainConfig':
        """
        Create WhitenStrainConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary (can be empty for this transform)

        Returns
        -------
        WhitenStrainConfig
            Configuration instance

        Examples
        --------
        >>> config = WhitenStrainConfig.from_dict({})
        >>> isinstance(config, WhitenStrainConfig)
        True
        """
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Empty dictionary (this transform has no parameters)

        Examples
        --------
        >>> config = WhitenStrainConfig()
        >>> config.to_dict()
        {}
        """
        return {}


class WhitenStrain(WaveformTransform[WhitenStrainConfig]):
    """
    Whiten the strain data by dividing w.r.t. the corresponding ASDs.

    This transform divides each detector's strain by its ASD to produce
    whitened data. The input sample must contain both 'waveform' and 'asds'
    dictionaries with matching detector keys.

    The whitening operation is:
        whitened_strain[ifo] = strain[ifo] / asd[ifo]

    Examples
    --------
    >>> import numpy as np
    >>> from dingo_waveform.transform.transforms.noise import WhitenStrain, WhitenStrainConfig
    >>>
    >>> config = WhitenStrainConfig()
    >>> transform = WhitenStrain.from_config(config)
    >>>
    >>> sample = {
    ...     'waveform': {
    ...         'H1': np.array([1.0, 2.0, 3.0]),
    ...         'L1': np.array([4.0, 5.0, 6.0])
    ...     },
    ...     'asds': {
    ...         'H1': np.array([0.5, 0.5, 0.5]),
    ...         'L1': np.array([2.0, 2.0, 2.0])
    ...     }
    ... }
    >>> result = transform(sample)
    >>> result['waveform']['H1']
    array([2., 4., 6.])
    >>> result['waveform']['L1']
    array([2. , 2.5, 3. ])

    Notes
    -----
    This transform expects that ASDs have already been sampled via
    SampleNoiseASD transform. The detectors in 'waveform' and 'asds'
    must match exactly.

    See Also
    --------
    SampleNoiseASD : Samples random ASDs for detectors
    WhitenAndScaleStrain : Whitening with additional scaling factor
    WhitenFixedASD : Whitening with fixed ASD from file
    """

    def __init__(self, config: WhitenStrainConfig):
        """
        Initialize WhitenStrain transform.

        Parameters
        ----------
        config : WhitenStrainConfig
            Configuration (has no parameters for this transform)
        """
        super().__init__(config)

    def __call__(self, input_sample: NoiseASDSample) -> NoiseASDSample:  # type: ignore[override]
        """
        Apply whitening transform.

        Divides each detector's strain by its ASD to produce whitened data.

        Parameters
        ----------
        input_sample : NoiseASDSample
            Input sample with waveform and asds dicts for each detector

        Returns
        -------
        NoiseASDSample
            Sample with whitened waveform (asds unchanged)

        Raises
        ------
        ValueError
            If detectors in 'waveform' and 'asds' don't match
        KeyError
            If 'waveform' or 'asds' keys are missing

        Examples
        --------
        >>> config = WhitenStrainConfig()
        >>> transform = WhitenStrain.from_config(config)
        >>> sample = {
        ...     'waveform': {'H1': np.array([2.0])},
        ...     'asds': {'H1': np.array([0.5])}
        ... }
        >>> result = transform(sample)
        >>> result['waveform']['H1']
        array([4.])
        """
        sample = input_sample.copy()
        ifos = sample["waveform"].keys()

        # Validate that detectors match
        if ifos != sample["asds"].keys():
            raise ValueError(
                f"Detectors of strain data, {ifos}, do not match "
                f'those of asds, {sample["asds"].keys()}.'
            )

        # Whiten strains by dividing by ASDs
        whitened_strains = {
            ifo: sample["waveform"][ifo] / sample["asds"][ifo] for ifo in ifos
        }
        sample["waveform"] = whitened_strains

        return sample
