"""
CropMaskStrainRandom: Apply random frequency cropping to strain.

This complex transform randomly crops strain data by masking waveforms outside
randomly sampled frequency bounds. Supports both stochastic and deterministic
cropping modes with extensive configuration options.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import numpy as np
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import DomainProtocol, TransformSample


@dataclass(frozen=True)
class CropMaskStrainRandomConfig(WaveformTransformConfig):
    """
    Configuration for CropMaskStrainRandom transform.

    Attributes
    ----------
    domain : Any
        Domain object (UniformFrequencyDomain or MultibandedFrequencyDomain)
    f_min_upper : Optional[float]
        Upper bound for random f_min sampling: [domain.f_min, f_min_upper]
    f_max_lower : Optional[float]
        Lower bound for random f_max sampling: [f_max_lower, domain.f_max]
    cropping_probability : float
        Probability for a sample to be cropped. Default 1.0.
    independent_detectors : bool
        If True, crop boundaries sampled independently per detector. Default True.
    independent_lower_upper : bool
        If True, cropping probability applied to lower/upper independently. Default True.
    deterministic_fmin_fmax : Optional[Union[List, List[List]]]
        Fixed [f_min, f_max] bounds. If provided, disables random sampling.
        Can be single pair or list of pairs for multiple detectors.
    """

    domain: DomainProtocol
    f_min_upper: Optional[float] = None
    f_max_lower: Optional[float] = None
    cropping_probability: float = 1.0
    independent_detectors: bool = True
    independent_lower_upper: bool = True
    deterministic_fmin_fmax: Optional[Union[List, List[List]]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain

        if self.domain is None:
            raise ValueError("domain cannot be None")

        # Basic domain type check
        if not hasattr(self.domain, '__call__'):
            raise ValueError(f"Domain must be callable, got {type(self.domain)}")

        # Validate cropping_probability
        if not isinstance(self.cropping_probability, (int, float)):
            raise TypeError(
                f"cropping_probability must be numeric, got {type(self.cropping_probability)}"
            )

        # Validate deterministic vs stochastic parameters
        if self.deterministic_fmin_fmax is not None:
            if self.f_min_upper is not None or self.f_max_lower is not None:
                raise ValueError(
                    "If deterministic_fmin_fmax is set, f_min_upper and f_max_lower "
                    "should not be set."
                )
            if self.cropping_probability < 1.0:
                raise ValueError(
                    f"cropping_probability must be 1.0 when deterministic_fmin_fmax is set, "
                    f"got {self.cropping_probability}."
                )
        else:
            if not 0 <= self.cropping_probability <= 1.0:
                raise ValueError(
                    f"cropping_probability must be in [0, 1], got {self.cropping_probability}."
                )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CropMaskStrainRandomConfig':
        """Create config from dictionary."""
        from dingo.gw.domains import build_domain

        domain = config_dict['domain']
        if isinstance(domain, dict):
            domain = build_domain(domain)

        return cls(
            domain=domain,
            f_min_upper=config_dict.get('f_min_upper', None),
            f_max_lower=config_dict.get('f_max_lower', None),
            cropping_probability=config_dict.get('cropping_probability', 1.0),
            independent_detectors=config_dict.get('independent_detectors', True),
            independent_lower_upper=config_dict.get('independent_lower_upper', True),
            deterministic_fmin_fmax=config_dict.get('deterministic_fmin_fmax', None)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'domain': self.domain.domain_dict,
            'f_min_upper': self.f_min_upper,
            'f_max_lower': self.f_max_lower,
            'cropping_probability': self.cropping_probability,
            'independent_detectors': self.independent_detectors,
            'independent_lower_upper': self.independent_lower_upper,
            'deterministic_fmin_fmax': self.deterministic_fmin_fmax
        }


class CropMaskStrainRandom(WaveformTransform[CropMaskStrainRandomConfig]):
    """
    Apply random cropping of strain by masking outside frequency bounds.

    This transform masks waveform data outside randomly sampled frequency
    ranges. Supports:
    - Stochastic cropping with configurable probability
    - Deterministic cropping with fixed bounds
    - Independent cropping per detector
    - Independent lower/upper bound sampling

    Examples
    --------
    >>> import numpy as np
    >>> from dingo.gw.domains import UniformFrequencyDomain
    >>> from dingo_waveform.transform.transforms.waveform import (
    ...     CropMaskStrainRandom,
    ...     CropMaskStrainRandomConfig
    ... )
    >>>
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>>
    >>> # Stochastic cropping
    >>> config = CropMaskStrainRandomConfig(
    ...     domain=domain,
    ...     f_min_upper=50.0,    # Sample f_min in [20, 50] Hz
    ...     f_max_lower=512.0,   # Sample f_max in [512, 1024] Hz
    ...     cropping_probability=0.8,
    ...     independent_detectors=True
    ... )
    >>> transform = CropMaskStrainRandom.from_config(config)
    >>>
    >>> # Deterministic cropping
    >>> config_det = CropMaskStrainRandomConfig(
    ...     domain=domain,
    ...     deterministic_fmin_fmax=[30.0, 600.0]  # Fixed bounds
    ... )

    Notes
    -----
    Stochastic mode:
    - f_min sampled from [domain.f_min, f_min_upper]
    - f_max sampled from [f_max_lower, domain.f_max]
    - Cropping applied with probability cropping_probability

    Deterministic mode:
    - Fixed bounds specified in deterministic_fmin_fmax
    - Can provide single pair or per-detector pairs
    - Always applied (cropping_probability must be 1.0)

    Waveform is set to zero outside sampled/fixed bounds.
    """

    def __init__(self, config: CropMaskStrainRandomConfig):
        """Initialize transform."""
        super().__init__(config)

        self.frequencies = config.domain()[config.domain.min_idx:]
        self.len_domain = len(self.frequencies)

        # Stochastic mode: pre-compute frequency indices
        if config.deterministic_fmin_fmax is None:
            self._idx_bound_f_max = self._get_domain_idx(
                config.f_max_lower, self.len_domain - 1
            )
            self._idx_bound_f_min = self._get_domain_idx(config.f_min_upper, 0)
            self._deterministic_mask = None
        else:
            # Deterministic mode: pre-compute mask
            self._deterministic_mask = self._initialize_deterministic_mask(
                config.deterministic_fmin_fmax
            )

    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Apply random cropping transform.

        Generic transform that works on any pipeline stage with waveform data.

        Parameters
        ----------
        input_sample : TransformSample
            Input sample (any stage) with 'waveform' array

        Returns
        -------
        TransformSample
            Sample with cropped waveform
        """
        sample = input_sample.copy()
        strain = sample["waveform"]

        if strain.shape[-1] != self.len_domain:
            raise ValueError(
                f"Expected waveform shape [..., {self.len_domain}], "
                f"got {strain.shape}."
            )

        # Deterministic mode
        if self._deterministic_mask is not None:
            if self._deterministic_mask.shape[1] not in [1, strain.shape[1]]:
                raise ValueError(
                    f"Deterministic mask must match detector count. "
                    f"Expected 1 or {strain.shape[1]} detectors, "
                    f"got {self._deterministic_mask.shape[1]}."
                )
            strain = np.where(self._deterministic_mask, strain, 0)
            sample["waveform"] = strain
            return sample

        # Stochastic mode
        constant_ax = 3 - self.config.independent_detectors
        lower = self._sample_lower_bound_indices(strain.shape[:-constant_ax])
        upper = self._sample_upper_bound_indices(strain.shape[:-constant_ax])

        # Apply cropping probability
        if self.config.cropping_probability < 1:
            mask = np.random.uniform(size=lower.shape) <= self.config.cropping_probability
            lower = np.where(mask, lower, 0)
            if self.config.independent_lower_upper:
                mask = np.random.uniform(size=lower.shape) <= self.config.cropping_probability
            upper = np.where(mask, upper, self.len_domain)

        # Broadcast and apply cropping
        mask_lower = np.arange(self.len_domain) >= lower[(...,) + (None,) * constant_ax]
        mask_upper = np.arange(self.len_domain) <= upper[(...,) + (None,) * constant_ax]
        strain = np.where(mask_lower, strain, 0)
        strain = np.where(mask_upper, strain, 0)
        sample["waveform"] = strain

        return sample

    def _sample_lower_bound_indices(self, shape: tuple) -> np.ndarray:
        """Sample indices for lower crop boundaries."""
        return np.random.randint(0, self._idx_bound_f_min + 1, shape)

    def _sample_upper_bound_indices(self, shape: tuple) -> np.ndarray:
        """Sample indices for upper crop boundaries."""
        return np.random.randint(self._idx_bound_f_max, self.len_domain, shape)

    def _get_domain_idx(self, f_value: Optional[float], default_value: int) -> int:
        """Get frequency index closest to f_value."""
        if f_value is not None:
            return np.argmin(np.abs(f_value - self.frequencies)).item()
        else:
            return default_value

    def _initialize_deterministic_mask(self, deterministic_fmin_fmax) -> np.ndarray:
        """Initialize deterministic cropping mask."""
        # Convert single pair to list of pairs
        if not isinstance(deterministic_fmin_fmax[0], list):
            deterministic_fmin_fmax = [deterministic_fmin_fmax]

        masking_indices = []
        for f_min, f_max in deterministic_fmin_fmax:
            self._check_fmin_fmax(f_min, f_max)
            idx_min = self._get_domain_idx(f_min, 0)
            idx_max = self._get_domain_idx(f_max, self.len_domain - 1)
            if idx_min >= idx_max:
                raise ValueError(f"f_min ({f_min}) must be less than f_max ({f_max}).")
            masking_indices.append([idx_min, idx_max])

        masking_indices = np.array(masking_indices)
        lower = masking_indices[:, 0]
        upper = masking_indices[:, 1]
        mask_lower = np.arange(len(self.frequencies))[None, :] >= lower[:, None]
        mask_upper = np.arange(len(self.frequencies))[None, :] <= upper[:, None]
        mask = mask_lower * mask_upper

        # Broadcast: (detectors, freq) => (batch, detectors, channels, freq)
        return mask[None, :, None, :]

    def _check_fmin_fmax(self, f_min: Optional[float], f_max: Optional[float]) -> None:
        """Validate f_min and f_max are within domain bounds."""
        domain_f_min = self.frequencies[0]
        domain_f_max = self.frequencies[-1]

        if f_min is not None and f_max is not None:
            if not (domain_f_min < f_min < f_max < domain_f_max):
                raise ValueError(
                    f"Expected {domain_f_min} < {f_min} < {f_max} < {domain_f_max}"
                )
        elif f_min is not None:
            if not (domain_f_min < f_min < domain_f_max):
                raise ValueError(f"Expected {f_min} in ({domain_f_min}, {domain_f_max})")
        elif f_max is not None:
            if not (domain_f_min < f_max < domain_f_max):
                raise ValueError(f"Expected {f_max} in ({domain_f_min}, {domain_f_max})")
