"""
Type definitions for Transform configuration.

This module provides proper dataclasses to replace Dict/List/Any types
in Transform and TransformConfig, following the dingo-waveform design principle
of using well-typed dataclasses instead of opaque dictionaries.
"""

from dataclasses import dataclass, field
from typing import Union, Optional, Dict, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports
    from dingo.gw.domains import Domain


@dataclass(frozen=True)
class StandardizationConfig:
    """
    Configuration for parameter standardization (mean/std normalization).

    Attributes
    ----------
    mean : Dict[str, float]
        Mean values for each parameter
    std : Dict[str, float]
        Standard deviation values for each parameter
    """
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that mean and std have same keys."""
        if set(self.mean.keys()) != set(self.std.keys()):
            raise ValueError(
                "mean and std must have the same parameter keys. "
                f"mean keys: {set(self.mean.keys())}, std keys: {set(self.std.keys())}"
            )


@dataclass(frozen=True)
class RandomStrainCroppingConfig:
    """
    Configuration for random strain cropping transform.

    This configuration specifies how to randomly crop frequency ranges
    during training to improve model robustness.

    Attributes
    ----------
    f_min_upper : Optional[float]
        Upper bound for sampling new f_min. New f_min sampled in [domain.f_min, f_min_upper].
    f_max_lower : Optional[float]
        Lower bound for sampling new f_max. New f_max sampled in [f_max_lower, domain.f_max].
    cropping_probability : float
        Probability of cropping being applied to a sample (default 1.0).
    independent_detectors : bool
        If True, crop boundaries sampled independently per detector (default True).
    independent_lower_upper : bool
        If True, cropping probability applied independently to lower/upper bounds (default True).
    deterministic_fmin_fmax : Optional[Union[List[Optional[float]], List[List[Optional[float]]]]]
        If not None, use fixed f_min/f_max values instead of random sampling.
        Can be [f_min, f_max] or [[f_min1, f_max1], [f_min2, f_max2], ...] for multiple crops.
    """
    f_min_upper: Optional[float] = None
    f_max_lower: Optional[float] = None
    cropping_probability: float = 1.0
    independent_detectors: bool = True
    independent_lower_upper: bool = True
    deterministic_fmin_fmax: Optional[
        Union[List[Optional[float]], List[List[Optional[float]]]]
    ] = None

    def __post_init__(self) -> None:
        """Validate cropping configuration."""
        if not 0.0 <= self.cropping_probability <= 1.0:
            raise ValueError(
                f"cropping_probability must be in [0, 1], got {self.cropping_probability}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "RandomStrainCroppingConfig":
        """
        Create RandomStrainCroppingConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict
            Dictionary with cropping configuration parameters

        Returns
        -------
        RandomStrainCroppingConfig
            Validated configuration instance
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary format (for compatibility with existing code).

        Returns
        -------
        Dict
            Dictionary representation with all non-None fields
        """
        result = {}
        if self.f_min_upper is not None:
            result["f_min_upper"] = self.f_min_upper
        if self.f_max_lower is not None:
            result["f_max_lower"] = self.f_max_lower
        result["cropping_probability"] = self.cropping_probability
        result["independent_detectors"] = self.independent_detectors
        result["independent_lower_upper"] = self.independent_lower_upper
        if self.deterministic_fmin_fmax is not None:
            result["deterministic_fmin_fmax"] = self.deterministic_fmin_fmax
        return result


@dataclass(frozen=True)
class GNPETimeShiftsConfig:
    """
    Configuration for GNPE (Group Neural Posterior Estimation) time shifts.

    GNPE applies group-equivariant transformations to improve inference robustness.
    This configuration specifies the kernel for sampling time shift perturbations
    and whether to enforce exact global time translation equivariance.

    Attributes
    ----------
    kernel : str
        Kernel type for sampling GNPE perturbations (e.g., "uniform", "gaussian")
    exact_equiv : bool
        Whether to enforce exact global time translation equivariance.
        If True, subtracts one proxy (first detector) from all others for exact equivariance.

    References
    ----------
    [1] GNPE paper: arxiv.org/abs/2111.13139
    """
    kernel: str
    exact_equiv: bool = True

    def __post_init__(self) -> None:
        """Validate GNPE configuration."""
        if not isinstance(self.kernel, str) or not self.kernel:
            raise ValueError(f"kernel must be a non-empty string, got {self.kernel}")

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "GNPETimeShiftsConfig":
        """
        Create GNPETimeShiftsConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict
            Dictionary with GNPE configuration. Must contain 'kernel' key.
            May contain 'exact_equiv' key (defaults to True).

        Returns
        -------
        GNPETimeShiftsConfig
            Validated configuration instance
        """
        # Handle both 'exact_equiv' and 'exact_global_equivariance' keys for backward compatibility
        exact_equiv = config_dict.get("exact_equiv", config_dict.get("exact_global_equivariance", True))
        return cls(
            kernel=config_dict["kernel"],
            exact_equiv=exact_equiv
        )

    def to_dict(self) -> Dict:
        """
        Convert to dictionary format (for compatibility with existing code).

        Returns
        -------
        Dict
            Dictionary representation with 'kernel' and 'exact_equiv' keys
        """
        return {
            "kernel": self.kernel,
            "exact_equiv": self.exact_equiv
        }


@dataclass(frozen=True)
class DomainUpdateConfig:
    """
    Configuration for updating frequency domain bounds.

    Specifies new minimum and/or maximum frequencies for narrowing
    the domain during inference or dataset generation.

    Attributes
    ----------
    f_min : Optional[float]
        New minimum frequency (Hz). If None, uses original domain f_min.
    f_max : Optional[float]
        New maximum frequency (Hz). If None, uses original domain f_max.
    """
    f_min: Optional[float] = None
    f_max: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate domain update configuration."""
        if self.f_min is not None and self.f_min <= 0:
            raise ValueError(f"f_min must be positive, got {self.f_min}")
        if self.f_max is not None and self.f_max <= 0:
            raise ValueError(f"f_max must be positive, got {self.f_max}")
        if self.f_min is not None and self.f_max is not None:
            if self.f_min >= self.f_max:
                raise ValueError(
                    f"f_min must be less than f_max, got f_min={self.f_min}, f_max={self.f_max}"
                )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, float]) -> "DomainUpdateConfig":
        """
        Create DomainUpdateConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, float]
            Dictionary with 'f_min' and/or 'f_max' keys

        Returns
        -------
        DomainUpdateConfig
            Validated configuration instance
        """
        return cls(
            f_min=config_dict.get("f_min"),
            f_max=config_dict.get("f_max")
        )

    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary format (for compatibility with existing code).

        Returns
        -------
        Dict[str, float]
            Dictionary with non-None frequency bounds
        """
        result = {}
        if self.f_min is not None:
            result["f_min"] = self.f_min
        if self.f_max is not None:
            result["f_max"] = self.f_max
        return result


@dataclass(frozen=True)
class ExtrinsicPriorConfig:
    """
    Configuration for extrinsic parameter priors.

    Extrinsic parameters describe how the signal projects onto detectors:
    sky location (ra, dec), orientation (psi), distance, and coalescence time.

    Each parameter is specified as either:
    - A string representing a bilby prior (e.g., "bilby.core.prior.Uniform(...)")
    - A float for a fixed value
    - The string "default" to use dingo's default prior for that parameter

    Attributes
    ----------
    dec : Union[str, float]
        Declination prior or value
    ra : Union[str, float]
        Right ascension prior or value
    geocent_time : Union[str, float]
        Geocentric coalescence time prior or value (relative to ref_time)
    psi : Union[str, float]
        Polarization angle prior or value
    luminosity_distance : Union[str, float]
        Luminosity distance (Mpc) prior or value

    Notes
    -----
    The get_extrinsic_prior_dict() function in dingo.gw.gwutils processes this
    configuration by replacing "default" values with dingo's default priors.

    Examples
    --------
    >>> config = ExtrinsicPriorConfig(
    ...     dec="default",
    ...     ra="default",
    ...     geocent_time="bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
    ...     psi="default",
    ...     luminosity_distance="bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0)"
    ... )
    """
    dec: Union[str, float] = "default"
    ra: Union[str, float] = "default"
    geocent_time: Union[str, float] = "default"
    psi: Union[str, float] = "default"
    luminosity_distance: Union[str, float] = "default"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Union[str, float]]) -> "ExtrinsicPriorConfig":
        """
        Create ExtrinsicPriorConfig from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Union[str, float]]
            Dictionary mapping parameter names to prior specifications

        Returns
        -------
        ExtrinsicPriorConfig
            Configuration instance (missing keys use "default")
        """
        return cls(
            dec=config_dict.get("dec", "default"),
            ra=config_dict.get("ra", "default"),
            geocent_time=config_dict.get("geocent_time", "default"),
            psi=config_dict.get("psi", "default"),
            luminosity_distance=config_dict.get("luminosity_distance", "default")
        )

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """
        Convert to dictionary format (for compatibility with existing code).

        Returns
        -------
        Dict[str, Union[str, float]]
            Dictionary mapping parameter names to prior specifications
        """
        return {
            "dec": self.dec,
            "ra": self.ra,
            "geocent_time": self.geocent_time,
            "psi": self.psi,
            "luminosity_distance": self.luminosity_distance
        }
