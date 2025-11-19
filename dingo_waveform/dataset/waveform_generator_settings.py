"""Settings for waveform generator configuration.

This module provides a serializable configuration class for waveform generation,
intended for use in YAML/JSON configuration files. For runtime parameters with
the domain object, use WaveformGeneratorParameters instead.
"""

from dataclasses import dataclass
from typing import Optional

from ..approximant import Approximant
from ..waveform_generator_parameters import _validate_waveform_generator_params


@dataclass
class WaveformGeneratorSettings:
    """
    Serializable configuration settings for waveform generation.

    This class represents the configuration subset of WaveformGeneratorParameters
    that can be stored in YAML/JSON files. It excludes runtime-specific fields
    like the domain object, mode_list, lal_params, and transform.

    To convert to runtime parameters, use:
        WaveformGeneratorParameters.from_settings(settings, domain)

    Attributes
    ----------
    approximant
        Waveform approximant to use (e.g., IMRPhenomD, IMRPhenomXPHM)
    f_ref
        Reference frequency in Hz
    spin_conversion_phase
        Phase for spin conversion. If None, uses the phase from waveform parameters
    f_start
        Starting frequency for waveform generation in Hz. If specified, overrides
        domain f_min. Useful for EOB waveforms

    See Also
    --------
    dingo_waveform.waveform_generator_parameters.WaveformGeneratorParameters :
        Full runtime parameters class (includes domain and other runtime fields)
    dingo_waveform.waveform_generator_parameters.WaveformGeneratorParameters.from_settings :
        Method to convert these settings to runtime parameters

    Examples
    --------
    >>> settings = WaveformGeneratorSettings(
    ...     approximant="IMRPhenomXPHM",
    ...     f_ref=20.0,
    ...     spin_conversion_phase=0.0
    ... )
    >>> # Convert to runtime parameters
    >>> params = WaveformGeneratorParameters.from_settings(settings, domain)
    """

    approximant: Approximant
    f_ref: float
    spin_conversion_phase: Optional[float] = None
    f_start: Optional[float] = None

    def __post_init__(self):
        """Validate settings."""
        # Validate f_ref and f_start (approximant is already the correct type)
        _validate_waveform_generator_params(
            self.approximant, self.f_ref, self.f_start
        )

    def to_dict(self):
        """Convert to dictionary format."""
        result = {
            "approximant": str(self.approximant),
            "f_ref": self.f_ref,
        }
        if self.spin_conversion_phase is not None:
            result["spin_conversion_phase"] = self.spin_conversion_phase
        if self.f_start is not None:
            result["f_start"] = self.f_start
        return result
