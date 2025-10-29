"""Settings for waveform generator configuration."""

from dataclasses import dataclass
from typing import Optional

from ..approximant import Approximant


@dataclass
class WaveformGeneratorSettings:
    """
    Configuration settings for waveform generation.

    Attributes
    ----------
    approximant
        Waveform approximant to use (e.g., IMRPhenomD, IMRPhenomXPHM).
    f_ref
        Reference frequency in Hz.
    spin_conversion_phase
        Phase for spin conversion. If None, uses the phase from waveform parameters.
    f_start
        Starting frequency for waveform generation in Hz. If specified, overrides
        domain f_min. Useful for EOB waveforms.
    """

    approximant: Approximant
    f_ref: float
    spin_conversion_phase: Optional[float] = None
    f_start: Optional[float] = None

    def __post_init__(self):
        """Validate settings and convert approximant if needed."""
        # Convert string to Approximant enum if necessary
        if isinstance(self.approximant, str):
            self.approximant = Approximant(self.approximant)

        # Validate f_ref is positive
        if self.f_ref <= 0:
            raise ValueError(f"f_ref must be positive, got {self.f_ref}")

        # Validate f_start if specified
        if self.f_start is not None and self.f_start <= 0:
            raise ValueError(f"f_start must be positive, got {self.f_start}")

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
