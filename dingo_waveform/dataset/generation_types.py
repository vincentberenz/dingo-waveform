"""Type definitions for dataset generation module.

This module defines dataclasses that provide type-safe alternatives to Dict types
used in parallel waveform generation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WaveformGeneratorConfig:
    """Configuration for building a WaveformGenerator.

    Attributes
    ----------
    approximant : str
        Waveform approximant name (e.g., 'IMRPhenomXPHM')
    f_ref : float
        Reference frequency in Hz
    spin_conversion_phase : float
        Phase for spin conversion
    """
    approximant: str
    f_ref: float
    spin_conversion_phase: float

    def to_dict(self) -> dict:
        """Convert to dictionary for legacy compatibility."""
        return {
            "approximant": self.approximant,
            "f_ref": self.f_ref,
            "spin_conversion_phase": self.spin_conversion_phase,
        }


@dataclass
class WaveformResult:
    """Result of generating a single waveform.

    Attributes
    ----------
    h_plus : Optional[np.ndarray]
        Plus polarization array, or None if generation failed
    h_cross : Optional[np.ndarray]
        Cross polarization array, or None if generation failed
    success : bool
        Whether waveform generation succeeded
    error_message : Optional[str]
        Error message if generation failed, None otherwise
    """
    h_plus: Optional[np.ndarray]
    h_cross: Optional[np.ndarray]
    success: bool = True
    error_message: Optional[str] = None

    @classmethod
    def success_result(cls, h_plus: np.ndarray, h_cross: np.ndarray) -> "WaveformResult":
        """Create a successful result."""
        return cls(h_plus=h_plus, h_cross=h_cross, success=True, error_message=None)

    @classmethod
    def failure_result(cls, error_message: str) -> "WaveformResult":
        """Create a failed result."""
        return cls(h_plus=None, h_cross=None, success=False, error_message=error_message)

    def to_dict(self) -> dict:
        """Convert to dictionary for legacy compatibility."""
        return {
            "h_plus": self.h_plus,
            "h_cross": self.h_cross,
        }
