from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import lal

from .approximant import Approximant
from .domains import Domain
from .polarizations import Polarization
from .types import Modes


@dataclass
class WaveformGeneratorParameters:
    """
    Container class for parameters controlling gravitational wave waveform generation.

    Attributes
    ----------
    approximant :
        The waveform approximant model to use (e.g., SEOBNRv5, IMRPhenomD)
    domain :
        The computational domain for the waveform generation
    f_ref :
        Reference frequency for the waveform generation
    f_start :
        Starting frequency for the waveform generation
    spin_conversion_phase :
        Phase angle used for converting spins
    convert_to_SI :
        Flag indicating whether to perform unit conversions to SI system
    mode_list :
        List of (ell, m) tuples specifying the spherical harmonic modes to include
        in the waveform calculation
    lal_params :
        Additional LAL parameters dictionary for waveform generation
    transform :
        Optional transformation function to apply to the waveform polarizations
    """

    approximant: Approximant
    domain: Domain
    f_ref: float
    f_start: Optional[float]
    spin_conversion_phase: Optional[float]
    convert_to_SI: bool
    mode_list: Optional[List[Modes]]
    lal_params: Optional[lal.Dict]
    transform: Optional[Callable[[Polarization], Polarization]] = None
