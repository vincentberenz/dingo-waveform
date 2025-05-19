import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import lal
import tomli

from .approximant import Approximant
from .domains import Domain, build_domain
from .imports import read_file
from .logs import TableStr
from .polarizations import Polarization
from .types import Modes


@dataclass
class WaveformGeneratorParameters(TableStr):
    """
    Container class for parameters controlling gravitational waveform generation.

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
    mode_list: Optional[List[Modes]]
    lal_params: Optional[lal.Dict]
    transform: Optional[Callable[[Polarization], Polarization]] = None

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], domain: Domain
    ) -> "WaveformGeneratorParameters":
        params = read_file(file_path)
        params["domain"] = domain
        return cls(**params)
