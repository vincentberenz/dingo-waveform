from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import lal

from .approximant import Approximant
from .domains import Domain
from .polarizations import Polarization
from .types import Modes


@dataclass
class WaveformGeneratorParameters:

    approximant: Approximant
    domain: Domain
    f_ref: float
    f_start: Optional[float]
    spin_conversion_phase: Optional[float]
    convert_to_SI: bool
    mode_list: Optional[List[Modes]]
    lal_params: Optional[lal.Dict]
    transform: Optional[Callable[[Polarization], Polarization]] = None
