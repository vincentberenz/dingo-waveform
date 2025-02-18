from dataclasses import dataclass
from typing import Dict, Optional

import lal

from .approximant import Approximant
from .spins import Spins
from .types import FrequencySeries, Iota, Mode


@dataclass
class InspiralChooseFDModesParameters(Spins):
    mass1: float
    mass2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    phase: float
    r: float
    iota: Iota
    lal_params: Optional[lal.Dict]
    approximant: Approximant

    def convert_J_to_L0_frame(
        self, hlm_J: Dict[Mode, FrequencySeries], spin_conversion_phase: Optional[float]
    ) -> Dict[Mode, FrequencySeries]:
        phase = self.phase
        if spin_conversion_phase is not None:
            phase = 0.0
        converted_to_SI = True
        return self._convert_J_to_L0_frame(
            hlm_J, self.mass1, self.mass2, converted_to_SI, self.f_ref, phase
        )
