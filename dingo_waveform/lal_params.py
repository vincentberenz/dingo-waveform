from typing import List

import lal
import lalsimulation as LS

from dingo_waveform.types import Modes


def get_lal_params(mode_list: List[Modes]) -> lal.Dict:
    lal_params = lal.CreateDict()
    ma = LS.SimInspiralCreateModeArray()
    for ell, m in mode_list:
        LS.SimInspiralModeArrayActivateMode(ma, ell, m)
    LS.SimInspiralWaveformParamsInsertModeArray(lal_params, ma)
    return lal_params
