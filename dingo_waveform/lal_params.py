from typing import List

import lal
import lalsimulation as LS

from dingo_waveform.types import Modes


def get_lal_params(mode_list: List[Modes]) -> lal.Dict:
    """
    Create and return a LAL dictionary with the specified modes activated.

    Parameters
    ----------
    mode_list
        List of modes to be activated in the LAL dictionary.

    Returns
    -------
    LAL dictionary with the specified modes activated.
    """
    lal_params = lal.CreateDict()
    ma = LS.SimInspiralCreateModeArray()
    for ell, m in mode_list:
        LS.SimInspiralModeArrayActivateMode(ma, ell, m)
    LS.SimInspiralWaveformParamsInsertModeArray(lal_params, ma)
    return lal_params
