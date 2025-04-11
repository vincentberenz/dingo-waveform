from typing import List

import lal
import lalsimulation as LS

from dingo_waveform.types import Modes


def get_lal_params(mode_list: List[Modes]) -> lal.Dict:
    """
    Create a LAL dictionary containing mode array parameters
    for gravitational wave signal generation.

    Parameters
    ----------
    mode_list :
        List of (ell, m) tuples specifying the spherical harmonic modes to include
        in the waveform calculation. Each tuple contains:
        - ell: The degree of the spherical harmonic mode
        - m: The order of the spherical harmonic mode

    Returns
    -------
    A LAL dictionary object containing the configured mode array parameters,
    suitable for use with waveform generation functions.
    """

    lal_params = lal.CreateDict()
    ma = LS.SimInspiralCreateModeArray()
    for ell, m in mode_list:
        LS.SimInspiralModeArrayActivateMode(ma, ell, m)
    LS.SimInspiralWaveformParamsInsertModeArray(lal_params, ma)
    return lal_params
