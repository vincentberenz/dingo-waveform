from dataclasses import dataclass
from typing import Dict

import lal
import numpy as np

from dingo_waveform.types import FrequencySeries, Iota, Mode


@dataclass
class Polarization:
    h_plus: np.ndarray
    h_cross: np.ndarray


def get_polarizations_from_fd_modes_m(
    hlm_fd: Dict[Mode, FrequencySeries], iota: Iota, phase: float
):
    pol_m = {}
    polarizations = ["h_plus", "h_cross"]

    for (l, m), h in hlm_fd.items():
        if m not in pol_m:
            pol_m[m] = {k: 0.0 for k in polarizations}
            pol_m[-m] = {k: 0.0 for k in polarizations}

        # In the L0 frame, we compute the polarizations from the modes using the
        # spherical harmonics below.
        ylm = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m)
        ylmstar = ylm.conjugate()

        # Modes (l,m) are defined on domain -f_max,...,-f_min,...0,...,f_min,...,f_max.
        # This splits up the frequency series into positive and negative frequency parts.
        if len(h) % 2 != 1:
            raise ValueError(
                "Even number of bins encountered, should be odd: -f_max,...,0,...,f_max."
            )
        offset = len(h) // 2
        h1 = h[offset:]
        h2 = h[offset::-1].conj()

        # Organize the modes such that pol_m[m] transforms as e^{- 1j * m * phase}.
        # This differs from the usual way, e.g.,
        #   https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
        #   _l_a_l_sim_inspiral_8c_source.html#l04801
        pol_m[m]["h_plus"] += 0.5 * h1 * ylm
        pol_m[-m]["h_plus"] += 0.5 * h2 * ylmstar
        pol_m[m]["h_cross"] += 0.5 * 1j * h1 * ylm
        pol_m[-m]["h_cross"] += -0.5 * 1j * h2 * ylmstar

    return pol_m
