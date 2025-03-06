from dataclasses import dataclass, fields
from typing import Dict, List

import lal
import numpy as np
from rich.console import Console
from rich.table import Table

from dingo_waveform.types import FrequencySeries, Iota, Mode


@dataclass
class Polarization:
    h_plus: FrequencySeries
    h_cross: FrequencySeries


def sum_contributions_m(
    x_m: Dict[int, Polarization], phase_shift: float = 0.0
) -> Polarization:
    """
    Sum the contributions over m-components, optionally introducing a phase shift.
    """
    result = Polarization(h_plus=0.0, h_cross=0.0)  # type: ignore
    for mode in x_m.keys():
        result.h_plus += x_m[mode].h_plus * np.exp(-1j * mode * phase_shift)
        result.h_cross += x_m[mode].h_cross * np.exp(-1j * mode * phase_shift)
    return result


def get_polarizations_from_fd_modes_m(
    hlm_fd: Dict[Mode, FrequencySeries], iota: Iota, phase: float
) -> Dict[int, Polarization]:
    pol_m: Dict[int, Dict[str, FrequencySeries]] = {}
    polarizations: List[str] = [f.name for f in fields(Polarization)]

    for (_, m), __ in hlm_fd.items():
        pol_m[m] = {"h_plus": 0.0, "h_cross": 0.0}  # type: ignore
        pol_m[-m] = {"h_plus": 0.0, "h_cross": 0.0}  # type: ignore

    for (l, m), h in hlm_fd.items():

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

    # Convert pol_m to a Dict[int, Polarization]
    return {
        m: Polarization(h_plus=pol["h_plus"], h_cross=pol["h_cross"])
        for m, pol in pol_m.items()
    }


def polarizations_to_table(pol: Dict[int, Polarization]) -> str:
    console = Console()
    table = Table(title="Polarizations")

    # Add columns
    table.add_column("Mode (m)", style="bold")
    table.add_column("h_plus", style="dim")
    table.add_column("h_cross", style="dim")

    # Add rows
    for m, polarization in pol.items():

        h_plus_repr = f"Array({polarization.h_plus.shape})"
        h_cross_repr = f"Array({polarization.h_cross.shape})"
        table.add_row(str(m), h_plus_repr, h_cross_repr)

    # Capture the output as a string
    with console.capture() as capture:
        console.print(table)

    return capture.get()
