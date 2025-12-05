from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Protocol, Union, runtime_checkable

import lal
import numpy as np
from rich.console import Console
from rich.table import Table

from dingo_waveform.types import BatchFrequencySeries, FrequencySeries, Iota, Mode, Modes


@runtime_checkable
class PolarizationProtocol(Protocol):
    """
    Protocol defining the interface for polarization data structures.

    Both Polarization (single waveform) and BatchPolarizations (multiple waveforms)
    implement this protocol, allowing functions to accept either type.
    """
    h_plus: Union[FrequencySeries, BatchFrequencySeries]
    h_cross: Union[FrequencySeries, BatchFrequencySeries]


@dataclass
class Polarization:
    """
    A dataclass representing the polarizations of a single gravitational wave.

    This class is used for single waveform generation and manipulation.

    Parameters
    ----------
    h_plus :
        The plus polarization component of the gravitational wave.
    h_cross :
        The cross polarization component of the gravitational wave.
    """

    h_plus: FrequencySeries
    h_cross: FrequencySeries


@dataclass
class BatchPolarizations:
    """
    A dataclass representing batched polarizations of gravitational waves.

    This is used for storing multiple waveforms as numpy arrays, primarily
    in dataset generation and batch processing contexts.

    Attributes
    ----------
    h_plus
        Array of plus polarization components with shape (num_waveforms, frequency_bins).
    h_cross
        Array of cross polarization components with shape (num_waveforms, frequency_bins).
    """

    h_plus: BatchFrequencySeries
    h_cross: BatchFrequencySeries

    def __post_init__(self):
        """Validate that h_plus and h_cross have the same shape."""
        if self.h_plus.shape != self.h_cross.shape:
            raise ValueError(
                f"h_plus and h_cross must have the same shape. "
                f"Got h_plus: {self.h_plus.shape}, h_cross: {self.h_cross.shape}"
            )

    def __len__(self) -> int:
        """Return the number of waveforms (first dimension of arrays)."""
        return self.h_plus.shape[0]

    @property
    def num_waveforms(self) -> int:
        """Return the number of waveforms."""
        return len(self)

    @property
    def num_frequency_bins(self) -> int:
        """Return the number of frequency bins."""
        return self.h_plus.shape[1] if self.h_plus.ndim > 1 else self.h_plus.shape[0]

    @classmethod
    def from_polarizations(cls, polarizations: List[Polarization]) -> BatchPolarizations:
        """
        Create a BatchPolarizations from a list of single Polarization objects.

        Parameters
        ----------
        polarizations
            List of Polarization objects to batch together.

        Returns
        -------
        A BatchPolarizations object containing all input polarizations.
        """
        if not polarizations:
            raise ValueError("Cannot create BatchPolarizations from empty list")

        h_plus = np.array([p.h_plus for p in polarizations])
        h_cross = np.array([p.h_cross for p in polarizations])
        return cls(h_plus=h_plus, h_cross=h_cross)


def sum_contributions_m(
    x_m: Dict[Mode, Polarization], phase_shift: float = 0.0
) -> Polarization:
    """
    Sum the contributions over m-components, optionally introducing a phase shift.

    Parameters
    ----------
    x_m
        Dictionary mapping m values (int) to their corresponding Polarization objects.
    phase_shift
        Optional phase shift to apply to each mode.

    Returns
    -------
    The resulting Polarization after summing contributions with the phase shift applied.
    """
    result = Polarization(h_plus=0.0, h_cross=0.0)  # type: ignore
    for mode in x_m.keys():
        # mode is an int (m value), use directly for phase shift
        result.h_plus += x_m[mode].h_plus * np.exp(-1j * mode * phase_shift)
        result.h_cross += x_m[mode].h_cross * np.exp(-1j * mode * phase_shift)
    return result


def get_polarizations_from_fd_modes_m(
    hlm_fd: Dict[Modes, FrequencySeries], iota: Iota, phase: float
) -> Dict[Mode, Polarization]:
    """
    Compute polarizations from frequency domain modes.

    Parameters
    ----------
    hlm_fd
        Dictionary of frequency domain modes.
    iota
        Inclination angle.
    phase
        Phase angle.

    Returns
    -------
    Dictionary mapping m values to their corresponding Polarization objects.
    Note: The function groups contributions by m value (summing over all l),
    and returns a dict with int keys (just m, not (l,m) tuples).
    """
    pol_m: Dict[Mode, Dict[str, FrequencySeries]] = {}
    polarizations: List[str] = [f.name for f in fields(Polarization)]

    for (_, m), __ in hlm_fd.items():
        if m not in pol_m:
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

    # Convert pol_m to Dict[Mode, Polarization]
    return {
        m: Polarization(h_plus=pol["h_plus"], h_cross=pol["h_cross"])
        for m, pol in pol_m.items()
    }


def polarizations_to_table(pol: Dict[Mode, Polarization]) -> str:
    """
    Convert polarizations to a formatted table string.

    Parameters
    ----------
    pol
        Dictionary mapping modes to their corresponding Polarization objects.

    Returns
    -------
    A string representing the polarizations as a formatted table.
    """
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
