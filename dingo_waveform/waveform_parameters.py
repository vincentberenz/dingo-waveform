from dataclasses import dataclass
from typing import Any, Optional

import lal

from .logging import TableStr


@dataclass
class WaveformParameters(TableStr):
    """
    Configuration dataclass for generating waveforms.
    """

    # TODO: Is this the exhaustive liste of possible fields ?
    luminosity_distance: Optional[float] = None
    redshift: Optional[float] = None
    comoving_distance: Optional[float] = None
    chi_1: Optional[float] = None
    chi_2: Optional[float] = None
    chi_1_in_plane: Optional[float] = None
    chi_2_in_plane: Optional[float] = None
    a_1: Optional[float] = None
    a_2: Optional[float] = None
    phi_jl: Optional[float] = None
    phi_12: Optional[float] = None
    tilt_1: Optional[float] = None
    tilt_2: Optional[float] = None
    dec: Optional[float] = None
    ra: Optional[float] = None

    geocent_time: Optional[float] = None

    # are those the same ?
    delta_phase: Optional[float] = None
    phase: Optional[float] = None

    psi: Optional[float] = None
    theta_jn: Optional[float] = None

    # Mass parameters
    mass_1: Optional[float] = None
    mass_2: Optional[float] = None
    total_mass: Optional[float] = None
    chirp_mass: Optional[float] = None
    mass_ratio: Optional[float] = None
    symmetric_mass_ratio: Optional[float] = None

    # Source frame mass parameters
    mass_1_source: Optional[float] = None
    mass_2_source: Optional[float] = None
    total_mass_source: Optional[float] = None
    chirp_mass_source: Optional[float] = None

    l_max: Optional[float] = None

    # new interface and SEOBNRv5 specific parameters
    postadiabatic: Optional[Any] = None
    postadiabatic_type: Optional[Any] = None
    lmax_nyquist: int = 2
