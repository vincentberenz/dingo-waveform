from dataclasses import dataclass
from typing import Any, Optional

import lal

from .logging import TableStr


@dataclass
class WaveformParameters(TableStr):
    """
    Configuration dataclass for generating waveforms.

    This class contains all parameters necessary for waveform generation,
    organized by physical categories. All parameters are optional and default
    to None.

    Parameters
    ----------
    Distance and Redshift Parameters
    -------------------------------
    luminosity_distance :
        Luminosity distance to the source in Mpc
    redshift :
        Redshift of the source
    comoving_distance :
        Comoving distance to the source

    Spin Parameters
    --------------
    chi_1 :
        Dimensionless spin magnitude of object 1
    chi_2 :
        Dimensionless spin magnitude of object 2
    chi_1_in_plane :
        In-plane component of spin 1
    chi_2_in_plane :
        In-plane component of spin 2
    a_1 :
        Alternative representation of spin 1 magnitude
    a_2 :
        Alternative representation of spin 2 magnitude

    Angular Parameters
    -----------------
    phi_jl :
        Angle between total angular momentum and line of sight
    phi_12 :
        Angle between spins
    tilt_1 :
        Tilt angle of spin 1
    tilt_2 :
        Tilt angle of spin 2
    dec :
        Declination angle (in radians)
    ra :
        Right ascension angle (in radians)
    geocent_time :
        Geocentric time (in seconds)
    delta_phase :
        Phase difference between h+ and h× polarizations
    phase :
        Orbital phase at reference frequency
    psi :
        Polarization angle
    theta_jn :
        Angle between total angular momentum and line of sight

    Mass Parameters
    --------------
    mass_1 :
        Mass of object 1 (in solar masses)
    mass_2 :
        Mass of object 2 (in solar masses)
    total_mass :
        Total mass of the system
    chirp_mass :
        Chirp mass of the system
    mass_ratio :
        Mass ratio q = m1/m2
    symmetric_mass_ratio :
        Symmetric mass ratio η = m1*m2/(m1+m2)²

    Source Frame Mass Parameters
    ---------------------------
    mass_1_source :
        Source frame mass of object 1
    mass_2_source :
        Source frame mass of object 2
    total_mass_source :
        Source frame total mass
    chirp_mass_source :
        Source frame chirp mass
    l_max :
        Maximum harmonic order to include

    SEOBNRv5 Specific Parameters
    ---------------------------
    postadiabatic :
        Post-adiabatic correction parameters for SEOBNRv5
    postadiabatic_type :
        Type specification for post-adiabatic corrections
    lmax_nyquist :
        Maximum harmonic index for Nyquist sampling
    """

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
    lmax_nyquist: Optional[int] = None
