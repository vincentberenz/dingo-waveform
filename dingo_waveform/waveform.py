from dataclasses import asdict, dataclass
from typing import Optional

import lal
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

from dingo_waveform.binary_black_holes import BinaryBlackHoleParameters


@dataclass
class WaveformParams:
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
    cos_tilt_1: Optional[float] = None
    cos_tilt_2: Optional[float] = None

    # are those the same ?
    delta_phase: Optional[float] = None
    phase: Optional[float] = None

    psi: Optional[float] = None
    theta_jn: Optional[float] = None
    f_ref: Optional[float] = None

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

    def to_binary_black_hole_parameters(
        self, convert_to_SI: bool
    ) -> BinaryBlackHoleParameters:
        params = asdict(self)
        converted_params, _ = convert_to_lal_binary_black_hole_parameters(params)
        if convert_to_SI:
            converted_params.mass_1 *= lal.MSUN_SI
            converted_params.mass_2 *= lal.MSUN_SI
            converted_params.luminosity_distance *= 1e6 * lal.PC_SI
        return BinaryBlackHoleParameters(**converted_params)
