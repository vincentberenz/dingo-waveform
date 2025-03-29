import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import astropy
import astropy.units
import lal

from .binary_black_holes import BinaryBlackHoleParameters
from .domains import DomainParameters
from .logging import TableStr
from .spins import Spins
from .types import Mode
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class GwSignalParameters(TableStr):
    mass1: float
    mass2: float
    spin1x: float
    spin1y: float
    spin1z: float
    spin2x: float
    spin2y: float
    spin2z: float
    deltaT: float
    f22_start: float
    f22_ref: float
    f_max: float
    deltaF: float
    phi_ref: float
    distance: float
    inclination: float
    condition: int 
    lmax_nyquist: Optional[int] = None
    postadiabatic: Optional[Any] = None
    postadiabatic_type: Optional[Any] = None

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float] = None,
        f_start: Optional[float] = None, 
        condition: int = 1,
        lmax_nyquist: Optional[int] = None,
        postadiabatic: Optional[Any] = None,
        postadiabatic_type: Optional[Any] = None
    )->"GwSignalParameters":
        
        # used to declare the value of f22_start below
        f_min = f_start if f_start is not None else domain_params.f_min

        spins: Spins = bbh_params.get_spins(spin_conversion_phase)

        return cls(
            mass1 = bbh_params.mass_1 * astropy.units.solMass,
            mass2 = bbh_params.mass_2 * astropy.units.solMass,
            spin1x = spins.s1x * astropy.units.dimensionless_unscaled,
            spin1y = spins.s1y * astropy.units.dimensionless_unscaled,
            spin1z = spins.s1z * astropy.units.dimensionless_unscaled,
            spin2x = spins.s2x * astropy.units.dimensionless_unscaled,
            spin2y = spins.s2y * astropy.units.dimensionless_unscaled,
            spin2z = spins.s2z * astropy.units.dimensionless_unscaled,
            deltaT = domain_params.delta_t * astropy.units.s,
            f22_start = f_min * astropy.units.Hz,
            f22_ref = bbh_params.f_ref * astropy.units.Hz,
            f_max = domain_params.f_max * astropy.units.Hz,
            deltaF = domain_params.delta_f * astropy.units.Hz,
            phi_ref = bbh_params.phase * astropy.units.rad,
            distance = bbh_params.luminosity_distance*astropy.units.Mpc,
            inclination = spins.iota * astropy.units.rad,
            condition = condition,
            postadiabatic = postadiabatic,  # SEOBNRv5 specific parameters
            postadiabatic_type = postadiabatic_type,  # SEOBNRv5 specific parameters
            lmax_nyquist = lmax_nyquist # SEOBNRv5 specific parameters
        )

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_params: WaveformParameters,
        domain_params: DomainParameters,
        f_ref: float,
        spin_conversion_phase: Optional[float] = None,
        f_start: Optional[float] = None, 
        convert_to_SI: bool = True
    )->"GwSignalParameters":

        bbh = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_params, f_ref, convert_to_SI
        )
        return cls.from_binary_black_hole_parameters(
            bbh, domain_params, spin_conversion_phase, 
            f_start, 
            lmax_nyquist=waveform_params.lmax_nyquist, 
            postadiabatic=waveform_params.postadiabatic, 
            postadiabatic_type=waveform_params.postadiabatic_type
        )
