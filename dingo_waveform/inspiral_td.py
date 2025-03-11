import logging
from dataclasses import asdict, astuple, dataclass
from typing import Dict, Optional, Tuple

import lal
import lalsimulation as LS

from . import wfg_utils
from .approximant import Approximant, TD_Approximant
from .binary_black_holes import BinaryBlackHoleParameters
from .domains import DomainParameters
from .lal_params import lal
from .logging import TableStr
from .polarizations import Polarization
from .spins import Spins
from .types import FrequencySeries, Iota, Mode
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class InspiralTDParameters(TableStr):
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    r: float
    iota: float
    phase: float
    longAscNode: float
    eccentricity: float
    meanPerAno: float
    delta_t: float
    f_min: float
    f_ref: float
    lal_params: Optional[lal.Dict]
    approximant: Optional[Approximant]

    def get_spins(self) -> Spins:
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> "InspiralTDParameters":
        spins: Spins = bbh_parameters.get_spins(spin_conversion_phase)
        params = asdict(spins)
        for attr in ("longAscNode", "eccentricity", "meanPerAny"):
            params[attr] = 0.0
        for attr in ("delta_t", "f_min", "f_ref"):
            params[attr] = asdict(domain_params)[attr]
        params["phase"] = bbh_parameters.phase
        params["lal_params"] = lal_params
        params["approximant"] = approximant
        instance = cls(**params)
        _logger.debug(instance.to_table("generated inspiral td parameters"))
        return instance

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_params: WaveformParameters,
        f_ref: float,
        convert_to_SI: bool,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> "InspiralTDParameters":
        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_params, f_ref, convert_to_SI
        )
        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            domain_params,
            spin_conversion_phase,
            lal_params,
            approximant,
        )

    def apply(self) -> Polarization:

        parameters = list(astuple(self))
        hp, hc = LS.SimInspiralTD(*parameters)
        return Polarization(h_plus=hp.data.data, h_cross=hc.data.data)
