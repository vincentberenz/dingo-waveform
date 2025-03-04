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
from .spins import Spins
from .types import FrequencySeries, Iota, Mode
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class InspiralChooseTDModesParameters(TableStr):
    phiRef: float
    delta_t: float
    mass_1: float
    mass_2: float
    iota: Iota
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    f_min: float
    f_ref: float
    distance: float
    lal_params: Optional[lal.Dict]
    l_max: float
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
    ) -> "InspiralChooseTDModesParameters":
        spins: Spins = bbh_parameters.get_spins(spin_conversion_phase)
        params = asdict(spins)
        params["phiRef"] = 0.0
        for attr in ("delta_t", "f_min"):
            params[attr] = getattr(domain_params, attr)
        params["f_ref"] = bbh_parameters.f_ref
        params["distance"] = bbh_parameters.luminosity_distance
        params["l_max"] = (
            bbh_parameters.l_max if bbh_parameters.l_max is not None else 0
        )
        params["mass_1"] = bbh_parameters.mass_1
        params["mass_2"] = bbh_parameters.mass_2
        params["lal_params"] = lal_params
        params["approximant"] = approximant
        instance = cls(**params)
        _logger.debug(
            instance.to_table("generated inspiral choose td modes parameters")
        )
        return instance

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_params: WaveformParameters,
        convert_to_SI: bool,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> "InspiralChooseTDModesParameters":
        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_params, convert_to_SI
        )
        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            domain_params,
            spin_conversion_phase,
            lal_params,
            approximant,
        )

    def apply(self) -> Tuple[Dict[Mode, FrequencySeries], Iota]:

        _logger.debug(
            "calling LS.SimInspiralChooseTDModes with arguments:"
            f"{', '.join(astuple(self))}"
        )

        hlm_ll: LS.SphHarmFrequencySeries = LS.SimInspiralChooseTDModes(
            list(astuple(self))
        )

        # Convert linked list of modes into dictionary with keys (l,m)
        # todo: is the type of data really lal.COMPLEX16FrequencySeries ?
        hlm_: Dict[Mode, lal.COMPLEX16FrequencySeries] = (
            wfg_utils.linked_list_modes_to_dict_modes(hlm_ll)
        )

        # taper the time domain modes in place
        hlm: Dict[Mode, FrequencySeries] = wfg_utils.taper_td_modes_in_place(hlm_)

        return hlm, self.iota
