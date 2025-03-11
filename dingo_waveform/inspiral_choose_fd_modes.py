import logging
from dataclasses import asdict, astuple, dataclass
from typing import Dict, Optional, Tuple

import lal
import lalsimulation as LS

from . import wfg_utils
from .approximant import Approximant
from .binary_black_holes import BinaryBlackHoleParameters
from .domains import DomainParameters
from .logging import TableStr
from .spins import Spins
from .types import FrequencySeries, Iota, Mode
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class InspiralChooseFDModesParameters(TableStr):
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    phase: float
    r: float
    iota: Iota
    lal_params: Optional[lal.Dict]
    approximant: Approximant

    def get_spins(self) -> Spins:
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    def convert_J_to_L0_frame(
        self, hlm_J: Dict[Mode, FrequencySeries]
    ) -> Dict[Mode, FrequencySeries]:
        converted_to_SI = True
        return self.get_spins().convert_J_to_L0_frame(
            hlm_J, self.mass_1, self.mass_2, converted_to_SI, self.f_ref, self.phase
        )

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> "InspiralChooseFDModesParameters":
        spins: Spins = bbh_parameters.get_spins(spin_conversion_phase)
        # adding iota, s1x, ..., s2x, ...
        parameters = asdict(spins)
        # direct mapping from this instance
        for k in ("mass_1", "mass_2", "phase"):
            parameters[k] = getattr(bbh_parameters, k)
        # adding domain related params
        domain_dict = asdict(domain_params)
        for k in ("delta_f", "f_min", "f_max"):
            parameters[k] = domain_dict[k]
        # other params
        parameters["f_ref"] = bbh_parameters.f_ref
        parameters["r"] = bbh_parameters.luminosity_distance
        parameters["lal_params"] = lal_params
        parameters["approximant"] = approximant
        instance = cls(**parameters)
        _logger.debug(
            instance.to_table("generated inspiral choose fd modes parameters")
        )
        return instance

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_parameters: WaveformParameters,
        f_ref: float,
        convert_to_SI: bool,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> "InspiralChooseFDModesParameters":
        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_parameters, f_ref, convert_to_SI
        )
        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            domain_params,
            spin_conversion_phase,
            lal_params,
            approximant,
        )

    def apply(
        self, spin_conversion_phase: Optional[float]
    ) -> Tuple[Dict[Mode, FrequencySeries], Iota]:

        # It is confusing that "spin_conversion_phase"
        # is needed as argument here.
        # It has already been used to setup the value of
        # "phase" in `from_waveform_parameters` or
        # `from_binary_black_hole_parameters`.

        # Calling the lal simulation method
        arguments = list(astuple(self))
        hlm_fd___: LS.SphHarmFrequencySeries = LS.SimInspiralChooseFDModes(*arguments)

        # "Converting" to frequency series
        hlm_fd__: Dict[Mode, lal.COMPLEX16FrequencySeries] = (
            wfg_utils.linked_list_modes_to_dict_modes(hlm_fd___)
        )
        hlm_fd_: Dict[Mode, FrequencySeries] = {
            k: v.data.data for k, v in hlm_fd__.items()
        }

        # "Converting" to L0 frame
        hlm_fd: Dict[Mode, FrequencySeries] = self.convert_J_to_L0_frame(hlm_fd_)

        return hlm_fd, self.iota
