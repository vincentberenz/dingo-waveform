import logging
from copy import deepcopy
from dataclasses import asdict, astuple, dataclass
from typing import Dict, Optional, cast
from venv import logger

import lal
import lalsimulation as LS

from ..approximant import Approximant, get_approximant
from ..binary_black_holes_parameters import BinaryBlackHoleParameters
from ..domains import DomainParameters
from ..logs import TableStr
from ..polarizations import Polarization, get_polarizations_from_fd_modes_m
from ..spins import Spins
from ..types import FrequencySeries, Iota, Mode, Modes
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import WaveformParameters
from . import polarization_modes_utils

_logger = logging.getLogger(__name__)


@dataclass
class _InspiralChooseFDModesParameters(TableStr):

    # Order matters !
    # The list of arguments will be generated via the 'astuple' dataclass
    # function which will 'cast' an instance of InspiralChooseFDModeParameters
    # to a tuple, which value order will be based on the order below.
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
    approximant: int

    def get_spins(self) -> Spins:
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    def convert_J_to_L0_frame(
        self, hlm_J: Dict[Modes, FrequencySeries]
    ) -> Dict[Modes, FrequencySeries]:

        return self.get_spins().convert_J_to_L0_frame(
            hlm_J, self.mass_1, self.mass_2, self.f_ref, self.phase
        )

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        f_start: Optional[float],
    ) -> "_InspiralChooseFDModesParameters":
        # Creates an instance from binary black hole parameters.

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
        if f_start is not None:
            parameters["f_min"] = f_start
        # other params
        parameters["f_ref"] = bbh_parameters.f_ref
        parameters["r"] = bbh_parameters.luminosity_distance
        parameters["lal_params"] = lal_params
        parameters["approximant"] = get_approximant(approximant)
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
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        f_start: Optional[float],
    ) -> "_InspiralChooseFDModesParameters":
        # Creates an instance from waveform parameters.

        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_parameters, f_ref
        )
        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            domain_params,
            spin_conversion_phase,
            lal_params,
            approximant,
            f_start,
        )

    def apply(self) -> Dict[Mode, Polarization]:

        # for SimInspiralChooseFDModes, SI units are required
        params: "_InspiralChooseFDModesParameters" = deepcopy(self)
        params.mass_1 *= lal.MSUN_SI
        params.mass_2 *= lal.MSUN_SI
        params.r *= 1e6 * lal.PC_SI

        # informing the user which arguments we pass to the function
        _logger.debug(
            params.to_table(f"calling LS.SimInspiralChooseFDModes with arguments:")
        )

        # Calling the lal simulation method
        arguments = list(astuple(params))
        hlm_fd___: LS.SphHarmFrequencySeries = LS.SimInspiralChooseFDModes(*arguments)

        # "Converting" to frequency series
        hlm_fd__: Dict[Modes, lal.COMPLEX16FrequencySeries] = (
            polarization_modes_utils.linked_list_modes_to_dict_modes(hlm_fd___)
        )
        hlm_fd_: Dict[Modes, FrequencySeries] = {
            k: v.data.data for k, v in hlm_fd__.items()
        }

        # "Converting" to L0 frame
        hlm_fd: Dict[Modes, FrequencySeries] = self.convert_J_to_L0_frame(hlm_fd_)

        return get_polarizations_from_fd_modes_m(hlm_fd, self.iota, self.phase)


def inspiral_choose_FD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: WaveformParameters,
) -> Dict[Mode, Polarization]:
    """
    Wrapper over lalsimulation.SimInspiralChooseFDModes

    Arguments
    ---------
    waveform_gen_params
      waveform generation configuration
    waveform_params
      waveform configuration

    Returns
    -------
    Dictionary mode / polarizations

    Raises
    ------
    ValueError
      if the phase parameter is not specified
    """

    if waveform_params.phase is None:
        raise ValueError(
            f"inspiral_choose_FD_modes: phase parameter should not be None"
        )

    instance = cast(
        _InspiralChooseFDModesParameters,
        _InspiralChooseFDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.f_ref,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.spin_conversion_phase,
            waveform_gen_params.lal_params,
            waveform_gen_params.approximant,
            waveform_gen_params.f_start,
        ),
    )

    return instance.apply()
