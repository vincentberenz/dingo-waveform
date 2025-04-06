import logging
from dataclasses import asdict, astuple, dataclass
from typing import Dict, Optional, cast

import lal
import lalsimulation as LS

from ..approximant import Approximant, get_approximant
from ..binary_black_holes import BinaryBlackHoleParameters
from ..domains import DomainParameters
from ..logging import TableStr
from ..polarizations import Polarization, get_polarizations_from_fd_modes_m
from ..spins import Spins
from ..types import FrequencySeries, Iota, Mode, Modes
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import WaveformParameters
from . import polarization_modes_utils

_logger = logging.getLogger(__name__)


@dataclass
class _InspiralChooseFDModesParameters(TableStr):
    """
    Dataclass which attributes will be the arguments passed to
    LS.SimInspiralChooseFDModes.
    """

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
        """
        Returns the spins.

        Returns:
            Spins: The spins of the system.
        """
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    def convert_J_to_L0_frame(
        self, hlm_J: Dict[Modes, FrequencySeries]
    ) -> Dict[Modes, FrequencySeries]:
        """
        Converts the given frequency series to the L0 frame.

        Parameters:
            hlm_J: The frequency series in the J frame.

        Returns:
            Dict[Modes, FrequencySeries]: The frequency series in the L0 frame.
        """
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
        approximant: Approximant,
    ) -> "_InspiralChooseFDModesParameters":
        """
        Creates an instance from binary black hole parameters.

        Parameters:
            bbh_parameters: The binary black hole parameters.
            domain_params: The domain parameters.
            spin_conversion_phase: The phase for spin conversion.
            lal_params: The LAL parameters.
            approximant: The approximant.

        Returns:
            InspiralChooseFDModesParameters: The created instance.
        """
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
        convert_to_SI: bool,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
    ) -> "_InspiralChooseFDModesParameters":
        """
        Creates an instance from waveform parameters.

        Parameters:
            waveform_parameters: The waveform parameters.
            f_ref: The reference frequency.
            convert_to_SI: Whether to convert to SI units.
            domain_params: The domain parameters.
            spin_conversion_phase: The phase for spin conversion.
            lal_params: The LAL parameters.
            approximant: The approximant.

        Returns:
            InspiralChooseFDModesParameters: The created instance.
        """
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

    def apply(self, phase: float) -> Dict[Mode, Polarization]:
        """
        Applies the LAL simulation method and converts the result to the L0 frame.

        Returns:
            Tuple[Dict[Modes, FrequencySeries], Iota]: The frequency series in the L0 frame and the iota.
        """
        # Calling the lal simulation method
        arguments = list(astuple(self))
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

        return get_polarizations_from_fd_modes_m(hlm_fd, self.iota, phase)


def inspiral_choose_FD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: WaveformParameters,
) -> Dict[Mode, Polarization]:

    if waveform_params.phase is None:
        raise ValueError(f"generate_TD_modes_LO: phase parameter should not be None")

    instance = cast(
        _InspiralChooseFDModesParameters,
        _InspiralChooseFDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.f_ref,
            waveform_gen_params.convert_to_SI,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.spin_conversion_phase,
            waveform_gen_params.lal_params,
            waveform_gen_params.approximant,
        ),
    )

    return instance.apply(waveform_params.phase)
