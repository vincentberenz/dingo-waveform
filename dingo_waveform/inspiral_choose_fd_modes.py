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
from .types import FrequencySeries, Iota, Mode, Modes
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class InspiralChooseFDModesParameters(TableStr):
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
    approximant: Approximant

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
        approximant: Optional[Approximant],
    ) -> "InspiralChooseFDModesParameters":
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

    def apply(self) -> Tuple[Dict[Modes, FrequencySeries], Iota]:
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
            wfg_utils.linked_list_modes_to_dict_modes(hlm_fd___)
        )
        hlm_fd_: Dict[Modes, FrequencySeries] = {
            k: v.data.data for k, v in hlm_fd__.items()
        }

        # "Converting" to L0 frame
        hlm_fd: Dict[Modes, FrequencySeries] = self.convert_J_to_L0_frame(hlm_fd_)

        return hlm_fd, self.iota
