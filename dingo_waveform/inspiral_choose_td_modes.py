"""
This module defines InspiralChooseTDModesParameters, a wrapper over
lalsimulation.SimInspiralChooseTDModes.
"""

import logging
from dataclasses import InitVar, asdict, astuple, dataclass
from typing import Dict, Optional, Tuple

import lal
import lalsimulation as LS

from . import wfg_utils
from .approximant import Approximant, TD_Approximant
from .binary_black_holes import BinaryBlackHoleParameters
from .domains import DomainParameters, FrequencyDomain
from .lal_params import lal
from .logging import TableStr
from .spins import Spins
from .types import FrequencySeries, Iota, Mode, Modes
from .waveform_parameters import WaveformParameters
from .wfg_utils import td_modes_to_fd_modes

_logger = logging.getLogger(__name__)


@dataclass
class InspiralChooseTDModesParameters(TableStr):
    """Dataclass for storing parameters for
    lal simulation's SimInspiralChooseTDModes function.
    """

    # Warning: order matters ! The arguments will be generated
    # in the order below:
    phiRef: float
    delta_t: float
    mass_1: float
    mass_2: float
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
    l_max: int
    approximant: Optional[Approximant]
    iota: InitVar[float] = 0

    def __post_init__(self, iota):
        # iota is required (used by self.apply) but is not an argument
        # for LS.SimInspiralChooseTDModes, and therefore should be 'excluded'
        # from the 'astuple' method.
        # Defining it as 'InitVar' and setting it up in '__post_init__' allow
        # for this.
        self.iota = iota

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
        l_max_default: int = 5,
    ) -> "InspiralChooseTDModesParameters":
        """Creates an instance from binary black hole parameters.

        Parameters
        ----------
        bbh_parameters : BinaryBlackHoleParameters
            The binary black hole parameters.
        domain_params : DomainParameters
            The domain parameters.
        spin_conversion_phase : Optional[float]
            The phase for spin conversion.
        lal_params : Optional[lal.Dict]
            The LAL parameters.
        approximant : Optional[Approximant]
            The approximant.
        l_max_default : int
            Default maximum l value for modes.

        Returns
        -------
        InspiralChooseTDModesParameters
            The created instance.
        """
        spins: Spins = bbh_parameters.get_spins(spin_conversion_phase)
        params = asdict(spins)
        params["phiRef"] = 0.0
        for attr in ("delta_t", "f_min"):
            params[attr] = getattr(domain_params, attr)
        params["f_ref"] = bbh_parameters.f_ref
        params["distance"] = bbh_parameters.luminosity_distance
        params["l_max"] = (
            bbh_parameters.l_max if bbh_parameters.l_max is not None else l_max_default
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
        f_ref: float,
        convert_to_SI: bool,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> "InspiralChooseTDModesParameters":
        """Creates an instance from waveform parameters.

        Parameters
        ----------
        waveform_params : WaveformParameters
            The waveform parameters.
        f_ref : float
            The reference frequency.
        convert_to_SI : bool
            Whether to convert to SI units.
        domain_params : DomainParameters
            The domain parameters.
        spin_conversion_phase : Optional[float]
            The phase for spin conversion.
        lal_params : Optional[lal.Dict]
            The LAL parameters.
        approximant : Optional[Approximant]
            The approximant.

        Returns
        -------
        InspiralChooseTDModesParameters
            The created instance.
        """
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

    def apply(
        self, domain: FrequencyDomain
    ) -> Tuple[Dict[Modes, FrequencySeries], Iota]:
        """Applies the LAL simulation method and converts the result to the frequency domain.

        Parameters
        ----------
        domain : FrequencyDomain
            The frequency domain to apply the transformation.

        Returns
        -------
        Tuple[Dict[Modes, FrequencySeries], Iota]
            The frequency series in the frequency domain and the iota.
        """
        _logger.debug(
            "calling LS.SimInspiralChooseTDModes with arguments:"
            f"{', '.join([str(v) for v in astuple(self)])}"
        )

        # Note: because self.iota is an "InitVar", it is excluded
        #   from the 'astuple' function.
        hlm__: LS.SphHarmFrequencySeries = LS.SimInspiralChooseTDModes(
            *list(astuple(self))
        )

        # Convert linked list of modes into dictionary with keys (l,m)
        # todo: is the type of data really lal.COMPLEX16FrequencySeries ?
        hlm_: Dict[Modes, lal.COMPLEX16FrequencySeries] = (
            wfg_utils.linked_list_modes_to_dict_modes(hlm__)
        )

        # taper the time domain modes in place
        wfg_utils.taper_td_modes_in_place(hlm_)

        hlm: Dict[Modes, FrequencySeries] = td_modes_to_fd_modes(hlm_, domain)

        # type ignore: mypy fails to see that self has a iota attribute,
        # likely because of its 'InitVar' definition.
        return hlm, self.iota  # type: ignore
