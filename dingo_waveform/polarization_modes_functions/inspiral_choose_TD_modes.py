import logging
from dataclasses import InitVar, asdict, astuple, dataclass
from typing import Dict, Optional, cast

import lal
import lalsimulation as LS

from ..approximant import Approximant, get_approximant
from ..binary_black_holes_parameters import BinaryBlackHoleParameters
from ..domains import DomainParameters, FrequencyDomain
from ..lal_params import lal
from ..logs import TableStr
from ..polarizations import Polarization, get_polarizations_from_fd_modes_m
from ..spins import Spins
from ..types import FrequencySeries, Iota, Mode, Modes
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import WaveformParameters
from . import polarization_modes_utils
from .polarization_modes_utils import td_modes_to_fd_modes

_logger = logging.getLogger(__name__)


@dataclass
class _InspiralChooseTDModesParameters(TableStr):
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
    iota: InitVar[Iota] = 0

    def __post_init__(self, iota: Iota) -> None:
        # iota is required (used by self.apply) but is not an argument
        # for LS.SimInspiralChooseTDModes, and therefore should be 'excluded'
        # from the 'astuple' method.
        # Defining it as 'InitVar' and setting it up in '__post_init__' allows
        # for this.
        self.iota = iota  # type: ignore

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        l_max_default: int = 5,
    ) -> "_InspiralChooseTDModesParameters":
        # Creates an instance from binary black hole parameters.

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
        params["approximant"] = get_approximant(approximant)
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
        convert_to_SI: Optional[bool],
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
    ) -> "_InspiralChooseTDModesParameters":
        # Creates an instance from waveform parameters.

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

    def apply(self, domain: FrequencyDomain, phase: float) -> Dict[Mode, Polarization]:
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
            polarization_modes_utils.linked_list_modes_to_dict_modes(hlm__)
        )

        # taper the time domain modes in place
        polarization_modes_utils.taper_td_modes_in_place(hlm_)

        hlm: Dict[Modes, FrequencySeries] = td_modes_to_fd_modes(hlm_, domain)

        pol: Dict[Mode, Polarization] = get_polarizations_from_fd_modes_m(
            hlm, self.iota, phase  # type: ignore
        )

        return pol


def inspiral_choose_TD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: WaveformParameters,
) -> Dict[Mode, Polarization]:
    """
    Wrapper over lalsimulation.SimInspiralChooseTDModes

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
      - if the specified domain is not an instance of FrequencyDomain
      - if the phase parameter is not specified
    """

    if not isinstance(waveform_gen_params.domain, FrequencyDomain):
        raise ValueError(
            "inspiral_choose_TD_modes can only be applied using on a FrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    if waveform_params.phase is None:
        raise ValueError(
            f"inspiral_choose_TD_modes: phase parameter should not be None"
        )

    instance = cast(
        _InspiralChooseTDModesParameters,
        _InspiralChooseTDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.f_ref,
            waveform_gen_params.convert_to_SI,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.spin_conversion_phase,
            waveform_gen_params.lal_params,
            waveform_gen_params.approximant,
        ),
    )

    return instance.apply(waveform_gen_params.domain, waveform_params.phase)
