from copy import deepcopy
from dataclasses import astuple
from typing import Dict, List, Literal, Optional, Tuple

import lal
import lalsimulation as LS

import dingo_waveform.wfg_utils as wfg_utils
from dingo_waveform.approximant import (
    Approximant,
    FD_Approximant,
    TD_Approximant,
    get_approximant,
    get_approximant_description,
)
from dingo_waveform.domains import Domain, DomainParameters, FrequencyDomain
from dingo_waveform.inspiral_choose_fd_modes import (
    InspiralChooseFDModesParameters,
)
from dingo_waveform.inspiral_choose_td_modes import (
    InspiralChooseTDModesParameters,
)
from dingo_waveform.lal_params import get_lal_params
from dingo_waveform.polarizations import (
    Polarization,
    get_polarizations_from_fd_modes_m,
)
from dingo_waveform.types import FrequencySeries, Iota, Mode
from dingo_waveform.waveform import WaveformParams


def inspiral_choose_TD_modes(
    params: BinaryBlackHoleParameters,
    domain_params: DomainParameters,
    spin_conversion_phase: Optional[float],
    lal_params: Optional[lal.Dict],
) -> Tuple[Dict[Mode, FrequencySeries], Iota]:

    # only approximant supported by this function
    approximant = TD_Approximant

    # "Converting' to the parameters required for the SimInspiralChooseTDModes
    # method
    inspiral_choose_td_modes_params: InspiralChooseTDModesParameters = (
        bbh_parameters.to_InspiralChooseTDModesParameters(
            domain_params, spin_conversion_phase, lal_params, approximant
        )
    )

    hlm_ll: LS.SphHarmFrequencySeries = LS.SimInspiralChooseTDModes(
        list(astuple(inspiral_choose_td_modes_params))
    )

    # Convert linked list of modes into dictionary with keys (l,m)
    # todo: is the type of data really lal.COMPLEX16FrequencySeries ?
    hlm_: Dict[Mode, lal.COMPLEX16FrequencySeries] = (
        wfg_utils.linked_list_modes_to_dict_modes(hlm_ll)
    )

    # taper the time domain modes in place
    hlm: Dict[Mode, FrequencySeries] = wfg_utils.taper_td_modes_in_place(hlm_)

    return hlm, inspiral_choose_td_modes_params.iota


def inspiral_choose_FD_modes(
    params: BinaryBlackHoleParameters,
    domain_params: DomainParameters,
    spin_conversion_phase: Optional[float],
    lal_params: Optional[lal.Dict],
) -> Tuple[Dict[Mode, FrequencySeries], Iota]:
    """
    Wrapper over the lal simulation method: SimInspiralChooseFDModes.
    This method:
    - generates the binary black holes parameters based on the waveform parameters
    - calls the SimInspiralChooseFDModes method
    - returns the corresponding complex frequency series.
    """

    # The only approximant supported
    approximant = FD_Approximant

    # "Converting' to the parameters required for the SimInspiralChooseFDModes
    # method
    inspiral_choose_fd_modes_params: InspiralChooseFDModesParameters = (
        bbh_parameters.to_InspiralChooseFDModesParameters(
            domain_params, spin_conversion_phase, lal_params, approximant
        )
    )

    # Calling the lal simulation method
    hlm_fd___: LS.SphHarmFrequencySeries = LS.SimInspiralChooseFDModes(
        list(astuple(inspiral_choose_fd_modes_params))
    )

    # "Converting" to frequency series
    hlm_fd__: Dict[Mode, lal.COMPLEX16FrequencySeries] = (
        wfg_utils.linked_list_modes_to_dict_modes(hlm_fd___)
    )
    hlm_fd_: Dict[Mode, FrequencySeries] = {
        k: v.data.data for k, v in hlm_fd__.items()
    }

    # "Converting" to L0 frame
    hlm_fd: Dict[Mode, FrequencySeries] = (
        inspiral_choose_fd_modes_params.convert_J_to_L0_frame(
            hlm_fd, spin_conversion_phase
        )
    )

    return hlm_fd, inspiral_choose_fd_modes_params.iota


def generate_hplus_hcross_m(
    domain: FrequencyDomain,
    params: BinaryBlackHoleParameters,
    approximant: Approximant,
    spin_conversion_phase: Optional[float],
    lal_params: Optional[lal.Dict],
    phase: Optional[float],
) -> Dict[int, Polarization]:

    hlm: Dict[Mode, FrequencySeries]
    iota: Iota

    # note: so far only the 101 (FD mode) and the 52 (TD modes)
    # approximants are supported. If supports to other approximant
    # get implemented, these functions may be useful:
    # - LS.SimInspiralImplementedFDApproximants
    # - LS.SimInspiralImplementedTDApproximants

    if approximant == TD_Approximant:

        hlm, iota = inspiral_choose_FD_modes(
            params,
            domain.get_parameters(),
            spin_conversion_phase,
            lal_params,
        )

    elif approximant == FD_Approximant:

        hlm, iota = inspiral_choose_TD_modes(
            params,
            domain.get_parameters(),
            spin_conversion_phase,
            lal_params,
        )

    else:
        # the approximant passed as argument is not supported,
        # raising error.
        try:
            desc = f" ({get_approximant_description(approximant)})"
        except Exception as e:
            desc = ""
        raise ValueError(
            "generate_hplus_hcross_m: only the approximants "
            f"{TD_Approximant} ({get_approximant_description(TD_Approximant)}) and "
            f"{FD_Approximant} ({get_approximant_description(FD_Approximant)})"
            f"are currently supported. {approximant}{desc} not supported)"
        )

    if phase is None:
        raise ValueError(
            "generate_hplus_hcross: the parameter 'phase' should be provided"
        )

    return get_polarizations_from_fd_modes_m(hlm, iota, phase)


class WaveformGenerator:
    """Generate polarizations using LALSimulation routines in the specified domain for a
    single GW coalescence given a set of waveform parameters.
    """

    def __init__(
        self,
        approximant: str,
        domain: Domain,
        f_ref: float,
        f_start: Optional[float] = None,
        mode_list: Optional[List[Mode]] = None,
        transform=None,
        spin_conversion_phase: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        approximant : str
            Waveform "approximant" string understood by lalsimulation
            This is defines which waveform model is used.
        domain : Domain
            Domain object that specifies on which physical domain the
            waveform polarizations will be generated, e.g. Fourier
            domain, time domain.
        f_ref : float
            Reference frequency for the waveforms
        f_start : float
            Starting frequency for waveform generation. This is optional, and if not
            included, the starting frequency will be set to f_min. This exists so that
            EOB waveforms can be generated starting from a lower frequency than f_min.
        mode_list : List[Tuple]
            A list of waveform (ell, m) modes to include when generating
            the polarizations.
        spin_conversion_phase : float = None
            Value for phiRef when computing cartesian spins from bilby spins via
            bilby_to_lalsimulation_spins. The common convention is to use the value of
            the phase parameter here, which is also used in the spherical harmonics
            when combining the different modes. If spin_conversion_phase = None,
            this default behavior is adapted.
            For dingo, this convention for the phase parameter makes it impossible to
            treat the phase as an extrinsic parameter, since we can only account for
            the change of phase in the spherical harmonics when changing the phase (in
            order to also change the cartesian spins -- specifically, to rotate the spins
            by phase in the sx-sy plane -- one would need to recompute the modes,
            which is expensive).
            By setting spin_conversion_phase != None, we impose the convention to always
            use phase = spin_conversion_phase when computing the cartesian spins.
        """
        self._approximant_str = approximant
        self._lal_params: Optional[lal.Dict] = None
        self._approximant: Optional[Approximant] = None

        if "SEOBNRv5" not in approximant:
            self._approximant = get_approximant(approximant)
            if mode_list is not None:
                self._lal_params = get_lal_params(mode_list)

        self._domain = domain
        self._f_ref = f_ref
        self._f_start = f_start
        self._transform = transform
        self._spin_conversion_phase = spin_conversion_phase

    def generate_hplus_hcross_m(
        self, parameters: WaveformParams
    ) -> Dict[int, Polarization]:

        if not isinstance(self._domain, FrequencyDomain):
            raise ValueError(
                "waveform generator generate_hplus_hcross_m: only "
                f"FrequencyDomain is supported (not {type(self._domain)})"
            )

        if not self._approximant in (FD_Approximant, TD_Approximant):
            try:
                desc = f" ({get_approximant_description(approximant)})"
            except Exception as e:
                desc = ""
            raise ValueError(
                "generate_hplus_hcross_m: only the approximants "
                f"{TD_Approximant} ({get_approximant_description(TD_Approximant)}) and "
                f"{FD_Approximant} ({get_approximant_description(FD_Approximant)})"
                f"are currently supported. {approximant}{desc} not supported)"
            )

        convert_to_SI = True

        # "Conversion" to binary black holes parameters
        params_ = deepcopy(parameters)
        params_.f_ref = self._f_ref
        bbh_parameters: BinaryBlackHoleParameters = (
            params_.to_binary_black_hole_parameters(convert_to_SI)
        )

        # Creating "empty" lal parameters if necessary
        if self._lal_params is None:
            lal_params = get_lal_params(mode_list)
        else:
            lal_params = self._lal_params

        # generating the polarizations. Calling either
        # - SimInspiralChooseTDModes (TD approximant) or
        # - SimInspiralChooseFDModes (FD approximant)
        pol: Dict[int, Polarization] = generate_hplus_hcross_m(
            domain,
            bbh_parameters,
            self._approximant,
            self._spin_conversion_phase,
            lal_params,
            phase,
        )

        return pol
