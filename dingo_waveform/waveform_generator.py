import logging
from copy import deepcopy
from typing import Dict, List, Optional

import lal

from .approximant import (  # Added missing imports
    Approximant,
    FD_Approximant,
    TD_Approximant,
    get_approximant,
    get_approximant_description,
)
from .domains import Domain, FrequencyDomain
from .inspiral_choose_fd_modes import InspiralChooseFDModesParameters
from .inspiral_choose_td_modes import InspiralChooseTDModesParameters
from .lal_params import get_lal_params
from .polarizations import (
    Polarization,
    get_polarizations_from_fd_modes_m,
    polarizations_to_table,
)
from .types import FrequencySeries, Iota, Mode
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


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
        _logger.info(
            f"creating waveform generator with domain {type(domain)} "
            f"and approximant {approximant}"
        )

        self._approximant_str = approximant
        self._lal_params: Optional[lal.Dict] = None
        self._approximant: Optional[Approximant] = None

        if "SEOBNRv5" not in approximant:
            self._approximant = get_approximant(approximant)
            # to check: confusing defining self._mode_list
            # is required. It is used in `generate_hplus_hcross_m`,
            # but in a way that feels redundant to what is happening here.
            self._mode_list = mode_list
            if mode_list is not None:
                logging.debug(
                    f"waveform generator: generating lal parameters based on mode list {mode_list}"
                )
                self._lal_params = get_lal_params(mode_list)
        else:
            # todo: what happens then ?
            ...

        self._domain = domain
        self._f_ref = f_ref
        self._f_start = f_start
        self._transform = transform
        self._spin_conversion_phase = spin_conversion_phase

    def generate_hplus_hcross_m(
        self, parameters: WaveformParameters
    ) -> Dict[int, Polarization]:

        _logger.info(
            parameters.to_table("generate hplus/hcross m with waveform parameters")
        )

        required_keys = ("phase",)
        for rq in required_keys:
            if getattr(parameters, rq) is None:
                raise ValueError(
                    f"generate_hplus_hcross_m: the parameters must specify a value for '{rq}'"
                )

        if not isinstance(self._domain, FrequencyDomain):
            raise ValueError(
                "waveform generator generate_hplus_hcross_m: only "
                f"FrequencyDomain is supported (not {type(self._domain)})"
            )

        if not self._approximant in (FD_Approximant, TD_Approximant):
            if self._approximant is None:
                raise ValueError(
                    "'None' approximant not supported for generate_hplus_hcross_m"
                )
            try:
                desc = f" ({get_approximant_description(self._approximant)})"
            except Exception as e:
                desc = ""
            raise ValueError(
                "generate_hplus_hcross_m: only the approximants "
                f"{TD_Approximant} ({get_approximant_description(TD_Approximant)}) and "
                f"{FD_Approximant} ({get_approximant_description(FD_Approximant)})"
                f"are currently supported. {self._approximant}{desc} not supported)"
            )

        waveform_params = deepcopy(parameters)
        _logger.debug(
            "waveform parameters: overwritting value of f_ref to {self._f_ref}"
        )
        waveform_params.f_ref = self._f_ref

        convert_to_SI = True
        hlm: Dict[Mode, FrequencySeries]
        iota: Iota

        if self._approximant == TD_Approximant:
            _logger.info(
                "generating hplus/hcross m using inspiral choose TD modes parameters"
            )
            inspiral_choose_td_modes_parameters = (
                InspiralChooseTDModesParameters.from_waveform_parameters(
                    waveform_params,
                    convert_to_SI,
                    self._domain.get_parameters(),
                    self._spin_conversion_phase,
                    self._lal_params,
                    self._approximant,
                )
            )
            hlm, iota = inspiral_choose_td_modes_parameters.apply()

        elif self._approximant == FD_Approximant:
            _logger.info(
                "generating hplus/hcross m using inspiral choose FD modes parameters"
            )
            inspiral_choose_fd_modes_parameters = (
                InspiralChooseFDModesParameters.from_waveform_parameters(
                    waveform_params,
                    convert_to_SI,
                    self._domain.get_parameters(),
                    self._spin_conversion_phase,
                    self._lal_params,
                    self._approximant,
                )
            )
            hlm, iota = inspiral_choose_fd_modes_parameters.apply(
                self._spin_conversion_phase
            )

        # type ignore: (we know parameters.phase is not None, checked earlier in this function)
        pol: Dict[int, Polarization] = get_polarizations_from_fd_modes_m(
            hlm, iota, parameters.phase  # type: ignore
        )

        _logger.debug(f"generated polarizations:\n{polarizations_to_table(pol)}")

        return pol
