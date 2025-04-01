import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union, cast

import lal

from .approximant import (  # Added missing imports
    Approximant,
    FD_Approximant,
    TD_Approximant,
    get_approximant,
    get_approximant_description,
)
from .domains import Domain, DomainParameters, FrequencyDomain, TimeDomain
from .imports import check_function_signature, import_entity
from .inspiral_choose_fd_modes import InspiralChooseFDModesParameters
from .inspiral_choose_td_modes import InspiralChooseTDModesParameters
from .inspiral_fd import InspiralFDParameters
from .inspiral_td import InspiralTDParameters
from .lal_params import get_lal_params
from .polarizations import (
    Polarization,
    get_polarizations_from_fd_modes_m,
    polarizations_to_table,
)
from .types import FrequencySeries, Iota, Mode, Modes
from .waveform_generator_parameters import WaveformGeneratorParameters
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


class WaveformGenerator:
    """Generate polarizations using LALSimulation routines in the specified domain for a
    single GW coalescence given a set of waveform parameters.
    """

    def __init__(
        self,
        approximant: Approximant,
        domain: Domain,
        f_ref: float,
        f_start: Optional[float] = None,
        spin_conversion_phase: Optional[float] = None,
        convert_to_SI: bool = True,
        mode_list: Optional[List[Modes]] = None,
        transform: Optional[Callable[[Polarization], Polarization]] = None,
    ):
        """
        Parameters
        ----------
        approximant
            Waveform "approximant" string understood by lalsimulation
            This is defines which waveform model is used.
        domain
            Domain object that specifies on which physical domain the
            waveform polarizations will be generated, e.g. Fourier
            domain, time domain.
        f_ref
            Reference frequency for the waveforms
        f_start
            Starting frequency for waveform generation. This is optional, and if not
            included, the starting frequency will be set to f_min. This exists so that
            EOB waveforms can be generated starting from a lower frequency than f_min.
        mode_list
            A list of waveform (ell, m) modes to include when generating
            the polarizations.
        spin_conversion_phase
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

        lal_params: Optional[lal.Dict]
        if mode_list is not None:
            lal_params = get_lal_params(mode_list)
        else:
            lal_params = None

        # packaging all attributes into an instance of WaveformGeneratorParameters
        # allows to pass them as arguments to the various functions called by
        # generate_hplus_hcross and generate_hplus_hcross_m while avoiding circular
        # dependency.
        self._waveform_generator_params = WaveformGeneratorParameters(
            approximant=approximant,
            domain=domain,
            f_ref=f_ref,
            f_start=f_start,
            spin_conversion_phase=spin_conversion_phase,
            convert_to_SI=convert_to_SI,
            mode_list=mode_list,
            lal_params=lal_params,
            transform=transform,
        )

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        """
        Generate hplus and hcross polarizations for given waveform parameters.

        Parameters
        ----------
        waveform_parameters
            The waveform parameters used to generate the polarizations.

        Returns
        -------
        Dict[Mode, Polarization]
            A dictionary mapping modes to their corresponding polarizations.
        """

        # printing in the log the waveform parameters
        _logger.info(
            waveform_parameters.to_table(
                f"generate hplus/hcross m with waveform parameters (f_ref={self._f_ref})"
            )
        )

        # ensuring the phase field of the waveform parameters is not None
        required_keys = ("phase",)
        for rq in required_keys:
            if getattr(waveform_parameters, rq) is None:
                raise ValueError(
                    f"generate_hplus_hcross_m: the parameters must specify a value for '{rq}'"
                )

        # for now, only frequency domains are supported
        if not isinstance(self._domain, FrequencyDomain):
            raise ValueError(
                "waveform generator generate_hplus_hcross_m: only "
                f"FrequencyDomain is supported (not {type(self._domain)})"
            )

        # only FD_Approximant (IMRPhenomXPHM) and TD_Approximant (SEOBNRv4PHM) are supported
        if not self._approximant in (FD_Approximant, TD_Approximant):
            if self._approximant is None:
                raise ValueError(
                    "'None' approximant not supported for generate_hplus_hcross_m"
                )
            try:
                desc = f" ({self._approximant})"
            except Exception as e:
                desc = ""
            raise ValueError(
                "generate_hplus_hcross_m: only the approximants "
                f"{TD_Approximant} ({TD_Approximant}) and "
                f"{FD_Approximant} ({FD_Approximant})"
                f"are currently supported. {self._approximant}{desc} not supported)"
            )

        convert_to_SI = True
        hlm: Dict[Modes, FrequencySeries]
        iota: Iota

        # For TD approximant, we apply the SimInspiralChooseTDModes function
        if self._approximant == TD_Approximant:
            _logger.info(
                "generating hplus/hcross m using inspiral choose TD modes parameters"
            )
            inspiral_choose_td_modes_parameters = (
                InspiralChooseTDModesParameters.from_waveform_parameters(
                    waveform_parameters,
                    self._f_ref,
                    convert_to_SI,
                    self._domain.get_parameters(),
                    self._spin_conversion_phase,
                    self._lal_params,
                    self._approximant,
                )
            )
            hlm, iota = inspiral_choose_td_modes_parameters.apply(self._domain)

        # For the FD approximant, we apply the SimInspiralChooseFDModes function
        elif self._approximant == FD_Approximant:
            _logger.info(
                "generating hplus/hcross m using inspiral choose FD modes parameters"
            )
            inspiral_choose_fd_modes_parameters = (
                InspiralChooseFDModesParameters.from_waveform_parameters(
                    waveform_parameters,
                    self._f_ref,
                    convert_to_SI,
                    self._domain.get_parameters(),
                    self._spin_conversion_phase,
                    self._lal_params,
                    self._approximant,
                )
            )
            hlm, iota = inspiral_choose_fd_modes_parameters.apply()

        # type ignore: (we know parameters.phase is not None, checked earlier in this function)
        pol: Dict[Mode, Polarization] = get_polarizations_from_fd_modes_m(
            hlm, iota, waveform_parameters.phase  # type: ignore
        )

        # logging the generated polarizations as a nice looking table.
        _logger.debug(f"generated polarizations:\n{polarizations_to_table(pol)}")

        return pol

    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        """
        Generate hplus and hcross polarizations for given waveform parameters.

        Parameters
        ----------
        waveform_parameters
            The waveform parameters used to generate the polarizations.

        Returns
        -------
        Polarization
            The generated polarization.
        """

        # logging the waveform parameters as a nice looking table
        _logger.info(
            waveform_parameters.to_table(
                f"generate polarization hplus/hcross with waveform parameters (f_ref={self._f_ref})"
            )
        )

        # For now only frequency and time domains are supported
        # TODO: I do not think TimeDomain has been finalized and would work
        if not isinstance(self._domain, FrequencyDomain) and not isinstance(
            self._domain, TimeDomain
        ):
            raise ValueError(
                "generate_hplus_hcross: domain must be an instance of FrequencyDomain or TimeDomain "
                f"{type(self._domain)} not supported"
            )

        convert_to_SI = True
        pol: Polarization
        domain_params: DomainParameters = self._domain.get_parameters()

        # for frequency domain, we call the SimInspiralFD function
        if isinstance(self._domain, FrequencyDomain):
            inspiral_fd_parameters = InspiralFDParameters.from_waveform_parameters(
                waveform_parameters,
                self._f_ref,
                convert_to_SI,
                domain_params,
                self._spin_conversion_phase,
                self._lal_params,
                self._approximant,
            )

            frequency_array = self._domain.sample_frequencies()

            pol = inspiral_fd_parameters.apply(frequency_array)

        # for the time domain, we call the SimInspiralTD function
        # TODO: does it work ? also in the original dingo repo
        elif isinstance(self._domain, TimeDomain):

            inspiral_td_parameters = InspiralTDParameters.from_waveform_parameters(
                waveform_parameters,
                self._f_ref,
                convert_to_SI,
                domain_params,
                self._spin_conversion_phase,
                self._lal_params,
                self._approximant,
            )

            pol = inspiral_td_parameters.apply()

        if self._transform is not None:
            _logger.debug(
                f"applying transform {self._transform.__name__} to polarization"
            )
            return self._transform(pol)

        return pol


@dataclass
class WaveformGeneratorParameters:
    approximant: Union[str, Approximant]
    f_ref: float
    f_start: Optional[float]
    mode_list: Optional[List[Modes]] = None
    transform: Optional[Union[Callable[[Polarization], Polarization], str]] = None
    spin_conversion_phase: Optional[float] = None


def build_waveform_generator(
    parameters: Union[WaveformGeneratorParameters, Dict], domain: Domain
) -> WaveformGenerator:

    # if as dict as been passed as argument, 'casting' is to an instance of
    # WaveformGeneratorParameters
    if not isinstance(parameters, WaveformGeneratorParameters):
        try:
            domain_parameters = WaveformGeneratorParameters(**parameters)
        except Exception as e:
            raise ValueError(
                f"Constructing domain: failed to construct from dictionary {repr(domain_parameters)}. {type(e)}: {e}"
            )
    parameters = cast(WaveformGeneratorParameters, parameters)

    # if transform is not None and is a string, we expect it to be an import path
    # to a transform function. We import it here.
    transform: Optional[Callable[[Polarization], Polarization]]
    if parameters.transform is not None:
        if type(parameters.transform) == str:
            transform, _, _ = import_entity(parameters.transform)
            # type ignore: we know transform is not None
            if not check_function_signature(transform, [Polarization], Polarization):  # type: ignore
                raise TypeError(
                    "waveform generator transform function should take an instance of Polarization as argument "
                    f"and return an instance of polarization. The function {transform.__name__} "  # type: ignore
                    "has a different signature."
                )
    else:
        transform = None

    approximant: str
    if type(parameters.approximant) != str:
        # type ignore: mypy fails to see the check right above
        approximant = get_approximant(parameters.approximant)  # type: ignore
    else:
        approximant = parameters.approximant

    return WaveformGenerator(
        Approximant(approximant),
        domain,
        parameters.f_ref,
        f_start=parameters.f_start,
        mode_list=parameters.mode_list,
        transform=transform,
        spin_conversion_phase=parameters.spin_conversion_phase,
    )
