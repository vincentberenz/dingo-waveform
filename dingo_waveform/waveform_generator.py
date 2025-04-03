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
from .generate_fd_modes import generate_FD_modes
from .generate_fd_modes_LO import generate_FD_modes_LO
from .generate_td_modes import generate_TD_modes
from .generate_td_modes_LO import generate_TD_modes_LO
from .generate_td_modes_LO_cond_extra_time import generate_TD_modes_LO_cond_extra_time
from .imports import check_function_signature, import_entity
from .inspiral_choose_fd_modes import inspiral_choose_FD_modes
from .inspiral_choose_td_modes import inspiral_choose_TD_modes
from .inspiral_fd import inspiral_FD
from .inspiral_td import inspiral_TD
from .lal_params import get_lal_params
from .polarizations import Polarization
from .types import GwPolarizationMethod, Mode, Modes, PolarizationMethod
from .waveform_generator_parameters import WaveformGeneratorParameters
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


GwPolarizationMethods: Dict[str, GwPolarizationMethod] = {
    "inspiral_choose_TD_modes": inspiral_choose_TD_modes,
    "inspiral_choose_FD_modes": inspiral_choose_FD_modes,
    "generate_FD_modes_LO": generate_FD_modes_LO,
    "generate_TD_modes_LO_cond_extra_time": generate_TD_modes_LO,
    "generate_TD_modes_LO": generate_TD_modes_LO,
}


PolarizationMethods: Dict[str, PolarizationMethod] = {
    "inspiral_TD": inspiral_TD,
    "inspiral_FD": inspiral_FD,
    "generate_FD_modes": generate_FD_modes,
    "generate_TD_modes": generate_TD_modes,
}


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
        polarization_method: Optional[PolarizationMethod] = None,
        gw_polarization_method: Optional[GwPolarizationMethod] = None,
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

        lal_params: Optional[lal.Dict]
        if mode_list is not None:
            lal_params = get_lal_params(mode_list)
        else:
            lal_params = None

        # packaging all attributes into an instance of WaveformGeneratorParameters
        # is not very elegant, but it allows to pass them as arguments
        # to the various functions called by generate_hplus_hcross and
        # generate_hplus_hcross_m; while avoiding circular dependency.
        self._waveform_gen_params = WaveformGeneratorParameters(
            approximant=approximant,
            domain=domain,
            f_ref=f_ref,
            f_start=f_start,
            spin_conversion_phase=spin_conversion_phase,
            convert_to_SI=convert_to_SI,
            mode_list=mode_list,
            lal_params=lal_params,
        )

        # if not None, these methods will be used by
        # generate_hplus_hcross / generate_hplus_hcross_m.
        # If None, the method will be chosen based on
        # the approximant
        self._polarization_method: Optional[PolarizationMethod] = polarization_method
        self._gw_polarization_method: Optional[GwPolarizationMethod] = (
            gw_polarization_method
        )

        self._transform = transform

    @classmethod
    def _get_gw_polarization_method(
        cls, approximant: Approximant
    ) -> GwPolarizationMethod:

        # Vincent note:
        # The below uses the "old" interface for SEOBNRv4PHM and the "new"
        # interface for any other approximant.
        # Methods for using the "old" interface for IMRPhenomXPHM is also implemented,
        # but is unused.
        # Let me know if this should be changed.

        # "new" interface
        # IMRPhenomXPHM approximant
        # (calling: gwsignal_get_waveform_generator and waveform.GenerateFDModes)
        if approximant == Approximant("IMRPhenomXPHM"):
            return generate_FD_modes_LO

        # "new" interface
        # SEOBNRv5PHM or SEOBNRv5HM approximant
        # (calling gwsignal_get_waveform_generator and GenerateFDModes)
        if approximant in (
            Approximant("SEOBNRv5PHM"),
            Approximant("SEOBNRv5HM"),
        ):
            return generate_TD_modes_LO_cond_extra_time

        # "new" interface
        # any other approximant
        # (calling gwsignal_get_waveform_generator and GenerateFDModes)
        return generate_TD_modes_LO

    def _generate_hplus_hcross_m_checks(
        self, waveform_parameters: WaveformParameters
    ) -> None:

        # for now, only frequency domain is supported
        if not isinstance(self._waveform_gen_params.domain, FrequencyDomain):
            raise ValueError(
                "generate_hplus_hcross_m: only FrequencyDomain are supported "
                f"({type(self._waveform_gen_params.domain)} not supported)"
            )

        # ensuring the phase field of the waveform parameters is not None
        required_keys = ("phase",)
        for rq in required_keys:
            if getattr(waveform_parameters, rq) is None:
                raise ValueError(
                    f"generate_hplus_hcross_m: the parameters must specify a value for '{rq}'"
                )

        # for now, only frequency domains are supported
        if not isinstance(self._waveform_gen_params.domain, FrequencyDomain):
            raise ValueError(
                "waveform generator generate_hplus_hcross_m: only "
                f"FrequencyDomain is supported (not {type(self._waveform_gen_params.domain)})"
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
                f"generate hplus/hcross m with waveform parameters (f_ref={self._waveform_gen_params.f_ref})"
            )
        )

        # checking the configuration is suitable for calling generate_hplus_hcross_m.
        # A ValueError will be raised if not.
        # In a separate method for readability only.
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        # getting the generator function.
        # In a separated method for readability only
        gw_function: GwPolarizationMethod
        if self._gw_polarization_method is not None:
            gw_function = self._gw_polarization_method
        else:
            gw_function = self._get_gw_polarization_method(
                self._waveform_gen_params.approximant
            )

        # generating the waveforms
        _logger.info(f"generating waveforms using function {gw_function.__name__}")
        return gw_function(self._waveform_gen_params, waveform_parameters)

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
                "generate polarization hplus/hcross with waveform parameters "
                f"(f_ref={self._waveform_gen_params.f_ref})"
            )
        )

        # For now only frequency and time domains are supported
        # TODO: I do not think TimeDomain has been finalized and would work
        if not isinstance(
            self._waveform_gen_params.domain, FrequencyDomain
        ) and not isinstance(self._waveform_gen_params.domain, TimeDomain):
            raise ValueError(
                "generate_hplus_hcross: domain must be an instance of FrequencyDomain or TimeDomain "
                f"{type(self._waveform_gen_params.domain)} not supported"
            )

        polarization_method: PolarizationMethod
        polarization: Polarization

        if self._polarization_method is not None:
            polarization_method = self._polarization_method
        elif isinstance(self._waveform_gen_params.domain, FrequencyDomain):
            polarization_method = inspiral_TD
        elif isinstance(self._waveform_gen_params.domain, TimeDomain):
            polarization_method = inspiral_TD

        _logger.info(
            f"generating waveforms using function {polarization_method.__name__}"
        )
        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )

        if self._transform is not None:
            _logger.debug(
                f"applying transform {self._transform.__name__} to polarization"
            )
            return self._transform(polarization)

        return polarization


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
