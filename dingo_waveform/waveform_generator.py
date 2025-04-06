import logging
from typing import Callable, Dict, List, Optional, TypeAlias, Union, cast

import lal

from . import polarization_functions, polarization_modes_functions
from .approximant import Approximant, get_approximant
from .domains import Domain, FrequencyDomain, TimeDomain
from .imports import check_function_signature, import_entity, import_function
from .lal_params import get_lal_params
from .polarizations import Polarization
from .types import Mode, Modes
from .waveform_generator_parameters import WaveformGeneratorParameters
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


PolarizationModesFunction: TypeAlias = Callable[
    [WaveformGeneratorParameters, WaveformParameters], Dict[Mode, Polarization]
]

PolarizationFunction: TypeAlias = Callable[
    [WaveformGeneratorParameters, WaveformParameters], Polarization
]


PolarizationModesFunctions: Dict[str, PolarizationModesFunction] = {
    "inspiral_choose_TD_modes": polarization_modes_functions.inspiral_choose_TD_modes,
    "inspiral_choose_FD_modes": polarization_modes_functions.inspiral_choose_FD_modes,
    "generate_FD_modes_LO": polarization_modes_functions.generate_FD_modes_LO,
    "generate_TD_modes_LO_cond_extra_time": polarization_modes_functions.generate_TD_modes_LO,
    "generate_TD_modes_LO": polarization_modes_functions.generate_TD_modes_LO,
}


PolarizationFunctions: Dict[str, PolarizationFunction] = {
    "inspiral_TD": polarization_functions.inspiral_TD,
    "inspiral_FD": polarization_functions.inspiral_FD,
    "generate_FD_modes": polarization_functions.generate_FD_modes,
    "generate_TD_modes": polarization_functions.generate_TD_modes,
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
        polarization_function: Optional[Union[str, PolarizationFunction]] = None,
        polarization_modes_function: Optional[
            Union[str, PolarizationModesFunction]
        ] = None,
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

        # generating the lal_params if requested
        lal_params: Optional[lal.Dict]
        if mode_list is not None:
            lal_params = get_lal_params(mode_list)
        else:
            lal_params = None

        # checking that the transform function
        # (used in generate_hplus_hcross)
        # has the proper signature (if not None)
        if transform is not None:
            if not check_function_signature(
                transform,
                [Polarization],
                Polarization,
            ):
                raise ValueError(
                    f"waveform_generator: can not use {transform} as polarization transform function, "
                    "as it does not have the required signature (args: Polarization, return type: Polarization)"
                )

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
            transform=transform,
        )

        # the argument polarization_method can be either:
        # - None: the method used by generate_hplus_hcross will be selected based
        #         on the approximant
        # - str: it is assumed to be the import path of a function to be imported and
        #        used by generate_hplus_hcross
        # - a callable: it is assumed to be the function to be used b generate_hplus_hcross
        # The output the import_function will either be either None or the function
        # to call (imported if required)
        self._polarization_function: Optional[PolarizationFunction] = import_function(
            polarization_function,
            [WaveformGeneratorParameters, WaveformParameters],
            Polarization,
        )

        # same for generate_hplus_hcross_m
        self._gw_polarization_method: Optional[PolarizationModesFunction] = (
            import_function(
                polarization_modes_function,
                [WaveformGeneratorParameters, WaveformParameters],
                Dict[Mode, Polarization],
            )
        )

    @classmethod
    def _get_polarization_modes_function(
        cls, approximant: Approximant
    ) -> PolarizationModesFunction:
        # Called by generate_hplus_hcross_m.
        # Returns the generator function to be used
        # based on the approximant.
        # (note: if the user passed a function as argument to the
        #        constructor, _get_gw_polarization_method will not be
        #        called and the user function will be used instead)

        # Vincent note:
        # The below uses the "old" interface for SEOBNRv4PHM and the "new"
        # interface for any other approximant.
        # Methods for using the "old" interface for IMRPhenomXPHM is also implemented,
        # but is unused. Users may pass them as argument to the constructor instead.
        # Let me know if this should be changed.

        if approximant == Approximant("SEOBNRv4PHM"):
            return polarization_modes_functions.generate_TD_modes_LO

        # "new" interface
        # IMRPhenomXPHM approximant
        # (calling: gwsignal_get_waveform_generator and waveform.GenerateFDModes)
        if approximant == Approximant("IMRPhenomXPHM"):
            return polarization_modes_functions.generate_FD_modes_LO

        # "new" interface
        # SEOBNRv5PHM or SEOBNRv5HM approximant
        # (calling gwsignal_get_waveform_generator and GenerateFDModes)
        if approximant in (
            Approximant("SEOBNRv5PHM"),
            Approximant("SEOBNRv5HM"),
        ):
            return polarization_modes_functions.generate_TD_modes_LO_cond_extra_time

        # "new" interface
        # any other approximant
        # (calling gwsignal_get_waveform_generator and GenerateFDModes)
        return polarization_modes_functions.generate_TD_modes_LO

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
        polarization_modes_function: PolarizationModesFunction
        if self._gw_polarization_method is not None:
            # we use the one provided by the user in the constructor
            polarization_modes_function = self._gw_polarization_method
        else:
            # we infer from the approximant
            # (code in a separate function for readability only)
            polarization_modes_function = self._get_polarization_modes_function(
                self._waveform_gen_params.approximant
            )

        # generating the waveforms
        _logger.info(
            f"generating waveforms using function {polarization_modes_function}"
        )
        return polarization_modes_function(
            self._waveform_gen_params, waveform_parameters
        )

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

        polarization_method: PolarizationFunction
        polarization: Polarization

        if self._polarization_function is not None:
            polarization_method = self._polarization_function
        elif isinstance(self._waveform_gen_params.domain, FrequencyDomain):
            polarization_method = polarization_functions.inspiral_FD
        elif isinstance(self._waveform_gen_params.domain, TimeDomain):
            polarization_method = polarization_functions.inspiral_TD

        _logger.info(
            f"generating waveforms using function {polarization_method.__name__}"
        )
        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )

        if self._waveform_gen_params.transform is not None:
            _logger.debug(
                f"applying transform {self._waveform_gen_params.transform} to polarization"
            )
            return self._waveform_gen_params.transform(polarization)

        return polarization


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
            # type ignore: we know parameters.transform is a string
            transform, _, _ = import_entity(parameters.transform)  # type: ignore
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
