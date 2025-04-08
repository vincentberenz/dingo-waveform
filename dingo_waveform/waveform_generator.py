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

PolarizationFunction: TypeAlias = Callable[
    [WaveformGeneratorParameters, WaveformParameters], Polarization
]
"""
Type alias for functions that generate a single polarization.

See related functions in subpackage dingo_waveform.polarization_functions

Parameters
----------
WaveformGeneratorParameters : 
    Parameters controlling the waveform generation process
WaveformParameters : 
    Parameters specific to the waveform being generated

Returns
-------
A single polarization value for the specified parameters.
"""


PolarizationModesFunction: TypeAlias = Callable[
    [WaveformGeneratorParameters, WaveformParameters], Dict[Mode, Polarization]
]
"""
Type alias for functions that generate multiple polarization modes.

See related functions in subpackage dingo_waveform.polarization_modes_functions

Parameters
----------
WaveformGeneratorParameters : 
    Parameters controlling the waveform generation process
WaveformParameters : 
    Parameters specific to the waveform being generated

Returns
-------
Dictionary mapping each mode (ell, m) to its corresponding polarization
values. This allows for the generation of multiple spherical harmonic
modes in a single function call.
"""

PolarizationFunctions: Dict[str, PolarizationFunction] = {
    "inspiral_TD": polarization_functions.inspiral_TD,
    "inspiral_FD": polarization_functions.inspiral_FD,
    "generate_FD_modes": polarization_functions.generate_FD_modes,
    "generate_TD_modes": polarization_functions.generate_TD_modes,
}
"""
Exhaustive list of PolarizationFunctions implemented by the dingo-waveform package.
"""


PolarizationModesFunctions: Dict[str, PolarizationModesFunction] = {
    "inspiral_choose_TD_modes": polarization_modes_functions.inspiral_choose_TD_modes,
    "inspiral_choose_FD_modes": polarization_modes_functions.inspiral_choose_FD_modes,
    "generate_FD_modes_LO": polarization_modes_functions.generate_FD_modes_LO,
    "generate_TD_modes_LO_cond_extra_time": polarization_modes_functions.generate_TD_modes_LO,
    "generate_TD_modes_LO": polarization_modes_functions.generate_TD_modes_LO,
}
"""
Exhaustive list of PolarizationModesFunctions implemented by the dingo-waveform package.
"""


class WaveformGenerator:
    """
    A class for generating gravitational wave polarizations using various waveform
    approximants and domains. This class serves as a wrapper around the
    PolarizationFunction and PolarizationModesFunction types, automatically
    selecting the appropriate function based on the specified approximant and domain.

    Methods
    -------
    generate_hplus_hcross(waveform_parameters)
        Generate h+ and h× polarizations for a given set of parameters
    generate_hplus_hcross_m(waveform_parameters)
        Generate h+ and h× polarizations for multiple modes
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
        Initialize the WaveformGenerator with the necessary parameters.

        The constructor prepares the waveform generator by:
        1. Creating the necessary LAL parameters if mode_list is provided
        2. Validating the transform function signature if provided
        3. Storing all parameters in a WaveformGeneratorParameters instance
        4. Importing and storing the specified polarization functions

        Parameters
        ----------
        approximant :
            The waveform approximant model to use (e.g., SEOBNRv5, IMRPhenomD)
        domain :
            The computational domain for the waveform generation
        f_ref :
            Reference frequency for the waveform generation
        f_start :
            Starting frequency for the waveform generation
        spin_conversion_phase :
            Phase angle used for converting spins
        convert_to_SI :
            Flag indicating whether to perform unit conversions to SI system
        mode_list :
            List of (ell, m) tuples specifying the spherical harmonic modes
        transform :
            Optional transformation function to apply to the generated polarizations
        polarization_function :
            Either a string representing the import path of a function to use, or
            the function itself to generate single polarization waveforms
        polarization_modes_function :
            Either a string representing the import path of a function to use, or
            the function itself to generate multiple polarization modes
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
        #         on the approximant / domain
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

    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        """
        Generate h+ and h× polarizations for a given set of waveform parameters.

        This method selects the appropriate polarization function based on:
        - User-provided function (if specified)
        - Domain type (FrequencyDomain or TimeDomain)

        It also applies any specified transform to the result (if any)

        Parameters
        ----------
        waveform_parameters :
            Parameters specific to the waveform being generated

        Returns
        -------
        The generated h+ and h× polarizations

        Raises
        ------
        ValueError
            If the domain is not FrequencyDomain or TimeDomain
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

    @classmethod
    def _get_polarization_modes_function(
        cls, approximant: Approximant
    ) -> PolarizationModesFunction:
        # Called by generate_hplus_hcross_m.
        # Returns the generator function to be used
        # based on the approximant.
        # (note: if the user passed a function as argument to the
        #        constructor, _get_polarization_modes_function will not be
        #        called and the user function will be used instead)

        # Vincent note:
        # The below uses the "old" interface for SEOBNRv4PHM and the "new"
        # interface for any other approximant.
        # Methods for using the "old" interface for IMRPhenomXPHM is also implemented,
        # but are unused. Users may pass them as argument to the constructor instead.
        # Let me know if this should be changed.

        # "new" interface
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
        # will be called by generate_hplus_hcross_m.
        # It raises an error if there is any configuration issue,
        # - improper domain (only FrequencyDomain is supported)
        # - the phase field of the waveform parameters is None

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
        Generate h+ and h× polarizations for multiple modes.

        Selects the appropriate polarization modes function based on:
        - User-provided function (if specified)
        - Approximant type

        Parameters
        ----------
        waveform_parameters :
            Parameters specific to the waveform being generated

        Returns
        -------
        Dictionary mapping each mode (ell, m) to its corresponding
        h+ and h× polarizations
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
            # we use the method provided by the user in the constructor
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


def build_waveform_generator(
    parameters: Union[WaveformGeneratorParameters, Dict], domain: Domain
) -> WaveformGenerator:
    """
    Factory function to create a WaveformGenerator instance from parameters.

    Parameters
    ----------
    parameters :
        Either an instance of WaveformGeneratorParameters or a dictionary
        containing the necessary parameters.
    domain : Domain
        The computational domain for the waveform generation

    Returns
    -------
    A configured WaveformGenerator instance ready for use

    Raises
    ------
    ValueError
        If the dictionary cannot be converted to WaveformGeneratorParameters
    TypeError
        If the transform function has an invalid signature
    """

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
