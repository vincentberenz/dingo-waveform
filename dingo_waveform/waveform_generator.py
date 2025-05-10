import json
import logging
from pathlib import Path, PosixPath
from typing import Callable, Dict, List, Optional, TypeAlias, Union, cast

import lal
import numpy as np
import tomli
from multipledispatch import dispatch

from . import polarization_functions, polarization_modes_functions
from .approximant import Approximant, get_approximant
from .domains import Domain, FrequencyDomain, TimeDomain, build_domain
from .imports import check_function_signature, import_entity, import_function, read_file
from .lal_params import get_lal_params
from .polarizations import Polarization, polarizations_to_table
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
        convert_to_SI: Optional[bool] = None,
        mode_list: Optional[List[Modes]] = None,
        transform: Optional[Union[str, Callable[[Polarization], Polarization]]] = None,
        polarization_function: Optional[Union[str, PolarizationFunction]] = None,
        polarization_modes_function: Optional[
            Union[str, PolarizationModesFunction]
        ] = None,
    ):
        """
        Initialize the WaveformGenerator with the necessary parameters.

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
            Optional transformation function to apply to the generated polarizations.
            Passed as the function itself or as an import path.
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
            if type(transform) == "str":
                transform = import_function(transform, [Polarization], Polarization)
            else:
                transform = cast(Callable, transform)
                if not check_function_signature(
                    transform,
                    [Polarization],
                    Polarization,
                ):
                    raise ValueError(
                        f"waveform_generator: can not use {transform} as polarization transform function, "
                        "as it does not have the required signature (args: Polarization, return type: Polarization)"
                    )
        transform = cast(Callable[[Polarization], Polarization], transform)

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

        # summarizing things for the user
        _logger.info(
            self._waveform_gen_params.to_table(
                "instantiated waveform generator with parameters:"
            )
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
        - The approximant

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

        # For now only frequency and time domains are supported
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
            # we use the polarization set by the user in the constructor
            polarization_method = self._polarization_function

        # "new" interface
        # SEOBNRv5PHM or SEOBNRv5HM approximant
        if self._waveform_gen_params.approximant in (
            Approximant("SEOBNRv5PHM"),
            Approximant("SEOBNRv5HM"),
        ):
            polarization_method = (
                polarization_functions.generate_FD_modes
                if isinstance(self._waveform_gen_params.domain, FrequencyDomain)
                else polarization_functions.generate_TD_modes
            )
        # "old" interface (any other approximant)
        else:
            polarization_method = (
                polarization_functions.inspiral_FD
                if isinstance(self._waveform_gen_params.domain, FrequencyDomain)
                else polarization_functions.inspiral_TD
            )

        # logging the waveform parameters as a nice looking table
        _logger.info(
            waveform_parameters.to_table(
                f"generating waveforms using function {polarization_method.__name__} "
                f"and waveform parameters (f_ref={self._waveform_gen_params.f_ref}):"
            )
        )

        # generating the waveforms
        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )

        # transforming the waveform using the user custom function
        # (if any)
        if self._waveform_gen_params.transform is not None:
            _logger.debug(
                f"applying transform {self._waveform_gen_params.transform} to polarization"
            )
            return self._waveform_gen_params.transform(polarization)
        else:
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
        # This uses the "old" interface for SEOBNRv4PHM and IMRPhenomXPHM;
        # and the "new" interface for any other approximant (including SEOBNRv5PHM and SEOBNRv5HM)
        # only reason: the test_wfg_m test uses the "old" interface for SEOBNRv4PHM and IMRPhenomXPHM,
        # and the "new" interface for SEOBNRv5PHM" and SEOBNRv5HM.
        # But this can be easily changed by modifying the code below.

        # "old" interface
        if approximant == Approximant("SEOBNRv4PHM"):
            return polarization_modes_functions.inspiral_choose_TD_modes

        # "old" interface
        # IMRPhenomXPHM approximant
        # (calling: gwsignal_get_waveform_generator and waveform.GenerateFDModes)
        if approximant == Approximant("IMRPhenomXPHM"):
            return polarization_modes_functions.inspiral_choose_FD_modes

        # "new" interface
        # SEOBNRv5PHM or SEOBNRv5HM approximant
        # (calling gwsignal_get_waveform_generator and GenerateFDModes + some processing)
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
            f"waveform generator: generating waveforms using function {polarization_modes_function.__name__}"
        )
        polarization_modes: Dict[Mode, Polarization] = polarization_modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        # logging the generated polarizations as a nice looking table.
        _logger.debug(
            f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
        )

        return polarization_modes


@dispatch(dict, Domain)
def build_waveform_generator(params: Dict, domain: Domain) -> WaveformGenerator:

    for key in ("approximant", "f_ref"):
        if key not in params.keys():
            raise ValueError(
                f"the key '{key}' is required to build a waveform generator from a dictionary"
            )

    approximant = str(params["approximant"])
    f_ref = float(params["f_ref"])

    spin_conversion_phase = params.get("spin_conversion_phase", None)
    if spin_conversion_phase is not None:
        spin_conversion_phase = bool(spin_conversion_phase)

    f_start = params.get("f_start", None)
    if f_start is not None:
        f_start = float(f_start)

    convert_to_SI = params.get("convert_to_SI", None)
    if convert_to_SI is not None:
        convert_to_SI = bool(convert_to_SI)

    mode_list = params.get("mode_list", None)

    transform = params.get("transform", None)
    if transform is not None:
        transform = str(transform)

    polarization_function = params.get("polarization_function", None)
    if polarization_function is not None:
        polarization_function = str(polarization_function)

    polarization_modes_function = params.get("polarization_mode_function", None)
    if polarization_modes_function is not None:
        polarization_modes_function = str(polarization_modes_function)

    return WaveformGenerator(
        approximant,
        domain,
        f_ref,
        f_start=f_start,
        spin_conversion_phase=spin_conversion_phase,
        convert_to_SI=convert_to_SI,
        mode_list=mode_list,
        transform=transform,
        polarization_function=polarization_function,
        polarization_modes_function=polarization_modes_function,
    )


@dispatch(dict)
def build_waveform_generator(params: Dict) -> WaveformGenerator:

    for key in ("domain", "waveform_generator"):
        if key not in params.keys():
            raise ValueError(
                f"the key '{key}' is required to build a waveform generator from a dictionary"
            )
    domain_params = params["domain"]
    domain: Domain = build_domain(domain_params)

    waveform_params = params["waveform_generator"]
    return build_waveform_generator(waveform_params, domain)


@dispatch(Path)
def build_waveform_generator(file_path: Path) -> WaveformGenerator:
    params = read_file(file_path)
    return build_waveform_generator(params)


@dispatch(str)
def build_waveform_generator(file_path: Path) -> WaveformGenerator:
    return build_waveform_generator(Path(file_path))
