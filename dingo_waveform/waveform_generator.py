import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeAlias, Union, cast

import lal
from multipledispatch import dispatch

from dingo_waveform import approximant

from . import polarization_functions, polarization_modes_functions
from .approximant import Approximant
from .domains import Domain, BaseFrequencyDomain, UniformFrequencyDomain, TimeDomain, build_domain
from .imports import check_function_signature, import_function, read_file
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
    "inspiral_TD": polarization_functions.lalsim_inspiral_TD,
    "inspiral_FD": polarization_functions.lalsim_inspiral_FD,
    "generate_FD_modes": polarization_functions.gwsignal_generate_FD_modes,
    "generate_TD_modes": polarization_functions.gwsignal_generate_TD_modes,
    "random_inspiral_FD": polarization_functions.random_inspiral_FD,
}
"""
Exhaustive list of PolarizationFunctions implemented by the dingo-waveform package.
"""


PolarizationModesFunctions: Dict[str, PolarizationModesFunction] = {
    "inspiral_choose_TD_modes": polarization_modes_functions.lalsim_inspiral_choose_TD_modes,
    "inspiral_choose_FD_modes": polarization_modes_functions.lalsim_inspiral_choose_FD_modes,
    "generate_FD_modes_LO": polarization_modes_functions.gwsignal_generate_FD_modes,
    "generate_TD_modes_LO_cond_extra_time": polarization_modes_functions.gwsignal_generate_TD_modes,
    "generate_TD_modes_LO": polarization_modes_functions.gwsignal_generate_TD_modes,
    "random_inspiral_FD_modes": polarization_modes_functions.random_inspiral_FD_modes,
}
"""
Exhaustive list of PolarizationModesFunctions implemented by the dingo-waveform package.
"""

polarization_modes_approximants: Tuple[Approximant, ...] = (
    Approximant("SEOBNRv4PHM"),
    Approximant("IMRPhenomXPHM"),
    Approximant("SEOBNRv5PHM"),
    Approximant("SEOBNRv5HM"),
    Approximant("RandomApproximant"),
)
"""
Exhaustive list of approximants supported by generate_hplus_hcross_m.
"""


class WaveformGenerator(ABC):
    """
    Abstract base class for generating gravitational wave polarizations using
    various waveform approximants and domains.

    Subclasses implement generate_hplus_hcross() with the appropriate backend.
    Subclasses that support mode-separated generation also define
    generate_hplus_hcross_m().

    Use build_waveform_generator() to construct the appropriate subclass
    based on the approximant name.
    """

    def __init__(
        self,
        approximant: Approximant,
        domain: Domain,
        f_ref: float,
        f_start: Optional[float] = None,
        spin_conversion_phase: Optional[float] = None,
        mode_list: Optional[List[Modes]] = None,
        transform: Optional[Union[str, Callable[[Polarization], Polarization]]] = None,
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
        mode_list :
            List of (ell, m) tuples specifying the spherical harmonic modes
        transform :
            Optional transformation function to apply to the generated polarizations.
            Passed as the function itself or as an import path.
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
            if isinstance(transform, str):
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
        self._waveform_gen_params = WaveformGeneratorParameters(
            approximant=approximant,
            domain=domain,
            f_ref=f_ref,
            f_start=f_start,
            spin_conversion_phase=spin_conversion_phase,
            mode_list=mode_list,
            lal_params=lal_params,
            transform=transform,
        )

        # summarizing things for the user
        if _logger.isEnabledFor(logging.INFO):
            _logger.info(
                self._waveform_gen_params.to_table(
                    "instantiated waveform generator with parameters:"
                )
            )

        # Batch transform pipeline (for compression, whitening, etc.)
        # This is applied after generation and operates on dictionaries of arrays
        self.transform = None

    @abstractmethod
    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        """
        Generate h+ and h× polarizations for a given set of waveform parameters.

        Parameters
        ----------
        waveform_parameters :
            Parameters specific to the waveform being generated

        Returns
        -------
        The generated h+ and h× polarizations
        """
        ...

    def _validate_domain_for_polarization(self) -> None:
        """Validate that the domain is a supported type for polarization generation."""
        if not isinstance(
            self._waveform_gen_params.domain, BaseFrequencyDomain
        ) and not isinstance(self._waveform_gen_params.domain, TimeDomain):
            raise ValueError(
                "generate_hplus_hcross: domain must be an instance of "
                "BaseFrequencyDomain or TimeDomain, "
                f"{type(self._waveform_gen_params.domain)} not supported"
            )

    def _apply_post_generation(self, polarization: Polarization) -> Polarization:
        """Apply domain-specific waveform transform and user transform."""
        # domain specific waveform transform
        # (most domains: does nothing, MultibandedFrequencyDomain: decimate)
        polarization = self._waveform_gen_params.domain.waveform_transform(polarization)

        # transforming the waveform using the user custom function (if any)
        if self._waveform_gen_params.transform is not None:
            _logger.debug(
                f"applying transform {self._waveform_gen_params.transform} to polarization"
            )
            return self._waveform_gen_params.transform(polarization)
        return polarization

    def _log_generation_start(
        self,
        waveform_parameters: WaveformParameters,
        function_name: str,
    ) -> None:
        """Log the start of waveform generation."""
        if _logger.isEnabledFor(logging.INFO):
            _logger.info(
                waveform_parameters.to_table(
                    f"starting to generate waveforms for approximant "
                    f"{self._waveform_gen_params.approximant} "
                    f"using function {function_name} "
                    f"and waveform parameters "
                    f"(f_ref={self._waveform_gen_params.f_ref}):"
                )
            )

    def _generate_hplus_hcross_m_checks(
        self, waveform_parameters: WaveformParameters
    ) -> None:
        """Validate configuration for mode-separated generation."""
        # for now, only frequency domain is supported
        if not isinstance(self._waveform_gen_params.domain, BaseFrequencyDomain):
            raise ValueError(
                "generate_hplus_hcross_m: only frequency-domain types are supported "
                f"({type(self._waveform_gen_params.domain)} not supported)"
            )

        # ensuring the phase field of the waveform parameters is not None
        required_keys = ("phase",)
        for rq in required_keys:
            if getattr(waveform_parameters, rq, None) is None:
                raise ValueError(
                    f"generate_hplus_hcross_m: the parameters must specify a value for '{rq}'"
                )


class LALSimWaveformGenerator(WaveformGenerator):
    """
    Waveform generator using the LALSimulation backend.

    Supports any LALSimulation approximant via SimInspiralFD/SimInspiralTD.
    Does not support mode-separated generation (generate_hplus_hcross_m).
    """

    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        self._validate_domain_for_polarization()

        polarization_method: PolarizationFunction
        if isinstance(self._waveform_gen_params.domain, BaseFrequencyDomain):
            polarization_method = polarization_functions.lalsim_inspiral_FD
        else:
            polarization_method = polarization_functions.lalsim_inspiral_TD

        self._log_generation_start(waveform_parameters, polarization_method.__name__)

        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )
        return self._apply_post_generation(polarization)


class SEOBNRv4PHMWaveformGenerator(LALSimWaveformGenerator):
    """
    Waveform generator for SEOBNRv4PHM.

    Inherits generate_hplus_hcross from LALSimWaveformGenerator.
    Adds generate_hplus_hcross_m using lalsim_inspiral_choose_TD_modes.
    """

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        """
        Generate h+ and h× polarizations for multiple modes.

        Parameters
        ----------
        waveform_parameters :
            Parameters specific to the waveform being generated

        Returns
        -------
        Dictionary mapping each mode (ell, m) to its corresponding
        h+ and h× polarizations
        """
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        modes_function = polarization_modes_functions.lalsim_inspiral_choose_TD_modes

        self._log_generation_start(waveform_parameters, modes_function.__name__)

        polarization_modes: Dict[Mode, Polarization] = modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
            )

        return polarization_modes


class IMRPhenomXPHMWaveformGenerator(LALSimWaveformGenerator):
    """
    Waveform generator for IMRPhenomXPHM.

    Inherits generate_hplus_hcross from LALSimWaveformGenerator.
    Adds generate_hplus_hcross_m using lalsim_inspiral_choose_FD_modes.
    """

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        """
        Generate h+ and h× polarizations for multiple modes.

        Parameters
        ----------
        waveform_parameters :
            Parameters specific to the waveform being generated

        Returns
        -------
        Dictionary mapping each mode (ell, m) to its corresponding
        h+ and h× polarizations
        """
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        modes_function = polarization_modes_functions.lalsim_inspiral_choose_FD_modes

        self._log_generation_start(waveform_parameters, modes_function.__name__)

        polarization_modes: Dict[Mode, Polarization] = modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
            )

        return polarization_modes


class GWSignalWaveformGenerator(WaveformGenerator):
    """
    Waveform generator using the GWSignal backend (SEOBNRv5PHM, SEOBNRv5HM).

    Supports both generate_hplus_hcross and generate_hplus_hcross_m.
    """

    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        self._validate_domain_for_polarization()

        polarization_method: PolarizationFunction
        if isinstance(self._waveform_gen_params.domain, BaseFrequencyDomain):
            polarization_method = polarization_functions.gwsignal_generate_FD_modes
        else:
            polarization_method = polarization_functions.gwsignal_generate_TD_modes

        self._log_generation_start(waveform_parameters, polarization_method.__name__)

        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )
        return self._apply_post_generation(polarization)

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        """
        Generate h+ and h× polarizations for multiple modes.

        Parameters
        ----------
        waveform_parameters :
            Parameters specific to the waveform being generated

        Returns
        -------
        Dictionary mapping each mode (ell, m) to its corresponding
        h+ and h× polarizations
        """
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        modes_function = (
            polarization_modes_functions.gwsignal_generate_TD_modes_SEOBNRv5
        )

        self._log_generation_start(waveform_parameters, modes_function.__name__)

        polarization_modes: Dict[Mode, Polarization] = modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
            )

        return polarization_modes


class RandomWaveformGenerator(WaveformGenerator):
    """
    Waveform generator for RandomApproximant.

    This is a developer example showing how to implement a WaveformGenerator
    subclass. It generates synthetic waveforms without calling any external
    waveform generation library.

    Supports both generate_hplus_hcross and generate_hplus_hcross_m.
    """

    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        self._validate_domain_for_polarization()

        polarization_method = polarization_functions.random_inspiral_FD

        self._log_generation_start(waveform_parameters, polarization_method.__name__)

        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )
        return self._apply_post_generation(polarization)

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        """
        Generate h+ and h× polarizations for multiple modes.

        Parameters
        ----------
        waveform_parameters :
            Parameters specific to the waveform being generated

        Returns
        -------
        Dictionary mapping each mode to its corresponding
        h+ and h× polarizations
        """
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        modes_function = polarization_modes_functions.random_inspiral_FD_modes

        self._log_generation_start(waveform_parameters, modes_function.__name__)

        polarization_modes: Dict[Mode, Polarization] = modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
            )

        return polarization_modes


# Mapping from approximant string to WaveformGenerator subclass.
# Approximants not in this mapping default to LALSimWaveformGenerator.
_APPROXIMANT_CLASS_MAP: Dict[str, Type[WaveformGenerator]] = {
    "SEOBNRv4PHM": SEOBNRv4PHMWaveformGenerator,
    "IMRPhenomXPHM": IMRPhenomXPHMWaveformGenerator,
    "SEOBNRv5PHM": GWSignalWaveformGenerator,
    "SEOBNRv5HM": GWSignalWaveformGenerator,
    "RandomApproximant": RandomWaveformGenerator,
}


def _get_waveform_generator_class(
    approximant: Approximant,
) -> Type[WaveformGenerator]:
    """Return the appropriate WaveformGenerator subclass for the given approximant."""
    return _APPROXIMANT_CLASS_MAP.get(str(approximant), LALSimWaveformGenerator)


@dispatch(dict, Domain)
def build_waveform_generator(params: Dict, domain: Domain) -> WaveformGenerator:

    for key in ("approximant", "f_ref"):
        if key not in params.keys():
            raise ValueError(
                f"the key '{key}' is required to build a waveform generator from a dictionary"
            )

    approximant = Approximant(str(params["approximant"]))
    f_ref = float(params["f_ref"])

    spin_conversion_phase = params.get("spin_conversion_phase", None)
    if spin_conversion_phase is not None:
        spin_conversion_phase = float(spin_conversion_phase)

    f_start = params.get("f_start", None)
    if f_start is not None:
        f_start = float(f_start)

    mode_list = params.get("mode_list", None)

    transform = params.get("transform", None)
    if transform is not None:
        transform = str(transform)

    cls = _get_waveform_generator_class(approximant)

    return cls(
        approximant,
        domain,
        f_ref,
        f_start=f_start,
        spin_conversion_phase=spin_conversion_phase,
        mode_list=mode_list,
        transform=transform,
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
