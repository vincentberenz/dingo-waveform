import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import lal
import tomli

from .approximant import Approximant
from .domains import Domain, build_domain
from .imports import read_file
from .logs import TableStr
from .polarizations import Polarization
from .types import Modes


def _validate_waveform_generator_params(
    approximant: Union[Approximant, str],
    f_ref: float,
    f_start: Optional[float] = None,
) -> Approximant:
    """
    Validate waveform generator parameters.

    Shared validation logic for WaveformGeneratorParameters and WaveformGeneratorSettings.

    Note: Approximant is a NewType alias for str (type hint only), so no actual
    type conversion occurs. The validation focuses on f_ref and f_start.

    Parameters
    ----------
    approximant
        Waveform approximant string
    f_ref
        Reference frequency in Hz
    f_start
        Starting frequency in Hz (optional)

    Returns
    -------
    Approximant
        Validated approximant (same as input, since Approximant is NewType for str)

    Raises
    ------
    ValueError
        If f_ref or f_start are not positive
    """
    # Validate f_ref is positive
    if f_ref <= 0:
        raise ValueError(f"f_ref must be positive, got {f_ref}")

    # Validate f_start if specified
    if f_start is not None and f_start <= 0:
        raise ValueError(f"f_start must be positive, got {f_start}")

    # Approximant is NewType for str, so just return as-is
    return approximant


@dataclass
class WaveformGeneratorParameters(TableStr):
    """
    Container class for parameters controlling gravitational waveform generation.

    This class is used at runtime and includes the Domain object. For serializable
    configuration (YAML/JSON), use WaveformGeneratorSettings from the dataset module.

    Attributes
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
        List of (ell, m) tuples specifying the spherical harmonic modes to include
        in the waveform calculation
    lal_params :
        Additional LAL parameters dictionary for waveform generation
    transform :
        Optional transformation function to apply to the waveform polarizations

    See Also
    --------
    dingo_waveform.dataset.waveform_generator_settings.WaveformGeneratorSettings :
        Serializable configuration class (subset of this class without domain/runtime fields)
    """

    approximant: Approximant
    domain: Domain
    f_ref: float
    f_start: Optional[float]
    spin_conversion_phase: Optional[float]
    mode_list: Optional[List[Modes]]
    lal_params: Optional[lal.Dict]
    transform: Optional[Callable[[Polarization], Polarization]] = None

    def __post_init__(self):
        """Validate parameters."""
        # Validate f_ref and f_start (approximant is already the correct type)
        _validate_waveform_generator_params(
            self.approximant, self.f_ref, self.f_start
        )

    @classmethod
    def from_settings(
        cls,
        settings: "WaveformGeneratorSettings",  # type: ignore
        domain: Domain,
        mode_list: Optional[List[Modes]] = None,
        lal_params: Optional[lal.Dict] = None,
        transform: Optional[Callable[[Polarization], Polarization]] = None,
    ) -> "WaveformGeneratorParameters":
        """
        Create WaveformGeneratorParameters from WaveformGeneratorSettings.

        This is the recommended way to create runtime parameters from serializable settings.

        Parameters
        ----------
        settings
            Serializable waveform generator settings (from YAML/JSON config)
        domain
            Built domain object
        mode_list
            Optional list of modes to include
        lal_params
            Optional LAL parameters
        transform
            Optional transform function

        Returns
        -------
        WaveformGeneratorParameters
            Runtime parameters ready for waveform generation
        """
        return cls(
            approximant=settings.approximant,
            domain=domain,
            f_ref=settings.f_ref,
            f_start=settings.f_start,
            spin_conversion_phase=settings.spin_conversion_phase,
            mode_list=mode_list,
            lal_params=lal_params,
            transform=transform,
        )

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], domain: Domain
    ) -> "WaveformGeneratorParameters":
        """Load parameters from file and combine with domain."""
        params = read_file(file_path)
        params["domain"] = domain
        return cls(**params)
