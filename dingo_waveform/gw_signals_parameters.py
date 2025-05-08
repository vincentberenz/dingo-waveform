import logging
from dataclasses import dataclass
from numbers import Number
from typing import Any, List, Optional, Union

import astropy
import astropy.units
import numpy as np
from astropy.units import Quantity

from .binary_black_holes_parameters import BinaryBlackHoleParameters
from .domains import DomainParameters
from .logging import TableStr
from .spins import Spins
from .types import Mode
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class GwSignalParameters(TableStr):
    """
    Parameters in a format suitable for calling functions from the
    lalsimulation.gwsignal.core package.

    The optional parameters condition,
    lmax_nyquist, postadiabatic, and postadiabatic_type are specifically used
    with the SEOBNRv5 model.
    """

    mass1: Quantity
    mass2: Quantity
    spin1x: Quantity
    spin1y: Quantity
    spin1z: Quantity
    spin2x: Quantity
    spin2y: Quantity
    spin2z: Quantity
    deltaT: Quantity
    f22_start: Quantity
    f22_ref: Quantity
    f_max: Quantity
    deltaF: Quantity
    phi_ref: Quantity
    distance: Quantity
    inclination: Quantity
    condition: int
    lmax_nyquist: Optional[int] = None
    postadiabatic: Optional[Any] = None
    postadiabatic_type: Optional[Any] = None

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float] = None,
        f_start: Optional[float] = None,
        condition: int = 1,
        lmax_nyquist: Optional[int] = None,
        postadiabatic: Optional[Any] = None,
        postadiabatic_type: Optional[Any] = None,
    ) -> "GwSignalParameters":
        """
        Create an instance of GwSignalParameters from binary black hole parameters.

        Parameters
        ----------
        bbh_params
            Parameters describing the binary black hole system
        domain_params
            Parameters defining the computational domain
        spin_conversion_phase
            Phase angle for converting spins
        f_start
            Starting frequency for waveform generation
        condition
            Numerical conditioning flag controlling numerical precision and stability
            in waveform calculations
        lmax_nyquist
            Maximum harmonic index for Nyquist sampling; determines the highest
            angular resolution included in the waveform calculation
        postadiabatic
            Post-adiabatic correction parameters for SEOBNRv5 waveform model;
            controls higher-order corrections beyond the adiabatic approximation
        postadiabatic_type
            Type specification for post-adiabatic corrections in SEOBNRv5;
            determines the specific implementation of post-adiabatic terms

        Returns
        -------
        GwSignalParameters
            An instance containing all necessary parameters for gravitational wave signal generation
        """

        # used to declare the value of f22_start below
        f_min = f_start if f_start is not None else domain_params.f_min

        spins: Spins = bbh_params.get_spins(spin_conversion_phase)

        params = {
            "mass1": bbh_params.mass_1 * astropy.units.solMass,
            "mass2": bbh_params.mass_2 * astropy.units.solMass,
            "spin1x": spins.s1x * astropy.units.dimensionless_unscaled,
            "spin1y": spins.s1y * astropy.units.dimensionless_unscaled,
            "spin1z": spins.s1z * astropy.units.dimensionless_unscaled,
            "spin2x": spins.s2x * astropy.units.dimensionless_unscaled,
            "spin2y": spins.s2y * astropy.units.dimensionless_unscaled,
            "spin2z": spins.s2z * astropy.units.dimensionless_unscaled,
            "deltaT": domain_params.delta_t * astropy.units.s,
            "f22_start": f_min * astropy.units.Hz,
            "f22_ref": bbh_params.f_ref * astropy.units.Hz,
            "f_max": domain_params.f_max * astropy.units.Hz,
            "deltaF": domain_params.delta_f * astropy.units.Hz,
            "phi_ref": bbh_params.phase * astropy.units.rad,
            "distance": bbh_params.luminosity_distance * astropy.units.Mpc,
            "inclination": spins.iota * astropy.units.rad,
            "condition": condition,
            "postadiabatic": postadiabatic,  # SEOBNRv5 specific parameters
            "postadiabatic_type": postadiabatic_type,  # SEOBNRv5 specific parameters
            "lmax_nyquist": lmax_nyquist,  # SEOBNRv5 specific parameters
        }

        return cls(**params)

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_params: WaveformParameters,
        domain_params: DomainParameters,
        f_ref: float,
        spin_conversion_phase: Optional[float] = None,
        f_start: Optional[float] = None,
        convert_to_SI: Optional[bool] = False,
    ) -> "GwSignalParameters":
        """
        Create an instance of GwSignalParameters from waveform parameters.

        This method provides a convenient interface for creating GwSignalParameters
        instances directly from WaveformParameters. It internally performs a two-step
        conversion process:

        1. First converts WaveformParameters to BinaryBlackHoleParameters using the
        reference frequency f_ref and unit conversion flag convert_to_SI
        2. Then creates GwSignalParameters using the previously documented
        from_binary_black_hole_parameters method

        The SEOBNRv5 specific parameters (lmax_nyquist, postadiabatic, and
        postadiabatic_type) are automatically propagated from waveform_params if present.

        Parameters
        ----------
        waveform_params
            Source parameters describing the gravitational wave signal
        domain_params
            Parameters defining the computational domain
        f_ref
            Reference frequency for the waveform generation
        spin_conversion_phase
            Phase angle used for converting spins (default: None)
        f_start
            Starting frequency for waveform generation (default: None)
        convert_to_SI
            Flag indicating whether to perform unit conversions to SI system
            (default: True)

        Returns
        -------
        GwSignalParameters
            An instance containing all necessary parameters for gravitational wave signal generation
        """

        # for "new" interface, we do not convert by default
        # to the masses to SI
        if convert_to_SI is None:
            convert_to_SI = False

        bbh = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_params, f_ref, convert_to_SI
        )
        instance = cls.from_binary_black_hole_parameters(
            bbh,
            domain_params,
            spin_conversion_phase,
            f_start,
            lmax_nyquist=waveform_params.lmax_nyquist,
            postadiabatic=waveform_params.postadiabatic,
            postadiabatic_type=waveform_params.postadiabatic_type,
        )

        _logger.debug(
            instance.to_table(
                f"created an instance of {instance.__class__.__name__} with parameters:"
            )
        )

        return instance
