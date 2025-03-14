import logging
from dataclasses import asdict, dataclass
from numbers import Number
from typing import List, Optional, Tuple, Union

import lal
import numpy as np
from bilby.gw.conversion import (
    bilby_to_lalsimulation_spins,
    convert_to_lal_binary_black_hole_parameters,
)

from .approximant import Approximant
from .domains import DomainParameters
from .logging import TableStr, to_table
from .spins import Spins
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


def _convert_to_float(x: Union[np.ndarray, Number, float]) -> float:
    """
    Convert a single element array to a number.

    Parameters
    ----------
    x:
        Array or float

    Returns
    -------
    A number
    """
    if isinstance(x, np.ndarray):
        if x.shape == () or x.shape == (1,):
            return float(x.item())
        else:
            raise ValueError(
                f"Expected an array of length one, but go shape = {x.shape}"
            )
    else:
        return float(x)  # type: ignore


@dataclass
class BinaryBlackHoleParameters(TableStr):
    luminosity_distance: float
    a_1: float
    a_2: float
    phi_jl: float
    phi_12: float
    tilt_1: float
    tilt_2: float
    phase: float
    theta_jn: float
    f_ref: float
    chirp_mass: float
    mass_ratio: float
    total_mass: float
    mass_1: float
    mass_2: float
    geocent_time: Optional[float] = None
    l_max: Optional[float] = None
    chi_1: Optional[float] = None
    chi_2: Optional[float] = None
    cos_tilt_1: Optional[float] = None
    cos_tilt_2: Optional[float] = None

    @classmethod
    def from_waveform_parameters(
        cls, waveform_params: WaveformParameters, f_ref: float, convert_to_SI: bool
    ) -> "BinaryBlackHoleParameters":
        params = asdict(waveform_params)
        params["f_ref"] = f_ref
        params = {k: v for k, v in params.items() if v is not None}
        _logger.debug(
            "calling convert_to_lal_binary_black_hole_parameters with parameters:\n"
            f"{to_table(params)}"
        )
        converted_params, _ = convert_to_lal_binary_black_hole_parameters(params)
        _logger.debug(
            "output of convert_to_lal_binary_black_hole_parameters:\n"
            f"{to_table(converted_params)}"
        )
        if convert_to_SI:
            _logger.debug("converting to SI units")
            converted_params["mass_1"] *= lal.MSUN_SI
            converted_params["mass_2"] *= lal.MSUN_SI
            converted_params["luminosity_distance"] *= 1e6 * lal.PC_SI
        instance = cls(**converted_params)
        _logger.debug(instance.to_table("generated binary black hole parameters"))
        return instance

    def get_spins(self, spin_conversion_phase: Optional[float]) -> Spins:
        keys: Tuple[str, ...] = (
            "theta_jn",
            "phi_jl",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "a_1",
            "a_2",
            "mass_1",
            "mass_2",
            "f_ref",
            "phase",
        )
        args: List[float] = [getattr(self, k) for k in keys]
        if spin_conversion_phase is not None:
            args[-1] = spin_conversion_phase
        _logger.debug(
            "calling bilby_to_lalsimulation_spins with arguments:\n"
            f"{to_table({k: a for k,a in zip(keys,args)})}"
        )
        iota_and_cart_spins: List[float] = [
            float(_convert_to_float(value))
            for value in bilby_to_lalsimulation_spins(*args)
        ]
        instance = Spins(*iota_and_cart_spins)
        # type ignore : for some reason I do not understand, instance is not
        # recognized by mypy as a dataclass type.
        _logger.debug("generated spins:\n" f"{to_table(instance)}")  # type: ignore
        return instance
