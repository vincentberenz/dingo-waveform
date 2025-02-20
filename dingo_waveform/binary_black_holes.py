from dataclasses import asdict, dataclass
from numbers import Number
from typing import List, Optional, Tuple, Union

import lal
import numpy as np
from bilby.gw.conversion import (
    bilby_to_lalsimulation_spins,
    convert_to_lal_binary_black_hole_parameters,
)

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import DomainParameters
from dingo_waveform.inspiral_choose_fd_modes import InspiralChooseFDModesParameters
from dingo_waveform.inspiral_choose_td_modes import InspiralChooseTDModesParameters
from dingo_waveform.spins import Spins


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
class BinaryBlackHoleParameters:
    luminosity_distance: float
    redshift: float
    a_1: float
    a_2: float
    cos_tilt_1: float
    cos_tilt_2: float
    phi_jl: float
    phi_12: float
    phase: float
    tilt_1: float
    tilt_2: float
    theta_jn: float
    f_ref: float

    # Mass parameters
    mass_1: float
    mass_2: float
    total_mass: float
    chirp_mass: float
    mass_ratio: float
    symmetric_mass_ratio: float

    # Source frame mass parameters
    mass_1_source: float
    mass_2_source: float
    total_mass_source: float
    chirp_mass_source: float

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
            ",ass_2",
            "f_ref",
            "phase",
        )
        args: List[float] = [getattr(self, k) for k in keys]
        if spin_conversion_phase is not None:
            args[-1] = spin_conversion_phase
        iota_and_cart_spins: List[float] = [
            float(_convert_to_float(value))
            for value in bilby_to_lalsimulation_spins(args)
        ]
        return Spins(*iota_and_cart_spins)

    def to_InspiralChooseFDModesParameters(
        self,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> InspiralChooseFDModesParameters:
        spins: Spins = self.get_spins(spin_conversion_phase)
        # adding iota, s1x, ..., s2x, ...
        parameters = asdict(spins)
        # direct mapping from this instance
        for k in ("mass_1", "mass_2", "phase"):
            parameters[k] = getattr(self, k)
        # adding domain related params
        domain_dict = asdict(domain_params)
        for k in ("delta_t", "f_min", "f_max", "f_ref"):
            parameters[k] = domain_dict[k]
        # other params
        parameters["r"] = self.luminosity_distance
        parameters["lal_params"] = lal_params
        parameters["approximant"] = approximant
        return InspiralChooseFDModesParameters(**parameters)

    def to_InspiralChooseTDModesParameters(
        self,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> InspiralChooseTDModesParameters: ...
