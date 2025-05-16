import contextlib
import json
import os
import tempfile
from distutils.command import build
from typing import Dict, Optional, Tuple

from dingo_waveform import waveform_parameters
from dingo_waveform.approximant import Approximant
from dingo_waveform.polarizations import Polarization
from dingo_waveform.prior import IntrinsicPriors
from dingo_waveform.types import Mode
from dingo_waveform.waveform_generator import (
    WaveformGenerator,
    build_waveform_generator,
    polarization_modes_approximants,
)
from dingo_waveform.waveform_parameters import WaveformParameters


def get_configuration_dict(approximant: str, f_start: Optional[float]) -> Dict:
    d: Dict = {
        "domain": {
            "type": "FrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,
        },
        "waveform_generator": {
            "approximant": approximant,
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
        "intrinsic_prior": {
            "mass_1": 50.0,
            "mass_2": 25.0,
            "chirp_mass": 60.0,
            "mass_ratio": 0.5,
            "phase": 2.5811112632546123,
            "a_1": 0.5,
            "a_2": 0.6,
            "tilt_1": 1.8222778934660213,
            "tilt_2": 1.3641458250460199,
            "phi_12": 4.469204665688967,
            "phi_jl": 3.021398659177057,
            "theta_jn": 1.4262724019800959,
            "luminosity_distance": 100.0,  # Mpc
            "geocent_time": 0.0,  # s
        },
    }
    if f_start is not None:
        d["waveform_generator"]["f_start"] = f_start
    return d


def test_IMRPhenomXPHM():
    # only checking no exception occur.
    approximant = Approximant("IMRPhenomXPHM")
    config = get_configuration_dict(approximant, f_start=None)
    waveform_generator = build_waveform_generator(config)
    waveform_parameters = IntrinsicPriors(**config["intrinsic_prior"]).sample()
    waveform_generator.generate_hplus_hcross(waveform_parameters)
    # if approximant in polarization_modes_approximants:
    #    waveform_generator.generate_hplus_hcross_m(waveform_parameters)
