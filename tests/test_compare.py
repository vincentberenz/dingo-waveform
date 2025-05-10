import json
from typing import Dict, Optional

import numpy as np
import pytest
from dingo import gw
from dingo.gw.domains import build_domain as original_build_domain
from dingo.gw.prior import (
    build_prior_with_defaults as original_build_prior_with_defaults,
)
from dingo.gw.waveform_generator import WaveformGenerator as OriginalWaveformGenerator

from dingo_waveform.polarizations import Polarization
from dingo_waveform.prior import IntrinsicPriors
from dingo_waveform.waveform_generator import (
    WaveformGenerator,
    build_waveform_generator,
)
from dingo_waveform.waveform_parameters import WaveformParameters

_approximants = ("IMRPhenomPv2", "IMRPhenomXPHM")
_f_start = (None, 15.0)


def _same(a: np.ndarray, b: np.ndarray, tolerance=1e-25) -> None:
    # assert a and b are the same up to the tolerance.
    # 'same': same shape, dtype and values.
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert np.allclose(a, b, atol=tolerance)


@pytest.fixture
def config_file_json(config_dict, tmp_path):
    """Create a temporary JSON config file."""
    file_path = tmp_path / "config.json"
    with open(file_path, "w") as f:
        json.dump(config_dict, f)
    return file_path


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


def get_original_waveform_generator(config_dict: Dict) -> OriginalWaveformGenerator:
    domain_params = config_dict["domain"]
    waveform_generator_params = config_dict["waveform_generator"]

    domain = original_build_domain(domain_params)
    return OriginalWaveformGenerator(
        waveform_generator_params["approximant"],
        domain,
        waveform_generator_params["f_ref"],
        spin_conversion_phase=waveform_generator_params["spin_conversion_phase"],
    )


def get_original_priors(config_dict: Dict):
    return original_build_prior_with_defaults(config_dict["intrinsic_prior"]).sample()


def get_new_waveform_generator(config_dict: Dict) -> WaveformGenerator:
    return build_waveform_generator(config_dict)


def get_new_priors(config_dict: Dict) -> WaveformParameters:
    return IntrinsicPriors(**config_dict["intrinsic_prior"]).sample()


def get_original_polarizations(approximant: str, f_start: Optional[float]) -> Dict:
    config_dict = get_configuration_dict(approximant, f_start)
    original_wfg = get_original_waveform_generator(config_dict)
    priors = get_original_priors(config_dict)
    return original_wfg.generate_hplus_hcross(priors)


def get_new_polarizations(approximant: str, f_start: Optional[float]) -> Polarization:
    config_dict = get_configuration_dict(approximant, f_start)
    wfg = get_new_waveform_generator(config_dict)
    priors = get_new_priors(config_dict)
    return wfg.generate_hplus_hcross(priors)


@pytest.mark.parametrize("approximant", _approximants)
@pytest.mark.parametrize("f_start", _f_start)
def test_compare_polarizations(approximant: str, f_start: Optional[float]):
    original_pols = get_original_polarizations(approximant, f_start)
    new_pols = get_new_polarizations(approximant, f_start)

    _same(original_pols["h_plus"], new_pols.h_plus)
    _same(original_pols["h_cross"], new_pols.h_cross)
