import pickle
from pathlib import Path
from typing import Any, Optional

from dingo_waveform.domains import FrequencyDomain
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.waveform_parameters import WaveformParameters


def _restore_ground_truth(name: str) -> Any:
    pickled_path = Path(__file__).parent / "ground_truths" / name
    with open(pickled_path, "rb") as f:
        return pickle.load(f)


def test_IMRPhenomXPHM_approximant() -> None:

    # domain settings (type: FrequencyDomain)
    f_min = 10.0
    f_max = 2048.0
    delta_f = 0.125

    p = {
        "mass_ratio": 0.3501852584069329,
        "chirp_mass": 31.709276525188667,
        "luminosity_distance": 1000.0,
        "theta_jn": 1.3663250108421872,
        "phase": 2.3133395191342094,
        "a_1": 0.9082488389607664,
        "a_2": 0.23195443013657285,
        "tilt_1": 2.2991912365076708,
        "tilt_2": 2.2878677821511086,
        "phi_12": 2.3726027637572384,
        "phi_jl": 1.5356479043406908,
        "geocent_time": 0.0,
    }

    approximant = "IMRPhenomXPHM"
    f_ref = 20.0
    f_start = 10.0
    spin_conversion_phase = 0.0

    domain = FrequencyDomain(
        delta_f=delta_f,
        f_min=f_min,
        f_max=f_max,
    )

    parameters = WaveformParameters(**p)

    wfg = WaveformGenerator(
        approximant, domain, f_ref, f_start, spin_conversion_phase=spin_conversion_phase
    )

    pol_m = wfg.generate_hplus_hcross_m(parameters)

    convert_to_lal_binary_black_hole_parameters = {
        "mass_ratio": 0.3501852584069329,
        "chirp_mass": 31.709276525188667,
        "luminosity_distance": 1000.0,
        "theta_jn": 1.3663250108421872,
        "phase": 2.3133395191342094,
        "a_1": 0.9082488389607664,
        "a_2": 0.23195443013657285,
        "tilt_1": 2.2991912365076708,
        "tilt_2": 2.2878677821511086,
        "phi_12": 2.3726027637572384,
        "phi_jl": 1.5356479043406908,
        "geocent_time": 0.0,
        "f_ref": 20.0,
    }

    bilby_to_lalsimulation_spin = {
        "theta_jn": 1.3663250108421872,
        "phi_jl": 1.5356479043406908,
        "tilt_1": 2.2991912365076708,
        "tilt_2": 2.2878677821511086,
        "phi_12": 2.3726027637572384,
        "a_1": 0.9082488389607664,
        "a_2": 0.23195443013657285,
        "mass_1": 1.2565859519276645e32,
        "mass_2": 4.40037876286311e31,
        "f_ref": 20.0,
        "phase": 0.0,
    }

    lal_parameter_tuple = (
        1.2565859519276645e32,
        4.40037876286311e31,
        -0.055470980343691426,
        -0.6755013274624259,
        -0.6045964607982675,
        0.1314543246986317,
        0.11526461124169105,
        -0.1524358474024755,
        0.125,
        10.0,
        2048.0,
        20.0,
        2.3133395191342094,
        3.0856775814913673e25,
        0.6179131357160138,
        None,
        101,
    )

    parameters_to_siminspiral_lal_fd_modes = (
        1.2565859519276645e32,
        4.40037876286311e31,
        -0.055470980343691426,
        -0.6755013274624259,
        -0.6045964607982675,
        0.1314543246986317,
        0.11526461124169105,
        -0.1524358474024755,
        0.125,
        10.0,
        2048.0,
        20.0,
        2.3133395191342094,
        3.0856775814913673e25,
        0.6179131357160138,
        None,
        101,
    )

    assert pol_m is not None

    # ground_truth = "fd_IMRPhenomXPHM.pickled"
