import pickle
from dataclasses import astuple
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
from lal import COMPLEX16FrequencySeries
from lalsimulation import SimInspiralChooseFDModes

from dingo_waveform import wfg_utils
from dingo_waveform.approximant import get_approximant
from dingo_waveform.domains import FrequencyDomain
from dingo_waveform.inspiral_choose_fd_modes import InspiralChooseFDModesParameters
from dingo_waveform.logging import to_table
from dingo_waveform.polarizations import Polarization, sum_contributions_m
from dingo_waveform.types import FrequencySeries, Mode
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.waveform_parameters import WaveformParameters


def _restore_ground_truth(name: str) -> Any:
    pickled_path = Path(__file__).parent / "ground_truths" / name
    with open(pickled_path, "rb") as f:
        return pickle.load(f)


def _same(a: np.ndarray, b: np.ndarray, tolerance=1e-25) -> None:
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert np.allclose(a, b, atol=tolerance)


def _compare_frequency_series(
    series1: COMPLEX16FrequencySeries,
    series2: COMPLEX16FrequencySeries,
    tolerance=1e-25,
):
    assert abs(series1.epoch.gpsSeconds - series2.epoch.gpsSeconds) <= tolerance
    assert abs(series1.f0 - series2.f0) <= tolerance
    assert abs(series1.deltaF - series2.deltaF) <= tolerance
    _same(series1.data.data, series2.data.data, tolerance=tolerance)


def _same_frequency_series_dict(
    a: Dict[Mode, COMPLEX16FrequencySeries], b: Dict[Mode, COMPLEX16FrequencySeries]
) -> None:
    assert set(a.keys()) == set(b.keys())
    for mode in a.keys():
        _compare_frequency_series(a[mode], b[mode])


def _same_dict(a: Dict[Mode, np.ndarray], b: Dict[Mode, np.ndarray]) -> None:
    assert set(a.keys()) == set(b.keys())
    for mode in a.keys():
        _same(a[mode], b[mode])


def test_inspiral_choose_fd_modes_parameters() -> None:

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
    f_ref: float = 20.0

    # ground truth arguments to SimInspiralChooseFDModes
    args_ground_truth = (
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
    waveform_parameters = WaveformParameters(**p)

    convert_to_SI = True
    f_min = 10.0
    f_max = 2048.0
    delta_f = 0.125
    spin_conversion_phase = 0.0
    lal_params = None
    approximant = get_approximant("IMRPhenomXPHM")

    domain = FrequencyDomain(
        delta_f=delta_f,
        f_min=f_min,
        f_max=f_max,
    )

    inspiral_choose_fd_modes_parameters = (
        InspiralChooseFDModesParameters.from_waveform_parameters(
            waveform_parameters,
            f_ref,
            convert_to_SI,
            domain.get_parameters(),
            spin_conversion_phase,
            lal_params,
            approximant,
        )
    )

    args = astuple(inspiral_choose_fd_modes_parameters)

    assert len(args_ground_truth) == len(args)
    for a1, a2 in zip(args, args_ground_truth):
        assert type(a1) == type(a2)
        assert a1 == a2

    d = _restore_ground_truth("fd_convert_J_to_L0.pickled")
    ground_truth_hlm_fd_ = d["hlm_fd_"]
    ground_truth_hlm_fd__ = d["hlm_fd__"]
    ground_truth_hlm_fd___ = d["hlm_fd___"]
    ground_truth_iota = d["iota"]

    # hlm_fd = LS.SimInspiralChooseFDModes(*parameters_lal_fd_modes)
    # unpack linked list, convert lal objects to arrays
    # hlm_fd_ = wfg_utils.linked_list_modes_to_dict_modes(hlm_fd)
    # hlm_fd__ = {k: v.data.data for k, v in hlm_fd_.items()}
    # For the waveform models considered here (e.g., IMRPhenomXPHM), the modes
    # are returned in the J frame (where the observer is at inclination=theta_JN,
    # azimuth=0). In this frame, the dependence on the reference phase enters
    # via the modes themselves. We need to convert to the L0 frame so that the
    # dependence on phase enters via the spherical harmonics.
    # hlm_fd___ = frame_utils.convert_J_to_L0_frame(
    #     hlm_fd__,
    #     parameters,
    #     self,
    #     spin_conversion_phase=self.spin_conversion_phase,
    # )
    # import pickle
    # d = {"hlm_fd_": hlm_fd_, "hlm_fd__": hlm_fd__, "hlm_fd___": hlm_fd___, "iota": iota}
    # with open("/tmp/fd_convert_J_to_L0.pickled", "wb") as f:
    #     pickle.dump(d, f)

    hlm_fd: SimInspiralChooseFDModes = SimInspiralChooseFDModes(*args)
    hlm_fd_: Dict[Mode, COMPLEX16FrequencySeries] = (
        wfg_utils.linked_list_modes_to_dict_modes(hlm_fd)
    )
    hlm_fd__: Dict[Mode, FrequencySeries] = {k: v.data.data for k, v in hlm_fd_.items()}
    hlm_fd___: Dict[Mode, FrequencySeries] = (
        inspiral_choose_fd_modes_parameters.convert_J_to_L0_frame(hlm_fd__)
    )
    assert inspiral_choose_fd_modes_parameters.iota == ground_truth_iota

    _same_frequency_series_dict(hlm_fd_, ground_truth_hlm_fd_)
    _same_dict(hlm_fd__, ground_truth_hlm_fd__)
    _same_dict(hlm_fd___, ground_truth_hlm_fd___)


def test_IMRPhenomXPHM_approximant() -> None:

    # This test replicates the executable part of the module dingo.gw.waveform_generator.waveform_generator,
    # which has been run and its "outputs" pickled to the file fd_IMRPhenomXPHM.pickled.
    # This test checks the results obtained with this refactored version of waveform_generator are identical.

    # reading the results obtained by the "original" dingo code, i.e. the outputs of
    # generate_hplus_hcross_m (pol_m)
    # sum_contributions_m (pol)
    # generate_hplus_hcross (pol_ref)
    ground_truth = _restore_ground_truth("fd_IMRPhenomXPHM.pickled")
    pol_m_ground_truth = cast(
        Dict[int, Dict[str, FrequencySeries]],
        ground_truth["pol_m"],
    )
    phase_shift = cast(float, ground_truth["phase_shift"])

    pol_ground_truth = cast(Dict[str, FrequencySeries], ground_truth["pol"])
    pol_ref_ground_truth = ground_truth["pol_ref"]

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

    waveform_parameters = WaveformParameters(**p)

    wfg = WaveformGenerator(
        approximant, domain, f_ref, f_start, spin_conversion_phase=spin_conversion_phase
    )

    pol_m: Dict[int, Polarization] = wfg.generate_hplus_hcross_m(waveform_parameters)
    pol: Polarization = sum_contributions_m(pol_m, phase_shift)
    # type ignore: we know phase is not None
    waveform_parameters.phase += phase_shift  # type: ignore
    pol_ref = wfg.generate_hplus_hcross(waveform_parameters)

    # checking our result is the same as the ground truth

    # checking pol_m
    assert len(pol_m) == len(pol_m_ground_truth)
    assert set(pol_m.keys()) == set(pol_m_ground_truth.keys())
    for mode in pol_m.keys():
        _same(pol_m[mode].h_cross, pol_m_ground_truth[mode]["h_cross"])
        _same(pol_m[mode].h_plus, pol_m_ground_truth[mode]["h_plus"])

    # checking pol
    _same(pol.h_cross, pol_ground_truth["h_cross"])
    _same(pol.h_plus, pol_ground_truth["h_plus"])
    _same(pol.h_cross.real, pol_ground_truth["h_cross"].real)
    _same(pol.h_plus.real, pol_ground_truth["h_plus"].real)

    # checking pol_ref
    _same(pol_ref.h_cross, pol_ref_ground_truth["h_cross"])
    _same(pol_ref.h_plus, pol_ref_ground_truth["h_plus"])
