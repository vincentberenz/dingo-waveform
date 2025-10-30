import numpy as np
import pytest
from scipy.interpolate import interp1d
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from bilby.gw.detector import PowerSpectralDensity

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import FrequencyDomain, MultibandedFrequencyDomain
from dingo_waveform.prior import IntrinsicPriors
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.polarizations import Polarization
from dingo_waveform.types import Mode


# Helper copied from tests/test_wfg_m.py

def _get_mismatch(
    a: np.ndarray,
    b: np.ndarray,
    domain,
    asd_file: Optional[str] = None,
) -> float:
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.
    """
    if asd_file is not None:
        # whiten a and b, such that we can use flat-spectrum inner products below
        psd = PowerSpectralDensity(asd_file=asd_file)
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        asd_array = asd_interp(domain())
        a = a / asd_array
        b = b / asd_array
    min_idx = domain.min_idx
    inner_ab = np.sum((np.conj(a) * b)[..., min_idx:], axis=-1).real
    inner_aa = np.sum((np.conj(a) * a)[..., min_idx:], axis=-1).real
    inner_bb = np.sum((np.conj(b) * b)[..., min_idx:], axis=-1).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap


_approximants = ("IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM", "SEOBNRv5HM")


@pytest.fixture
def mfd():
    base_domain = FrequencyDomain(delta_f=0.0625, f_min=20.0, f_max=1038.0)
    domain = MultibandedFrequencyDomain(
        nodes=[20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        delta_f_initial=0.0625,
        base_domain=base_domain,
    )
    return domain


@pytest.fixture(params=_approximants)
def approximant(request):
    return Approximant(request.param)


@pytest.fixture
def intrinsic_prior(approximant: Approximant):
    # Align with tests/test_wfg_m.get_intrinsic_prior
    intrinsic_dict: Dict[str, float | str]
    if "PHM" in approximant:
        intrinsic_dict = {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "geocent_time": 0.0,
        }
    else:
        intrinsic_dict = {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "chi_1": 'bilby.gw.prior.AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.99))',
            "chi_2": 'bilby.gw.prior.AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.99))',
            "geocent_time": 0.0,
        }
    return IntrinsicPriors(**intrinsic_dict)


@pytest.fixture
def wfg_mfd(mfd, approximant):
    return WaveformGenerator(
        approximant=approximant,
        domain=mfd,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )


@pytest.fixture
def wfg_ufd(mfd, approximant):
    return WaveformGenerator(
        approximant=approximant,
        domain=mfd.base_domain,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )


@pytest.fixture
def num_evaluations(approximant: Approximant):
    if approximant == Approximant("SEOBNRv4PHM"):
        return 1
    else:
        return 10


@pytest.fixture
def tolerances(approximant: Approximant) -> Tuple[float, float]:
    # Return max mismatches in MFD, UFD.
    if approximant == Approximant("IMRPhenomXPHM"):
        return 1e-4, 1e-3
    else:
        return 1e-9, 1e-3


@pytest.mark.parametrize("approximant", _approximants)
def test_decimation(
    intrinsic_prior, wfg_mfd, wfg_ufd, mfd, num_evaluations, tolerances
):
    mismatches_mfd: List[List[float]] = []
    mismatches_ufd: List[List[float]] = []
    for _ in range(num_evaluations):
        p = intrinsic_prior.sample()

        pol_mfd: Polarization = wfg_mfd.generate_hplus_hcross(p)
        pol_ufd: Polarization = wfg_ufd.generate_hplus_hcross(p)

        # Compare UFD waveforms decimated to MFD against waveforms generated in MFD.
        pol_ufd_decimated = Polarization(
            h_plus=mfd.decimate(pol_ufd.h_plus),
            h_cross=mfd.decimate(pol_ufd.h_cross),
        )

        mismatches_mfd.append(
            [
                _get_mismatch(
                    asdict(pol_mfd)[pol],
                    asdict(pol_ufd_decimated)[pol],
                    mfd,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                for pol in ["h_plus", "h_cross"]
            ]
        )

        # Also compare UFD waveforms against MFD waveforms interpolated to UFD.
        ufd: FrequencyDomain = mfd.base_domain
        pol_mfd_interpolated = Polarization(
            h_plus=ufd.update_data(
                interp1d(mfd(), pol_mfd.h_plus, fill_value="extrapolate")(ufd())
            ),
            h_cross=ufd.update_data(
                interp1d(mfd(), pol_mfd.h_cross, fill_value="extrapolate")(ufd())
            ),
        )

        mismatches_ufd.append(
            [
                _get_mismatch(
                    asdict(pol_ufd)[pol],
                    asdict(pol_mfd_interpolated)[pol],
                    ufd,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                for pol in ["h_plus", "h_cross"]
            ]
        )

    mismatches_mfd_arr = np.array(mismatches_mfd)
    mismatches_ufd_arr = np.array(mismatches_ufd)

    assert np.max(mismatches_mfd_arr) < tolerances[0]
    assert np.max(mismatches_ufd_arr) < tolerances[1]


@pytest.mark.parametrize("approximant", _approximants)
def test_decimation_m(
    intrinsic_prior, wfg_mfd, wfg_ufd, mfd, num_evaluations, tolerances
):
    mismatches_mfd: List[List[List[float]]] = []
    mismatches_ufd: List[List[List[float]]] = []
    for _ in range(num_evaluations):
        p = intrinsic_prior.sample()

        wf_mfd: Dict[Mode, Polarization] = wfg_mfd.generate_hplus_hcross_m(p)
        wf_ufd: Dict[Mode, Polarization] = wfg_ufd.generate_hplus_hcross_m(p)
        modes = list(wf_mfd.keys())

        # UFD decimated to MFD
        wf_ufd_decimated: Dict[Mode, Polarization] = {
            m: Polarization(
                h_plus=mfd.decimate(pol.h_plus),
                h_cross=mfd.decimate(pol.h_cross),
            )
            for m, pol in wf_ufd.items()
        }

        mismatches_mfd.append(
            [
                [
                    _get_mismatch(
                        wf_mfd[m].h_plus if pol == "h_plus" else wf_mfd[m].h_cross,
                        wf_ufd_decimated[m].h_plus
                        if pol == "h_plus"
                        else wf_ufd_decimated[m].h_cross,
                        mfd,
                        asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                    )
                    for pol in ["h_plus", "h_cross"]
                ]
                for m in modes
            ]
        )

        # MFD interpolated to UFD
        ufd = mfd.base_domain
        wf_mfd_interpolated: Dict[Mode, Polarization] = {
            m: Polarization(
                h_plus=ufd.update_data(
                    interp1d(mfd(), pol.h_plus, fill_value="extrapolate")(ufd())
                ),
                h_cross=ufd.update_data(
                    interp1d(mfd(), pol.h_cross, fill_value="extrapolate")(ufd())
                ),
            )
            for m, pol in wf_mfd.items()
        }

        mismatches_ufd.append(
            [
                [
                    _get_mismatch(
                        wf_ufd[m].h_plus if pol == "h_plus" else wf_ufd[m].h_cross,
                        wf_mfd_interpolated[m].h_plus
                        if pol == "h_plus"
                        else wf_mfd_interpolated[m].h_cross,
                        ufd,
                        asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                    )
                    for pol in ["h_plus", "h_cross"]
                ]
                for m in modes
            ]
        )

    mismatches_mfd_arr = np.array(mismatches_mfd)
    mismatches_ufd_arr = np.array(mismatches_ufd)

    assert np.max(mismatches_mfd_arr) < tolerances[0]

    # Some of the negative m modes do not do well, so we exclude by taking the median.
    assert np.median(mismatches_ufd_arr) < 10 * tolerances[1]
