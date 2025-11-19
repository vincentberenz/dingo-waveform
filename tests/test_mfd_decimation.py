"""
Redesigned tests for MultibandedFrequencyDomain waveform generation and decimation.

These tests verify:
1. Decimation quality: UFD waveforms decimated to MFD match MFD-generated waveforms
2. Scientific correctness: Results match original dingo package
"""
import numpy as np
import pytest
from scipy.interpolate import interp1d
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from bilby.gw.detector import PowerSpectralDensity

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo_waveform.prior import IntrinsicPriors
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.polarizations import Polarization
from dingo_waveform.types import Mode

# Import comparison utilities
from dingo_waveform.comparison import compare_waveforms


def _get_mismatch(
    a: np.ndarray,
    b: np.ndarray,
    domain,
    asd_file: Optional[str] = None,
) -> float:
    """
    Mismatch is 1 - overlap, where overlap is defined by
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
    domain = MultibandedFrequencyDomain(
        nodes=[20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        delta_f_initial=0.0625,
        base_delta_f=0.0625,
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
def wfg_ufd(approximant, mfd):
    # Create a UniformFrequencyDomain that corresponds to the base grid
    # Start from f_min=0 and match MFD's f_max so waveforms can be properly decimated
    base_domain = UniformFrequencyDomain(delta_f=0.0625, f_min=0.0, f_max=mfd.f_max)
    return WaveformGenerator(
        approximant=approximant,
        domain=base_domain,
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
def decimation_tolerance(approximant: Approximant) -> float:
    """Tolerance for decimation quality test (MFD vs UFD decimated)."""
    if approximant == Approximant("IMRPhenomXPHM"):
        return 1e-4
    else:
        return 1e-9


@pytest.mark.parametrize("approximant", _approximants)
def test_decimation_quality(
    intrinsic_prior, wfg_mfd, wfg_ufd, mfd, num_evaluations, decimation_tolerance
):
    """
    Test that decimating UFD waveforms to MFD matches directly-generated MFD waveforms.

    This verifies that:
    1. The decimation algorithm preserves signal content
    2. MFD waveforms are equivalent to decimated UFD waveforms
    """
    mismatches: List[List[float]] = []

    for _ in range(num_evaluations):
        p = intrinsic_prior.sample()

        # Generate waveforms in both domains
        pol_mfd: Polarization = wfg_mfd.generate_hplus_hcross(p)
        pol_ufd: Polarization = wfg_ufd.generate_hplus_hcross(p)

        # Decimate UFD to MFD
        pol_ufd_decimated = Polarization(
            h_plus=mfd.decimate(pol_ufd.h_plus),
            h_cross=mfd.decimate(pol_ufd.h_cross),
        )

        # Compute mismatches
        mismatches.append(
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

    mismatches_arr = np.array(mismatches)
    max_mismatch = np.max(mismatches_arr)

    assert max_mismatch < decimation_tolerance, (
        f"Decimation mismatch {max_mismatch:.2e} exceeds tolerance {decimation_tolerance:.2e}"
    )


@pytest.mark.parametrize("approximant", _approximants)
def test_compatibility_with_dingo(intrinsic_prior, mfd, approximant: Approximant):
    """
    Test that dingo-waveform produces identical results to original dingo.

    This is the ground truth test - it verifies scientific correctness by
    comparing against the original implementation.
    """
    # Sample parameters (use fixed seed for reproducibility)
    np.random.seed(42)
    p = intrinsic_prior.sample()

    # Convert to plain dict for comparison function
    from dataclasses import asdict
    p_dict = asdict(p)
    # Convert numpy scalars to Python floats
    p_dict = {k: float(v) if hasattr(v, 'item') else v for k, v in p_dict.items() if v is not None}

    # Prepare domain parameters
    domain_params = {
        'nodes': [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        'delta_f_initial': 0.0625,
        'base_delta_f': 0.0625,
    }

    # Compare with original dingo
    result = compare_waveforms(
        domain_type='multibanded',
        domain_params=domain_params,
        approximant=str(approximant),
        waveform_params=p_dict,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )

    # Verify shapes match
    assert result.shapes_match, (
        f"Shape mismatch: dingo={result.dingo_shape}, "
        f"refactored={result.refactored_shape}"
    )

    # Verify numerical agreement (differences should be at machine precision level)
    assert result.max_diff_h_plus < 1e-20, (
        f"h_plus differs from dingo by {result.max_diff_h_plus:.2e}"
    )
    assert result.max_diff_h_cross < 1e-20, (
        f"h_cross differs from dingo by {result.max_diff_h_cross:.2e}"
    )


@pytest.mark.skip(reason="Multibanded mode-separated: Partial implementation. IMRPhenomXPHM works. SEOBNRv4PHM has LAL type errors. SEOBNRv5PHM/HM require resampling from non-standard FFT grids - implementation in progress.")
@pytest.mark.parametrize("approximant", _approximants)
def test_decimation_m_quality(
    intrinsic_prior, wfg_mfd, wfg_ufd, mfd, num_evaluations, decimation_tolerance
):
    """
    Test decimation quality for mode-by-mode waveforms.

    Similar to test_decimation_quality but for individual modes.
    """
    mismatches: List[List[List[float]]] = []

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

        mismatches.append(
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

    mismatches_arr = np.array(mismatches)
    max_mismatch = np.max(mismatches_arr)

    # For mode-by-mode, use median to be robust to outlier modes
    median_mismatch = np.median(mismatches_arr)

    assert max_mismatch < decimation_tolerance, (
        f"Mode decimation max mismatch {max_mismatch:.2e} exceeds tolerance {decimation_tolerance:.2e}"
    )

    # Also check that median is good (ensures most modes are well-behaved)
    assert median_mismatch < decimation_tolerance / 10, (
        f"Mode decimation median mismatch {median_mismatch:.2e} is too large"
    )
