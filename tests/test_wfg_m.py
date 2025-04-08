"""
This tests the method WaveformGenerator.generate_hplus_hcross_m, that returns the
polarzations disentangled into contributions m in [-l_max, ...,0, ...,l_max],
that transform as exp(-1j * m * phase_shift) under phase shifts. This is important when
treating the phase parameter as an extrinsic parameter.

Note: this only accounts for the modified argument in the spherical harmonics, not for
the rotation of phase_shift of the cartesian spins in xy plane. Our workaround is to
set wfg.spin_conversion_phase = 0.0, which sets a constant phase 0 when converting PE
spins to cartesian spins. This means that phi_12 and phi_jl have different definitions,
which needs to be accounted for in postprocessing. The tests below all use
wfg.spin_conversion_phase = 0.0.
"""

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pytest
from bilby.gw.detector import PowerSpectralDensity
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import FrequencyDomain
from dingo_waveform.polarizations import Polarization, sum_contributions_m
from dingo_waveform.prior import IntrinsicPriors
from dingo_waveform.types import FrequencySeries, Mode
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.waveform_parameters import WaveformParameters

_approximants = ("IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM", "SEOBNRv5HM")


def _get_mismatch(
    a: FrequencySeries,
    b: FrequencySeries,
    domain: FrequencyDomain,
    asd_file: Optional[str] = None,
) -> float:
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.

    Parameters
    ----------
    a
    b
    domain
    asd_file

    Returns
    -------
    The mismatch score

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
    inner_ab = np.sum((a.conj() * b)[..., min_idx:], axis=-1).real
    inner_aa = np.sum((a.conj() * a)[..., min_idx:], axis=-1).real
    inner_bb = np.sum((b.conj() * b)[..., min_idx:], axis=-1).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap


@pytest.fixture(params=_approximants)
def approximant(request):
    return request.param


def get_uniform_fd_domain() -> FrequencyDomain:
    return FrequencyDomain(
        delta_f=0.125,
        f_min=10.0,
        f_max=2048.0,
    )


def get_intrinsic_prior(approximant: Approximant) -> IntrinsicPriors:
    intrinsic_dict: Dict[str, Union[str, float]]
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
        # Aligned spins
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


def get_waveform_generator(approximant: Approximant) -> WaveformGenerator:
    uniform_fd_domain = get_uniform_fd_domain()
    return WaveformGenerator(
        approximant=approximant,
        domain=uniform_fd_domain,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )


def get_num_evaluations(approximant: Approximant) -> int:
    if "Phenom" in approximant:
        return 10
    elif approximant == "SEOBNRv4PHM":
        return 1
    else:
        return 10


def get_tolerances(approximant: Approximant) -> Tuple[float, float]:
    # Return (max, median) mismatches expected.
    if approximant == Approximant("IMRPhenomXPHM"):
        # The mismatches are typically be of order 1e-5 to 1e-9. This comes from the
        # calculation of the magnitude of the orbital angular momentum, which we calculate
        # to a different order the IMRPhenomXPHM. It's tricky to get this exactly right,
        # since there are many different methods for this. But the small mismatches we do
        # get should not have a big effect in practice.
        return 2e-2, 1e-5

    elif approximant == Approximant("SEOBNRv4PHM"):
        # The mismatches are typically be of order 1e-5. This is exclusively due to
        # different tapering. The reference polarizations are tapered and FFTed on the
        # level of polarizations, while for generate_hplus_hcross_m, the tapering and FFT
        # happens on the level of complex modes.
        # We tested the mismatches for 20k waveforms, and the largest mismatch encountered
        # was 7e-4, while almost all mismatches were of order 1e-5.
        return 5e-4, 5e-4

    elif approximant in (Approximant("SEOBNRv5PHM"), Approximant("SEOBNRv5HM")):
        # Tested on 1000 mismatches.
        return 1e-9, 1e-12

    else:
        return 1e-5, 1e-5


@pytest.mark.parametrize("approximant", _approximants)
def test_generate_hplus_hcross_m(approximant) -> None:
    intrinsic_prior = get_intrinsic_prior(approximant)
    wfg = get_waveform_generator(approximant)
    num_evaluations = get_num_evaluations(approximant)
    tolerances = get_tolerances(approximant)

    mismatches: List[List[float]] = []
    for idx in range(num_evaluations):
        p: WaveformParameters = intrinsic_prior.sample()
        phase_shift = np.random.uniform(high=2 * np.pi)

        pol_m: Dict[Mode, Polarization] = wfg.generate_hplus_hcross_m(p)
        pol: Polarization = sum_contributions_m(pol_m, phase_shift=phase_shift)
        if p.phase is None:
            raise RuntimeError(
                "test_generate_hplus_hcross_m requires a non None phase parameter"
            )
        p.phase += phase_shift
        pol_ref: Polarization = wfg.generate_hplus_hcross(p)

        domain = cast(FrequencyDomain, wfg._waveform_gen_params.domain)

        mismatches.append(
            [
                _get_mismatch(
                    asdict(pol)[pol_name],
                    asdict(pol_ref)[pol_name],
                    domain,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                for pol_name in ("h_plus", "h_cross")
            ]
        )

    mismatches_ = np.array(mismatches)

    assert np.max(mismatches_) < tolerances[0]
    assert np.median(mismatches_) < tolerances[1]
