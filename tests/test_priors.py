from dataclasses import asdict, fields

import numpy as np
import pytest
from bilby.gw.prior import BBHPriorDict

from dingo_waveform.prior import (
    BBHExtrinsicPriorDict,
    ExtrinsicPriors,
    IntrinsicPriors,
    Priors,
    default_extrinsic_dict,
    prior_split,
)


def test_sample_and_split() -> None:

    extrinsic_priors = ExtrinsicPriors(geocent_time=0.05, psi=np.pi / 2)
    waveform_parameters = extrinsic_priors.sample()

    assert waveform_parameters.geocent_time == 0.05
    assert waveform_parameters.psi == np.pi / 2

    intrinsic_priors = IntrinsicPriors(mass_2=20.0, tilt_1=1.0)
    waveform_parameters = intrinsic_priors.sample()

    assert waveform_parameters.mass_2 == 20.0
    assert waveform_parameters.tilt_1 == 1.0

    priors = Priors(phi_12=0.1, luminosity_distance=500.0)

    waveform_parameters = priors.sample()

    assert waveform_parameters.phi_12 == 0.1
    assert waveform_parameters.tilt_1 == 1.0

    intrinsic_wf, extrinsic_wf = prior_split(waveform_parameters)

    assert intrinsic_wf.phi_12 == 0.1
    assert extrinsic_wf.tilt_1 == 1.0

    assert set([k for k, v in asdict(intrinsic_wf).items() if v is not None]) == set(
        fields(IntrinsicPriors)
    )

    assert set([k for k, v in asdict(extrinsic_wf).items() if v is not None]) == set(
        fields(ExtrinsicPriors)
    )


def test_prior_constraint():
    dict = {
        "mass_1": "bilby.core.prior.Uniform(minimum=10.0, maximum=80.0)",
        "mass_2": "bilby.core.prior.Uniform(minimum=10.0, maximum=80.0)",
        "mass_ratio": "bilby.core.prior.Constraint(minimum=0.125, maximum=1.0)",
    }
    prior = BBHPriorDict(dict)
    samples = prior.sample(1000)
    assert np.all(samples["mass_1"] > samples["mass_2"])


def test_mean_std():
    num_samples = 100000
    eps = 0.01
    keys = ["ra", "dec", "luminosity_distance"]
    prior = BBHExtrinsicPriorDict(default_extrinsic_dict)
    mean_exact, std_exact = prior.mean_std(keys)
    mean_approx, std_approx = prior.mean_std(
        keys, sample_size=num_samples, force_numerical=True
    )
    ratios_exact = np.array(list(mean_exact.values())) / np.array(
        list(std_exact.values())
    )
    ratios_approx = np.array(list(mean_approx.values())) / np.array(
        list(std_approx.values())
    )
    assert list(mean_exact.keys()) == keys
    assert np.allclose(ratios_exact, ratios_approx, atol=eps, rtol=eps)
