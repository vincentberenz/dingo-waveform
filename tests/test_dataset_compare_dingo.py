"""
Tests comparing dataset generation between dingo-waveform and original dingo.

These tests verify that the new dingo-waveform dataset generation produces
identical results to the original dingo implementation.
"""

import numpy as np
import pytest

# Import from dingo-waveform
from dingo_waveform.dataset import DatasetSettings, generate_waveform_dataset

# Import from original dingo
try:
    from dingo.gw.waveform_generator import WaveformGenerator as DingoWaveformGenerator
    from dingo.gw.domains import build_domain as build_domain_dingo

    DINGO_AVAILABLE = True
except ImportError:
    DINGO_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not DINGO_AVAILABLE, reason="Original dingo package not installed"
)


@pytest.fixture
def basic_domain_config():
    """Basic frequency domain configuration."""
    return {
        "type": "UniformFrequencyDomain",
        "f_min": 20.0,
        "f_max": 512.0,
        "delta_f": 0.125,
    }


@pytest.fixture
def basic_waveform_config():
    """Basic waveform generator configuration."""
    return {
        "approximant": "IMRPhenomD",
        "f_ref": 20.0,
    }


@pytest.fixture
def aligned_spin_prior():
    """Intrinsic prior with aligned spins for IMRPhenomD."""
    return {
        "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=50.0)",
        "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=50.0)",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.2, maximum=1.0)",
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
        "luminosity_distance": 1000.0,
        "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "phase": "bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary='periodic')",
        "chi_1": "bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.88))",
        "chi_2": "bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.88))",
        "geocent_time": 0.0,
    }


@pytest.fixture
def generic_spin_prior():
    """Intrinsic prior with generic spins for precessing approximants."""
    return {
        "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=50.0)",
        "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=50.0)",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.2, maximum=1.0)",
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
        "luminosity_distance": 1000.0,
        "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "phase": "bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary='periodic')",
        "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.88)",
        "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.88)",
        "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "phi_12": "bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary='periodic')",
        "phi_jl": "bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary='periodic')",
        "geocent_time": 0.0,
    }


class TestCompareWithDingo:
    """Compare dingo-waveform dataset generation with original dingo."""

    def test_aligned_spin_imrphenomd(
        self, basic_domain_config, basic_waveform_config, aligned_spin_prior
    ):
        """Test IMRPhenomD with aligned spins matches original dingo."""
        # Generate with dingo-waveform
        settings_new = DatasetSettings(
            domain=basic_domain_config,
            waveform_generator=basic_waveform_config,
            intrinsic_prior=aligned_spin_prior,
            num_samples=3,
        )
        dataset_new = generate_waveform_dataset(settings_new, num_processes=1)

        # Generate with original dingo using same waveform generator settings
        domain_dingo = build_domain_dingo(basic_domain_config)
        wfg_dingo = DingoWaveformGenerator(
            approximant=basic_waveform_config["approximant"],
            domain=domain_dingo,
            f_ref=basic_waveform_config["f_ref"],
        )

        # Generate waveforms from same parameters
        polarizations_dingo = []
        for _, params_row in dataset_new.parameters.iterrows():
            params_dict = params_row.to_dict()
            try:
                h_dict = wfg_dingo.generate_hplus_hcross(params_dict)
                polarizations_dingo.append(h_dict)
            except Exception as e:
                pytest.skip(f"Original dingo failed to generate waveform: {e}")

        # Compare waveforms
        for i, h_dingo in enumerate(polarizations_dingo):
            h_plus_new = dataset_new.polarizations.h_plus[i]
            h_cross_new = dataset_new.polarizations.h_cross[i]

            h_plus_dingo = h_dingo["h_plus"]
            h_cross_dingo = h_dingo["h_cross"]

            # Check shapes match
            assert h_plus_new.shape == h_plus_dingo.shape
            assert h_cross_new.shape == h_cross_dingo.shape

            # Check values are very close (allowing for minor numerical differences)
            np.testing.assert_allclose(
                h_plus_new,
                h_plus_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_plus mismatch for waveform {i}",
            )
            np.testing.assert_allclose(
                h_cross_new,
                h_cross_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_cross mismatch for waveform {i}",
            )

    def test_precessing_imrphenompv2(
        self, basic_domain_config, generic_spin_prior
    ):
        """Test IMRPhenomPv2 with generic spins matches original dingo."""
        waveform_config = {
            "approximant": "IMRPhenomPv2",
            "f_ref": 20.0,
        }

        # Generate with dingo-waveform
        settings_new = DatasetSettings(
            domain=basic_domain_config,
            waveform_generator=waveform_config,
            intrinsic_prior=generic_spin_prior,
            num_samples=3,
        )
        dataset_new = generate_waveform_dataset(settings_new, num_processes=1)

        # Generate with original dingo
        domain_dingo = build_domain_dingo(basic_domain_config)
        wfg_dingo = DingoWaveformGenerator(
            approximant=waveform_config["approximant"],
            domain=domain_dingo,
            f_ref=waveform_config["f_ref"],
        )

        # Generate waveforms from same parameters
        polarizations_dingo = []
        for _, params_row in dataset_new.parameters.iterrows():
            params_dict = params_row.to_dict()
            try:
                h_dict = wfg_dingo.generate_hplus_hcross(params_dict)
                polarizations_dingo.append(h_dict)
            except Exception as e:
                pytest.skip(f"Original dingo failed to generate waveform: {e}")

        # Compare waveforms
        for i, h_dingo in enumerate(polarizations_dingo):
            h_plus_new = dataset_new.polarizations.h_plus[i]
            h_cross_new = dataset_new.polarizations.h_cross[i]

            h_plus_dingo = h_dingo["h_plus"]
            h_cross_dingo = h_dingo["h_cross"]

            # Check shapes match
            assert h_plus_new.shape == h_plus_dingo.shape
            assert h_cross_new.shape == h_cross_dingo.shape

            # Check values are very close
            np.testing.assert_allclose(
                h_plus_new,
                h_plus_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_plus mismatch for waveform {i}",
            )
            np.testing.assert_allclose(
                h_cross_new,
                h_cross_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_cross mismatch for waveform {i}",
            )

    def test_higher_frequency_domain(
        self, basic_waveform_config, aligned_spin_prior
    ):
        """Test with higher frequency range."""
        domain_config = {
            "type": "UniformFrequencyDomain",
            "f_min": 15.0,
            "f_max": 1024.0,
            "delta_f": 0.25,
        }

        # Generate with dingo-waveform
        settings_new = DatasetSettings(
            domain=domain_config,
            waveform_generator=basic_waveform_config,
            intrinsic_prior=aligned_spin_prior,
            num_samples=2,
        )
        dataset_new = generate_waveform_dataset(settings_new, num_processes=1)

        # Generate with original dingo
        domain_dingo = build_domain_dingo(domain_config)
        wfg_dingo = DingoWaveformGenerator(
            approximant=basic_waveform_config["approximant"],
            domain=domain_dingo,
            f_ref=basic_waveform_config["f_ref"],
        )

        # Generate waveforms from same parameters
        polarizations_dingo = []
        for _, params_row in dataset_new.parameters.iterrows():
            params_dict = params_row.to_dict()
            try:
                h_dict = wfg_dingo.generate_hplus_hcross(params_dict)
                polarizations_dingo.append(h_dict)
            except Exception as e:
                pytest.skip(f"Original dingo failed to generate waveform: {e}")

        # Compare waveforms
        for i, h_dingo in enumerate(polarizations_dingo):
            h_plus_new = dataset_new.polarizations.h_plus[i]
            h_cross_new = dataset_new.polarizations.h_cross[i]

            h_plus_dingo = h_dingo["h_plus"]
            h_cross_dingo = h_dingo["h_cross"]

            # Check shapes match
            assert h_plus_new.shape == h_plus_dingo.shape
            assert h_cross_new.shape == h_cross_dingo.shape

            # Check values are very close
            np.testing.assert_allclose(
                h_plus_new,
                h_plus_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_plus mismatch for waveform {i}",
            )
            np.testing.assert_allclose(
                h_cross_new,
                h_cross_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_cross mismatch for waveform {i}",
            )

    def test_different_reference_frequency(
        self, basic_domain_config, aligned_spin_prior
    ):
        """Test with different reference frequency."""
        waveform_config = {
            "approximant": "IMRPhenomD",
            "f_ref": 50.0,  # Different from usual 20 Hz
        }

        # Generate with dingo-waveform
        settings_new = DatasetSettings(
            domain=basic_domain_config,
            waveform_generator=waveform_config,
            intrinsic_prior=aligned_spin_prior,
            num_samples=2,
        )
        dataset_new = generate_waveform_dataset(settings_new, num_processes=1)

        # Generate with original dingo
        domain_dingo = build_domain_dingo(basic_domain_config)
        wfg_dingo = DingoWaveformGenerator(
            approximant=waveform_config["approximant"],
            domain=domain_dingo,
            f_ref=waveform_config["f_ref"],
        )

        # Generate waveforms from same parameters
        polarizations_dingo = []
        for _, params_row in dataset_new.parameters.iterrows():
            params_dict = params_row.to_dict()
            try:
                h_dict = wfg_dingo.generate_hplus_hcross(params_dict)
                polarizations_dingo.append(h_dict)
            except Exception as e:
                pytest.skip(f"Original dingo failed to generate waveform: {e}")

        # Compare waveforms
        for i, h_dingo in enumerate(polarizations_dingo):
            h_plus_new = dataset_new.polarizations.h_plus[i]
            h_cross_new = dataset_new.polarizations.h_cross[i]

            h_plus_dingo = h_dingo["h_plus"]
            h_cross_dingo = h_dingo["h_cross"]

            # Check shapes match
            assert h_plus_new.shape == h_plus_dingo.shape
            assert h_cross_new.shape == h_cross_dingo.shape

            # Check values are very close
            np.testing.assert_allclose(
                h_plus_new,
                h_plus_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_plus mismatch for waveform {i}",
            )
            np.testing.assert_allclose(
                h_cross_new,
                h_cross_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_cross mismatch for waveform {i}",
            )

    def test_extreme_mass_ratios(self, basic_domain_config, basic_waveform_config):
        """Test with extreme mass ratios."""
        prior_config = {
            "mass_1": "bilby.core.prior.Constraint(minimum=30.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=5.0, maximum=15.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=0.5)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=15.0, maximum=50.0)",
            "luminosity_distance": 500.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": "bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary='periodic')",
            "chi_1": "bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.99))",
            "chi_2": "bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.99))",
            "geocent_time": 0.0,
        }

        # Generate with dingo-waveform
        settings_new = DatasetSettings(
            domain=basic_domain_config,
            waveform_generator=basic_waveform_config,
            intrinsic_prior=prior_config,
            num_samples=2,
        )
        dataset_new = generate_waveform_dataset(settings_new, num_processes=1)

        # Generate with original dingo
        domain_dingo = build_domain_dingo(basic_domain_config)
        wfg_dingo = DingoWaveformGenerator(
            approximant=basic_waveform_config["approximant"],
            domain=domain_dingo,
            f_ref=basic_waveform_config["f_ref"],
        )

        # Generate waveforms from same parameters
        polarizations_dingo = []
        for _, params_row in dataset_new.parameters.iterrows():
            params_dict = params_row.to_dict()
            try:
                h_dict = wfg_dingo.generate_hplus_hcross(params_dict)
                polarizations_dingo.append(h_dict)
            except Exception as e:
                pytest.skip(f"Original dingo failed to generate waveform: {e}")

        # Compare waveforms
        for i, h_dingo in enumerate(polarizations_dingo):
            h_plus_new = dataset_new.polarizations.h_plus[i]
            h_cross_new = dataset_new.polarizations.h_cross[i]

            h_plus_dingo = h_dingo["h_plus"]
            h_cross_dingo = h_dingo["h_cross"]

            # Check shapes match
            assert h_plus_new.shape == h_plus_dingo.shape
            assert h_cross_new.shape == h_cross_dingo.shape

            # Check values are very close
            np.testing.assert_allclose(
                h_plus_new,
                h_plus_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_plus mismatch for waveform {i}",
            )
            np.testing.assert_allclose(
                h_cross_new,
                h_cross_dingo,
                rtol=1e-10,
                atol=1e-15,
                err_msg=f"h_cross mismatch for waveform {i}",
            )
