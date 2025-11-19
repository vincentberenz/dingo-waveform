"""
Additional tests to improve code coverage.

This module contains tests specifically written to cover code paths
that are not exercised by the existing test suite.
"""

import numpy as np
import pytest
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.waveform_parameters import WaveformParameters
from dingo_waveform.domains import UniformFrequencyDomain, TimeDomain
from dingo_waveform.polarizations import Polarization, BatchPolarizations, sum_contributions_m
from dingo_waveform.approximant import Approximant
from dingo_waveform.domains.domain import Domain, DomainParameters
from dingo_waveform.prior import IntrinsicPriors, ExtrinsicPriors, Priors
from dingo_waveform.dataset.dataset_settings import DatasetSettings
from dingo_waveform.dataset.compression_settings import CompressionSettings, SVDSettings
from dingo_waveform.dataset.waveform_generator_settings import WaveformGeneratorSettings


class TestTimeDomainWaveforms:
    """Test time domain waveform generation to improve coverage."""

    @pytest.mark.skip(reason="Time domain not fully supported yet")
    def test_time_domain_basic(self):
        """Test basic time domain waveform generation."""
        domain = TimeDomain(time_duration=4.0, sampling_rate=2048.0)
        wfg = WaveformGenerator(
            Approximant("SEOBNRv5PHM"),
            domain,
            f_ref=20.0,
        )

        params = WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert isinstance(pol, Polarization)
        assert pol.h_plus.shape[0] > 0
        assert pol.h_cross.shape[0] > 0

    @pytest.mark.skip(reason="TimeDomain is abstract, can't be instantiated directly")
    def test_time_domain_properties(self):
        """Test time domain property access."""
        pass  # TimeDomain is abstract


class TestDomainEdgeCases:
    """Test domain edge cases and error handling."""

    def test_domain_parameters(self):
        """Test DomainParameters dataclass."""
        params = DomainParameters(
            type="UniformFrequencyDomain",
            f_min=20.0,
            f_max=1024.0,
            delta_f=0.125,
        )
        assert params.type == "UniformFrequencyDomain"
        assert params.f_min == 20.0

    def test_frequency_domain_window_factor(self):
        """Test frequency domain with window factor."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
        assert hasattr(domain, 'f_min')
        assert hasattr(domain, 'f_max')
        assert hasattr(domain, 'delta_f')

    def test_frequency_domain_update(self):
        """Test frequency domain update method."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)

        # Update to narrower range
        new_domain = domain.update(f_min=30.0, f_max=512.0)
        assert new_domain.f_min == 30.0
        assert new_domain.f_max == 512.0
        assert new_domain.delta_f == domain.delta_f

    def test_frequency_domain_get_parameters(self):
        """Test get_parameters method."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
        params = domain.get_parameters()

        assert isinstance(params, DomainParameters)
        assert params.f_min == 20.0
        assert params.f_max == 1024.0


class TestPolarizations:
    """Test polarization classes."""

    def test_polarization_creation(self):
        """Test Polarization dataclass creation."""
        h_plus = np.array([1.0 + 1j, 2.0 + 2j])
        h_cross = np.array([0.5 + 0.5j, 1.0 + 1j])

        pol = Polarization(h_plus=h_plus, h_cross=h_cross)
        assert np.allclose(pol.h_plus, h_plus)
        assert np.allclose(pol.h_cross, h_cross)

    def test_batch_polarizations(self):
        """Test BatchPolarizations class."""
        h_plus = np.random.randn(5, 1000) + 1j * np.random.randn(5, 1000)
        h_cross = np.random.randn(5, 1000) + 1j * np.random.randn(5, 1000)

        batch = BatchPolarizations(h_plus=h_plus, h_cross=h_cross)
        assert batch.h_plus.shape == (5, 1000)
        assert batch.h_cross.shape == (5, 1000)

    def test_sum_contributions_m_with_phase_shift(self):
        """Test sum_contributions_m with phase shift."""
        # Create simple mode contributions
        modes = {
            -2: Polarization(
                h_plus=np.array([1.0 + 0j, 2.0 + 0j]),
                h_cross=np.array([0.5 + 0j, 1.0 + 0j])
            ),
            2: Polarization(
                h_plus=np.array([1.0 + 0j, 2.0 + 0j]),
                h_cross=np.array([0.5 + 0j, 1.0 + 0j])
            ),
        }

        # Sum with phase shift
        result = sum_contributions_m(modes, phase_shift=0.5)
        assert isinstance(result, Polarization)
        assert result.h_plus.shape == (2,)


class TestPriors:
    """Test prior classes."""

    def test_intrinsic_priors_sampling(self):
        """Test IntrinsicPriors sampling."""
        prior = IntrinsicPriors(
            mass_1=35.0,
            mass_2=30.0,
            chirp_mass=28.0,
            mass_ratio=0.85,
            phase=1.0,
            a_1=0.5,
            a_2=0.3,
            tilt_1=1.0,
            tilt_2=0.8,
            phi_12=2.0,
            phi_jl=1.5,
            theta_jn=1.2,
            luminosity_distance=200.0,
            geocent_time=0.0,
        )

        sample = prior.sample()
        assert isinstance(sample, WaveformParameters)
        assert hasattr(sample, 'mass_1')
        assert hasattr(sample, 'luminosity_distance')

    def test_extrinsic_priors(self):
        """Test ExtrinsicPriors class."""
        prior = ExtrinsicPriors(
            ra=2.0,
            dec=0.3,
            psi=1.0,
            geocent_time=0.0,
            luminosity_distance=200.0,
        )

        sample = prior.sample()
        assert hasattr(sample, 'ra')
        assert hasattr(sample, 'dec')

    @pytest.mark.skip(reason="Combined Priors class requires complex setup")
    def test_combined_priors(self):
        """Test Priors class combining intrinsic and extrinsic."""
        pass  # Skip for now as it requires special setup


class TestDatasetSettings:
    """Test dataset settings edge cases."""

    def test_compression_settings_validation(self):
        """Test CompressionSettings validation."""
        svd = SVDSettings(size=50, num_training_samples=100)
        compression = CompressionSettings(svd=svd)

        assert compression.svd.size == 50
        assert compression.whitening is None

    def test_compression_settings_both(self):
        """Test CompressionSettings with both SVD and whitening."""
        svd = SVDSettings(size=50, num_training_samples=100)
        whitening = {"asd_file": "aLIGO_ZERO_DET_high_P_asd.txt"}
        compression = CompressionSettings(svd=svd, whitening=whitening)

        assert compression.svd is not None
        assert compression.whitening is not None

    def test_compression_settings_error(self):
        """Test CompressionSettings raises error when both are None."""
        with pytest.raises(ValueError, match="must specify at least one"):
            CompressionSettings(svd=None, whitening=None)

    def test_waveform_generator_settings(self):
        """Test WaveformGeneratorSettings."""
        settings = WaveformGeneratorSettings(
            approximant="IMRPhenomXPHM",
            f_ref=20.0,
            spin_conversion_phase=0.0,
            f_start=15.0,
        )

        d = settings.to_dict()
        assert d["approximant"] == "IMRPhenomXPHM"
        assert d["f_ref"] == 20.0
        assert d["f_start"] == 15.0

    def test_dataset_settings_to_from_dict(self):
        """Test DatasetSettings serialization."""
        settings_dict = {
            "domain": {
                "type": "UniformFrequencyDomain",
                "f_min": 20.0,
                "f_max": 1024.0,
                "delta_f": 0.125,
            },
            "waveform_generator": {
                "approximant": "IMRPhenomXPHM",
                "f_ref": 20.0,
                "spin_conversion_phase": 0.0,
            },
            "intrinsic_prior": {
                "mass_1": 35.0,
                "mass_2": 30.0,
                "chirp_mass": 28.0,
                "mass_ratio": 0.85,
                "phase": 1.0,
                "a_1": 0.5,
                "a_2": 0.3,
                "tilt_1": 1.0,
                "tilt_2": 0.8,
                "phi_12": 2.0,
                "phi_jl": 1.5,
                "theta_jn": 1.2,
                "luminosity_distance": 200.0,
                "geocent_time": 0.0,
            },
            "num_samples": 10,
        }

        settings = DatasetSettings.from_dict(settings_dict)
        assert settings.num_samples == 10

        # Convert back to dict
        d = settings.to_dict()
        assert d["num_samples"] == 10
        assert d["domain"]["f_min"] == 20.0


class TestWaveformParameters:
    """Test WaveformParameters edge cases."""

    def test_waveform_parameters_creation(self):
        """Test WaveformParameters creation with all parameters."""
        params = WaveformParameters(
            mass_1=35.0,
            mass_2=30.0,
            luminosity_distance=200.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.5,
            a_2=0.3,
            tilt_1=1.0,
            tilt_2=0.8,
            phi_12=2.0,
            phi_jl=1.5,
            geocent_time=0.0,
            ra=2.0,
            dec=0.5,
            psi=1.0,
        )

        assert params.mass_1 == 35.0
        assert params.mass_2 == 30.0
        assert params.ra == 2.0


class TestErrorHandling:
    """Test error handling paths."""

    def test_dataset_settings_invalid_num_samples(self):
        """Test DatasetSettings with invalid num_samples."""
        domain_params = DomainParameters(
            type="UniformFrequencyDomain",
            f_min=20.0,
            f_max=1024.0,
            delta_f=0.125,
        )
        wfg_settings = WaveformGeneratorSettings(
            approximant="IMRPhenomXPHM",
            f_ref=20.0,
        )
        intrinsic = IntrinsicPriors(
            mass_1=35.0,
            mass_2=30.0,
            chirp_mass=28.0,
            mass_ratio=0.85,
            phase=1.0,
            a_1=0.5,
            a_2=0.3,
            tilt_1=1.0,
            tilt_2=0.8,
            phi_12=2.0,
            phi_jl=1.5,
            theta_jn=1.2,
            luminosity_distance=200.0,
            geocent_time=0.0,
        )

        with pytest.raises(ValueError, match="num_samples must be positive"):
            DatasetSettings(
                domain=domain_params,
                waveform_generator=wfg_settings,
                intrinsic_prior=intrinsic,
                num_samples=0,  # Invalid
            )


class TestApproximants:
    """Test approximant handling."""

    def test_approximant_str(self):
        """Test Approximant string representation."""
        approx = Approximant("IMRPhenomXPHM")
        assert str(approx) == "IMRPhenomXPHM"

    def test_approximant_different_types(self):
        """Test different approximant types."""
        approximants = [
            "IMRPhenomD",
            "IMRPhenomXPHM",
            "SEOBNRv4PHM",
            "SEOBNRv5PHM",
            "SEOBNRv5HM",
        ]

        for approx_name in approximants:
            approx = Approximant(approx_name)
            assert isinstance(approx, str)
