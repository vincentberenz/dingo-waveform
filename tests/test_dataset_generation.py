"""Tests for dataset generation functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dingo_waveform.dataset import (
    DatasetSettings,
    WaveformDataset,
    generate_waveform_dataset,
)
from dingo_waveform.domains import FrequencyDomain


@pytest.fixture
def domain_config():
    """Fixture providing basic domain configuration."""
    return {
        "type": "FrequencyDomain",
        "f_min": 20.0,
        "f_max": 512.0,
        "delta_f": 0.125,
    }


@pytest.fixture
def waveform_generator_config():
    """Fixture providing waveform generator configuration."""
    return {
        "approximant": "IMRPhenomD",
        "f_ref": 20.0,
    }


@pytest.fixture
def intrinsic_prior_config():
    """Fixture providing intrinsic prior configuration."""
    # Use aligned spins (chi_1, chi_2) instead of generic spins for IMRPhenomD
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
def basic_settings(domain_config, waveform_generator_config, intrinsic_prior_config):
    """Fixture providing basic dataset settings."""
    return DatasetSettings(
        domain=domain_config,
        waveform_generator=waveform_generator_config,
        intrinsic_prior=intrinsic_prior_config,
        num_samples=5,
    )


class TestDatasetSettings:
    """Tests for DatasetSettings dataclass."""

    def test_creation(self, basic_settings):
        """Test basic creation of settings."""
        assert basic_settings.num_samples == 5
        assert isinstance(basic_settings.domain, dict)
        assert basic_settings.domain["type"] == "FrequencyDomain"

    def test_validation_success(self, basic_settings):
        """Test that validation passes for valid settings."""
        basic_settings.validate()  # Should not raise

    def test_validation_missing_domain(
        self, waveform_generator_config, intrinsic_prior_config
    ):
        """Test validation fails with missing domain type."""
        with pytest.raises(ValueError, match="type"):
            settings = DatasetSettings(
                domain={},  # Empty dict, no 'type' field
                waveform_generator=waveform_generator_config,
                intrinsic_prior=intrinsic_prior_config,
                num_samples=10,
            )
            settings.validate()

    def test_validation_invalid_num_samples(
        self, domain_config, waveform_generator_config, intrinsic_prior_config
    ):
        """Test validation fails with invalid num_samples."""
        with pytest.raises(ValueError, match="num_samples.*positive"):
            settings = DatasetSettings(
                domain=domain_config,
                waveform_generator=waveform_generator_config,
                intrinsic_prior=intrinsic_prior_config,
                num_samples=0,
            )
            settings.validate()

    def test_to_dict(self, basic_settings):
        """Test conversion to dictionary."""
        d = basic_settings.to_dict()
        assert isinstance(d, dict)
        assert d["num_samples"] == 5
        assert "domain" in d
        assert "waveform_generator" in d


class TestWaveformDataset:
    """Tests for WaveformDataset class."""

    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameter DataFrame."""
        return pd.DataFrame(
            {
                "mass_1": [30.0, 35.0, 40.0],
                "mass_2": [25.0, 30.0, 35.0],
                "phase": [0.0, 1.0, 2.0],
            }
        )

    @pytest.fixture
    def sample_polarizations(self):
        """Create sample polarization arrays."""
        return {
            "h_plus": np.random.randn(3, 100) + 1j * np.random.randn(3, 100),
            "h_cross": np.random.randn(3, 100) + 1j * np.random.randn(3, 100),
        }

    def test_creation(self, sample_parameters, sample_polarizations):
        """Test basic dataset creation."""
        dataset = WaveformDataset(sample_parameters, sample_polarizations)
        assert len(dataset) == 3
        assert dataset.parameters.shape == (3, 3)

    def test_validation_mismatch_h_plus(self, sample_parameters, sample_polarizations):
        """Test validation fails when h_plus length doesn't match parameters."""
        bad_polarizations = sample_polarizations.copy()
        bad_polarizations["h_plus"] = bad_polarizations["h_plus"][:2]  # Only 2 rows
        with pytest.raises(ValueError, match="Mismatch.*h_plus"):
            WaveformDataset(sample_parameters, bad_polarizations)

    def test_validation_mismatch_h_cross(self, sample_parameters, sample_polarizations):
        """Test validation fails when h_cross length doesn't match parameters."""
        bad_polarizations = sample_polarizations.copy()
        bad_polarizations["h_cross"] = bad_polarizations["h_cross"][:2]  # Only 2 rows
        with pytest.raises(ValueError, match="Mismatch.*h_cross"):
            WaveformDataset(sample_parameters, bad_polarizations)

    def test_repr(self, sample_parameters, sample_polarizations):
        """Test string representation."""
        dataset = WaveformDataset(sample_parameters, sample_polarizations)
        repr_str = repr(dataset)
        assert "num_waveforms=3" in repr_str
        assert "num_parameters=3" in repr_str
        assert "waveform_length=100" in repr_str

    def test_save_and_load(self, sample_parameters, sample_polarizations):
        """Test saving and loading dataset."""
        dataset = WaveformDataset(
            sample_parameters,
            sample_polarizations,
            settings={"domain": {"f_max": 512.0}},
        )

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            dataset.save(temp_path)
            assert temp_path.exists()

            # Load
            loaded = WaveformDataset.load(temp_path)
            assert len(loaded) == len(dataset)
            assert loaded.parameters.shape == dataset.parameters.shape
            assert np.allclose(loaded.polarizations["h_plus"], dataset.polarizations["h_plus"])
            assert np.allclose(loaded.polarizations["h_cross"], dataset.polarizations["h_cross"])
            assert loaded.settings == dataset.settings

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_with_none_settings(self, sample_parameters, sample_polarizations):
        """Test saving dataset with None values in settings."""
        dataset = WaveformDataset(
            sample_parameters,
            sample_polarizations,
            settings={"window_factor": None, "other": 42},
        )

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            temp_path = Path(f.name)

        try:
            dataset.save(temp_path)
            loaded = WaveformDataset.load(temp_path)
            assert loaded.settings["window_factor"] is None
            assert loaded.settings["other"] == 42

        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestGenerateWaveformDataset:
    """Tests for generate_waveform_dataset function."""

    def test_generate_sequential(self, basic_settings):
        """Test generating dataset with sequential execution."""
        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        assert isinstance(dataset, WaveformDataset)
        assert len(dataset) <= basic_settings.num_samples  # May be less if some fail
        assert "h_plus" in dataset.polarizations
        assert "h_cross" in dataset.polarizations
        assert dataset.polarizations["h_plus"].shape[0] == len(dataset)
        assert dataset.polarizations["h_cross"].shape[0] == len(dataset)

        # Check that waveforms are not all zeros (actual generation happened)
        # Use max absolute value since GW strains are very small (~ 1e-23)
        assert np.abs(dataset.polarizations["h_plus"]).max() > 0.0
        assert np.abs(dataset.polarizations["h_cross"]).max() > 0.0

    def test_generate_parallel(self, basic_settings):
        """Test generating dataset with parallel execution."""
        dataset = generate_waveform_dataset(basic_settings, num_processes=2)

        assert isinstance(dataset, WaveformDataset)
        assert len(dataset) <= basic_settings.num_samples
        assert "h_plus" in dataset.polarizations
        assert "h_cross" in dataset.polarizations

        # Check that waveforms are not all zeros (actual generation happened)
        assert np.abs(dataset.polarizations["h_plus"]).max() > 0.0
        assert np.abs(dataset.polarizations["h_cross"]).max() > 0.0

    def test_waveform_shapes(self, basic_settings):
        """Test that generated waveforms have correct shapes."""
        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        expected_length = int(
            basic_settings.domain["f_max"] / basic_settings.domain["delta_f"]
        ) + 1

        assert dataset.polarizations["h_plus"].shape == (len(dataset), expected_length)
        assert dataset.polarizations["h_cross"].shape == (len(dataset), expected_length)

    def test_parameter_columns(self, basic_settings):
        """Test that generated parameters have expected columns."""
        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        # Bilby sampling may add extra derived parameters (e.g., mass_1, mass_2)
        # Check for some core parameters that should always be present
        core_params = {'chirp_mass', 'mass_ratio', 'luminosity_distance', 'phase'}
        actual_params = set(dataset.parameters.columns)

        # Should at least have the core parameters
        assert core_params.issubset(actual_params)

    def test_settings_stored(self, basic_settings):
        """Test that settings are stored in the dataset."""
        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        assert dataset.settings is not None
        assert "domain" in dataset.settings
        assert "waveform_generator" in dataset.settings
        assert "num_samples" in dataset.settings

    def test_round_trip_save_load(self, basic_settings):
        """Test generating, saving, and loading a dataset."""
        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            temp_path = Path(f.name)

        try:
            dataset.save(temp_path)
            loaded = WaveformDataset.load(temp_path)

            assert len(loaded) == len(dataset)
            assert np.allclose(loaded.polarizations["h_plus"], dataset.polarizations["h_plus"])
            assert np.allclose(loaded.polarizations["h_cross"], dataset.polarizations["h_cross"])
            # Column order may differ after save/load, so check columns match
            pd.testing.assert_frame_equal(
                loaded.parameters.sort_index(axis=1),
                dataset.parameters.sort_index(axis=1)
            )

        finally:
            if temp_path.exists():
                temp_path.unlink()
