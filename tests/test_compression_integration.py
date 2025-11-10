"""Integration tests for compression in dataset generation."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from dingo_waveform.dataset import (
    CompressionSettings,
    DatasetSettings,
    SVDSettings,
    WaveformDataset,
    generate_waveform_dataset,
)
from dingo_waveform.dataset.waveform_generator_settings import WaveformGeneratorSettings
from dingo_waveform.domains import DomainParameters
from dingo_waveform.prior import IntrinsicPriors


@pytest.fixture
def basic_settings():
    """Create basic dataset settings without compression."""
    return DatasetSettings(
        domain=DomainParameters(
            type="UniformFrequencyDomain",
            f_min=20.0,
            f_max=512.0,
            delta_f=0.5,
        ),
        waveform_generator=WaveformGeneratorSettings(
            approximant="IMRPhenomXPHM",
            f_ref=20.0,
        ),
        intrinsic_prior=IntrinsicPriors(
            mass_1="bilby.core.prior.Constraint(minimum=35.0, maximum=40.0)",
            mass_2="bilby.core.prior.Constraint(minimum=30.0, maximum=35.0)",
            mass_ratio="bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            chirp_mass="bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            phase='bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            a_1="bilby.core.prior.Uniform(minimum=0.0, maximum=0.2)",
            a_2="bilby.core.prior.Uniform(minimum=0.0, maximum=0.2)",
            tilt_1="bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            tilt_2="bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            phi_12='bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            phi_jl='bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            theta_jn="bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            luminosity_distance="bilby.core.prior.Uniform(minimum=100.0, maximum=1000.0)",
        ),
        num_samples=5,
    )


class TestCompressionSettings:
    """Test CompressionSettings dataclass."""

    def test_svd_only(self):
        """Test compression with SVD only."""
        settings = CompressionSettings(
            svd=SVDSettings(
                size=50,
                num_training_samples=10,
                num_validation_samples=5,
            )
        )
        assert settings.svd.size == 50
        assert settings.whitening is None

    def test_whitening_only(self, tmp_path):
        """Test compression with whitening only."""
        asd_path = tmp_path / "asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=np.ones(100))

        settings = CompressionSettings(whitening=str(asd_path))
        assert settings.whitening == asd_path
        assert settings.svd is None

    def test_both_svd_and_whitening(self, tmp_path):
        """Test compression with both SVD and whitening."""
        asd_path = tmp_path / "asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=np.ones(100))

        settings = CompressionSettings(
            svd=SVDSettings(size=50, num_training_samples=10),
            whitening=str(asd_path),
        )
        assert settings.svd is not None
        assert settings.whitening is not None

    def test_error_no_compression(self):
        """Test error when neither SVD nor whitening specified."""
        with pytest.raises(ValueError, match="must specify at least one"):
            CompressionSettings()


@pytest.mark.slow
class TestDatasetGenerationWithCompression:
    """Test dataset generation with compression enabled."""

    def test_generate_with_svd_compression(self, basic_settings):
        """Test generating dataset with SVD compression."""
        # Add SVD compression
        basic_settings.compression = CompressionSettings(
            svd=SVDSettings(
                size=30,
                num_training_samples=20,  # Need at least 15 to get 30 components (15*2=30 total)
                num_validation_samples=5,
            )
        )
        basic_settings.num_samples = 5

        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        # Check dataset was created
        assert len(dataset) == 5
        assert dataset.svd_basis is not None
        assert dataset.svd_basis.n_components == 30

        # Check waveforms are compressed (smaller than original domain)
        from dingo_waveform.domains import build_domain
        domain = build_domain(basic_settings.domain)
        original_length = len(domain)
        compressed_length = dataset.polarizations.h_plus.shape[1]
        assert compressed_length == 30
        assert compressed_length < original_length

    def test_svd_basis_saved_and_loaded(self, basic_settings, tmp_path):
        """Test that SVD basis is saved and loaded correctly."""
        # Generate compressed dataset
        basic_settings.compression = CompressionSettings(
            svd=SVDSettings(size=25, num_training_samples=15)  # Need at least 13 for 25 components
        )
        basic_settings.num_samples = 5

        dataset1 = generate_waveform_dataset(basic_settings, num_processes=1)

        # Save dataset
        save_path = tmp_path / "compressed_dataset.hdf5"
        dataset1.save(save_path)

        # Load dataset
        dataset2 = WaveformDataset.load(save_path)

        # Check SVD basis was preserved
        assert dataset2.svd_basis is not None
        assert dataset2.svd_basis.n_components == 25
        assert np.allclose(dataset2.svd_basis.V, dataset1.svd_basis.V)

        # Check polarizations match
        assert np.allclose(dataset2.polarizations.h_plus, dataset1.polarizations.h_plus)

    def test_load_pretrained_svd(self, basic_settings, tmp_path):
        """Test loading pre-trained SVD basis from file."""
        # First, create and save an SVD basis
        from dingo_waveform.compression.svd import SVDBasis
        from dingo_waveform.domains import build_domain

        # Get the actual domain length
        domain = build_domain(basic_settings.domain)
        domain_length = len(domain)

        np.random.seed(42)
        # Need at least 40 training samples to get 40 components
        train_data = np.random.randn(50, domain_length) + 1j * np.random.randn(50, domain_length)
        basis = SVDBasis()
        basis.generate_basis(train_data, n_components=40, method="scipy")

        basis_path = tmp_path / "pretrained_basis.hdf5"
        basis.save(basis_path)

        # Generate dataset using pre-trained basis
        basic_settings.compression = CompressionSettings(
            svd=SVDSettings(
                size=40,  # Should be ignored when loading from file
                num_training_samples=1,  # Should be ignored
                file=basis_path,
            )
        )
        basic_settings.num_samples = 3

        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        # Check it used the pre-trained basis
        assert dataset.svd_basis is not None
        assert dataset.svd_basis.n_components == 40
        assert np.allclose(dataset.svd_basis.V, basis.V)

    def test_compression_reduces_file_size(self, basic_settings, tmp_path):
        """Test that compression reduces dataset file size."""
        # Generate uncompressed dataset
        basic_settings.num_samples = 20
        dataset_uncompressed = generate_waveform_dataset(basic_settings, num_processes=1)

        path_uncompressed = tmp_path / "uncompressed.hdf5"
        dataset_uncompressed.save(path_uncompressed)

        # Generate compressed dataset
        basic_settings.compression = CompressionSettings(
            svd=SVDSettings(size=20, num_training_samples=15)
        )
        dataset_compressed = generate_waveform_dataset(basic_settings, num_processes=1)

        path_compressed = tmp_path / "compressed.hdf5"
        dataset_compressed.save(path_compressed)

        # Compressed file should be smaller
        size_uncompressed = path_uncompressed.stat().st_size
        size_compressed = path_compressed.stat().st_size

        # With 20 components vs 985 bins, should see significant reduction
        assert size_compressed < size_uncompressed

    @pytest.mark.skip(reason="Requires ASD file")
    def test_generate_with_whitening(self, basic_settings, tmp_path):
        """Test generating dataset with whitening."""
        # Create dummy ASD file
        domain_length = len(basic_settings.domain)
        asd = np.ones(domain_length) * 1e-23

        asd_path = tmp_path / "asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=asd)

        # Add whitening
        basic_settings.compression = CompressionSettings(whitening=asd_path)
        basic_settings.num_samples = 3

        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        # Dataset should be generated (whitened waveforms)
        assert len(dataset) == 3

    @pytest.mark.skip(reason="Requires ASD file")
    def test_generate_with_whitening_and_svd(self, basic_settings, tmp_path):
        """Test generating dataset with both whitening and SVD."""
        # Create dummy ASD file
        domain_length = len(basic_settings.domain)
        asd = np.ones(domain_length) * 1e-23

        asd_path = tmp_path / "asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=asd)

        # Add both whitening and SVD
        basic_settings.compression = CompressionSettings(
            whitening=asd_path,
            svd=SVDSettings(size=25, num_training_samples=10),
        )
        basic_settings.num_samples = 3

        dataset = generate_waveform_dataset(basic_settings, num_processes=1)

        # Dataset should be whitened AND compressed
        assert len(dataset) == 3
        assert dataset.svd_basis is not None
        assert dataset.polarizations.h_plus.shape[1] == 25


class TestBackwardCompatibility:
    """Test backward compatibility with datasets without compression."""

    def test_load_old_dataset_without_svd(self, tmp_path):
        """Test loading dataset that doesn't have SVD basis."""
        # Create a simple dataset without SVD
        from dingo_waveform.polarizations import BatchPolarizations
        import pandas as pd

        parameters = pd.DataFrame({
            "mass_1": [35.0, 36.0],
            "mass_2": [30.0, 31.0],
        })
        polarizations = BatchPolarizations(
            h_plus=np.random.randn(2, 100) + 1j * np.random.randn(2, 100),
            h_cross=np.random.randn(2, 100) + 1j * np.random.randn(2, 100),
        )

        dataset = WaveformDataset(parameters, polarizations)

        # Save and load
        path = tmp_path / "old_dataset.hdf5"
        dataset.save(path)
        loaded = WaveformDataset.load(path)

        # Should load successfully with svd_basis=None
        assert loaded.svd_basis is None
        assert len(loaded) == 2
