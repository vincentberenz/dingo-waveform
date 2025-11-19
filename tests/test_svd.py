"""Tests for SVD compression functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dingo_waveform.compression.svd import SVDBasis


class TestSVDBasis:
    """Test cases for SVDBasis class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample waveform data for testing."""
        np.random.seed(42)
        # Simulate 100 waveforms with 500 frequency bins
        n_samples = 100
        n_features = 500
        # Create data with some structure (not purely random)
        basis_vectors = np.random.randn(n_features, 10) + 1j * np.random.randn(n_features, 10)
        coefficients = np.random.randn(n_samples, 10)
        data = coefficients @ basis_vectors.T + 0.01 * (
            np.random.randn(n_samples, n_features) + 1j * np.random.randn(n_samples, n_features)
        )
        return data.astype(np.complex128)

    def test_generate_basis_scipy(self, sample_data):
        """Test SVD basis generation using scipy method."""
        basis = SVDBasis()
        basis.generate_basis(sample_data, n_components=10, method="scipy")

        assert basis.V is not None
        assert basis.Vh is not None
        assert basis.s is not None
        assert basis.n_components == 10
        assert basis.V.shape == (sample_data.shape[1], 10)
        assert basis.Vh.shape == (10, sample_data.shape[1])

    def test_generate_basis_all_components(self, sample_data):
        """Test SVD with n_components=0 (keep all)."""
        basis = SVDBasis()
        basis.generate_basis(sample_data, n_components=0, method="scipy")

        # Should keep min(n_samples, n_features) components
        expected_n = min(sample_data.shape)
        assert basis.n_components == expected_n

    def test_generate_basis_randomized(self, sample_data):
        """Test SVD basis generation using randomized method."""
        basis = SVDBasis()
        try:
            basis.generate_basis(sample_data, n_components=10, method="randomized")
            assert basis.n_components == 10
        except ValueError as e:
            # Randomized SVD may fail with complex data in newer scikit-learn
            if "complex" in str(e).lower():
                pytest.skip("Randomized SVD not supported for complex data in this scikit-learn version")
            else:
                raise

    def test_compress_decompress(self, sample_data):
        """Test compression and decompression roundtrip."""
        basis = SVDBasis()
        basis.generate_basis(sample_data[:50], n_components=20, method="scipy")

        # Test on validation data
        test_data = sample_data[50:60]
        compressed = basis.compress(test_data)
        reconstructed = basis.decompress(compressed)

        assert compressed.shape == (10, 20)
        assert reconstructed.shape == test_data.shape

        # Check reconstruction is reasonably accurate
        mismatch = np.linalg.norm(test_data - reconstructed) / np.linalg.norm(test_data)
        assert mismatch < 0.1  # Less than 10% error

    def test_compress_single_waveform(self, sample_data):
        """Test compression of single waveform."""
        basis = SVDBasis()
        basis.generate_basis(sample_data, n_components=10, method="scipy")

        single = sample_data[0]
        compressed = basis.compress(single)
        reconstructed = basis.decompress(compressed)

        assert compressed.shape == (10,)
        assert reconstructed.shape == single.shape

    def test_compute_mismatches(self, sample_data):
        """Test mismatch computation for validation."""
        basis = SVDBasis()
        basis.generate_basis(sample_data[:50], n_components=20, method="scipy")

        # Compute mismatches on validation set
        val_data = sample_data[50:70]
        val_params = pd.DataFrame({"mass_1": np.random.uniform(10, 50, 20)})

        results = basis.compute_mismatches(val_data, val_params)

        assert len(results) == 20
        assert "mismatch_n=20" in results.columns
        assert "mass_1" in results.columns
        assert all(results["mismatch_n=20"] >= 0)
        assert all(results["mismatch_n=20"] <= 1)

    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading SVD basis."""
        # Create and save basis
        basis1 = SVDBasis()
        basis1.generate_basis(sample_data, n_components=15, method="scipy")

        save_path = tmp_path / "test_basis.hdf5"
        basis1.save(save_path)

        assert save_path.exists()

        # Load basis
        basis2 = SVDBasis.load(save_path)

        assert basis2.n_components == 15
        assert np.allclose(basis2.V, basis1.V)
        assert np.allclose(basis2.s, basis1.s)

        # Test that loaded basis works
        test_data = sample_data[0]
        compressed1 = basis1.compress(test_data)
        compressed2 = basis2.compress(test_data)
        assert np.allclose(compressed1, compressed2)

    def test_to_dict_from_dict(self, sample_data):
        """Test dictionary serialization."""
        basis1 = SVDBasis()
        basis1.generate_basis(sample_data, n_components=10, method="scipy")

        # Convert to dict and back
        basis_dict = basis1.to_dict()
        basis2 = SVDBasis.from_dict(basis_dict)

        assert basis2.n_components == 10
        assert np.allclose(basis2.V, basis1.V)
        assert np.allclose(basis2.s, basis1.s)

    def test_error_compress_before_generate(self, sample_data):
        """Test error when compressing before generating basis."""
        basis = SVDBasis()

        with pytest.raises(ValueError, match="has not been generated"):
            basis.compress(sample_data[0])

    def test_error_decompress_before_generate(self):
        """Test error when decompressing before generating basis."""
        basis = SVDBasis()

        with pytest.raises(ValueError, match="has not been generated"):
            basis.decompress(np.random.randn(10))

    def test_error_save_before_generate(self, tmp_path):
        """Test error when saving before generating basis."""
        basis = SVDBasis()
        save_path = tmp_path / "test.hdf5"

        with pytest.raises(ValueError, match="has not been generated"):
            basis.save(save_path)

    def test_error_invalid_method(self, sample_data):
        """Test error with invalid SVD method."""
        basis = SVDBasis()

        with pytest.raises(ValueError, match="Invalid method"):
            basis.generate_basis(sample_data, n_components=10, method="invalid")

    def test_error_empty_data(self):
        """Test error with empty data."""
        basis = SVDBasis()

        with pytest.raises(ValueError, match="cannot be empty"):
            basis.generate_basis(np.array([]), n_components=10)

    def test_error_load_nonexistent_file(self, tmp_path):
        """Test error loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            SVDBasis.load(tmp_path / "nonexistent.hdf5")

    def test_from_dict_empty(self):
        """Test from_dict with empty dictionary."""
        basis = SVDBasis.from_dict({})
        assert basis.V is None
        assert basis.Vh is None
        assert basis.s is None


class TestSVDBasisIntegration:
    """Integration tests for SVD basis with realistic waveform data."""

    @pytest.fixture
    def realistic_waveforms(self):
        """Create realistic-looking waveform data."""
        np.random.seed(123)
        n_waveforms = 50
        n_freq_bins = 1000

        # Simulate h_plus and h_cross with some structure
        frequencies = np.linspace(20, 1024, n_freq_bins)
        waveforms = []

        for _ in range(n_waveforms):
            # Simple damped sinusoid model
            amplitude = np.random.uniform(1e-22, 1e-21)
            frequency_peak = np.random.uniform(100, 300)
            damping = np.random.uniform(0.01, 0.05)

            signal = amplitude * np.exp(-damping * frequencies) * np.exp(
                2j * np.pi * frequency_peak * frequencies / 1000
            )
            waveforms.append(signal)

        return np.array(waveforms)

    def test_high_compression_ratio(self, realistic_waveforms):
        """Test SVD with high compression ratio."""
        basis = SVDBasis()
        # Compress 1000 bins to 50 components (20x compression)
        basis.generate_basis(realistic_waveforms, n_components=50, method="scipy")

        compressed = basis.compress(realistic_waveforms)
        reconstructed = basis.decompress(compressed)

        assert compressed.shape == (50, 50)
        assert reconstructed.shape == realistic_waveforms.shape

        # With structured data, SVD should achieve good compression
        relative_error = np.linalg.norm(realistic_waveforms - reconstructed) / np.linalg.norm(
            realistic_waveforms
        )
        assert relative_error < 0.2  # Less than 20% relative error

    def test_concatenated_polarizations(self, realistic_waveforms):
        """Test SVD on concatenated h_plus and h_cross (as used in practice)."""
        # Simulate both polarizations
        h_plus = realistic_waveforms
        h_cross = realistic_waveforms * 0.8  # Cross polarization typically smaller

        # Concatenate as done in generate.py
        combined = np.concatenate([h_plus, h_cross], axis=0)

        basis = SVDBasis()
        basis.generate_basis(combined, n_components=30, method="scipy")

        # Compress and decompress
        compressed = basis.compress(combined)
        reconstructed = basis.decompress(compressed)

        assert compressed.shape[0] == 100  # 50 h_plus + 50 h_cross
        assert compressed.shape[1] == 30
        assert reconstructed.shape == combined.shape
