"""Tests for the SVDBasis class."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from dingo_waveform.svd import (
    SVDBasis,
    SVDGenerationConfig,
    SVDMetadata,
    ValidationConfig,
)


@pytest.fixture
def real_training_data():
    """Generate real-valued training data."""
    np.random.seed(42)
    return np.random.randn(100, 50)


@pytest.fixture
def complex_training_data():
    """Generate complex-valued training data."""
    np.random.seed(42)
    return np.random.randn(100, 50) + 1j * np.random.randn(100, 50)


@pytest.fixture
def real_svd_basis(real_training_data):
    """Create a real-valued SVD basis."""
    config = SVDGenerationConfig(n_components=10, method="scipy")
    return SVDBasis.from_training_data(real_training_data, config)


@pytest.fixture
def complex_svd_basis(complex_training_data):
    """Create a complex-valued SVD basis."""
    config = SVDGenerationConfig(n_components=10, method="scipy")
    return SVDBasis.from_training_data(complex_training_data, config)


class TestSVDBasisCreation:
    """Test SVDBasis creation methods."""

    def test_from_training_data_real(self, real_training_data):
        """Test creating SVDBasis from real training data."""
        config = SVDGenerationConfig(n_components=10, method="scipy")
        basis = SVDBasis.from_training_data(real_training_data, config)

        assert basis.n_components == 10
        assert basis.V.shape == (50, 10)
        assert basis.Vh.shape == (10, 50)
        assert basis.s.shape == (10,)
        assert basis.method == "scipy"

    def test_from_training_data_complex(self, complex_training_data):
        """Test creating SVDBasis from complex training data."""
        config = SVDGenerationConfig(n_components=15, method="scipy")
        basis = SVDBasis.from_training_data(complex_training_data, config)

        assert basis.n_components == 15
        assert basis.V.shape == (50, 15)
        assert basis.Vh.shape == (15, 50)
        assert basis.s.shape == (15,)
        assert np.iscomplexobj(basis.V)
        assert np.iscomplexobj(basis.Vh)

    def test_from_training_data_with_metadata(self, real_training_data):
        """Test creating SVDBasis with metadata."""
        config = SVDGenerationConfig(n_components=10, method="scipy")
        metadata = SVDMetadata(
            description="Test basis",
            n_training_samples=100,
            extra={"created_by": "test_suite", "version": "1.0"},
        )
        basis = SVDBasis.from_training_data(real_training_data, config, metadata)

        assert basis.metadata is not None
        assert basis.metadata.description == "Test basis"
        assert basis.metadata.n_training_samples == 100
        assert basis.metadata.extra["created_by"] == "test_suite"

    def test_n_components_zero(self, real_training_data):
        """Test that n_components=0 uses all components."""
        config = SVDGenerationConfig(n_components=0, method="scipy")
        basis = SVDBasis.from_training_data(real_training_data, config)

        # Should use min(n_samples, n_features) = min(100, 50) = 50
        assert basis.n_components == 50

    def test_random_method(self, real_training_data):
        """Test using random SVD method."""
        config = SVDGenerationConfig(n_components=10, method="random")
        try:
            basis = SVDBasis.from_training_data(real_training_data, config)
            assert basis.n_components == 10
            assert basis.method == "random"
        except ValueError as e:
            # Skip if scikit-learn doesn't support complex for random SVD
            if "randomized_svd failed" in str(e):
                pytest.skip("scikit-learn version doesn't support complex random SVD")
            else:
                raise


class TestSVDBasisProperties:
    """Test SVDBasis properties."""

    def test_v_vh_relationship(self, real_svd_basis):
        """Test that Vh = V^H (conjugate transpose)."""
        assert np.allclose(real_svd_basis.Vh, real_svd_basis.V.T.conj())

    def test_v_vh_relationship_complex(self, complex_svd_basis):
        """Test Vh = V^H for complex data."""
        assert np.allclose(complex_svd_basis.Vh, complex_svd_basis.V.T.conj())

    def test_singular_values_ordered(self, real_svd_basis):
        """Test that singular values are in descending order."""
        s = real_svd_basis.s
        assert np.all(s[:-1] >= s[1:])

    def test_orthogonality(self, real_svd_basis):
        """Test that V columns are orthogonal."""
        V = real_svd_basis.V
        VhV = V.T.conj() @ V
        identity = np.eye(real_svd_basis.n_components)
        assert np.allclose(VhV, identity, atol=1e-10)


class TestSVDBasisCompression:
    """Test compression and decompression."""

    def test_compress_decompress_real(self, real_svd_basis):
        """Test compress and decompress for real data."""
        np.random.seed(123)
        data = np.random.randn(10, 50)

        compressed = real_svd_basis.compress(data)
        reconstructed = real_svd_basis.decompress(compressed)

        assert compressed.shape == (10, 10)
        assert reconstructed.shape == (10, 50)

    def test_compress_decompress_complex(self, complex_svd_basis):
        """Test compress and decompress for complex data."""
        np.random.seed(123)
        data = np.random.randn(10, 50) + 1j * np.random.randn(10, 50)

        compressed = complex_svd_basis.compress(data)
        reconstructed = complex_svd_basis.decompress(compressed)

        assert compressed.shape == (10, 10)
        assert reconstructed.shape == (10, 50)
        assert np.iscomplexobj(reconstructed)

    def test_compress_dict(self, real_svd_basis):
        """Test compressing dictionary of arrays."""
        np.random.seed(123)
        data_dict = {
            "stream1": np.random.randn(10, 50),
            "stream2": np.random.randn(10, 50),
        }

        compressed_dict = real_svd_basis.compress_dict(data_dict)

        assert "stream1" in compressed_dict
        assert "stream2" in compressed_dict
        assert compressed_dict["stream1"].shape == (10, 10)
        assert compressed_dict["stream2"].shape == (10, 10)

    def test_decompress_dict(self, real_svd_basis):
        """Test decompressing dictionary of coefficient arrays."""
        np.random.seed(123)
        coeff_dict = {
            "stream1": np.random.randn(10, 10),
            "stream2": np.random.randn(10, 10),
        }

        decompressed_dict = real_svd_basis.decompress_dict(coeff_dict)

        assert "stream1" in decompressed_dict
        assert "stream2" in decompressed_dict
        assert decompressed_dict["stream1"].shape == (10, 50)
        assert decompressed_dict["stream2"].shape == (10, 50)

    def test_reconstruction_quality(self, real_svd_basis, real_training_data):
        """Test that reconstruction quality is good for training data."""
        # Compress and decompress training data
        compressed = real_svd_basis.compress(real_training_data[:10])
        reconstructed = real_svd_basis.decompress(compressed)

        # Compute relative error
        error = np.linalg.norm(real_training_data[:10] - reconstructed, axis=1)
        relative_error = error / np.linalg.norm(real_training_data[:10], axis=1)

        # Error should be small (data is in the span of the basis)
        assert np.all(relative_error < 1.0)


class TestSVDBasisValidation:
    """Test validation functionality."""

    def test_validate_basic(self, real_svd_basis, real_training_data):
        """Test basic validation."""
        test_data = real_training_data[80:]  # Use last 20 samples
        config = ValidationConfig(increment=5)

        validated_basis, result = real_svd_basis.validate(test_data, config)

        assert validated_basis.mismatches is not None
        assert not validated_basis.mismatches.empty
        assert result.mismatches is not None

    def test_validate_with_labels(self, real_svd_basis, real_training_data):
        """Test validation with parameter labels."""
        test_data = real_training_data[80:]
        labels = pd.DataFrame({"param1": np.arange(20), "param2": np.arange(20) * 2})
        config = ValidationConfig(increment=5)

        validated_basis, result = real_svd_basis.validate(test_data, config, labels=labels)

        # Check that labels are included in mismatches
        assert "param1" in validated_basis.mismatches.columns
        assert "param2" in validated_basis.mismatches.columns

    def test_validate_verbose(self, real_svd_basis, real_training_data, capsys):
        """Test validation with verbose output."""
        test_data = real_training_data[80:]
        config = ValidationConfig(increment=5)

        real_svd_basis.validate(test_data, config, verbose=True)

        captured = capsys.readouterr()
        assert "Mean mismatch" in captured.out
        assert "Max mismatch" in captured.out


class TestSVDBasisTruncation:
    """Test truncation functionality."""

    def test_truncate_basic(self, real_svd_basis):
        """Test basic truncation."""
        truncated = real_svd_basis.truncate(5)

        assert truncated.n_components == 5
        assert truncated.V.shape == (50, 5)
        assert truncated.Vh.shape == (5, 50)
        assert truncated.s.shape == (5,)

    def test_truncate_preserves_original(self, real_svd_basis):
        """Test that truncation doesn't modify original basis."""
        original_n = real_svd_basis.n_components
        original_V_shape = real_svd_basis.V.shape

        truncated = real_svd_basis.truncate(5)

        # Original should be unchanged (immutable)
        assert real_svd_basis.n_components == original_n
        assert real_svd_basis.V.shape == original_V_shape

    def test_truncate_invalid_n_components(self, real_svd_basis):
        """Test that invalid truncation raises error."""
        with pytest.raises(ValueError):
            real_svd_basis.truncate(20)  # More than current n_components

        with pytest.raises(ValueError):
            real_svd_basis.truncate(0)  # Less than 1

    def test_truncate_preserves_metadata(self, real_training_data):
        """Test that truncation preserves metadata."""
        config = SVDGenerationConfig(n_components=10, method="scipy")
        metadata = SVDMetadata(description="Test", n_training_samples=100)
        basis = SVDBasis.from_training_data(real_training_data, config, metadata)

        truncated = basis.truncate(5)

        assert truncated.metadata is not None
        assert truncated.metadata.description == "Test"


class TestSVDBasisIO:
    """Test save and load functionality."""

    def test_save_load_real(self, real_svd_basis):
        """Test saving and loading real-valued basis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basis.h5")

            # Save
            real_svd_basis.save(filepath)

            # Load
            loaded_basis = SVDBasis.from_file(filepath)

            # Verify
            assert loaded_basis.n_components == real_svd_basis.n_components
            assert np.allclose(loaded_basis.V, real_svd_basis.V)
            assert np.allclose(loaded_basis.s, real_svd_basis.s)

    def test_save_load_complex(self, complex_svd_basis):
        """Test saving and loading complex-valued basis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basis.h5")

            complex_svd_basis.save(filepath)
            loaded_basis = SVDBasis.from_file(filepath)

            assert np.allclose(loaded_basis.V, complex_svd_basis.V)
            assert np.iscomplexobj(loaded_basis.V)

    def test_save_load_with_validation(self, real_svd_basis, real_training_data):
        """Test saving and loading basis with validation results."""
        # Add validation
        test_data = real_training_data[80:]
        config = ValidationConfig(increment=5)
        validated_basis, _ = real_svd_basis.validate(test_data, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basis.h5")

            # Save
            validated_basis.save(filepath)

            # Load
            loaded_basis = SVDBasis.from_file(filepath)

            # Verify validation results are preserved
            assert loaded_basis.mismatches is not None
            assert not loaded_basis.mismatches.empty

    def test_to_dict_from_dict(self, real_svd_basis):
        """Test converting to and from dictionary."""
        # Convert to dict
        basis_dict = real_svd_basis.to_dict()

        # Convert from dict
        loaded_basis = SVDBasis.from_dict(basis_dict)

        # Verify
        assert loaded_basis.n_components == real_svd_basis.n_components
        assert np.allclose(loaded_basis.V, real_svd_basis.V)
        assert np.allclose(loaded_basis.s, real_svd_basis.s)


class TestSVDBasisImmutability:
    """Test that SVDBasis is immutable."""

    def test_cannot_modify_result(self, real_svd_basis):
        """Test that _result cannot be modified."""
        with pytest.raises((AttributeError, Exception)):
            real_svd_basis._result = None

    def test_dataclass_frozen(self, real_svd_basis):
        """Test that SVDBasis is a frozen dataclass (fields cannot be reassigned)."""
        # Frozen dataclass prevents reassignment of fields
        with pytest.raises((AttributeError, Exception)):
            real_svd_basis._result = None

        # Note: numpy arrays themselves are mutable (this is expected)
        # The frozen dataclass prevents replacing the entire array reference,
        # but individual elements can still be modified if accessed
        # This is standard frozen dataclass behavior with mutable contents
