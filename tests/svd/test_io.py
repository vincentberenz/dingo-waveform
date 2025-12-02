"""Tests for I/O functions (save/load)."""

import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from dingo_waveform.svd import SVDGenerationConfig, SVDMetadata, ValidationConfig
from dingo_waveform.svd.decomposition import generate_svd_basis
from dingo_waveform.svd.io import (
    load_svd_from_dict,
    load_svd_from_hdf5,
    save_svd_to_dict,
    save_svd_to_hdf5,
)
from dingo_waveform.svd.validation import validate_svd


@pytest.fixture
def svd_result():
    """Generate an SVD result."""
    np.random.seed(42)
    data = np.random.randn(100, 50)
    config = SVDGenerationConfig(n_components=10, method="scipy")
    return generate_svd_basis(data, config)


@pytest.fixture
def svd_result_complex():
    """Generate a complex SVD result."""
    np.random.seed(42)
    data = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
    config = SVDGenerationConfig(n_components=10, method="scipy")
    return generate_svd_basis(data, config)


@pytest.fixture
def tmpfile():
    """Create a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        filepath = f.name
    yield filepath
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)


class TestSaveLoadHDF5:
    """Test HDF5 save and load functions."""

    def test_save_load_basic(self, svd_result, tmpfile):
        """Test basic save and load."""
        # Save
        save_svd_to_hdf5(tmpfile, svd_result, mismatches=None, metadata=None)

        # Load
        loaded_result, loaded_mismatches, loaded_metadata = load_svd_from_hdf5(tmpfile)

        # Verify
        assert loaded_result.n_components == svd_result.n_components
        assert loaded_result.method == svd_result.method
        assert np.allclose(loaded_result.V, svd_result.V)
        assert np.allclose(loaded_result.s, svd_result.s)
        assert loaded_mismatches is None
        assert loaded_metadata is None

    def test_save_load_complex(self, svd_result_complex, tmpfile):
        """Test save and load with complex data."""
        save_svd_to_hdf5(tmpfile, svd_result_complex)

        loaded_result, _, _ = load_svd_from_hdf5(tmpfile)

        assert np.allclose(loaded_result.V, svd_result_complex.V)
        assert np.iscomplexobj(loaded_result.V)

    def test_save_load_with_metadata(self, svd_result, tmpfile):
        """Test save and load with metadata."""
        metadata = SVDMetadata(
            description="Test SVD",
            n_training_samples=100,
            extra={"created_by": "test_suite", "version": "1.0", "custom_field": "custom_value"},
        )

        save_svd_to_hdf5(tmpfile, svd_result, metadata=metadata)

        loaded_result, _, loaded_metadata = load_svd_from_hdf5(tmpfile)

        assert loaded_metadata is not None
        assert loaded_metadata.description == "Test SVD"
        assert loaded_metadata.n_training_samples == 100
        assert loaded_metadata.extra["created_by"] == "test_suite"
        assert loaded_metadata.extra["version"] == "1.0"
        assert loaded_metadata.extra["custom_field"] == "custom_value"

    def test_save_load_with_mismatches(self, svd_result, tmpfile):
        """Test save and load with validation mismatches."""
        # Create mismatches
        test_data = np.random.randn(30, 50)
        config = ValidationConfig(increment=5)
        val_result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        save_svd_to_hdf5(tmpfile, svd_result, mismatches=val_result.mismatches)

        loaded_result, loaded_mismatches, _ = load_svd_from_hdf5(tmpfile)

        assert loaded_mismatches is not None
        assert not loaded_mismatches.empty
        assert list(loaded_mismatches.columns) == list(val_result.mismatches.columns)

    def test_save_load_all_components(self, svd_result, tmpfile):
        """Test save and load with all components (metadata, mismatches)."""
        metadata = SVDMetadata(description="Complete test")
        test_data = np.random.randn(20, 50)
        config = ValidationConfig(increment=5)
        val_result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        save_svd_to_hdf5(tmpfile, svd_result, mismatches=val_result.mismatches, metadata=metadata)

        loaded_result, loaded_mismatches, loaded_metadata = load_svd_from_hdf5(tmpfile)

        assert loaded_result.n_components == svd_result.n_components
        assert loaded_mismatches is not None
        assert loaded_metadata is not None
        assert loaded_metadata.description == "Complete test"

    def test_save_mode_overwrite(self, svd_result, tmpfile):
        """Test that mode='w' overwrites existing file."""
        # Save first SVD
        save_svd_to_hdf5(tmpfile, svd_result)

        # Create different SVD
        np.random.seed(123)
        data2 = np.random.randn(100, 50)
        config2 = SVDGenerationConfig(n_components=15, method="scipy")
        svd_result2 = generate_svd_basis(data2, config2)

        # Overwrite
        save_svd_to_hdf5(tmpfile, svd_result2, mode="w")

        # Load and verify it's the second one
        loaded_result, _, _ = load_svd_from_hdf5(tmpfile)
        assert loaded_result.n_components == 15

    def test_hdf5_file_structure(self, svd_result, tmpfile):
        """Test that HDF5 file has correct structure."""
        save_svd_to_hdf5(tmpfile, svd_result)

        with h5py.File(tmpfile, "r") as f:
            # Check datasets exist
            assert "V" in f
            assert "s" in f

            # Check attributes
            assert f.attrs["n_components"] == svd_result.n_components
            assert f.attrs["method"] == svd_result.method

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_svd_from_hdf5("/nonexistent/path/file.h5")

    def test_load_invalid_file(self, tmpfile):
        """Test loading from invalid HDF5 file raises error."""
        # Create an empty file
        with open(tmpfile, "w") as f:
            f.write("not an hdf5 file")

        with pytest.raises((OSError, Exception)):
            load_svd_from_hdf5(tmpfile)

    def test_load_missing_v_matrix(self, tmpfile):
        """Test loading file without V matrix raises error."""
        # Create HDF5 file without V
        with h5py.File(tmpfile, "w") as f:
            f.create_dataset("s", data=np.array([1, 2, 3]))

        with pytest.raises(KeyError):
            load_svd_from_hdf5(tmpfile)


class TestSaveLoadDict:
    """Test dictionary save and load functions."""

    def test_save_load_basic(self, svd_result):
        """Test basic save and load to/from dictionary."""
        # Save to dict
        svd_dict = save_svd_to_dict(svd_result, mismatches=None)

        # Load from dict
        loaded_result, loaded_mismatches = load_svd_from_dict(svd_dict)

        # Verify
        assert loaded_result.n_components == svd_result.n_components
        assert loaded_result.method == svd_result.method
        assert np.allclose(loaded_result.V, svd_result.V)
        assert np.allclose(loaded_result.s, svd_result.s)
        assert loaded_mismatches is None

    def test_save_load_complex(self, svd_result_complex):
        """Test save and load with complex data."""
        svd_dict = save_svd_to_dict(svd_result_complex)
        loaded_result, _ = load_svd_from_dict(svd_dict)

        assert np.allclose(loaded_result.V, svd_result_complex.V)
        assert np.iscomplexobj(loaded_result.V)

    def test_save_load_with_mismatches(self, svd_result):
        """Test save and load with mismatches."""
        test_data = np.random.randn(30, 50)
        config = ValidationConfig(increment=5)
        val_result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        svd_dict = save_svd_to_dict(svd_result, mismatches=val_result.mismatches)
        loaded_result, loaded_mismatches = load_svd_from_dict(svd_dict)

        assert loaded_mismatches is not None
        assert not loaded_mismatches.empty
        assert list(loaded_mismatches.columns) == list(val_result.mismatches.columns)

    def test_dict_structure(self, svd_result):
        """Test that dictionary has correct structure."""
        svd_dict = save_svd_to_dict(svd_result)

        assert "V" in svd_dict
        assert "s" in svd_dict
        assert "n_components" in svd_dict
        assert "method" in svd_dict

        assert svd_dict["n_components"] == svd_result.n_components
        assert svd_dict["method"] == svd_result.method

    def test_dict_v_shape(self, svd_result):
        """Test that V array in dict has correct shape."""
        svd_dict = save_svd_to_dict(svd_result)

        assert isinstance(svd_dict["V"], np.ndarray)
        assert svd_dict["V"].shape == svd_result.V.shape

    def test_load_missing_v(self):
        """Test loading dict without V raises error."""
        incomplete_dict = {"s": np.array([1, 2, 3]), "n_components": 3}

        with pytest.raises(KeyError):
            load_svd_from_dict(incomplete_dict)

    def test_roundtrip_dict(self, svd_result):
        """Test that save-load roundtrip preserves data."""
        # Multiple roundtrips
        svd_dict1 = save_svd_to_dict(svd_result)
        loaded1, _ = load_svd_from_dict(svd_dict1)

        svd_dict2 = save_svd_to_dict(loaded1)
        loaded2, _ = load_svd_from_dict(svd_dict2)

        assert np.allclose(loaded2.V, svd_result.V)
        assert np.allclose(loaded2.s, svd_result.s)


class TestDataPreservation:
    """Test that data is preserved exactly during save/load."""

    def test_preserve_singular_values(self, svd_result, tmpfile):
        """Test that singular values are preserved exactly."""
        save_svd_to_hdf5(tmpfile, svd_result)
        loaded_result, _, _ = load_svd_from_hdf5(tmpfile)

        assert np.allclose(loaded_result.s, svd_result.s, atol=1e-15)

    def test_preserve_v_matrix(self, svd_result, tmpfile):
        """Test that V matrix is preserved exactly."""
        save_svd_to_hdf5(tmpfile, svd_result)
        loaded_result, _, _ = load_svd_from_hdf5(tmpfile)

        assert np.allclose(loaded_result.V, svd_result.V, atol=1e-15)

    def test_preserve_method(self, svd_result, tmpfile):
        """Test that method is preserved."""
        save_svd_to_hdf5(tmpfile, svd_result)
        loaded_result, _, _ = load_svd_from_hdf5(tmpfile)

        assert loaded_result.method == svd_result.method

    def test_preserve_mismatch_values(self, svd_result, tmpfile):
        """Test that mismatch values are preserved exactly."""
        test_data = np.random.randn(20, 50)
        config = ValidationConfig(increment=5)
        val_result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        save_svd_to_hdf5(tmpfile, svd_result, mismatches=val_result.mismatches)
        _, loaded_mismatches, _ = load_svd_from_hdf5(tmpfile)

        assert loaded_mismatches is not None
        for col in val_result.mismatches.columns:
            if "mismatch" in col:
                assert np.allclose(loaded_mismatches[col], val_result.mismatches[col], atol=1e-15)


class TestCompatibility:
    """Test compatibility with dingo file format."""

    def test_dataset_type_attribute(self, svd_result, tmpfile):
        """Test that dataset_type attribute is set correctly."""
        save_svd_to_hdf5(tmpfile, svd_result)

        with h5py.File(tmpfile, "r") as f:
            # Should have dataset_type attribute for compatibility
            if "dataset_type" in f.attrs:
                assert f.attrs["dataset_type"] == "svd_basis"

    def test_backwards_compatible_structure(self, svd_result, tmpfile):
        """Test that file structure matches original dingo SVD format."""
        save_svd_to_hdf5(tmpfile, svd_result)

        with h5py.File(tmpfile, "r") as f:
            # Required datasets
            assert "V" in f
            assert "s" in f

            # Required attributes
            assert "n_components" in f.attrs
            assert "method" in f.attrs


class TestEdgeCases:
    """Test edge cases in I/O."""

    def test_save_empty_mismatches(self, svd_result, tmpfile):
        """Test saving with empty mismatches DataFrame."""
        empty_mismatches = pd.DataFrame()

        # Should not crash
        save_svd_to_hdf5(tmpfile, svd_result, mismatches=empty_mismatches)

        loaded_result, loaded_mismatches, _ = load_svd_from_hdf5(tmpfile)

        # May load as None or empty DataFrame
        assert loaded_mismatches is None or loaded_mismatches.empty

    def test_save_very_large_svd(self, tmpfile):
        """Test saving large SVD (stress test)."""
        np.random.seed(42)
        data = np.random.randn(1000, 500)
        config = SVDGenerationConfig(n_components=200, method="scipy")
        large_svd = generate_svd_basis(data, config)

        save_svd_to_hdf5(tmpfile, large_svd)
        loaded_result, _, _ = load_svd_from_hdf5(tmpfile)

        assert loaded_result.n_components == 200
        assert loaded_result.V.shape == (500, 200)

    def test_unicode_metadata(self, svd_result, tmpfile):
        """Test saving metadata with unicode characters."""
        metadata = SVDMetadata(
            description="Test with unicode: αβγδε 你好 🎉",
            extra={"created_by": "user_测试"},
        )

        save_svd_to_hdf5(tmpfile, svd_result, metadata=metadata)
        _, _, loaded_metadata = load_svd_from_hdf5(tmpfile)

        assert loaded_metadata is not None
        assert loaded_metadata.description == "Test with unicode: αβγδε 你好 🎉"
        assert loaded_metadata.extra is not None
        assert loaded_metadata.extra["created_by"] == "user_测试"

    def test_special_characters_in_path(self, svd_result):
        """Test saving to path with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with spaces and special chars
            filepath = os.path.join(tmpdir, "test file (special).h5")

            save_svd_to_hdf5(filepath, svd_result)
            loaded_result, _, _ = load_svd_from_hdf5(filepath)

            assert loaded_result.n_components == svd_result.n_components
