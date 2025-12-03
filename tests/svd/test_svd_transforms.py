"""Tests for transform framework."""

import numpy as np
import pytest

from dingo_waveform.svd import (
    ApplySVD,
    ComposeTransforms,
    SVDBasis,
    SVDGenerationConfig,
    Transform,
)


# Fixtures for test data
@pytest.fixture
def real_training_data():
    """Generate real-valued training data."""
    np.random.seed(42)
    return np.random.randn(100, 200)


@pytest.fixture
def complex_training_data():
    """Generate complex-valued training data."""
    np.random.seed(42)
    return np.random.randn(100, 200) + 1j * np.random.randn(100, 200)


@pytest.fixture
def real_svd_basis(real_training_data):
    """Create a trained real SVD basis."""
    config = SVDGenerationConfig(n_components=50, method="scipy")
    return SVDBasis.from_training_data(real_training_data, config)


@pytest.fixture
def complex_svd_basis(complex_training_data):
    """Create a trained complex SVD basis."""
    config = SVDGenerationConfig(n_components=50, method="scipy")
    return SVDBasis.from_training_data(complex_training_data, config)


# Test Transform abstract base class
class TestTransformBase:
    """Tests for Transform abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify that Transform cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Transform()  # type: ignore[abstract]

    def test_concrete_transform_works(self):
        """Verify that a concrete Transform implementation works."""

        class DoubleTransform(Transform):
            def __call__(self, data):
                return {key: value * 2 for key, value in data.items()}

        transform = DoubleTransform()
        data = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
        result = transform(data)

        assert "a" in result
        assert "b" in result
        np.testing.assert_array_equal(result["a"], np.array([2, 4, 6]))
        np.testing.assert_array_equal(result["b"], np.array([8, 10, 12]))


# Test ComposeTransforms
class TestComposeTransforms:
    """Tests for ComposeTransforms pipeline composition."""

    def test_empty_pipeline(self):
        """Test that an empty pipeline is a no-op."""
        pipeline = ComposeTransforms([])
        data = {"a": np.array([1, 2, 3])}
        result = pipeline(data)
        np.testing.assert_array_equal(result["a"], data["a"])

    def test_single_transform(self):
        """Test pipeline with a single transform."""

        class AddOneTransform(Transform):
            def __call__(self, data):
                return {key: value + 1 for key, value in data.items()}

        pipeline = ComposeTransforms([AddOneTransform()])
        data = {"a": np.array([1, 2, 3])}
        result = pipeline(data)
        np.testing.assert_array_equal(result["a"], np.array([2, 3, 4]))

    def test_multiple_transforms(self):
        """Test pipeline with multiple transforms chained."""

        class AddOneTransform(Transform):
            def __call__(self, data):
                return {key: value + 1 for key, value in data.items()}

        class MultiplyTwoTransform(Transform):
            def __call__(self, data):
                return {key: value * 2 for key, value in data.items()}

        # Pipeline: (x + 1) * 2
        pipeline = ComposeTransforms([AddOneTransform(), MultiplyTwoTransform()])
        data = {"a": np.array([1, 2, 3])}
        result = pipeline(data)
        # (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
        np.testing.assert_array_equal(result["a"], np.array([4, 6, 8]))

    def test_repr(self):
        """Test string representation of pipeline."""

        class DummyTransform(Transform):
            def __call__(self, data):
                return data

        pipeline = ComposeTransforms([DummyTransform(), DummyTransform()])
        repr_str = repr(pipeline)

        assert "ComposeTransforms" in repr_str
        assert "DummyTransform" in repr_str


# Test ApplySVD
class TestApplySVD:
    """Tests for ApplySVD transform."""

    def test_compression_mode_real(self, real_svd_basis):
        """Test compression with real-valued data."""
        transform = ApplySVD(real_svd_basis, inverse=False)

        # Create test data
        test_data = {
            "h_plus": np.random.randn(10, 200),
            "h_cross": np.random.randn(10, 200),
        }

        result = transform(test_data)

        # Check keys preserved
        assert "h_plus" in result
        assert "h_cross" in result

        # Check compressed shape
        assert result["h_plus"].shape == (10, 50)
        assert result["h_cross"].shape == (10, 50)

    def test_compression_mode_complex(self, complex_svd_basis):
        """Test compression with complex-valued data."""
        transform = ApplySVD(complex_svd_basis, inverse=False)

        # Create test data
        test_data = {
            "h_plus": np.random.randn(10, 200) + 1j * np.random.randn(10, 200),
            "h_cross": np.random.randn(10, 200) + 1j * np.random.randn(10, 200),
        }

        result = transform(test_data)

        # Check keys preserved
        assert "h_plus" in result
        assert "h_cross" in result

        # Check compressed shape
        assert result["h_plus"].shape == (10, 50)
        assert result["h_cross"].shape == (10, 50)

        # Check dtype preserved
        assert np.iscomplexobj(result["h_plus"])
        assert np.iscomplexobj(result["h_cross"])

    def test_decompression_mode(self, real_svd_basis):
        """Test decompression mode."""
        transform = ApplySVD(real_svd_basis, inverse=True)

        # Create compressed data (coefficients)
        compressed_data = {
            "h_plus": np.random.randn(10, 50),
            "h_cross": np.random.randn(10, 50),
        }

        result = transform(compressed_data)

        # Check keys preserved
        assert "h_plus" in result
        assert "h_cross" in result

        # Check decompressed shape
        assert result["h_plus"].shape == (10, 200)
        assert result["h_cross"].shape == (10, 200)

    def test_roundtrip(self, real_svd_basis):
        """Test compress -> decompress roundtrip."""
        compress_transform = ApplySVD(real_svd_basis, inverse=False)
        decompress_transform = ApplySVD(real_svd_basis, inverse=True)

        # Use data similar to training data for better reconstruction
        np.random.seed(123)
        original = {
            "h_plus": np.random.randn(10, 200),
            "h_cross": np.random.randn(10, 200),
        }

        # Compress then decompress
        compressed = compress_transform(original)
        reconstructed = decompress_transform(compressed)

        # Check shape matches
        assert reconstructed["h_plus"].shape == original["h_plus"].shape
        assert reconstructed["h_cross"].shape == original["h_cross"].shape

        # Check that compression actually happened (reduced dimensionality)
        assert compressed["h_plus"].shape == (10, 50)
        assert compressed["h_cross"].shape == (10, 50)

        # Just verify reconstruction exists and has some similarity
        # (with 50 components out of 200, some information loss is expected)
        for key in ["h_plus", "h_cross"]:
            # Compute normalized mismatch
            diff_norm = np.linalg.norm(original[key] - reconstructed[key])
            orig_norm = np.linalg.norm(original[key])
            relative_error = diff_norm / orig_norm

            # Should have some reconstruction (not completely random)
            assert relative_error < 1.0, f"Reconstruction for {key} seems completely off"

    def test_single_sample(self, real_svd_basis):
        """Test with single sample (1D arrays)."""
        compress_transform = ApplySVD(real_svd_basis, inverse=False)

        # Single sample (1D)
        single_sample = {
            "h_plus": np.random.randn(200),
            "h_cross": np.random.randn(200),
        }

        result = compress_transform(single_sample)

        # Check compressed to 1D with correct size
        assert result["h_plus"].shape == (50,)
        assert result["h_cross"].shape == (50,)

    def test_repr(self, real_svd_basis):
        """Test string representation."""
        compress_transform = ApplySVD(real_svd_basis, inverse=False)
        decompress_transform = ApplySVD(real_svd_basis, inverse=True)

        compress_str = repr(compress_transform)
        decompress_str = repr(decompress_transform)

        assert "ApplySVD" in compress_str
        assert "n_components=50" in compress_str
        assert "compress" in compress_str

        assert "ApplySVD" in decompress_str
        assert "n_components=50" in decompress_str
        assert "decompress" in decompress_str


# Integration tests
class TestTransformIntegration:
    """Integration tests for transform pipeline."""

    def test_realistic_pipeline(self, complex_svd_basis):
        """Test a realistic transform pipeline."""

        class NormalizeTransform(Transform):
            """Normalize each array to unit norm."""

            def __call__(self, data):
                return {
                    key: value / (np.linalg.norm(value) + 1e-10)
                    for key, value in data.items()
                }

        # Create pipeline: normalize -> compress
        pipeline = ComposeTransforms(
            [
                NormalizeTransform(),
                ApplySVD(complex_svd_basis, inverse=False),
            ]
        )

        # Test data
        test_data = {
            "h_plus": np.random.randn(5, 200) + 1j * np.random.randn(5, 200),
            "h_cross": np.random.randn(5, 200) + 1j * np.random.randn(5, 200),
        }

        result = pipeline(test_data)

        # Verify compression happened
        assert result["h_plus"].shape == (5, 50)
        assert result["h_cross"].shape == (5, 50)

    def test_compress_decompress_pipeline(self, real_svd_basis):
        """Test compress -> decompress as a pipeline."""
        pipeline = ComposeTransforms(
            [
                ApplySVD(real_svd_basis, inverse=False),
                ApplySVD(real_svd_basis, inverse=True),
            ]
        )

        original = {
            "h_plus": np.random.randn(10, 200),
            "h_cross": np.random.randn(10, 200),
        }

        result = pipeline(original)

        # Shape should be preserved
        assert result["h_plus"].shape == original["h_plus"].shape
        assert result["h_cross"].shape == original["h_cross"].shape

    def test_empty_dict(self, real_svd_basis):
        """Test that empty dictionary passes through."""
        transform = ApplySVD(real_svd_basis, inverse=False)
        result = transform({})
        assert result == {}

    def test_preserves_dict_keys(self, real_svd_basis):
        """Test that transform preserves all dictionary keys."""
        transform = ApplySVD(real_svd_basis, inverse=False)

        data = {
            "field1": np.random.randn(5, 200),
            "field2": np.random.randn(5, 200),
            "field3": np.random.randn(5, 200),
        }

        result = transform(data)

        assert set(result.keys()) == set(data.keys())
        for key in data.keys():
            assert result[key].shape == (5, 50)
