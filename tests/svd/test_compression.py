"""Tests for compression and decompression functions."""

import numpy as np
import pytest
from dingo_waveform.svd import (
    SVDGenerationConfig,
    compress,
    compress_dict,
    decompress,
    decompress_dict,
)
from dingo_waveform.svd.decomposition import generate_svd_basis


@pytest.fixture
def real_svd_result():
    """Generate a real-valued SVD result."""
    np.random.seed(42)
    data = np.random.randn(100, 50)
    config = SVDGenerationConfig(n_components=10, method="scipy")
    return generate_svd_basis(data, config)


@pytest.fixture
def complex_svd_result():
    """Generate a complex-valued SVD result."""
    np.random.seed(42)
    data = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
    config = SVDGenerationConfig(n_components=10, method="scipy")
    return generate_svd_basis(data, config)


class TestCompress:
    """Test compress function."""

    def test_compress_single_sample_real(self, real_svd_result):
        """Test compressing a single sample of real data."""
        data = np.random.randn(50)
        result = compress(data, real_svd_result.V)

        assert result.coefficients.shape == (10,)

    def test_compress_multiple_samples_real(self, real_svd_result):
        """Test compressing multiple samples of real data."""
        data = np.random.randn(20, 50)
        result = compress(data, real_svd_result.V)

        assert result.coefficients.shape == (20, 10)

    def test_compress_single_sample_complex(self, complex_svd_result):
        """Test compressing a single sample of complex data."""
        data = np.random.randn(50) + 1j * np.random.randn(50)
        result = compress(data, complex_svd_result.V)

        assert result.coefficients.shape == (10,)
        assert np.iscomplexobj(result.coefficients)

    def test_compress_multiple_samples_complex(self, complex_svd_result):
        """Test compressing multiple samples of complex data."""
        data = np.random.randn(20, 50) + 1j * np.random.randn(20, 50)
        result = compress(data, complex_svd_result.V)

        assert result.coefficients.shape == (20, 10)
        assert np.iscomplexobj(result.coefficients)

    def test_compress_correct_computation(self, real_svd_result):
        """Test that compression computes data @ V correctly."""
        data = np.random.randn(20, 50)
        result = compress(data, real_svd_result.V)

        expected = data @ real_svd_result.V
        assert np.allclose(result.coefficients, expected)

    def test_compress_wrong_shape(self, real_svd_result):
        """Test that wrong input shape raises error."""
        data = np.random.randn(20, 30)  # Wrong number of features

        with pytest.raises((ValueError, Exception)):
            compress(data, real_svd_result.V)


class TestDecompress:
    """Test decompress function."""

    def test_decompress_single_sample_real(self, real_svd_result):
        """Test decompressing a single sample of real coefficients."""
        coefficients = np.random.randn(10)
        result = decompress(coefficients, real_svd_result.Vh)

        assert result.data.shape == (50,)

    def test_decompress_multiple_samples_real(self, real_svd_result):
        """Test decompressing multiple samples of real coefficients."""
        coefficients = np.random.randn(20, 10)
        result = decompress(coefficients, real_svd_result.Vh)

        assert result.data.shape == (20, 50)

    def test_decompress_single_sample_complex(self, complex_svd_result):
        """Test decompressing a single sample of complex coefficients."""
        coefficients = np.random.randn(10) + 1j * np.random.randn(10)
        result = decompress(coefficients, complex_svd_result.Vh)

        assert result.data.shape == (50,)
        assert np.iscomplexobj(result.data)

    def test_decompress_multiple_samples_complex(self, complex_svd_result):
        """Test decompressing multiple samples of complex coefficients."""
        coefficients = np.random.randn(20, 10) + 1j * np.random.randn(20, 10)
        result = decompress(coefficients, complex_svd_result.Vh)

        assert result.data.shape == (20, 50)
        assert np.iscomplexobj(result.data)

    def test_decompress_correct_computation(self, real_svd_result):
        """Test that decompression computes coefficients @ Vh correctly."""
        coefficients = np.random.randn(20, 10)
        result = decompress(coefficients, real_svd_result.Vh)

        expected = coefficients @ real_svd_result.Vh
        assert np.allclose(result.data, expected)

    def test_decompress_wrong_shape(self, real_svd_result):
        """Test that wrong input shape raises error."""
        coefficients = np.random.randn(20, 5)  # Wrong number of components

        with pytest.raises((ValueError, Exception)):
            decompress(coefficients, real_svd_result.Vh)


class TestCompressDecompress:
    """Test compress and decompress together."""

    def test_roundtrip_real(self, real_svd_result):
        """Test that compress then decompress reconstructs data."""
        np.random.seed(123)
        # Generate test data in the span of the SVD basis
        coeffs = np.random.randn(10, 10)
        original_data = coeffs @ real_svd_result.Vh

        # Compress
        compressed = compress(original_data, real_svd_result.V)

        # Decompress
        decompressed = decompress(compressed.coefficients, real_svd_result.Vh)

        # Should perfectly reconstruct data in the span of the basis
        assert np.allclose(decompressed.data, original_data, atol=1e-10)

    def test_roundtrip_complex(self, complex_svd_result):
        """Test roundtrip with complex data."""
        np.random.seed(123)
        # Generate test data in the span of the SVD basis
        coeffs = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        original_data = coeffs @ complex_svd_result.Vh

        compressed = compress(original_data, complex_svd_result.V)
        decompressed = decompress(compressed.coefficients, complex_svd_result.Vh)

        assert np.allclose(decompressed.data, original_data, atol=1e-10)

    def test_approximate_reconstruction(self, real_svd_result):
        """Test reconstruction of data not in the span of the basis."""
        np.random.seed(123)
        data = np.random.randn(10, 50)

        compressed = compress(data, real_svd_result.V)
        decompressed = decompress(compressed.coefficients, real_svd_result.Vh)

        # Should be an approximation (not perfect reconstruction)
        # But error should be bounded
        error = np.linalg.norm(data - decompressed.data, axis=1)
        relative_error = error / np.linalg.norm(data, axis=1)

        # Relative error should be less than 100% (sanity check)
        assert np.all(relative_error < 1.0)


class TestCompressDict:
    """Test compress_dict function."""

    def test_compress_dict_basic(self, real_svd_result):
        """Test compressing a dictionary of arrays."""
        data_dict = {
            "H1": np.random.randn(20, 50),
            "L1": np.random.randn(20, 50),
            "V1": np.random.randn(20, 50),
        }

        compressed_dict = compress_dict(data_dict, real_svd_result.V)

        assert isinstance(compressed_dict, dict)
        assert set(compressed_dict.keys()) == {"H1", "L1", "V1"}
        assert compressed_dict["H1"].shape == (20, 10)
        assert compressed_dict["L1"].shape == (20, 10)
        assert compressed_dict["V1"].shape == (20, 10)

    def test_compress_dict_empty(self, real_svd_result):
        """Test compressing an empty dictionary."""
        compressed_dict = compress_dict({}, real_svd_result.V)

        assert compressed_dict == {}

    def test_compress_dict_single_key(self, real_svd_result):
        """Test compressing a dictionary with single key."""
        data_dict = {"H1": np.random.randn(20, 50)}

        compressed_dict = compress_dict(data_dict, real_svd_result.V)

        assert len(compressed_dict) == 1
        assert "H1" in compressed_dict
        assert compressed_dict["H1"].shape == (20, 10)

    def test_compress_dict_preserves_keys(self, real_svd_result):
        """Test that all keys are preserved."""
        data_dict = {f"ifo_{i}": np.random.randn(20, 50) for i in range(10)}

        compressed_dict = compress_dict(data_dict, real_svd_result.V)

        assert set(compressed_dict.keys()) == set(data_dict.keys())


class TestDecompressDict:
    """Test decompress_dict function."""

    def test_decompress_dict_basic(self, real_svd_result):
        """Test decompressing a dictionary of coefficient arrays."""
        coeff_dict = {
            "H1": np.random.randn(20, 10),
            "L1": np.random.randn(20, 10),
            "V1": np.random.randn(20, 10),
        }

        decompressed_dict = decompress_dict(coeff_dict, real_svd_result.Vh)

        assert isinstance(decompressed_dict, dict)
        assert set(decompressed_dict.keys()) == {"H1", "L1", "V1"}
        assert decompressed_dict["H1"].shape == (20, 50)
        assert decompressed_dict["L1"].shape == (20, 50)
        assert decompressed_dict["V1"].shape == (20, 50)

    def test_decompress_dict_empty(self, real_svd_result):
        """Test decompressing an empty dictionary."""
        decompressed_dict = decompress_dict({}, real_svd_result.Vh)

        assert decompressed_dict == {}

    def test_decompress_dict_single_key(self, real_svd_result):
        """Test decompressing a dictionary with single key."""
        coeff_dict = {"H1": np.random.randn(20, 10)}

        decompressed_dict = decompress_dict(coeff_dict, real_svd_result.Vh)

        assert len(decompressed_dict) == 1
        assert "H1" in decompressed_dict
        assert decompressed_dict["H1"].shape == (20, 50)


class TestCompressDecompressDict:
    """Test compress_dict and decompress_dict together."""

    def test_roundtrip_dict(self, real_svd_result):
        """Test compress then decompress for dictionaries."""
        # Generate data in the span of the basis
        coeffs_H1 = np.random.randn(10, 10)
        coeffs_L1 = np.random.randn(10, 10)

        data_dict = {
            "H1": coeffs_H1 @ real_svd_result.Vh,
            "L1": coeffs_L1 @ real_svd_result.Vh,
        }

        # Compress
        compressed = compress_dict(data_dict, real_svd_result.V)

        # Decompress
        decompressed = decompress_dict(compressed, real_svd_result.Vh)

        # Should perfectly reconstruct
        assert np.allclose(decompressed["H1"], data_dict["H1"], atol=1e-10)
        assert np.allclose(decompressed["L1"], data_dict["L1"], atol=1e-10)

    def test_roundtrip_dict_complex(self, complex_svd_result):
        """Test roundtrip with complex dictionary."""
        coeffs_H1 = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        coeffs_L1 = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

        data_dict = {
            "H1": coeffs_H1 @ complex_svd_result.Vh,
            "L1": coeffs_L1 @ complex_svd_result.Vh,
        }

        compressed = compress_dict(data_dict, complex_svd_result.V)
        decompressed = decompress_dict(compressed, complex_svd_result.Vh)

        assert np.allclose(decompressed["H1"], data_dict["H1"], atol=1e-10)
        assert np.allclose(decompressed["L1"], data_dict["L1"], atol=1e-10)


class TestEdgeCases:
    """Test edge cases."""

    def test_compress_1d_input(self, real_svd_result):
        """Test compressing 1D array (single sample)."""
        data = np.random.randn(50)
        result = compress(data, real_svd_result.V)

        # Should handle 1D input
        assert result.coefficients.shape == (10,)

    def test_decompress_1d_input(self, real_svd_result):
        """Test decompressing 1D array (single sample)."""
        coefficients = np.random.randn(10)
        result = decompress(coefficients, real_svd_result.Vh)

        # Should handle 1D input
        assert result.data.shape == (50,)

    def test_compress_3d_input(self, real_svd_result):
        """Test compressing 3D array."""
        data = np.random.randn(5, 4, 50)

        # Should work - compress along last axis
        result = compress(data, real_svd_result.V)
        assert result.coefficients.shape == (5, 4, 10)

    def test_different_dtypes(self, real_svd_result):
        """Test with different numpy dtypes."""
        # Float32
        data_f32 = np.random.randn(10, 50).astype(np.float32)
        result_f32 = compress(data_f32, real_svd_result.V.astype(np.float32))
        assert result_f32.coefficients.dtype in [np.float32, np.float64]

        # Complex64
        data_c64 = (np.random.randn(10, 50) + 1j * np.random.randn(10, 50)).astype(np.complex64)
        V_c64 = real_svd_result.V.astype(np.complex64)
        result_c64 = compress(data_c64, V_c64)
        assert result_c64.coefficients.dtype in [np.complex64, np.complex128]
