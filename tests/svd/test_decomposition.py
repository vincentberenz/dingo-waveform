"""Tests for SVD decomposition functions."""

import numpy as np
import pytest

from dingo_waveform.svd import SVDGenerationConfig, generate_svd_basis
from dingo_waveform.svd.decomposition import compute_svd_random, compute_svd_scipy
from dingo_waveform.svd.results import SVDDecompositionResult


@pytest.fixture
def real_data():
    """Generate real-valued data."""
    np.random.seed(42)
    return np.random.randn(100, 50)


@pytest.fixture
def complex_data():
    """Generate complex-valued data."""
    np.random.seed(42)
    return np.random.randn(100, 50) + 1j * np.random.randn(100, 50)


class TestComputeSVDScipy:
    """Test compute_svd_scipy function."""

    def test_basic_real(self, real_data):
        """Test basic SVD computation with real data."""
        result = compute_svd_scipy(real_data, n_components=10)

        assert isinstance(result, SVDDecompositionResult)
        assert result.V.shape == (50, 10)
        assert result.Vh.shape == (10, 50)
        assert result.s.shape == (10,)
        assert result.n_components == 10
        assert result.method == "scipy"

    def test_basic_complex(self, complex_data):
        """Test basic SVD computation with complex data."""
        result = compute_svd_scipy(complex_data, n_components=15)

        assert result.V.shape == (50, 15)
        assert result.Vh.shape == (15, 50)
        assert result.s.shape == (15,)
        assert np.iscomplexobj(result.V)
        assert np.iscomplexobj(result.Vh)

    def test_n_components_zero(self, real_data):
        """Test with n_components=0 (use all components)."""
        result = compute_svd_scipy(real_data, n_components=0)

        # Should use min(m, n) = min(100, 50) = 50
        assert result.n_components == 50
        assert result.V.shape == (50, 50)

    def test_n_components_exceeds_dimensions(self, real_data):
        """Test with n_components larger than matrix dimensions."""
        result = compute_svd_scipy(real_data, n_components=100)

        # Should cap at min(m, n) = 50
        assert result.n_components == 50

    def test_vh_is_conjugate_transpose(self, real_data):
        """Test that Vh = V^H."""
        result = compute_svd_scipy(real_data, n_components=10)

        assert np.allclose(result.Vh, result.V.T.conj())

    def test_orthogonality(self, real_data):
        """Test that V has orthonormal columns."""
        result = compute_svd_scipy(real_data, n_components=10)

        VhV = result.V.T.conj() @ result.V
        identity = np.eye(10)
        assert np.allclose(VhV, identity, atol=1e-10)

    def test_singular_values_positive(self, real_data):
        """Test that singular values are positive."""
        result = compute_svd_scipy(real_data, n_components=10)

        assert np.all(result.s > 0)

    def test_singular_values_decreasing(self, real_data):
        """Test that singular values are in descending order."""
        result = compute_svd_scipy(real_data, n_components=20)

        assert np.all(result.s[:-1] >= result.s[1:])

    def test_reconstruction(self, real_data):
        """Test that SVD can reconstruct the original data."""
        result = compute_svd_scipy(real_data, n_components=0)  # All components

        # Reconstruct: data ≈ U @ S @ Vh, but we only have V and Vh
        # Check: data @ V @ Vh ≈ data
        reconstructed = (real_data @ result.V) @ result.Vh

        assert np.allclose(real_data, reconstructed, atol=1e-10)


class TestComputeSVDRandom:
    """Test compute_svd_random function."""

    def test_basic_real(self, real_data):
        """Test basic randomized SVD with real data."""
        try:
            result = compute_svd_random(real_data, n_components=10, random_state=42)

            assert isinstance(result, SVDDecompositionResult)
            assert result.V.shape == (50, 10)
            assert result.Vh.shape == (10, 50)
            assert result.s.shape == (10,)
            assert result.n_components == 10
            assert result.method == "random"
        except ValueError as e:
            if "randomized_svd failed" in str(e):
                pytest.skip("scikit-learn version doesn't support required features")
            else:
                raise

    def test_random_state_reproducibility(self, real_data):
        """Test that random_state ensures reproducibility."""
        try:
            result1 = compute_svd_random(real_data, n_components=10, random_state=42)
            result2 = compute_svd_random(real_data, n_components=10, random_state=42)

            assert np.allclose(result1.V, result2.V)
            assert np.allclose(result1.s, result2.s)
        except ValueError as e:
            if "randomized_svd failed" in str(e):
                pytest.skip("scikit-learn version doesn't support required features")
            else:
                raise

    def test_different_random_states(self, real_data):
        """Test that different random states give different results."""
        try:
            result1 = compute_svd_random(real_data, n_components=10, random_state=42)
            result2 = compute_svd_random(real_data, n_components=10, random_state=123)

            # Results should be similar but not identical
            # (they should span similar subspaces)
            assert not np.allclose(result1.V, result2.V, atol=1e-6)
        except ValueError as e:
            if "randomized_svd failed" in str(e):
                pytest.skip("scikit-learn version doesn't support required features")
            else:
                raise

    def test_approximation_quality(self, real_data):
        """Test that randomized SVD gives good approximation."""
        try:
            result_random = compute_svd_random(real_data, n_components=10, random_state=42)
            result_exact = compute_svd_scipy(real_data, n_components=10)

            # Singular values should be reasonably close (randomized SVD is approximate)
            assert np.allclose(result_random.s, result_exact.s, rtol=0.01)
        except ValueError as e:
            if "randomized_svd failed" in str(e):
                pytest.skip("scikit-learn version doesn't support required features")
            else:
                raise


class TestGenerateSVDBasis:
    """Test generate_svd_basis function."""

    def test_scipy_method(self, real_data):
        """Test with scipy method."""
        config = SVDGenerationConfig(n_components=10, method="scipy")
        result = generate_svd_basis(real_data, config)

        assert result.method == "scipy"
        assert result.n_components == 10

    def test_random_method(self, real_data):
        """Test with random method."""
        config = SVDGenerationConfig(n_components=10, method="random")
        try:
            result = generate_svd_basis(real_data, config)
            assert result.method == "random"
            assert result.n_components == 10
        except ValueError as e:
            if "randomized_svd failed" in str(e):
                pytest.skip("scikit-learn version doesn't support required features")
            else:
                raise

    def test_invalid_method(self, real_data):
        """Test that invalid method raises error."""
        config = SVDGenerationConfig(n_components=10, method="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unsupported SVD method"):
            generate_svd_basis(real_data, config)

    def test_preserves_dtype_real(self, real_data):
        """Test that real data produces real SVD (in the returned V)."""
        config = SVDGenerationConfig(n_components=10, method="scipy")
        result = generate_svd_basis(real_data, config)

        # V should be real (or at least have negligible imaginary part)
        assert not np.iscomplexobj(result.V) or np.allclose(result.V.imag, 0)

    def test_preserves_dtype_complex(self, complex_data):
        """Test that complex data produces complex SVD."""
        config = SVDGenerationConfig(n_components=10, method="scipy")
        result = generate_svd_basis(complex_data, config)

        assert np.iscomplexobj(result.V)
        assert np.iscomplexobj(result.Vh)


class TestSVDDecompositionResult:
    """Test SVDDecompositionResult dataclass."""

    def test_creation(self):
        """Test creating an SVDDecompositionResult."""
        V = np.random.randn(50, 10)
        Vh = V.T.conj()
        s = np.random.rand(10)

        result = SVDDecompositionResult(V=V, Vh=Vh, s=s, n_components=10, method="scipy")

        assert result.n_components == 10
        assert result.method == "scipy"
        assert np.array_equal(result.V, V)
        assert np.array_equal(result.Vh, Vh)
        assert np.array_equal(result.s, s)

    def test_dataclass_attributes(self):
        """Test that SVDDecompositionResult has correct attributes."""
        V = np.random.randn(50, 10)
        Vh = V.T.conj()
        s = np.random.rand(10)

        result = SVDDecompositionResult(V=V, Vh=Vh, s=s, n_components=10, method="scipy")

        # Can modify attributes (dataclass is not frozen)
        # This is OK since it's a simple result container
        result.n_components = 20
        assert result.n_components == 20


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample(self):
        """Test SVD with single sample."""
        data = np.random.randn(1, 50)
        config = SVDGenerationConfig(n_components=1, method="scipy")

        result = generate_svd_basis(data, config)
        assert result.n_components == 1

    def test_single_feature(self):
        """Test SVD with single feature."""
        data = np.random.randn(100, 1)
        config = SVDGenerationConfig(n_components=1, method="scipy")

        result = generate_svd_basis(data, config)
        assert result.n_components == 1

    def test_wide_matrix(self):
        """Test SVD with more features than samples."""
        data = np.random.randn(10, 50)
        config = SVDGenerationConfig(n_components=5, method="scipy")

        result = generate_svd_basis(data, config)
        assert result.n_components == 5
        assert result.V.shape == (50, 5)

    def test_tall_matrix(self):
        """Test SVD with more samples than features."""
        data = np.random.randn(100, 10)
        config = SVDGenerationConfig(n_components=5, method="scipy")

        result = generate_svd_basis(data, config)
        assert result.n_components == 5
        assert result.V.shape == (10, 5)
