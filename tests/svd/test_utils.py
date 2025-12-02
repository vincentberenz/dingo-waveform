"""Tests for utility functions."""

import numpy as np
import pytest
from dingo_waveform.svd import SVDGenerationConfig
from dingo_waveform.svd.decomposition import generate_svd_basis
from dingo_waveform.svd.results import SVDDecompositionResult
from dingo_waveform.svd.utils import (
    compute_explained_variance_ratio,
    estimate_reconstruction_error,
    truncate_svd,
)


@pytest.fixture
def svd_result():
    """Generate an SVD result for testing."""
    np.random.seed(42)
    data = np.random.randn(100, 50)
    config = SVDGenerationConfig(n_components=20, method="scipy")
    return generate_svd_basis(data, config)


class TestTruncateSVD:
    """Test truncate_svd function."""

    def test_basic_truncation(self, svd_result):
        """Test basic truncation."""
        V_trunc, Vh_trunc, s_trunc = truncate_svd(
            svd_result.V, svd_result.Vh, svd_result.s, n_components=10
        )

        assert V_trunc.shape == (50, 10)
        assert Vh_trunc.shape == (10, 50)
        assert s_trunc.shape == (10,)

    def test_truncation_preserves_values(self, svd_result):
        """Test that truncation preserves the first n values."""
        V_trunc, Vh_trunc, s_trunc = truncate_svd(
            svd_result.V, svd_result.Vh, svd_result.s, n_components=10
        )

        # First 10 components should match
        assert np.allclose(V_trunc, svd_result.V[:, :10])
        assert np.allclose(Vh_trunc, svd_result.Vh[:10, :])
        assert np.allclose(s_trunc, svd_result.s[:10])

    def test_truncate_to_same_size(self, svd_result):
        """Test truncating to same size (no-op)."""
        V_trunc, Vh_trunc, s_trunc = truncate_svd(
            svd_result.V, svd_result.Vh, svd_result.s, n_components=20
        )

        assert s_trunc.shape[0] == 20
        assert np.allclose(V_trunc, svd_result.V)

    def test_truncate_to_one(self, svd_result):
        """Test truncating to single component."""
        V_trunc, Vh_trunc, s_trunc = truncate_svd(
            svd_result.V, svd_result.Vh, svd_result.s, n_components=1
        )

        assert s_trunc.shape[0] == 1
        assert V_trunc.shape == (50, 1)
        assert s_trunc.shape == (1,)


class TestEstimateReconstructionError:
    """Test estimate_reconstruction_error function."""

    def test_basic_estimation(self, svd_result):
        """Test basic error estimation."""
        error = estimate_reconstruction_error(svd_result.s, n_components=10)

        assert isinstance(error, float)
        assert error >= 0

    def test_error_decreases_with_components(self, svd_result):
        """Test that error decreases as n_components increases."""
        error_5 = estimate_reconstruction_error(svd_result.s, n_components=5)
        error_10 = estimate_reconstruction_error(svd_result.s, n_components=10)
        error_15 = estimate_reconstruction_error(svd_result.s, n_components=15)

        # Error should decrease (or stay same) as we use more components
        assert error_10 <= error_5
        assert error_15 <= error_10

    def test_error_zero_at_full_components(self, svd_result):
        """Test that error is zero when using all components."""
        error = estimate_reconstruction_error(svd_result.s, n_components=20)

        # Should be zero or very close to zero
        assert error < 1e-10

    def test_error_computation_method(self, svd_result):
        """Test that error is computed from singular values."""
        # Error should be relative error based on truncated singular values
        n = 10
        # The function computes: ||s[n:]||^2 / ||s||^2
        s_full_norm_sq = np.linalg.norm(svd_result.s) ** 2
        s_trunc_norm_sq = np.linalg.norm(svd_result.s[n:]) ** 2
        expected_error = s_trunc_norm_sq / s_full_norm_sq
        computed_error = estimate_reconstruction_error(svd_result.s, n_components=n)

        assert np.isclose(computed_error, expected_error)


class TestComputeExplainedVarianceRatio:
    """Test compute_explained_variance_ratio function."""

    def test_basic_computation(self, svd_result):
        """Test basic variance ratio computation."""
        ratios = compute_explained_variance_ratio(svd_result.s)

        assert len(ratios) == 20
        assert all(0 <= r <= 1 for r in ratios)

    def test_variance_ratios_are_cumulative(self, svd_result):
        """Test that function returns cumulative variance ratio."""
        ratios = compute_explained_variance_ratio(svd_result.s)

        # Should be cumulative and sum to 1
        assert np.isclose(ratios[-1], 1.0, atol=1e-10)

        # Should be increasing
        for i in range(len(ratios) - 1):
            assert ratios[i] <= ratios[i + 1]

    def test_first_component_positive(self, svd_result):
        """Test that first component explains some variance."""
        ratios = compute_explained_variance_ratio(svd_result.s)

        assert ratios[0] > 0


class TestEdgeCases:
    """Test edge cases for utility functions."""

    def test_truncate_single_component_svd(self):
        """Test truncating SVD with single component."""
        np.random.seed(42)
        data = np.random.randn(100, 50)
        config = SVDGenerationConfig(n_components=1, method="scipy")
        svd_result = generate_svd_basis(data, config)

        # Can only truncate to 1
        V_trunc, Vh_trunc, s_trunc = truncate_svd(
            svd_result.V, svd_result.Vh, svd_result.s, n_components=1
        )
        assert s_trunc.shape[0] == 1

    def test_variance_ratio_single_component(self):
        """Test variance ratio with single component."""
        np.random.seed(42)
        data = np.random.randn(100, 50)
        config = SVDGenerationConfig(n_components=1, method="scipy")
        svd_result = generate_svd_basis(data, config)

        ratios = compute_explained_variance_ratio(svd_result.s)

        assert len(ratios) == 1
        # Single component explains all its variance (cumulative = 1)
        assert np.isclose(ratios[0], 1.0)

    def test_error_estimation_all_components(self):
        """Test error estimation with all possible components."""
        np.random.seed(42)
        data = np.random.randn(50, 100)  # More features than samples
        config = SVDGenerationConfig(n_components=0, method="scipy")  # All components
        svd_result = generate_svd_basis(data, config)

        # Using all components should give zero error
        error = estimate_reconstruction_error(svd_result.s, n_components=svd_result.n_components)
        assert error < 1e-10


class TestComplexData:
    """Test utilities with complex-valued data."""

    def test_truncate_complex(self):
        """Test truncating complex SVD."""
        np.random.seed(42)
        data = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
        config = SVDGenerationConfig(n_components=20, method="scipy")
        svd_result = generate_svd_basis(data, config)

        V_trunc, Vh_trunc, s_trunc = truncate_svd(
            svd_result.V, svd_result.Vh, svd_result.s, n_components=10
        )

        assert s_trunc.shape[0] == 10
        assert np.iscomplexobj(V_trunc)
        assert np.iscomplexobj(Vh_trunc)

    def test_variance_ratio_complex(self):
        """Test variance ratio with complex data."""
        np.random.seed(42)
        data = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
        config = SVDGenerationConfig(n_components=20, method="scipy")
        svd_result = generate_svd_basis(data, config)

        ratios = compute_explained_variance_ratio(svd_result.s)

        # Singular values are always real, so ratios should be real
        assert all(np.isreal(r) for r in ratios)
        assert np.isclose(ratios[-1], 1.0)

    def test_error_estimation_complex(self):
        """Test error estimation with complex data."""
        np.random.seed(42)
        data = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
        config = SVDGenerationConfig(n_components=20, method="scipy")
        svd_result = generate_svd_basis(data, config)

        error = estimate_reconstruction_error(svd_result.s, n_components=10)

        # Error should be real and positive
        assert np.isreal(error)
        assert error >= 0
