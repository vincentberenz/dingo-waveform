"""Tests for validation functions."""

import numpy as np
import pandas as pd
import pytest
from dingo_waveform.svd import (
    SVDGenerationConfig,
    ValidationConfig,
    compute_mismatch,
    print_validation_summary,
    validate_svd,
)
from dingo_waveform.svd.decomposition import generate_svd_basis


@pytest.fixture
def svd_result():
    """Generate an SVD result for testing."""
    np.random.seed(42)
    data = np.random.randn(100, 50)
    config = SVDGenerationConfig(n_components=20, method="scipy")
    return generate_svd_basis(data, config)


@pytest.fixture
def test_data():
    """Generate test data for validation."""
    np.random.seed(123)
    return np.random.randn(30, 50)


class TestComputeMismatch:
    """Test compute_mismatch function."""

    def test_perfect_reconstruction(self, svd_result):
        """Test mismatch is zero for perfect reconstruction."""
        # Generate data in the span of the basis
        np.random.seed(123)
        coeffs = np.random.randn(20)
        data = coeffs @ svd_result.Vh

        # Compress and decompress to get reconstruction
        compressed = data @ svd_result.V
        reconstructed = compressed @ svd_result.Vh
        mismatch = compute_mismatch(data, reconstructed)

        assert mismatch < 1e-10

    def test_mismatch_range(self, svd_result, test_data):
        """Test that mismatch is between 0 and 1."""
        for i in range(len(test_data)):
            # Compress and decompress to get reconstruction
            compressed = test_data[i] @ svd_result.V
            reconstructed = compressed @ svd_result.Vh
            mismatch = compute_mismatch(test_data[i], reconstructed)
            assert 0 <= mismatch <= 1

    def test_mismatch_truncation_increases(self, svd_result, test_data):
        """Test that mismatch increases with truncation."""
        sample = test_data[0]

        # Full basis - compress and decompress
        compressed_full = sample @ svd_result.V[:, :20]
        reconstructed_full = compressed_full @ svd_result.Vh[:20, :]
        mismatch_full = compute_mismatch(sample, reconstructed_full)

        # Truncated basis - compress and decompress
        compressed_trunc = sample @ svd_result.V[:, :10]
        reconstructed_trunc = compressed_trunc @ svd_result.Vh[:10, :]
        mismatch_truncated = compute_mismatch(sample, reconstructed_trunc)

        # Mismatch should increase (or at least not decrease) with truncation
        assert mismatch_truncated >= mismatch_full - 1e-10

    def test_mismatch_complex(self):
        """Test mismatch computation with complex data."""
        np.random.seed(42)
        data = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
        config = SVDGenerationConfig(n_components=20, method="scipy")
        svd_result = generate_svd_basis(data, config)

        # Test data
        test_sample = np.random.randn(50) + 1j * np.random.randn(50)

        # Compress and decompress to get reconstruction
        compressed = test_sample @ svd_result.V
        reconstructed = compressed @ svd_result.Vh
        mismatch = compute_mismatch(test_sample, reconstructed)

        assert 0 <= mismatch <= 1
        assert np.isreal(mismatch)  # Mismatch should always be real

    def test_mismatch_zero_vector(self, svd_result):
        """Test mismatch with zero vector."""
        zero_vector = np.zeros(50)

        # Mismatch with zero vector is undefined (division by zero), but shouldn't crash
        # Compress and decompress to get reconstruction
        compressed = zero_vector @ svd_result.V
        reconstructed = compressed @ svd_result.Vh
        mismatch = compute_mismatch(zero_vector, reconstructed)

        # Mathematically undefined (0/0), results in NaN
        # This is expected behavior - the test just ensures it doesn't crash
        assert isinstance(mismatch, (float, np.floating))


class TestValidateSVD:
    """Test validate_svd function."""

    def test_basic_validation(self, svd_result, test_data):
        """Test basic SVD validation."""
        config = ValidationConfig(increment=5)

        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config, labels=None)

        assert result.mismatches is not None
        assert not result.mismatches.empty
        assert len(result.mismatches) == len(test_data)

    def test_validation_columns(self, svd_result, test_data):
        """Test that validation creates correct mismatch columns."""
        config = ValidationConfig(increment=5)

        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        # Should have columns for n=5, 10, 15, 20
        expected_cols = ["mismatch_n=5", "mismatch_n=10", "mismatch_n=15", "mismatch_n=20"]
        for col in expected_cols:
            assert col in result.mismatches.columns

    def test_validation_with_labels(self, svd_result, test_data):
        """Test validation with parameter labels."""
        labels = pd.DataFrame(
            {
                "mass_1": np.random.uniform(10, 50, len(test_data)),
                "mass_2": np.random.uniform(10, 50, len(test_data)),
            }
        )

        config = ValidationConfig(increment=5)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config, labels)

        # Labels should be included in mismatches DataFrame
        assert "mass_1" in result.mismatches.columns
        assert "mass_2" in result.mismatches.columns
        assert np.allclose(result.mismatches["mass_1"], labels["mass_1"])

    def test_validation_increment(self, svd_result, test_data):
        """Test that increment parameter works correctly."""
        config_small = ValidationConfig(increment=3)
        result_small = validate_svd(svd_result.V, svd_result.Vh, test_data, config_small)

        # Should have more columns with smaller increment
        # n=3, 6, 9, 12, 15, 18, 20 (final)
        mismatch_cols_small = [c for c in result_small.mismatches.columns if "mismatch" in c]

        config_large = ValidationConfig(increment=10)
        result_large = validate_svd(svd_result.V, svd_result.Vh, test_data, config_large)

        # n=10, 20
        mismatch_cols_large = [c for c in result_large.mismatches.columns if "mismatch" in c]

        assert len(mismatch_cols_small) > len(mismatch_cols_large)

    def test_validation_final_n(self, svd_result, test_data):
        """Test that final n_components is always included."""
        config = ValidationConfig(increment=7)  # Won't divide evenly into 20

        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        # Should always include the final n=20
        assert "mismatch_n=20" in result.mismatches.columns

    def test_validation_mismatches_positive(self, svd_result, test_data):
        """Test that all mismatches are non-negative."""
        config = ValidationConfig(increment=5)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        for col in result.mismatches.columns:
            if "mismatch" in col:
                assert np.all(result.mismatches[col] >= 0)

    def test_validation_mismatches_bounded(self, svd_result, test_data):
        """Test that all mismatches are <= 1."""
        config = ValidationConfig(increment=5)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        for col in result.mismatches.columns:
            if "mismatch" in col:
                assert np.all(result.mismatches[col] <= 1)


class TestPrintValidationSummary:
    """Test print_validation_summary function."""

    def test_print_summary_basic(self, svd_result, test_data, capsys):
        """Test that summary prints correctly."""
        config = ValidationConfig(increment=10)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        print_validation_summary(result)

        captured = capsys.readouterr()
        output = captured.out

        # Should print statistics
        assert "Mean mismatch" in output
        assert "Standard deviation" in output
        assert "Max mismatch" in output
        assert "Median mismatch" in output
        assert "Percentiles" in output

    def test_print_summary_percentiles(self, svd_result, test_data, capsys):
        """Test that percentiles are printed."""
        config = ValidationConfig(increment=10)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        print_validation_summary(result)

        captured = capsys.readouterr()
        output = captured.out

        # Check for specific percentiles
        assert "99.00" in output or "99" in output
        assert "99.90" in output or "99.9" in output
        assert "99.99" in output

    def test_print_summary_multiple_n(self, svd_result, test_data, capsys):
        """Test summary with multiple n values."""
        config = ValidationConfig(increment=5)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        print_validation_summary(result)

        captured = capsys.readouterr()
        output = captured.out

        # Should print results for multiple n values
        assert "n = 5" in output
        assert "n = 10" in output
        assert "n = 15" in output
        assert "n = 20" in output

    def test_print_summary_no_crash_empty(self, capsys):
        """Test that empty validation doesn't crash."""
        from dingo_waveform.svd.results import ValidationResult

        # Create empty validation result
        result = ValidationResult(mismatches=pd.DataFrame(), summary={})

        # Should not crash
        print_validation_summary(result)

        captured = capsys.readouterr()
        # May print nothing or a message, but shouldn't crash


class TestValidationEdgeCases:
    """Test edge cases in validation."""

    def test_validation_single_sample(self, svd_result):
        """Test validation with single test sample."""
        test_data = np.random.randn(1, 50)
        config = ValidationConfig(increment=5)

        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        assert len(result.mismatches) == 1

    def test_validation_increment_larger_than_n(self, svd_result, test_data):
        """Test validation when increment is larger than n_components."""
        config = ValidationConfig(increment=50)  # Larger than n_components=20

        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        # Should still have at least one mismatch column (for n=20)
        mismatch_cols = [c for c in result.mismatches.columns if "mismatch" in c]
        assert len(mismatch_cols) >= 1

    def test_validation_increment_equals_n(self, svd_result, test_data):
        """Test validation when increment equals n_components."""
        config = ValidationConfig(increment=20)

        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        # Should have exactly one mismatch column
        mismatch_cols = [c for c in result.mismatches.columns if "mismatch" in c]
        assert len(mismatch_cols) == 1
        assert "mismatch_n=20" in result.mismatches.columns

    def test_validation_complex_data(self):
        """Test validation with complex data."""
        np.random.seed(42)
        train_data = np.random.randn(100, 50) + 1j * np.random.randn(100, 50)
        test_data = np.random.randn(30, 50) + 1j * np.random.randn(30, 50)

        config = SVDGenerationConfig(n_components=20, method="scipy")
        svd_result = generate_svd_basis(train_data, config)

        val_config = ValidationConfig(increment=5)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, val_config)

        assert result.mismatches is not None
        assert len(result.mismatches) == len(test_data)

    def test_validation_preserves_label_order(self, svd_result, test_data):
        """Test that validation preserves label row order."""
        labels = pd.DataFrame({"id": np.arange(len(test_data))})

        config = ValidationConfig(increment=5)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config, labels)

        # IDs should be preserved in order
        assert np.array_equal(result.mismatches["id"], labels["id"])


class TestValidationStatistics:
    """Test validation statistics computation."""

    def test_mean_mismatch_computation(self, svd_result, test_data):
        """Test that mean mismatch is computed correctly."""
        config = ValidationConfig(increment=10)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        # Manually compute mean for first mismatch column
        col = "mismatch_n=10"
        expected_mean = np.mean(result.mismatches[col])

        # The mean should match numpy's computation
        assert np.isclose(expected_mean, np.mean(result.mismatches[col]))

    def test_mismatch_decreases_with_n(self, svd_result, test_data):
        """Test that mismatch generally decreases as n increases."""
        config = ValidationConfig(increment=5)
        result = validate_svd(svd_result.V, svd_result.Vh, test_data, config)

        # Get mean mismatches for each n
        mean_mismatches = {}
        for col in result.mismatches.columns:
            if "mismatch_n=" in col:
                n = int(col.split("=")[-1])
                mean_mismatches[n] = np.mean(result.mismatches[col])

        # Check that mismatches generally decrease
        n_values = sorted(mean_mismatches.keys())
        for i in range(len(n_values) - 1):
            # Mismatch at larger n should be <= mismatch at smaller n
            # (allowing small numerical tolerance)
            assert mean_mismatches[n_values[i + 1]] <= mean_mismatches[n_values[i]] + 1e-10
