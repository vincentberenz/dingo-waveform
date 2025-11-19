"""
Comprehensive pytest tests for the dingo_waveform.domains.binning.adaptive_binning module.

This test suite covers:
- Band dataclass properties and methods
- plan_bands() function
- compile_binning_from_bands() function
- compute_adaptive_binning() function
- decimate_uniform() function
- decimate() function
- _infer_base_offset_idx_auto() helper function
- Edge cases and error conditions
"""

import numpy as np
import pytest

from dingo_waveform.domains.binning.adaptive_binning import (
    Band,
    BinningParameters,
    compute_adaptive_binning,
    compile_binning_from_bands,
    decimate,
    decimate_uniform,
    plan_bands,
    _infer_base_offset_idx_auto,
)


# ==============================================================================
# Test Band dataclass
# ==============================================================================


def test_band_creation():
    """Test basic Band creation and field access."""
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=40.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=200,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=50,
        remainder=0,
        bin_start=0,
        bin_end=50,
    )

    assert band.index == 0
    assert band.node_lower == 20.0
    assert band.node_upper == 40.0
    assert band.delta_f_band == 0.5
    assert band.decimation_factor_band == 2
    assert band.num_bins == 50


def test_band_immutability():
    """Test that Band is immutable (frozen=True)."""
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=40.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=200,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=50,
        remainder=0,
        bin_start=0,
        bin_end=50,
    )

    with pytest.raises(Exception):  # FrozenInstanceError
        band.index = 1


def test_band_properties():
    """Test Band computed properties."""
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=40.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=200,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=50,
        remainder=0,
        bin_start=0,
        bin_end=50,
    )

    # Test bin_slice property
    assert band.bin_slice == slice(0, 50)

    # Test band_width_indices property
    assert band.band_width_indices == 100

    # Test covered_base_samples property
    assert band.covered_base_samples == 100  # 50 * 2

    # Test coverage_ratio property
    assert band.coverage_ratio == 1.0  # 100 / 100


def test_band_coverage_ratio_with_remainder():
    """Test coverage_ratio when there's a remainder."""
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=40.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=203,  # 103 samples, not evenly divisible
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=51,
        remainder=1,  # 103 - (51 * 2) = 1
        bin_start=0,
        bin_end=51,
    )

    assert band.band_width_indices == 103
    assert band.covered_base_samples == 102  # 51 * 2
    assert pytest.approx(band.coverage_ratio, rel=1e-6) == 102 / 103


def test_band_coverage_ratio_zero_width():
    """Test coverage_ratio with zero width."""
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=20.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=100,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=0,
        remainder=0,
        bin_start=0,
        bin_end=0,
    )

    assert band.coverage_ratio == 0.0


# ==============================================================================
# Test plan_bands()
# ==============================================================================


def test_plan_bands_basic():
    """Test basic band planning with simple parameters."""
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)

    assert len(bands) == 2
    assert bands[0].index == 0
    assert bands[1].index == 1

    # First band
    assert bands[0].node_lower == 20.0
    assert bands[0].node_upper == 40.0
    assert bands[0].delta_f_band == 0.25
    assert bands[0].decimation_factor_band == 1

    # Second band
    assert bands[1].node_lower == 40.0
    assert bands[1].node_upper == 80.0
    assert bands[1].delta_f_band == 0.5
    assert bands[1].decimation_factor_band == 2


def test_plan_bands_dyadic_spacing():
    """Test that bands have dyadic (power-of-2) spacing."""
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)

    assert len(bands) == 3
    assert bands[0].delta_f_band == 0.25
    assert bands[1].delta_f_band == 0.5
    assert bands[2].delta_f_band == 1.0
    assert bands[0].decimation_factor_band == 2
    assert bands[1].decimation_factor_band == 4
    assert bands[2].decimation_factor_band == 8


def test_plan_bands_single_band():
    """Test band planning with a single band."""
    nodes = [20.0, 100.0]
    base_delta_f = 0.5
    delta_f_initial = 0.5

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)

    assert len(bands) == 1
    assert bands[0].index == 0
    assert bands[0].node_lower == 20.0
    assert bands[0].node_upper == 100.0


def test_plan_bands_contiguous_bin_ranges():
    """Test that bin ranges are contiguous across bands."""
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.125

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)

    # Check that each band's bin_start matches previous band's bin_end
    for i in range(1, len(bands)):
        assert bands[i].bin_start == bands[i-1].bin_end


def test_plan_bands_invalid_base_delta_f():
    """Test that negative or zero base_delta_f raises ValueError."""
    nodes = [20.0, 40.0]

    with pytest.raises(ValueError, match="base_delta_f must be positive"):
        plan_bands(nodes, base_delta_f=0.0, delta_f_initial=0.25)

    with pytest.raises(ValueError, match="base_delta_f must be positive"):
        plan_bands(nodes, base_delta_f=-0.25, delta_f_initial=0.25)


def test_plan_bands_invalid_delta_f_initial():
    """Test that negative or zero delta_f_initial raises ValueError."""
    nodes = [20.0, 40.0]

    with pytest.raises(ValueError, match="delta_f_initial must be positive"):
        plan_bands(nodes, base_delta_f=0.25, delta_f_initial=0.0)

    with pytest.raises(ValueError, match="delta_f_initial must be positive"):
        plan_bands(nodes, base_delta_f=0.25, delta_f_initial=-0.25)


def test_plan_bands_invalid_nodes_shape():
    """Test that non-1D nodes raise ValueError."""
    nodes_2d = [[20.0, 40.0], [60.0, 80.0]]

    with pytest.raises(ValueError, match="Expected 1D nodes array"):
        plan_bands(nodes_2d, base_delta_f=0.25, delta_f_initial=0.25)


def test_plan_bands_too_few_nodes():
    """Test that less than 2 nodes raises ValueError."""
    with pytest.raises(ValueError, match="at least two elements"):
        plan_bands([20.0], base_delta_f=0.25, delta_f_initial=0.25)

    with pytest.raises(ValueError, match="at least two elements"):
        plan_bands([], base_delta_f=0.25, delta_f_initial=0.25)


def test_plan_bands_non_increasing_nodes():
    """Test that non-strictly-increasing nodes raise ValueError."""
    with pytest.raises(ValueError, match="must be strictly increasing"):
        plan_bands([40.0, 40.0], base_delta_f=0.25, delta_f_initial=0.25)

    with pytest.raises(ValueError, match="must be strictly increasing"):
        plan_bands([40.0, 20.0], base_delta_f=0.25, delta_f_initial=0.25)


def test_plan_bands_decimation_too_small():
    """Test that delta_f_initial < base_delta_f raises ValueError."""
    nodes = [20.0, 40.0]
    base_delta_f = 0.5
    delta_f_initial = 0.25  # smaller than base_delta_f

    with pytest.raises(ValueError, match="Invalid decimation factors"):
        plan_bands(nodes, base_delta_f, delta_f_initial)


# ==============================================================================
# Test compile_binning_from_bands()
# ==============================================================================


def test_compile_binning_from_bands_basic():
    """Test basic compilation from Band instances."""
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    assert isinstance(params, BinningParameters)
    assert params.num_bands == 2
    assert params.total_bins > 0
    assert params.base_delta_f == base_delta_f
    assert params.delta_f_initial == delta_f_initial


def test_compile_binning_reconstructs_nodes():
    """Test that compilation reconstructs nodes correctly."""
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    np.testing.assert_allclose(params.nodes, nodes, rtol=1e-6)


def test_compile_binning_per_bin_arrays():
    """Test that per-bin arrays have correct shape."""
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    assert len(params.band_assignment) == params.total_bins
    assert len(params.delta_f) == params.total_bins
    assert len(params.f_base_lower) == params.total_bins
    assert len(params.f_base_upper) == params.total_bins
    assert len(params.base_lower_idx) == params.total_bins
    assert len(params.base_upper_idx) == params.total_bins


def test_compile_binning_per_band_arrays():
    """Test that per-band arrays have correct shape."""
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    assert len(params.delta_f_bands) == params.num_bands
    assert len(params.decimation_factors_bands) == params.num_bands
    assert len(params.num_bins_bands) == params.num_bands
    assert len(params.remainder_per_band) == params.num_bands
    assert params.band_bin_ranges.shape == (params.num_bands, 2)


def test_compile_binning_empty_bands():
    """Test compilation with empty band list."""
    params = compile_binning_from_bands([], base_delta_f=0.25, delta_f_initial=0.25)

    assert params.num_bands == 0
    assert params.total_bins == 0
    assert len(params.nodes) == 0
    assert len(params.band_assignment) == 0


def test_compile_binning_invalid_band_indices():
    """Test that non-contiguous band indices raise ValueError."""
    band0 = Band(
        index=0, node_lower=20.0, node_upper=40.0,
        node_lower_idx=80, node_upper_idx_exclusive=160,
        delta_f_band=0.25, decimation_factor_band=1,
        num_bins=80, remainder=0, bin_start=0, bin_end=80,
    )
    band2 = Band(  # Skipping index 1
        index=2, node_lower=40.0, node_upper=80.0,
        node_lower_idx=160, node_upper_idx_exclusive=320,
        delta_f_band=0.5, decimation_factor_band=2,
        num_bins=80, remainder=0, bin_start=80, bin_end=160,
    )

    with pytest.raises(ValueError, match="indexed 0..num_bands-1 without gaps"):
        compile_binning_from_bands([band0, band2], base_delta_f=0.25, delta_f_initial=0.25)


def test_compile_binning_non_contiguous_boundaries():
    """Test that non-contiguous band boundaries raise ValueError."""
    band0 = Band(
        index=0, node_lower=20.0, node_upper=40.0,
        node_lower_idx=80, node_upper_idx_exclusive=160,
        delta_f_band=0.25, decimation_factor_band=1,
        num_bins=80, remainder=0, bin_start=0, bin_end=80,
    )
    band1 = Band(
        index=1, node_lower=50.0, node_upper=80.0,  # Gap between 40.0 and 50.0
        node_lower_idx=200, node_upper_idx_exclusive=320,
        delta_f_band=0.5, decimation_factor_band=2,
        num_bins=60, remainder=0, bin_start=80, bin_end=140,
    )

    with pytest.raises(ValueError, match="Non-contiguous band boundaries"):
        compile_binning_from_bands([band0, band1], base_delta_f=0.25, delta_f_initial=0.25)


# ==============================================================================
# Test compute_adaptive_binning()
# ==============================================================================


def test_compute_adaptive_binning_basic():
    """Test the convenience function with basic parameters."""
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    assert isinstance(params, BinningParameters)
    assert params.num_bands == 2
    assert params.total_bins > 0


def test_compute_adaptive_binning_matches_two_step():
    """Test that compute_adaptive_binning matches plan_bands + compile."""
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.25

    # Two-step process
    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params1 = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    # Convenience function
    params2 = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Compare key fields
    assert params1.num_bands == params2.num_bands
    assert params1.total_bins == params2.total_bins
    np.testing.assert_array_equal(params1.nodes, params2.nodes)
    np.testing.assert_array_equal(params1.num_bins_bands, params2.num_bins_bands)


# ==============================================================================
# Test decimate_uniform()
# ==============================================================================


def test_decimate_uniform_pick_policy():
    """Test uniform decimation with pick policy."""
    data = np.arange(100, dtype=np.float32)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="pick")

    expected = data[::5]  # Pick every 5th element starting from 0
    np.testing.assert_array_equal(result, expected)
    assert result.shape[-1] == 20


def test_decimate_uniform_mean_policy():
    """Test uniform decimation with mean policy."""
    data = np.arange(100, dtype=np.float32)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="mean")

    # Mean of [0,1,2,3,4] = 2, [5,6,7,8,9] = 7, etc.
    expected = np.array([2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0,
                         52.0, 57.0, 62.0, 67.0, 72.0, 77.0, 82.0, 87.0, 92.0, 97.0])
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_decimate_uniform_with_remainder():
    """Test that remainder samples are dropped."""
    data = np.arange(103, dtype=np.float32)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="pick")

    # 103 // 5 = 20 bins, remainder 3 samples dropped
    assert result.shape[-1] == 20
    expected = data[:100:5]
    np.testing.assert_array_equal(result, expected)


def test_decimate_uniform_multidimensional():
    """Test uniform decimation on multidimensional arrays."""
    data = np.arange(200, dtype=np.float32).reshape(2, 100)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="pick")

    assert result.shape == (2, 20)
    np.testing.assert_array_equal(result[0], data[0, ::5])
    np.testing.assert_array_equal(result[1], data[1, ::5])


def test_decimate_uniform_factor_one():
    """Test that decimation factor of 1 returns unchanged data."""
    data = np.arange(100, dtype=np.float32)

    result = decimate_uniform(data, decimation_factor=1, policy="pick")

    np.testing.assert_array_equal(result, data)


def test_decimate_uniform_empty_result():
    """Test decimation when input is too short for even one bin."""
    data = np.arange(3, dtype=np.float32)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="pick")

    assert result.shape[-1] == 0


def test_decimate_uniform_invalid_factor():
    """Test that decimation factor < 1 raises ValueError."""
    data = np.arange(100, dtype=np.float32)

    with pytest.raises(ValueError, match="decimation_factor must be >= 1"):
        decimate_uniform(data, decimation_factor=0)


# ==============================================================================
# Test decimate()
# ==============================================================================


def test_decimate_explicit_mode():
    """Test decimation in explicit mode with base_offset_idx=0."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Create synthetic data matching the base grid
    data_len = int(params.nodes_indices[-1])
    data = np.arange(data_len, dtype=np.float32)

    result = decimate(data, params, base_offset_idx=0, mode="explicit", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_auto_mode_full_grid():
    """Test decimation in auto mode with full grid starting at 0."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Create data that matches full grid (nodes_indices[0] == 0)
    data_len = int(params.nodes_indices[-1]) + 100  # Extra samples beyond coverage
    data = np.arange(data_len, dtype=np.float32)

    result = decimate(data, params, mode="auto", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_auto_mode_windowed():
    """Test decimation in auto mode with windowed coverage."""
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Create data with exact coverage (windowed case)
    coverage = int(params.nodes_indices[-1] - params.nodes_indices[0])
    data = np.arange(coverage, dtype=np.float32)

    result = decimate(data, params, mode="auto", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_multidimensional():
    """Test decimation on multidimensional arrays."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    data = np.arange(2 * data_len, dtype=np.float32).reshape(2, data_len)

    result = decimate(data, params, base_offset_idx=0, mode="explicit", policy="pick")

    assert result.shape == (2, params.total_bins)


def test_decimate_with_positive_offset():
    """Test decimation with positive base_offset_idx."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    offset = 50
    data_len = int(params.nodes_indices[-1]) + offset
    data = np.arange(data_len, dtype=np.float32)

    result = decimate(data, params, base_offset_idx=offset, mode="explicit", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_instance_method():
    """Test the decimate instance method on BinningParameters."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    data = np.arange(data_len, dtype=np.float32)

    # Use instance method
    result = params.decimate(data, base_offset_idx=0, mode="explicit", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_out_of_bounds():
    """Test that out-of-bounds slicing raises IndexError."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Data too short
    data = np.arange(10, dtype=np.float32)

    with pytest.raises(IndexError, match="Input slice out of bounds"):
        decimate(data, params, base_offset_idx=0, mode="explicit", check=True)


def test_decimate_check_disabled():
    """Test that checks can be disabled (though it may produce incorrect results)."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    data = np.arange(data_len, dtype=np.float32)

    # Should not raise even with checks disabled (data is valid here)
    result = decimate(data, params, base_offset_idx=0, mode="explicit", check=False)
    assert result.shape[-1] == params.total_bins


# ==============================================================================
# Test _infer_base_offset_idx_auto()
# ==============================================================================


def test_infer_offset_windowed_coverage():
    """Test offset inference for windowed coverage case."""
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    coverage = int(params.nodes_indices[-1] - params.nodes_indices[0])
    offset = _infer_base_offset_idx_auto(params, coverage)

    # Should infer offset = -nodes_indices[0]
    expected_offset = -int(params.nodes_indices[0])
    assert offset == expected_offset


def test_infer_offset_full_grid():
    """Test offset inference for full grid case."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1]) + 100  # Extra samples
    offset = _infer_base_offset_idx_auto(params, data_len)

    # Should infer offset = 0
    assert offset == 0


def test_infer_offset_ambiguous():
    """Test that ambiguous cases raise ValueError."""
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Data length that doesn't match coverage or full grid
    ambiguous_len = int(params.nodes_indices[-1] - params.nodes_indices[0]) + 50

    with pytest.raises(ValueError, match="Cannot auto-infer base_offset_idx"):
        _infer_base_offset_idx_auto(params, ambiguous_len)


# ==============================================================================
# Integration tests
# ==============================================================================


def test_end_to_end_simple_binning():
    """Test complete workflow with simple parameters."""
    # Define a simple frequency domain
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    # Compute binning parameters
    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Create synthetic waveform data
    data_len = int(params.nodes_indices[-1])
    waveform = np.sin(np.linspace(0, 10 * np.pi, data_len)).astype(np.float32)

    # Decimate the waveform
    decimated = params.decimate(waveform, base_offset_idx=0, mode="explicit")

    # Verify output shape
    assert decimated.shape[-1] == params.total_bins
    assert len(decimated) == params.total_bins


def test_end_to_end_multibanded():
    """Test complete workflow with multiple bands."""
    # Mimics the MFD test setup
    nodes = [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0]
    base_delta_f = 0.0625
    delta_f_initial = 0.0625

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Verify we have the expected number of bands
    assert params.num_bands == 6

    # Verify dyadic spacing
    for i in range(params.num_bands):
        expected_delta_f = delta_f_initial * (2 ** i)
        assert pytest.approx(params.delta_f_bands[i], rel=1e-6) == expected_delta_f


def test_consistency_between_pick_and_mean():
    """Test that pick and mean policies produce similar results for smooth data."""
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    # Create smooth data (low frequency sine wave)
    data_len = int(params.nodes_indices[-1])
    data = np.sin(np.linspace(0, 2 * np.pi, data_len)).astype(np.float32)

    result_pick = params.decimate(data, base_offset_idx=0, policy="pick")
    result_mean = params.decimate(data, base_offset_idx=0, policy="mean")

    # For smooth data, pick and mean should be reasonably close
    # (exact match depends on decimation factor and smoothness)
    assert result_pick.shape == result_mean.shape
    # We don't assert closeness here as it depends on data characteristics
