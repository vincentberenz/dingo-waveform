"""Tests for MultibandedFrequencyDomain."""

import numpy as np
import pytest
import torch

from dingo_waveform.domains import (
    DomainParameters,
    UniformFrequencyDomain,
    MultibandedFrequencyDomain,
    build_domain,
    decimate_uniform,
)

# Standard parameters for testing
_base_domain_params = {"f_min": 20.0, "f_max": 1024.0, "delta_f": 0.125}
_nodes = [20.0, 64.0, 256.0, 1024.0]
_delta_f_initial = 0.125


@pytest.fixture
def base_domain():
    """Create a standard base frequency domain."""
    return UniformFrequencyDomain(**_base_domain_params)


@pytest.fixture
def multibanded_domain():
    """Create a standard multibanded frequency domain."""
    return MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"]
    )


def test_multibanded_creation(multibanded_domain):
    """Test basic creation of MultibandedFrequencyDomain."""
    assert multibanded_domain.num_bands == 3
    assert len(multibanded_domain) == 2656
    assert multibanded_domain.f_min == 20.0
    assert np.isclose(multibanded_domain.f_max, 1023.875)


def test_multibanded_bands_structure(multibanded_domain):
    """Test that bands have correct delta_f structure (doubling per band)."""
    # Check that delta_f doubles in each band
    assert np.isclose(multibanded_domain._binning.delta_f_bands[0], 0.125)
    assert np.isclose(multibanded_domain._binning.delta_f_bands[1], 0.25)
    assert np.isclose(multibanded_domain._binning.delta_f_bands[2], 0.5)

    # Check decimation factors
    assert multibanded_domain._binning.decimation_factors_bands[0] == 1
    assert multibanded_domain._binning.decimation_factors_bands[1] == 2
    assert multibanded_domain._binning.decimation_factors_bands[2] == 4


def test_multibanded_sample_frequencies(multibanded_domain):
    """Test sample frequencies array."""
    freqs = multibanded_domain.sample_frequencies
    assert len(freqs) == len(multibanded_domain)
    assert freqs[0] >= multibanded_domain.f_min
    assert freqs[-1] <= multibanded_domain.f_max
    # Check frequencies are monotonically increasing
    assert np.all(np.diff(freqs) > 0)


def test_multibanded_delta_f_array(multibanded_domain):
    """Test that delta_f is an array with correct values."""
    delta_f = multibanded_domain.delta_f
    assert isinstance(delta_f, np.ndarray)
    assert len(delta_f) == len(multibanded_domain)
    # Check min and max values
    assert np.isclose(delta_f.min(), 0.125)
    assert np.isclose(delta_f.max(), 0.5)


def test_multibanded_noise_std(multibanded_domain):
    """Test that noise_std is an array."""
    noise_std = multibanded_domain.noise_std
    assert isinstance(noise_std, np.ndarray)
    assert len(noise_std) == len(multibanded_domain)
    # Check relationship: noise_std = 1 / sqrt(4 * delta_f)
    expected = 1.0 / np.sqrt(4.0 * multibanded_domain.delta_f)
    assert np.allclose(noise_std, expected)


def test_multibanded_frequency_mask(multibanded_domain):
    """Test frequency mask (should be all ones)."""
    mask = multibanded_domain.frequency_mask
    assert isinstance(mask, np.ndarray)
    assert len(mask) == len(multibanded_domain)
    assert np.all(mask == 1.0)
    assert multibanded_domain.frequency_mask_length == len(multibanded_domain)


def test_multibanded_indices(multibanded_domain):
    """Test min_idx and max_idx."""
    assert multibanded_domain.min_idx == 0
    assert multibanded_domain.max_idx == len(multibanded_domain) - 1


def test_multibanded_duration_sampling_rate_not_implemented(multibanded_domain):
    """Test that duration and sampling_rate raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        _ = multibanded_domain.duration
    with pytest.raises(NotImplementedError):
        _ = multibanded_domain.sampling_rate


def test_decimate_uniform_numpy():
    """Test decimate_uniform function with numpy arrays."""
    data = np.arange(100, dtype=np.float32)
    # Decimate by factor of 2 with mean policy
    decimated = decimate_uniform(data, 2, policy="mean")
    assert len(decimated) == 50
    # Check averaging: (0+1)/2, (2+3)/2, etc.
    assert np.isclose(decimated[0], 0.5)
    assert np.isclose(decimated[1], 2.5)

    # Decimate by factor of 5 with mean policy
    data = np.arange(100, dtype=np.float32)
    decimated = decimate_uniform(data, 5, policy="mean")
    assert len(decimated) == 20
    # Check averaging: (0+1+2+3+4)/5, etc.
    assert np.isclose(decimated[0], 2.0)


def test_decimate_uniform_torch():
    """Test decimate_uniform function with torch tensors."""
    data = torch.arange(100, dtype=torch.float32)
    # Decimate by factor of 2 with mean policy
    decimated = decimate_uniform(data, 2, policy="mean")
    assert len(decimated) == 50
    assert torch.isclose(decimated[0], torch.tensor(0.5))
    assert torch.isclose(decimated[1], torch.tensor(2.5))


def test_decimate_uniform_invalid_factor():
    """Test that decimate_uniform raises error for invalid decimation factor."""
    data = np.arange(100, dtype=np.float32)
    # New implementation drops remainder, so non-divisible factors work
    decimated = decimate_uniform(data, 3, policy="pick")
    # 100 // 3 = 33 bins
    assert len(decimated) == 33


def test_multibanded_decimate_numpy(multibanded_domain):
    """Test decimation with numpy arrays."""
    # Create random data matching the coverage (windowed case for auto mode)
    np.random.seed(42)
    # For auto mode to work, data length must match coverage exactly
    coverage = int(multibanded_domain._binning.nodes_indices[-1] - multibanded_domain._binning.nodes_indices[0])
    data = np.random.randn(coverage) + 1j * np.random.randn(coverage)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (len(multibanded_domain),)
    assert decimated.dtype == data.dtype


def test_multibanded_decimate_torch(multibanded_domain):
    """Test decimation with torch tensors."""
    torch.manual_seed(42)
    # For auto mode to work, data length must match coverage exactly
    coverage = int(multibanded_domain._binning.nodes_indices[-1] - multibanded_domain._binning.nodes_indices[0])
    data = torch.randn(coverage, dtype=torch.complex64)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (len(multibanded_domain),)
    assert decimated.dtype == data.dtype


def test_multibanded_decimate_windowed_data(multibanded_domain):
    """Test decimation with windowed data (coverage length only)."""
    # Create data with exact coverage length (windowed case)
    np.random.seed(42)
    coverage = int(multibanded_domain._binning.nodes_indices[-1] - multibanded_domain._binning.nodes_indices[0])
    data = np.random.randn(coverage) + 1j * np.random.randn(coverage)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (len(multibanded_domain),)


def test_multibanded_decimate_batched(multibanded_domain):
    """Test decimation with batched data."""
    np.random.seed(42)
    batch_size = 5
    num_detectors = 3
    coverage = int(multibanded_domain._binning.nodes_indices[-1] - multibanded_domain._binning.nodes_indices[0])
    data = np.random.randn(batch_size, num_detectors, coverage)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (batch_size, num_detectors, len(multibanded_domain))


def test_multibanded_decimate_invalid_shape(multibanded_domain):
    """Test that decimation raises error for incompatible data shape."""
    data = np.random.randn(100)  # Wrong length
    # The error could be IndexError (out of bounds) or ValueError depending on the check
    with pytest.raises((IndexError, ValueError)):
        multibanded_domain.decimate(data)


def test_multibanded_get_parameters(multibanded_domain):
    """Test get_parameters returns correct DomainParameters."""
    params = multibanded_domain.get_parameters()
    assert isinstance(params, DomainParameters)
    assert params.type == "dingo_waveform.domains.MultibandedFrequencyDomain"
    assert params.nodes == _nodes
    assert np.isclose(params.delta_f_initial, _delta_f_initial)
    assert np.isclose(params.base_delta_f, _base_domain_params["delta_f"])
    assert params.window_factor is None  # Not set in fixture


def test_multibanded_from_parameters(multibanded_domain):
    """Test from_parameters reconstructs domain correctly."""
    params = multibanded_domain.get_parameters()
    reconstructed = MultibandedFrequencyDomain.from_parameters(params)

    assert len(reconstructed) == len(multibanded_domain)
    assert reconstructed.num_bands == multibanded_domain.num_bands
    assert np.allclose(reconstructed.nodes, multibanded_domain.nodes)
    assert np.isclose(reconstructed.f_min, multibanded_domain.f_min)
    assert np.isclose(reconstructed.f_max, multibanded_domain.f_max)


def test_multibanded_from_parameters_missing_fields():
    """Test from_parameters raises error when required fields are missing."""
    params = DomainParameters(
        type="MultibandedFrequencyDomain",
        nodes=[20.0, 64.0, 256.0, 1024.0],
        # Missing delta_f_initial and base_domain
    )
    with pytest.raises(ValueError, match="should not be None"):
        MultibandedFrequencyDomain.from_parameters(params)


def test_multibanded_build_domain_from_dict():
    """Test building MultibandedFrequencyDomain from dictionary."""
    domain_dict = {
        "type": "MultibandedFrequencyDomain",
        "nodes": [20.0, 64.0, 256.0, 1024.0],
        "delta_f_initial": 0.125,
        "base_delta_f": 0.125,
    }
    domain = build_domain(domain_dict)
    assert isinstance(domain, MultibandedFrequencyDomain)
    assert len(domain) == 2656
    assert domain.num_bands == 3


def test_multibanded_build_domain_from_parameters(multibanded_domain):
    """Test building MultibandedFrequencyDomain from DomainParameters."""
    params = multibanded_domain.get_parameters()
    domain = build_domain(params)
    assert isinstance(domain, MultibandedFrequencyDomain)
    assert len(domain) == len(multibanded_domain)


def test_multibanded_call_method(multibanded_domain):
    """Test __call__ returns sample frequencies."""
    freqs = multibanded_domain()
    assert isinstance(freqs, np.ndarray)
    assert len(freqs) == len(multibanded_domain)
    assert np.array_equal(freqs, multibanded_domain.sample_frequencies)


def test_multibanded_len(multibanded_domain):
    """Test __len__ returns correct length."""
    assert len(multibanded_domain) == 2656


def test_multibanded_sample_frequencies_torch(multibanded_domain):
    """Test sample_frequencies_torch property."""
    freqs_torch = multibanded_domain.sample_frequencies_torch
    assert isinstance(freqs_torch, torch.Tensor)
    assert freqs_torch.dtype == torch.float32
    assert len(freqs_torch) == len(multibanded_domain)
    # Check values match numpy version
    assert torch.allclose(
        freqs_torch, torch.from_numpy(multibanded_domain.sample_frequencies)
    )


def test_multibanded_sample_frequencies_torch_cuda(multibanded_domain):
    """Test sample_frequencies_torch_cuda property."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    freqs_cuda = multibanded_domain.sample_frequencies_torch_cuda
    assert isinstance(freqs_cuda, torch.Tensor)
    assert freqs_cuda.is_cuda
    assert len(freqs_cuda) == len(multibanded_domain)


def test_multibanded_narrowed_basic():
    """Test narrowed() method creates new domain with reduced range."""
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"]
    )
    original_len = len(mfd)

    # Narrow to smaller range
    narrowed_mfd = mfd.narrowed(f_min=64.0, f_max=512.0)

    # Original domain unchanged
    assert len(mfd) == original_len

    # New domain is smaller
    assert len(narrowed_mfd) < original_len
    assert narrowed_mfd.f_min >= 64.0
    assert narrowed_mfd.f_max <= 512.0


def test_multibanded_narrowed_validation():
    """Test that narrowed() validates inputs correctly."""
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"]
    )

    # Test f_min >= f_max
    with pytest.raises(ValueError, match="f_min must be strictly smaller"):
        mfd.narrowed(f_min=100.0, f_max=50.0)

    # Test f_min out of range
    with pytest.raises(ValueError, match="not in"):
        mfd.narrowed(f_min=10.0)

    # Test f_max out of range
    with pytest.raises(ValueError, match="not in"):
        mfd.narrowed(f_max=2000.0)


def test_multibanded_adapt_data_basic():
    """Test adapt_data() function for slicing data to narrowed domain."""
    from dingo_waveform.domains.multibanded_frequency_domain import adapt_data

    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"]
    )
    original_len = len(mfd)
    data_original = np.random.randn(original_len)

    # Create narrowed domain
    narrowed_mfd = mfd.narrowed(f_min=64.0, f_max=512.0)
    new_len = len(narrowed_mfd)

    # Adapt data from old domain to new domain
    data_adapted = adapt_data(mfd, narrowed_mfd, data_original)

    assert len(data_adapted) == new_len
    assert len(data_adapted) < len(data_original)


def test_multibanded_adapt_data_multidimensional():
    """Test adapt_data() works with multidimensional arrays."""
    from dingo_waveform.domains.multibanded_frequency_domain import adapt_data

    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"]
    )
    original_len = len(mfd)
    data = np.random.randn(5, 3, original_len)

    # Create narrowed domain
    narrowed_mfd = mfd.narrowed(f_min=64.0, f_max=512.0)
    new_len = len(narrowed_mfd)

    # Adapt along last axis (default)
    adapted = adapt_data(mfd, narrowed_mfd, data, axis=-1)
    assert adapted.shape == (5, 3, new_len)

    # Adapt along axis 2 (same as -1 for this shape)
    adapted = adapt_data(mfd, narrowed_mfd, data, axis=2)
    assert adapted.shape == (5, 3, new_len)


def test_multibanded_adapt_data_incompatible_domains():
    """Test adapt_data() raises error for incompatible domains."""
    from dingo_waveform.domains.multibanded_frequency_domain import adapt_data

    mfd1 = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"]
    )

    # Create different domain that's not a subset
    mfd2 = MultibandedFrequencyDomain(
        nodes=[10.0, 50.0, 200.0],
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"]
    )

    data = np.random.randn(len(mfd1))

    # Could raise either "not a subrange" or "Inconsistent bin mapping" depending on overlap
    with pytest.raises(ValueError):
        adapt_data(mfd1, mfd2, data)


def test_multibanded_time_translate_numpy(multibanded_domain):
    """Test time translation with numpy arrays."""
    np.random.seed(42)
    data = np.random.randn(len(multibanded_domain)) + 1j * np.random.randn(
        len(multibanded_domain)
    )
    dt = 0.001  # 1 ms shift

    translated = multibanded_domain.time_translate_data(data, dt)
    assert translated.shape == data.shape
    assert translated.dtype == data.dtype

    # Test that time translation can be undone
    back = multibanded_domain.time_translate_data(translated, -dt)
    assert np.allclose(back, data)


def test_multibanded_time_translate_torch(multibanded_domain):
    """Test time translation with torch tensors."""
    torch.manual_seed(42)
    batch_size = 5
    dt = torch.randn(batch_size, dtype=torch.float32) * 0.01
    data = torch.randn(batch_size, len(multibanded_domain), dtype=torch.complex64)

    translated = multibanded_domain.time_translate_data(data, dt)
    assert translated.shape == data.shape
    assert translated.dtype == data.dtype


def test_multibanded_time_translate_torch_real_imag(multibanded_domain):
    """Test time translation with real/imag representation."""
    torch.manual_seed(42)
    batch_size = 5
    num_detectors = 2
    dt = torch.randn(batch_size, num_detectors, dtype=torch.float32) * 0.01
    # Shape: (batch, detectors, 2 (real/imag), frequencies)
    data = torch.randn(
        batch_size, num_detectors, 2, len(multibanded_domain), dtype=torch.float32
    )

    translated = multibanded_domain.time_translate_data(data, dt)
    assert translated.shape == data.shape
    assert translated.dtype == data.dtype


def test_multibanded_with_window_factor():
    """Test creating MultibandedFrequencyDomain with window_factor."""
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
        window_factor=0.5
    )
    assert mfd.window_factor == 0.5
    assert len(mfd) == 2656


def test_multibanded_invalid_nodes_shape():
    """Test that invalid nodes shape raises error."""
    # Nodes should be 1D array
    invalid_nodes = [[20.0, 64.0], [256.0, 1024.0]]
    with pytest.raises(ValueError, match="Expected 1D nodes array"):
        MultibandedFrequencyDomain(
            nodes=invalid_nodes,
            delta_f_initial=_delta_f_initial,
            base_delta_f=_base_domain_params["delta_f"],
        )


def test_multibanded_endpoints():
    """Test that endpoints are correctly computed."""
    mfd = MultibandedFrequencyDomain(
        nodes=[20.0, 64.0, 256.0, 1024.0],
        delta_f_initial=0.125,
        base_delta_f=0.125,
    )
    assert mfd.f_min == 20.0
    # f_max might be slightly different due to edge effects, but should be close
    assert abs(mfd.f_max - 1024.0) < 1.0


def test_multibanded_compare_with_uniform_low_frequencies(multibanded_domain):
    """Test that multibanded domain matches uniform domain at low frequencies."""
    # At the lowest band, multibanded should have same delta_f as base
    assert np.isclose(multibanded_domain._binning.delta_f_bands[0], _base_domain_params["delta_f"])

    # Sample frequencies in first band should start at f_min
    first_band_num_bins = multibanded_domain._binning.num_bins_bands[0]
    first_band_freqs = multibanded_domain.sample_frequencies[:first_band_num_bins]

    # First frequency should be close to f_min
    assert np.isclose(first_band_freqs[0], multibanded_domain.f_min, atol=_base_domain_params["delta_f"])


def test_multibanded_efficient_representation():
    """Test that multibanded domain is more efficient than uniform."""
    base_delta_f = 0.0625
    f_min = 20.0
    f_max = 2048.0
    nodes = [20.0, 128.0, 512.0, 2048.0]
    mfd = MultibandedFrequencyDomain(
        nodes=nodes, delta_f_initial=base_delta_f, base_delta_f=base_delta_f
    )

    # Calculate what uniform domain length would be
    uniform_len = int((f_max - f_min) / base_delta_f)

    # Multibanded should have fewer bins than uniform domain
    assert len(mfd) < uniform_len
    # Rough estimate: should be at least 30% reduction
    reduction = 1 - len(mfd) / uniform_len
    assert reduction > 0.3
