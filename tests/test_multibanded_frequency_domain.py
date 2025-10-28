"""Tests for MultibandedFrequencyDomain."""

import numpy as np
import pytest
import torch

from dingo_waveform.domains import (
    DomainParameters,
    FrequencyDomain,
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
    return FrequencyDomain(**_base_domain_params)


@pytest.fixture
def multibanded_domain(base_domain):
    """Create a standard multibanded frequency domain."""
    return MultibandedFrequencyDomain(
        nodes=_nodes, delta_f_initial=_delta_f_initial, base_domain=base_domain
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
    assert np.isclose(multibanded_domain._delta_f_bands[0], 0.125)
    assert np.isclose(multibanded_domain._delta_f_bands[1], 0.25)
    assert np.isclose(multibanded_domain._delta_f_bands[2], 0.5)

    # Check decimation factors
    assert multibanded_domain._decimation_factors_bands[0] == 1
    assert multibanded_domain._decimation_factors_bands[1] == 2
    assert multibanded_domain._decimation_factors_bands[2] == 4


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
    data = np.arange(100, dtype=float)
    # Decimate by factor of 2
    decimated = decimate_uniform(data, 2)
    assert len(decimated) == 50
    # Check averaging: (0+1)/2, (2+3)/2, etc.
    assert np.isclose(decimated[0], 0.5)
    assert np.isclose(decimated[1], 2.5)

    # Decimate by factor of 5
    data = np.arange(100, dtype=float)
    decimated = decimate_uniform(data, 5)
    assert len(decimated) == 20
    # Check averaging: (0+1+2+3+4)/5, etc.
    assert np.isclose(decimated[0], 2.0)


def test_decimate_uniform_torch():
    """Test decimate_uniform function with torch tensors."""
    data = torch.arange(100, dtype=torch.float32)
    # Decimate by factor of 2
    decimated = decimate_uniform(data, 2)
    assert len(decimated) == 50
    assert torch.isclose(decimated[0], torch.tensor(0.5))
    assert torch.isclose(decimated[1], torch.tensor(2.5))


def test_decimate_uniform_invalid_factor():
    """Test that decimate_uniform raises error for invalid decimation factor."""
    data = np.arange(100, dtype=float)
    with pytest.raises(ValueError, match="not a multiple"):
        decimate_uniform(data, 3)  # 100 is not divisible by 3


def test_multibanded_decimate_numpy(base_domain, multibanded_domain):
    """Test decimation with numpy arrays."""
    # Create random data in base domain
    np.random.seed(42)
    data = np.random.randn(len(base_domain)) + 1j * np.random.randn(len(base_domain))

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (len(multibanded_domain),)
    assert decimated.dtype == data.dtype


def test_multibanded_decimate_torch(base_domain, multibanded_domain):
    """Test decimation with torch tensors."""
    torch.manual_seed(42)
    data = torch.randn(len(base_domain), dtype=torch.complex64)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (len(multibanded_domain),)
    assert decimated.dtype == data.dtype


def test_multibanded_decimate_truncated_data(base_domain, multibanded_domain):
    """Test decimation with data that starts at f_min (truncated)."""
    # Create data starting from min_idx (no leading zeros)
    np.random.seed(42)
    data_len = len(base_domain) - base_domain.min_idx
    data = np.random.randn(data_len) + 1j * np.random.randn(data_len)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (len(multibanded_domain),)


def test_multibanded_decimate_batched(base_domain, multibanded_domain):
    """Test decimation with batched data."""
    np.random.seed(42)
    batch_size = 5
    num_detectors = 3
    data = np.random.randn(batch_size, num_detectors, len(base_domain))

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (batch_size, num_detectors, len(multibanded_domain))


def test_multibanded_decimate_invalid_shape(multibanded_domain):
    """Test that decimation raises error for incompatible data shape."""
    data = np.random.randn(100)  # Wrong length
    with pytest.raises(ValueError, match="incompatible"):
        multibanded_domain.decimate(data)


def test_multibanded_get_parameters(multibanded_domain):
    """Test get_parameters returns correct DomainParameters."""
    params = multibanded_domain.get_parameters()
    assert isinstance(params, DomainParameters)
    assert params.type == "dingo_waveform.domains.MultibandedFrequencyDomain"
    assert params.nodes == _nodes
    assert np.isclose(params.delta_f_initial, _delta_f_initial)
    assert isinstance(params.base_domain, dict)
    assert params.base_domain["type"] == "dingo_waveform.domains.FrequencyDomain"


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
        "base_domain": {
            "type": "FrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,
        },
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


def test_multibanded_update_with_dict(base_domain):
    """Test update method with dictionary."""
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes, delta_f_initial=_delta_f_initial, base_domain=base_domain
    )
    original_len = len(mfd)

    # Update to smaller range
    mfd.update({"f_min": 30.0, "f_max": 512.0})

    assert len(mfd) < original_len
    assert mfd.f_min >= 30.0
    assert mfd.f_max <= 512.0


def test_multibanded_update_prevents_second_update(base_domain):
    """Test that update cannot be called twice."""
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes, delta_f_initial=_delta_f_initial, base_domain=base_domain
    )
    mfd.update({"f_min": 30.0})

    with pytest.raises(ValueError, match="Can't update domain.*second time"):
        mfd.update({"f_max": 512.0})


def test_multibanded_set_new_range_validation(base_domain):
    """Test _set_new_range validates inputs correctly."""
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes, delta_f_initial=_delta_f_initial, base_domain=base_domain
    )

    # Test f_min >= f_max
    with pytest.raises(ValueError, match="f_min must not be larger"):
        mfd._set_new_range(f_min=100.0, f_max=50.0)

    # Test f_min out of range
    with pytest.raises(ValueError, match="not in expected range"):
        mfd._set_new_range(f_min=10.0)

    # Test f_max out of range
    with pytest.raises(ValueError, match="not in expected range"):
        mfd._set_new_range(f_max=2000.0)


def test_multibanded_update_data_no_change(multibanded_domain):
    """Test update_data returns data unchanged if already compatible."""
    data = np.random.randn(len(multibanded_domain))
    updated = multibanded_domain.update_data(data)
    assert np.array_equal(data, updated)


def test_multibanded_update_data_after_update(base_domain):
    """Test update_data truncates data after domain update."""
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes, delta_f_initial=_delta_f_initial, base_domain=base_domain
    )
    original_len = len(mfd)
    data_original = np.random.randn(original_len)

    # Update domain
    mfd.update({"f_min": 30.0, "f_max": 512.0})
    new_len = len(mfd)

    # Update data
    data_updated = mfd.update_data(data_original)
    assert len(data_updated) == new_len
    assert len(data_updated) < len(data_original)


def test_multibanded_update_data_incompatible_shape(multibanded_domain):
    """Test update_data raises error for incompatible data."""
    data = np.random.randn(100)  # Wrong shape
    with pytest.raises(ValueError, match="incompatible"):
        multibanded_domain.update_data(data)


def test_multibanded_update_data_different_axis(base_domain):
    """Test update_data works with different axis."""
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes, delta_f_initial=_delta_f_initial, base_domain=base_domain
    )
    original_len = len(mfd)
    data = np.random.randn(5, 3, original_len)

    mfd.update({"f_max": 512.0})
    new_len = len(mfd)

    # Update along last axis
    updated = mfd.update_data(data, axis=-1)
    assert updated.shape == (5, 3, new_len)

    # Update along axis 2
    updated = mfd.update_data(data, axis=2)
    assert updated.shape == (5, 3, new_len)


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


def test_multibanded_base_domain_as_dict():
    """Test creating MultibandedFrequencyDomain with base_domain as dict."""
    base_domain_dict = {
        "type": "FrequencyDomain",
        "f_min": 20.0,
        "f_max": 1024.0,
        "delta_f": 0.125,
    }
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes, delta_f_initial=_delta_f_initial, base_domain=base_domain_dict
    )
    assert isinstance(mfd.base_domain, FrequencyDomain)
    assert len(mfd) == 2656


def test_multibanded_invalid_nodes_shape():
    """Test that invalid nodes shape raises error."""
    base_domain = FrequencyDomain(**_base_domain_params)
    # Nodes should be 1D array
    invalid_nodes = [[20.0, 64.0], [256.0, 1024.0]]
    with pytest.raises(ValueError, match="Expected format.*for nodes"):
        MultibandedFrequencyDomain(
            nodes=invalid_nodes,
            delta_f_initial=_delta_f_initial,
            base_domain=base_domain,
        )


def test_multibanded_endpoints_in_base_domain(base_domain):
    """Test that endpoints validation works correctly."""
    # This should work - endpoints are in base domain
    mfd = MultibandedFrequencyDomain(
        nodes=[20.0, 64.0, 256.0, 1024.0],
        delta_f_initial=0.125,
        base_domain=base_domain,
    )
    assert mfd.f_min in base_domain()
    # f_max might be slightly different due to edge effects, but should be close
    assert abs(mfd.f_max - 1024.0) < 1.0


def test_multibanded_compare_with_uniform_low_frequencies(
    base_domain, multibanded_domain
):
    """Test that multibanded domain matches uniform domain at low frequencies."""
    # At the lowest band, multibanded should have same delta_f as base
    assert np.isclose(multibanded_domain._delta_f_bands[0], base_domain.delta_f)

    # Sample frequencies in first band should be similar to base domain
    first_band_freqs = multibanded_domain.sample_frequencies[
        : multibanded_domain._num_bins_bands[0]
    ]
    base_freqs = base_domain()[
        base_domain.min_idx : base_domain.min_idx + len(first_band_freqs)
    ]

    # Should be very close (within delta_f/2)
    assert np.allclose(first_band_freqs, base_freqs, atol=base_domain.delta_f / 2)


def test_multibanded_efficient_representation():
    """Test that multibanded domain is more efficient than uniform."""
    base_domain = FrequencyDomain(f_min=20.0, f_max=2048.0, delta_f=0.0625)
    nodes = [20.0, 128.0, 512.0, 2048.0]
    mfd = MultibandedFrequencyDomain(
        nodes=nodes, delta_f_initial=0.0625, base_domain=base_domain
    )

    # Multibanded should have fewer bins than base domain
    assert len(mfd) < len(base_domain)
    # Rough estimate: should be at least 30% reduction
    reduction = 1 - len(mfd) / len(base_domain)
    assert reduction > 0.3
