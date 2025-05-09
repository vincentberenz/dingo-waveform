import json

import pytest
import tomli

from dingo_waveform.domains import FrequencyDomain
from dingo_waveform.waveform_generator import (
    WaveformGenerator,
    build_waveform_generator,
)
from dingo_waveform.waveform_generator_parameters import WaveformGeneratorParameters


@pytest.fixture
def domain():
    """Create a test frequency domain."""
    return FrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)


@pytest.fixture
def generator_parameters():
    """Create test waveform generator parameters."""
    return WaveformGeneratorParameters(
        approximant="IMRPhenomPv2",
        f_ref=20.0,
        spin_conversion_phase=0.0,
    )


@pytest.fixture
def config_dict():
    """Create a test configuration dictionary."""
    return {
        "domain": {
            "type": "dingo_waveform.domains.FrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,
        },
        "waveform_generator": {
            "approximant": "IMRPhenomPv2",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
    }


@pytest.fixture
def config_file_json(config_dict, tmp_path):
    """Create a temporary JSON config file."""
    file_path = tmp_path / "config.json"
    with open(file_path, "w") as f:
        json.dump(config_dict, f)
    return file_path


@pytest.fixture
def config_file_toml(config_dict, tmp_path):
    """Create a temporary TOML config file."""
    file_path = tmp_path / "config.toml"
    with open(file_path, "wb") as f:
        tomli.dump(config_dict, f)
    return file_path


def test_build_waveform_generator_from_parameters(domain, generator_parameters):
    """Test building a waveform generator from parameters."""
    generator = build_waveform_generator(generator_parameters, domain)
    _assert_waveform_generator(generator)


def test_build_waveform_generator_from_dict(domain):
    """Test building a waveform generator from a dictionary."""
    params_dict = {
        "approximant": "IMRPhenomPv2",
        "f_ref": 20.0,
        "spin_conversion_phase": 0.0,
    }
    generator = build_waveform_generator(params_dict, domain)
    _assert_waveform_generator(generator)


def _assert_waveform_generator(generator):
    """Helper function to assert common properties of a waveform generator."""
    assert isinstance(generator, WaveformGenerator)
    assert generator._waveform_gen_params.approximant == "IMRPhenomPv2"
    assert generator._waveform_gen_params.f_ref == 20.0
    assert generator._waveform_gen_params.spin_conversion_phase == 0.0
    assert isinstance(generator, WaveformGenerator)
    assert generator._waveform_gen_params.approximant == "IMRPhenomPv2"
    assert generator._waveform_gen_params.f_ref == 20.0
    assert generator._waveform_gen_params.spin_conversion_phase == 0.0
    assert isinstance(generator._waveform_gen_params.domain, FrequencyDomain)
    assert generator._waveform_gen_params.domain.f_min == 20.0
    assert generator._waveform_gen_params.domain.f_max == 1024.0
    assert generator._waveform_gen_params.domain.delta_f == 0.125


def test_build_waveform_generator_from_json_file(config_file_json):
    """Test building a waveform generator from a JSON file."""
    generator = build_waveform_generator(config_file_json)
    _assert_waveform_generator(generator)


def test_build_waveform_generator_from_toml_file(config_file_toml):
    """Test building a waveform generator from a TOML file."""
    generator = build_waveform_generator(config_file_toml)
    _assert_waveform_generator(generator)


def test_build_waveform_generator_invalid_file_format(tmp_path):
    """Test that building from an invalid file format raises an error."""
    file_path = tmp_path / "config.txt"
    file_path.write_text("invalid format")
    with pytest.raises(ValueError, match="Unsupported file format"):
        build_waveform_generator(file_path)


def test_build_waveform_generator_missing_keys(tmp_path):
    """Test that building from a file with missing keys raises an error."""
    file_path = tmp_path / "config.json"
    with open(file_path, "w") as f:
        json.dump({"domain": {}}, f)
    with pytest.raises(KeyError, match="Missing required key"):
        build_waveform_generator(file_path)


def test_build_waveform_generator_invalid_dict(domain):
    """Test that building from an invalid dictionary raises an error."""
    with pytest.raises(ValueError):
        build_waveform_generator({"invalid": "params"}, domain)
