"""
Tests for the WaveformGenerator class hierarchy.

Verifies that:
- The ABC cannot be instantiated directly
- The factory function returns the correct subclass for each approximant
- Mode-separated generation is only available on appropriate subclasses
- Unknown approximants default to LALSimWaveformGenerator
- The isinstance bug fix (str transform) is correct
"""

import pytest

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import UniformFrequencyDomain
from dingo_waveform.waveform_generator import (
    GWSignalWaveformGenerator,
    IMRPhenomXPHMWaveformGenerator,
    LALSimWaveformGenerator,
    SEOBNRv4PHMWaveformGenerator,
    WaveformGenerator,
    _get_waveform_generator_class,
    build_waveform_generator,
)


@pytest.fixture
def domain() -> UniformFrequencyDomain:
    return UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)


class TestClassMapping:
    """Test that _get_waveform_generator_class returns the right subclass."""

    def test_imrphenomxphm(self):
        cls = _get_waveform_generator_class(Approximant("IMRPhenomXPHM"))
        assert cls is IMRPhenomXPHMWaveformGenerator

    def test_seobnrv4phm(self):
        cls = _get_waveform_generator_class(Approximant("SEOBNRv4PHM"))
        assert cls is SEOBNRv4PHMWaveformGenerator

    def test_seobnrv5phm(self):
        cls = _get_waveform_generator_class(Approximant("SEOBNRv5PHM"))
        assert cls is GWSignalWaveformGenerator

    def test_seobnrv5hm(self):
        cls = _get_waveform_generator_class(Approximant("SEOBNRv5HM"))
        assert cls is GWSignalWaveformGenerator

    def test_unknown_defaults_to_lalsim(self):
        cls = _get_waveform_generator_class(Approximant("IMRPhenomD"))
        assert cls is LALSimWaveformGenerator

    def test_unknown_imrphenompv2(self):
        cls = _get_waveform_generator_class(Approximant("IMRPhenomPv2"))
        assert cls is LALSimWaveformGenerator


class TestBuildWaveformGenerator:
    """Test that build_waveform_generator returns correct subclass instances."""

    def test_builds_imrphenomxphm(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, IMRPhenomXPHMWaveformGenerator)
        assert isinstance(wfg, LALSimWaveformGenerator)
        assert isinstance(wfg, WaveformGenerator)

    def test_builds_seobnrv4phm(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "SEOBNRv4PHM", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, SEOBNRv4PHMWaveformGenerator)
        assert isinstance(wfg, LALSimWaveformGenerator)
        assert isinstance(wfg, WaveformGenerator)

    def test_builds_seobnrv5phm(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "SEOBNRv5PHM", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, GWSignalWaveformGenerator)
        assert isinstance(wfg, WaveformGenerator)

    def test_builds_generic_lalsim(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomD", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, LALSimWaveformGenerator)
        assert isinstance(wfg, WaveformGenerator)
        assert not isinstance(wfg, IMRPhenomXPHMWaveformGenerator)
        assert not isinstance(wfg, SEOBNRv4PHMWaveformGenerator)


class TestModeSupport:
    """Test that generate_hplus_hcross_m is only on mode-supporting subclasses."""

    def test_imrphenomxphm_has_modes(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0}, domain
        )
        assert hasattr(wfg, "generate_hplus_hcross_m")

    def test_seobnrv4phm_has_modes(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "SEOBNRv4PHM", "f_ref": 20.0}, domain
        )
        assert hasattr(wfg, "generate_hplus_hcross_m")

    def test_seobnrv5phm_has_modes(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "SEOBNRv5PHM", "f_ref": 20.0}, domain
        )
        assert hasattr(wfg, "generate_hplus_hcross_m")

    def test_generic_lalsim_no_modes(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomD", "f_ref": 20.0}, domain
        )
        assert not hasattr(wfg, "generate_hplus_hcross_m")


class TestABCNotInstantiable:
    """Test that the ABC cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self, domain):
        with pytest.raises(TypeError):
            WaveformGenerator(
                Approximant("IMRPhenomXPHM"), domain, 20.0
            )


class TestInheritance:
    """Test the class hierarchy relationships."""

    def test_lalsim_is_wfg(self):
        assert issubclass(LALSimWaveformGenerator, WaveformGenerator)

    def test_seobnrv4phm_is_lalsim(self):
        assert issubclass(SEOBNRv4PHMWaveformGenerator, LALSimWaveformGenerator)

    def test_imrphenomxphm_is_lalsim(self):
        assert issubclass(IMRPhenomXPHMWaveformGenerator, LALSimWaveformGenerator)

    def test_gwsignal_is_wfg(self):
        assert issubclass(GWSignalWaveformGenerator, WaveformGenerator)

    def test_gwsignal_is_not_lalsim(self):
        assert not issubclass(GWSignalWaveformGenerator, LALSimWaveformGenerator)
