"""
Extended tests to improve code coverage for time domain and mode-separated functions.

This module adds tests that exercise code paths not heavily covered by existing tests.
"""

import numpy as np
import pytest
from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import UniformFrequencyDomain
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.waveform_parameters import WaveformParameters
from dingo_waveform.polarizations import Polarization


class TestTimeDomainWaveforms:
    """Test time domain waveform generation through WaveformGenerator."""

    def get_basic_params(self):
        """Get basic waveform parameters."""
        return WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

    def test_time_domain_imrphenomxphm(self):
        """Test time domain generation with IMRPhenomXPHM."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )
        params = self.get_basic_params()

        pol = wfg.generate_hplus_hcross(params)
        assert isinstance(pol, Polarization)
        assert pol.h_plus.shape[0] > 0
        assert np.any(np.abs(pol.h_plus) > 0)

    def test_time_domain_with_f_start(self):
        """Test time domain with f_start parameter."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
            f_start=15.0,
        )
        params = self.get_basic_params()

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_time_domain_seobnrv5phm(self):
        """Test time domain with SEOBNRv5PHM."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("SEOBNRv5PHM"),
            domain,
            f_ref=20.0,
        )
        params = self.get_basic_params()

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0
        assert np.any(np.abs(pol.h_plus) > 0)

    def test_time_domain_imrphenomd(self):
        """Test time domain with IMRPhenomD."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomD"),
            domain,
            f_ref=20.0,
        )
        # IMRPhenomD requires aligned spins
        params = WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            chi_1=0.3,
            chi_2=0.2,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_time_domain_aligned_spin(self):
        """Test time domain with aligned spins."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("SEOBNRv4"),
            domain,
            f_ref=20.0,
        )

        # Aligned spins
        params = WaveformParameters(
            mass_1=35.0,
            mass_2=30.0,
            luminosity_distance=200.0,
            theta_jn=1.0,
            phase=0.5,
            chi_1=0.3,
            chi_2=-0.2,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_time_domain_extreme_spins(self):
        """Test time domain with high spins."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        params = WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=150.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.95,
            a_2=0.90,
            tilt_1=1.5,
            tilt_2=1.2,
            phi_12=2.0,
            phi_jl=1.5,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_time_domain_high_mass_ratio(self):
        """Test time domain with high mass ratio."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        params = WaveformParameters(
            mass_1=50.0,
            mass_2=10.0,  # q = 5
            luminosity_distance=200.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_time_domain_various_inclinations(self):
        """Test time domain with different inclinations."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        for theta_jn in [0.1, np.pi/4, np.pi/2, 3*np.pi/4]:
            params = WaveformParameters(
                mass_1=30.0,
                mass_2=25.0,
                luminosity_distance=150.0,
                theta_jn=theta_jn,
                phase=0.5,
                a_1=0.3,
                a_2=0.2,
                tilt_1=0.5,
                tilt_2=0.3,
                phi_12=1.0,
                phi_jl=0.3,
                geocent_time=0.0,
            )

            pol = wfg.generate_hplus_hcross(params)
            assert pol.h_plus.shape[0] > 0


class TestModeSeparatedWaveforms:
    """Test mode-separated waveform generation."""

    def get_basic_params(self):
        """Get basic waveform parameters."""
        return WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

    def test_mode_separated_imrphenomxphm(self):
        """Test mode-separated with IMRPhenomXPHM."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )
        params = self.get_basic_params()

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert isinstance(pol_m, dict)
        assert len(pol_m) > 0

        for m, pol in pol_m.items():
            assert pol.h_plus.shape[0] > 0
            assert pol.h_cross.shape[0] > 0

    def test_mode_separated_different_masses(self):
        """Test mode-separated with different mass configuration."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )
        params = WaveformParameters(
            mass_1=40.0,
            mass_2=20.0,  # q = 2
            luminosity_distance=150.0,
            theta_jn=0.5,
            phase=1.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.6,
            tilt_2=0.4,
            phi_12=1.5,
            phi_jl=0.5,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_lower_frequency(self):
        """Test mode-separated with lower frequency range."""
        domain = UniformFrequencyDomain(f_min=15.0, f_max=256.0, delta_f=0.125)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=15.0,
        )
        params = WaveformParameters(
            mass_1=35.0,
            mass_2=30.0,
            luminosity_distance=200.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.5,
            a_2=0.3,
            tilt_1=0.8,
            tilt_2=0.6,
            phi_12=1.2,
            phi_jl=0.7,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_with_spin_conversion(self):
        """Test mode-separated with spin_conversion_phase."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
            spin_conversion_phase=0.5,
        )
        params = self.get_basic_params()

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_high_spins(self):
        """Test mode-separated with high spins."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        params = WaveformParameters(
            mass_1=35.0,
            mass_2=30.0,
            luminosity_distance=200.0,
            theta_jn=1.2,
            phase=1.5,
            a_1=0.9,
            a_2=0.85,
            tilt_1=1.5,
            tilt_2=1.0,
            phi_12=2.5,
            phi_jl=1.8,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_extreme_mass_ratio(self):
        """Test mode-separated with extreme mass ratio."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        params = WaveformParameters(
            mass_1=60.0,
            mass_2=10.0,  # q = 6
            luminosity_distance=300.0,
            theta_jn=0.8,
            phase=1.0,
            a_1=0.5,
            a_2=0.3,
            tilt_1=0.7,
            tilt_2=0.5,
            phi_12=1.5,
            phi_jl=0.8,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_face_on(self):
        """Test mode-separated with face-on orientation."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        params = WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=0.1,  # Face-on
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_edge_on(self):
        """Test mode-separated with edge-on orientation."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        params = WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=np.pi - 0.1,  # Nearly edge-on
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_various_phases(self):
        """Test mode-separated with various phase values."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        for phase in [0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
            params = WaveformParameters(
                mass_1=30.0,
                mass_2=25.0,
                luminosity_distance=100.0,
                theta_jn=1.0,
                phase=phase,
                a_1=0.3,
                a_2=0.2,
                tilt_1=0.5,
                tilt_2=0.3,
                phi_12=1.0,
                phi_jl=0.3,
                geocent_time=0.0,
            )

            pol_m = wfg.generate_hplus_hcross_m(params)
            assert len(pol_m) > 0

    def test_mode_separated_zero_spins(self):
        """Test mode-separated with zero spins."""
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = WaveformGenerator(
            Approximant("IMRPhenomXPHM"),
            domain,
            f_ref=20.0,
        )

        params = WaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.0,
            a_2=0.0,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0
