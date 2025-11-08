#!/usr/bin/env python3
"""
Quick test script for the plotting module.

This generates a simple waveform and creates various plots to verify
the plotting functions work correctly.
"""

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import UniformFrequencyDomain
from dingo_waveform.waveform_generator import WaveformGenerator
from dingo_waveform.waveform_parameters import WaveformParameters
from dingo_waveform.plotting import (
    plot_polarizations_frequency,
    plot_mode_amplitudes,
    plot_individual_modes,
)

def main():
    print("Testing dingo-waveform plotting module...")
    print("=" * 60)

    # Create a simple waveform
    print("\n1. Creating waveform generator...")
    domain = UniformFrequencyDomain(
        f_min=20.0,
        f_max=512.0,
        delta_f=0.25,
    )

    wfg = WaveformGenerator(
        approximant=Approximant("IMRPhenomXPHM"),
        domain=domain,
        f_ref=20.0,
        spin_conversion_phase=0.0,
    )
    print(f"   Domain: {len(domain)} frequency bins")

    # Generate waveform parameters
    print("\n2. Generating waveform parameters...")
    params = WaveformParameters(
        mass_1=36.0,
        mass_2=29.0,
        chirp_mass=28.095556,
        mass_ratio=0.805556,
        luminosity_distance=1000.0,
        theta_jn=0.5,
        phase=0.0,
        a_1=0.3,
        a_2=0.2,
        tilt_1=0.5,
        tilt_2=0.8,
        phi_12=1.7,
        phi_jl=0.3,
        geocent_time=0.0,
    )

    # Generate basic polarizations
    print("\n3. Generating basic polarizations (generate_hplus_hcross)...")
    pol = wfg.generate_hplus_hcross(params)
    print(f"   h_plus shape: {pol.h_plus.shape}")
    print(f"   h_cross shape: {pol.h_cross.shape}")

    # Test polarization plotting
    print("\n4. Creating polarization plots...")
    try:
        fig1 = plot_polarizations_frequency(pol, domain, plot_type="amplitude")
        print("   ✓ plot_polarizations_frequency(amplitude) created")

        fig2 = plot_polarizations_frequency(pol, domain, plot_type="both")
        print("   ✓ plot_polarizations_frequency(both) created")

        # Save to HTML
        fig1.write_html("test_polarization_amplitude.html")
        print("   ✓ Saved to test_polarization_amplitude.html")

        fig2.write_html("test_polarization_both.html")
        print("   ✓ Saved to test_polarization_both.html")
    except Exception as e:
        print(f"   ✗ Error creating polarization plots: {e}")
        import traceback
        traceback.print_exc()

    # Generate mode-separated polarizations
    print("\n5. Generating mode-separated polarizations (generate_hplus_hcross_m)...")
    try:
        modes = wfg.generate_hplus_hcross_m(params)
        print(f"   Number of modes: {len(modes)}")
        print(f"   Modes: {list(modes.keys())}")
    except Exception as e:
        print(f"   ✗ Error generating modes: {e}")
        return

    # Test mode plotting
    print("\n6. Creating mode plots...")
    try:
        fig3 = plot_mode_amplitudes(modes, domain)
        print("   ✓ plot_mode_amplitudes created")
        fig3.write_html("test_mode_amplitudes.html")
        print("   ✓ Saved to test_mode_amplitudes.html")

        fig4 = plot_individual_modes(modes, domain, domain_type="frequency")
        print("   ✓ plot_individual_modes created")
        fig4.write_html("test_individual_modes.html")
        print("   ✓ Saved to test_individual_modes.html")
    except Exception as e:
        print(f"   ✗ Error creating mode plots: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("\nGenerated HTML files:")
    print("  - test_polarization_amplitude.html")
    print("  - test_polarization_both.html")
    print("  - test_mode_amplitudes.html")
    print("  - test_individual_modes.html")
    print("\nOpen these files in a web browser to view the interactive plots.")

if __name__ == "__main__":
    main()
