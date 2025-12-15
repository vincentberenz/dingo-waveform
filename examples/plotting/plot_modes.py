#!/usr/bin/env python3
"""
Mode-separated waveform plotting example.

This script demonstrates how to:
1. Generate mode-separated waveforms (using generate_hplus_hcross_m)
2. Plot individual modes and their amplitudes
3. Compare mode contributions

Usage:
    python plot_modes.py [config_file]

If no config file is specified, uses config_modes.yaml

Note: Only certain approximants support mode separation (e.g., IMRPhenomXPHM, SEOBNRv4PHM)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import plotly.graph_objects as go

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import Domain
from dingo_waveform.imports import read_file
from dingo_waveform.plotting import (
    plot_mode_amplitudes,
    plot_individual_modes,
    plot_mode_reconstruction,
)
from dingo_waveform.polarizations import Polarization
from dingo_waveform.types import Mode
from dingo_waveform.waveform_generator import WaveformGenerator, build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters


def main() -> None:
    # Determine config file
    config_file: Path
    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    else:
        config_file = Path(__file__).parent / "config_modes.yaml"

    print(f"Loading configuration from: {config_file}")

    # Load configuration
    config: Dict[str, Any] = read_file(config_file)

    # Build waveform generator
    print("\nBuilding waveform generator...")
    wfg: WaveformGenerator = build_waveform_generator(config)

    # Check if approximant supports modes
    approximant_str: str = config["waveform_generator"]["approximant"]
    supported_approximants: List[str] = ["IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM", "SEOBNRv5HM"]
    if approximant_str not in supported_approximants:
        print(f"\nWARNING: {approximant_str} may not support mode separation.")
        print(f"Recommended approximants: {', '.join(supported_approximants)}")
        print("Continuing anyway...\n")

    # Create waveform parameters
    print("Creating waveform parameters...")
    params: WaveformParameters = WaveformParameters(**config["waveform_parameters"])

    # Generate mode-separated waveform
    print("\nGenerating mode-separated waveform...")
    modes: Dict[Mode, Polarization]
    try:
        modes = wfg.generate_hplus_hcross_m(params)
        print(f"Generated {len(modes)} modes: {list(modes.keys())}")
    except Exception as e:
        print(f"\nError: Failed to generate mode-separated waveform:")
        print(f"  {e}")
        print(f"\nThis approximant ({approximant_str}) may not support mode separation.")
        sys.exit(1)

    # Create plots
    print("\nCreating plots...")
    domain: Domain = wfg._waveform_gen_params.domain
    approximant: Approximant = wfg._waveform_gen_params.approximant

    # 1. Mode amplitudes comparison
    print("  - Mode amplitudes plot")
    fig_amplitudes: go.Figure = plot_mode_amplitudes(
        modes,
        domain,
        title=f"Mode Amplitudes: {approximant}",
    )
    fig_amplitudes.show()
    fig_amplitudes.write_html(config_file.stem + "_mode_amplitudes.html")

    # 2. Individual modes
    print("  - Individual modes plot")
    fig_individual: go.Figure = plot_individual_modes(
        modes,
        domain,
        title=f"Individual Modes: {approximant}",
    )
    fig_individual.show()
    fig_individual.write_html(config_file.stem + "_individual_modes.html")

    # 3. Mode reconstruction (shows how modes sum to total polarization)
    print("  - Mode reconstruction plot")
    fig_reconstruction: go.Figure = plot_mode_reconstruction(
        modes,
        domain,
        title=f"Mode Reconstruction: {approximant}",
    )
    fig_reconstruction.show()
    fig_reconstruction.write_html(config_file.stem + "_mode_reconstruction.html")

    print("\nAll plots saved to HTML files.")


if __name__ == "__main__":
    main()
