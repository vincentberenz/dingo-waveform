#!/usr/bin/env python3
"""
Basic waveform plotting example.

This script demonstrates how to:
1. Load a configuration file
2. Generate a waveform using dingo-waveform
3. Create interactive plots using the plotting module

Usage:
    python plot_basic_waveform.py [config_file]

If no config file is specified, uses config_basic_frequency.yaml
"""

import sys
from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import Domain
from dingo_waveform.imports import read_file
from dingo_waveform.plotting import plot_polarizations_frequency
from dingo_waveform.polarizations import Polarization
from dingo_waveform.waveform_generator import WaveformGenerator, build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters


def main() -> None:
    # Determine config file
    config_file: Path
    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    else:
        config_file = Path(__file__).parent / "config_basic_frequency.yaml"

    print(f"Loading configuration from: {config_file}")

    # Load configuration
    config: Dict[str, Any] = read_file(config_file)

    # Build waveform generator
    print("\nBuilding waveform generator...")
    wfg: WaveformGenerator = build_waveform_generator(config)

    # Create waveform parameters
    print("Creating waveform parameters...")
    params: WaveformParameters = WaveformParameters(**config["waveform_parameters"])

    # Generate waveform
    print("\nGenerating waveform...")
    polarization: Polarization = wfg.generate_hplus_hcross(params)

    # Create plot
    print("Creating interactive plot...")
    domain: Domain = wfg._waveform_gen_params.domain
    approximant: Approximant = wfg._waveform_gen_params.approximant
    fig: go.Figure = plot_polarizations_frequency(
        polarization,
        domain,
        title=f"Waveform: {approximant} (M1={params.mass_1}M☉, M2={params.mass_2}M☉)",
    )

    # Display plot
    print("\nDisplaying plot in browser...")
    print("(Close browser tab to continue)\n")
    fig.show()

    # Optionally save to HTML
    output_file: str = config_file.stem + "_plot.html"
    fig.write_html(output_file)
    print(f"Plot saved to: {output_file}")


if __name__ == "__main__":
    main()
