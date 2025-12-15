#!/usr/bin/env python3
"""
Multibanded frequency domain plotting example.

This script demonstrates how to:
1. Generate waveforms in multibanded frequency domain (dyadic spacing)
2. Visualize the non-uniform frequency sampling
3. Plot waveforms with multibanded structure

Usage:
    python plot_multibanded.py [config_file]

If no config file is specified, uses config_multibanded.yaml
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import Domain, MultibandedFrequencyDomain
from dingo_waveform.imports import read_file
from dingo_waveform.polarizations import Polarization
from dingo_waveform.waveform_generator import WaveformGenerator, build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters


def main() -> None:
    # Determine config file
    config_file: Path
    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    else:
        config_file = Path(__file__).parent / "config_multibanded.yaml"

    print(f"Loading configuration from: {config_file}")

    # Load configuration
    config: Dict[str, Any] = read_file(config_file)

    # Build waveform generator
    print("\nBuilding waveform generator...")
    wfg: WaveformGenerator = build_waveform_generator(config)

    # Get domain and verify it's multibanded
    domain: Domain = wfg._waveform_gen_params.domain
    if not isinstance(domain, MultibandedFrequencyDomain):
        print(f"\nError: Domain type is {type(domain).__name__}, not MultibandedFrequencyDomain")
        print("Please use a configuration with MultibandedFrequencyDomain")
        sys.exit(1)

    # Display domain info
    print(f"\nDomain information:")
    print(f"  Type: {type(domain).__name__}")
    print(f"  Frequency range: {domain.f_min:.1f} - {domain.f_max:.1f} Hz")
    print(f"  Number of samples: {len(domain.sample_frequencies)}")
    print(f"  Nodes: {domain.nodes}")

    # Create waveform parameters
    print("\nCreating waveform parameters...")
    params: WaveformParameters = WaveformParameters(**config["waveform_parameters"])

    # Generate waveform
    print("Generating waveform...")
    polarization: Polarization = wfg.generate_hplus_hcross(params)

    # Create subplots: top shows frequency sampling, bottom shows waveform
    print("\nCreating visualization...")
    fig: go.Figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Multibanded Frequency Sampling",
            "Waveform: h<sub>+</sub> Amplitude"
        ),
        vertical_spacing=0.12,
        row_heights=[0.3, 0.7],
    )

    frequencies: np.ndarray = domain.sample_frequencies
    h_plus_amp: np.ndarray = np.abs(polarization.h_plus)

    # Top plot: Show frequency sampling structure
    # Create step function showing delta_f at each frequency
    delta_fs: np.ndarray = np.diff(frequencies)
    delta_fs = np.append(delta_fs, delta_fs[-1])  # Extend last value

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=delta_fs,
            mode="lines",
            line=dict(color="#1f77b4", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="Δf",
            hovertemplate="f=%{x:.2f} Hz<br>Δf=%{y:.3f} Hz<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add vertical lines at nodes
    nodes: List[float] = domain.nodes
    for node in nodes:
        if domain.f_min <= node <= domain.f_max:
            fig.add_vline(
                x=node,
                line=dict(color="red", width=1, dash="dash"),
                opacity=0.5,
                row=1,
                col=1,
            )

    # Bottom plot: Show waveform amplitude
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=h_plus_amp,
            mode="lines",
            line=dict(color="#ff7f0e", width=1.5),
            name="|h<sub>+</sub>|",
            hovertemplate="f=%{x:.2f} Hz<br>|h<sub>+</sub>|=%{y:.2e}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Update axes
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Δf (Hz)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="|h<sub>+</sub>|", type="log", row=2, col=1)

    # Update layout
    approximant: Approximant = wfg._waveform_gen_params.approximant
    fig.update_layout(
        title=f"Multibanded Waveform: {approximant} (M1={params.mass_1}M☉, M2={params.mass_2}M☉)",
        height=800,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
    )

    # Display and save
    print("Displaying plot in browser...")
    fig.show()

    output_file: str = config_file.stem + "_multibanded_plot.html"
    fig.write_html(output_file)
    print(f"\nPlot saved to: {output_file}")


if __name__ == "__main__":
    main()
