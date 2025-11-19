#!/usr/bin/env python3
"""
Waveform comparison plotting example.

This script demonstrates how to:
1. Generate waveforms from multiple configuration files
2. Compare different approximants or parameters
3. Create overlay plots for comparison

Usage:
    python plot_comparison.py config1.yaml config2.yaml [config3.yaml ...]

Example:
    python plot_comparison.py config_basic_frequency.yaml config_precessing.yaml
"""

import sys
from pathlib import Path

import yaml
import plotly.graph_objects as go
import numpy as np

from dingo_waveform.waveform_generator import build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_comparison.py config1.yaml config2.yaml [config3.yaml ...]")
        print("\nExample:")
        print("  python plot_comparison.py config_basic_frequency.yaml config_precessing.yaml")
        sys.exit(1)

    config_files = [Path(f) for f in sys.argv[1:]]

    # Load and generate all waveforms
    waveforms = []
    labels = []

    for config_file in config_files:
        print(f"\nProcessing: {config_file.name}")

        # Load configuration
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Build generator and generate waveform
        wfg = build_waveform_generator(config)
        params = WaveformParameters(**config["waveform_parameters"])
        polarization = wfg.generate_hplus_hcross(params)

        # Store results
        domain = wfg._waveform_gen_params.domain
        waveforms.append((domain, polarization))

        # Create label
        approximant = config["waveform_generator"]["approximant"]
        mass_1 = params.mass_1
        mass_2 = params.mass_2
        label = f"{approximant} (M1={mass_1:.0f}, M2={mass_2:.0f})"
        labels.append(label)

    # Create comparison plot for h_plus
    print("\nCreating comparison plot...")
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, ((domain, polarization), label) in enumerate(zip(waveforms, labels)):
        frequencies = domain.sample_frequencies
        h_plus_amp = np.abs(polarization.h_plus)

        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=h_plus_amp,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"{label}<br>f=%{{x:.2f}} Hz<br>|h<sub>+</sub>|=%{{y:.2e}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Waveform Comparison: h<sub>+</sub> Amplitude",
        xaxis_title="Frequency (Hz)",
        yaxis_title="|h<sub>+</sub>|",
        xaxis_type="log",
        yaxis_type="log",
        height=600,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Show and save
    fig.show()
    fig.write_html("waveform_comparison.html")
    print("\nComparison plot saved to: waveform_comparison.html")


if __name__ == "__main__":
    main()
