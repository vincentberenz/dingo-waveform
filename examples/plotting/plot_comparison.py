#!/usr/bin/env python3
"""
Waveform comparison plotting example.

This script demonstrates how to:
1. Generate waveforms from multiple configuration files
2. Compare different approximants or parameters
3. Create overlay plots for comparison

Usage:
    python plot_comparison.py

The script compares waveforms from several default configurations.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go

from dingo_waveform.domains import Domain
from dingo_waveform.imports import read_file
from dingo_waveform.polarizations import Polarization
from dingo_waveform.waveform_generator import WaveformGenerator, build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters


def main() -> None:
    # Use default configuration files for comparison
    script_dir: Path = Path(__file__).parent
    config_files: List[Path] = [
        script_dir / "config_basic_frequency.yaml",
        script_dir / "config_precessing.yaml",
        script_dir / "config_high_mass.yaml",
    ]

    print("Comparing waveforms from the following configurations:")
    for config_file in config_files:
        print(f"  - {config_file.name}")
    print()

    # Load and generate all waveforms
    waveforms: List[Tuple[Domain, Polarization]] = []
    labels: List[str] = []

    for config_file in config_files:
        print(f"\nProcessing: {config_file.name}")

        # Load configuration
        config: Dict[str, Any] = read_file(config_file)

        # Build generator and generate waveform
        wfg: WaveformGenerator = build_waveform_generator(config)
        params: WaveformParameters = WaveformParameters(**config["waveform_parameters"])
        polarization: Polarization = wfg.generate_hplus_hcross(params)

        # Store results
        domain: Domain = wfg._waveform_gen_params.domain
        waveforms.append((domain, polarization))

        # Create label
        approximant: str = config["waveform_generator"]["approximant"]
        mass_1: float = params.mass_1
        mass_2: float = params.mass_2
        label: str = f"{approximant} (M1={mass_1:.0f}, M2={mass_2:.0f})"
        labels.append(label)

    # Create comparison plot for h_plus
    print("\nCreating comparison plot...")
    fig: go.Figure = go.Figure()

    colors: List[str] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, ((domain, polarization), label) in enumerate(zip(waveforms, labels)):
        frequencies: np.ndarray = domain.sample_frequencies()
        h_plus_amp: np.ndarray = np.abs(polarization.h_plus)

        color: str = colors[idx % len(colors)]
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
