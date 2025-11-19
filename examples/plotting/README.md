# Waveform Plotting Examples

This directory contains examples demonstrating how to generate and plot gravitational waveforms using `dingo-waveform`.

## Overview

The examples show how to:
- Load configuration files (YAML format)
- Generate waveforms with different approximants
- Create interactive plots using the `dingo_waveform.plotting` module
- Work with mode-separated waveforms
- Compare different waveforms
- Visualize multibanded frequency domains

## Requirements

All examples require the plotting dependencies:

```bash
pip install -e ".[dev]"
```

This installs: plotly, kaleido, matplotlib, gwpy

## Configuration Files

### `config_basic_frequency.yaml`
Basic frequency domain example with aligned spins using IMRPhenomXPHM.
- Domain: UniformFrequencyDomain (20-1024 Hz, Δf=0.125 Hz)
- Masses: M1=36 M☉, M2=29 M☉
- Spins: Aligned (tilt_1=tilt_2=0)

### `config_precessing.yaml`
Precessing binary example using SEOBNRv5PHM.
- Domain: UniformFrequencyDomain (20-1024 Hz)
- Masses: M1=40 M☉, M2=30 M☉
- Spins: Precessing (tilt_1=0.8, tilt_2=1.2)

### `config_modes.yaml`
Mode-separated waveform example using IMRPhenomXPHM.
- Demonstrates mode decomposition (ℓ,m) harmonics
- Masses: M1=50 M☉, M2=35 M☉

### `config_multibanded.yaml`
Multibanded (dyadic) frequency domain example.
- Domain: MultibandedFrequencyDomain with non-uniform spacing
- Efficient for SVD compression
- Shows frequency sampling structure

### `config_high_mass.yaml`
High-mass binary example.
- Domain: 30-2048 Hz (higher f_min, wider bandwidth)
- Masses: M1=80 M☉, M2=60 M☉
- Shorter waveform duration

## Python Scripts

### `plot_basic_waveform.py`

Generate and plot a basic waveform.

**Usage:**
```bash
python plot_basic_waveform.py [config_file]
```

**Example:**
```bash
# Use default config (config_basic_frequency.yaml)
python plot_basic_waveform.py

# Use specific config
python plot_basic_waveform.py config_precessing.yaml
```

**Output:**
- Interactive plotly figure in browser
- HTML file: `{config_name}_plot.html`

**What it does:**
1. Loads configuration
2. Builds `WaveformGenerator`
3. Generates waveform polarizations (h+, h×)
4. Plots frequency domain amplitude and phase

---

### `plot_modes.py`

Generate and plot mode-separated waveforms.

**Usage:**
```bash
python plot_modes.py [config_file]
```

**Example:**
```bash
# Use default config (config_modes.yaml)
python plot_modes.py

# Use IMRPhenomXPHM with custom parameters
python plot_modes.py config_basic_frequency.yaml
```

**Output:**
- Three HTML files:
  - `{config_name}_mode_amplitudes.html` - Compare mode amplitudes
  - `{config_name}_individual_modes.html` - Individual mode plots
  - `{config_name}_mode_reconstruction.html` - Mode reconstruction

**Supported approximants:**
- IMRPhenomXPHM ✓
- SEOBNRv4PHM ✓
- SEOBNRv5PHM ✓
- SEOBNRv5HM ✓

**What it does:**
1. Generates mode-separated waveforms using `generate_hplus_hcross_m()`
2. Creates visualizations of individual spherical harmonic modes (ℓ,m)
3. Shows how modes sum to create total polarization

---

### `plot_comparison.py`

Compare multiple waveforms.

**Usage:**
```bash
python plot_comparison.py config1.yaml config2.yaml [config3.yaml ...]
```

**Example:**
```bash
# Compare aligned vs precessing spins
python plot_comparison.py config_basic_frequency.yaml config_precessing.yaml

# Compare multiple approximants
python plot_comparison.py config_basic_frequency.yaml config_modes.yaml config_high_mass.yaml
```

**Output:**
- HTML file: `waveform_comparison.html`

**What it does:**
1. Loads multiple configurations
2. Generates all waveforms
3. Creates overlay plot comparing h+ amplitudes
4. Uses log-log scale for frequency/amplitude

---

### `plot_multibanded.py`

Visualize multibanded frequency domain structure.

**Usage:**
```bash
python plot_multibanded.py [config_file]
```

**Example:**
```bash
# Use default multibanded config
python plot_multibanded.py

# Use custom multibanded config
python plot_multibanded.py config_multibanded.yaml
```

**Output:**
- HTML file: `{config_name}_multibanded_plot.html`

**What it does:**
1. Generates waveform in MultibandedFrequencyDomain
2. Creates two-panel plot:
   - Top: Frequency spacing Δf vs frequency (shows dyadic structure)
   - Bottom: Waveform amplitude
3. Highlights band boundaries (nodes) with vertical lines

---

## Quick Start

Run all examples:

```bash
cd examples/plotting

# Basic waveform
python plot_basic_waveform.py

# Mode-separated waveform
python plot_modes.py

# Compare two waveforms
python plot_comparison.py config_basic_frequency.yaml config_precessing.yaml

# Multibanded visualization
python plot_multibanded.py
```

## API Usage

All scripts demonstrate the core API pattern:

```python
import yaml
from dingo_waveform.waveform_generator import build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters
from dingo_waveform.plotting import plot_polarizations_frequency

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Build generator
wfg = build_waveform_generator(config)

# Generate waveform
params = WaveformParameters(**config["waveform_parameters"])
polarization = wfg.generate_hplus_hcross(params)

# Plot
fig = plot_polarizations_frequency(polarization, wfg.domain)
fig.show()
```

## Available Plotting Functions

From `dingo_waveform.plotting`:

### Polarization plots (for `generate_hplus_hcross` output)
- `plot_polarizations_time()` - Time domain plot
- `plot_polarizations_frequency()` - Frequency domain plot
- `plot_polarization_spectrogram()` - Time-frequency spectrogram
- `plot_polarization_qtransform()` - Q-transform visualization

### Mode plots (for `generate_hplus_hcross_m` output)
- `plot_mode_amplitudes()` - Compare mode amplitudes
- `plot_individual_modes()` - Individual mode plots
- `plot_mode_comparison()` - Side-by-side mode comparison
- `plot_modes_grid()` - Grid of all modes
- `plot_mode_reconstruction()` - Show mode summation

### Converters (for advanced usage)
- `polarization_to_gwpy_timeseries()` - Convert to gwpy TimeSeries
- `polarization_to_gwpy_frequencyseries()` - Convert to gwpy FrequencySeries
- `gwpy_to_polarization()` - Convert from gwpy back to Polarization

## Creating Custom Configurations

YAML format structure:

```yaml
domain:
  type: UniformFrequencyDomain  # or MultibandedFrequencyDomain
  delta_f: 0.125
  f_min: 20.0
  f_max: 1024.0

waveform_generator:
  approximant: IMRPhenomXPHM
  f_ref: 20.0
  f_start: 20.0
  spin_conversion_phase: 0.0

waveform_parameters:
  mass_1: 36.0
  mass_2: 29.0
  luminosity_distance: 1000.0
  theta_jn: 0.5
  phase: 0.0
  a_1: 0.5
  a_2: 0.3
  tilt_1: 0.5
  tilt_2: 0.8
  phi_12: 1.7
  phi_jl: 0.3
  geocent_time: 0.0
```

## Tips

1. **For interactive exploration:** Use `fig.show()` to open in browser
2. **For presentations:** Save to HTML with `fig.write_html("plot.html")`
3. **For publications:** Use `fig.write_image("plot.png")` (requires kaleido)
4. **Modes:** Only certain approximants support mode separation
5. **Memory:** High-resolution domains (small Δf) can be memory-intensive
6. **Performance:** Multibanded domains are more efficient for long waveforms
