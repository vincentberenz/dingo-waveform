# dingo-plot

Generate and visualize waveforms from JSON configuration files.

## Usage

```bash
dingo-plot [OPTIONS]
```

## Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to JSON configuration file | `config.json` |
| `--modes` | `-m` | Plot mode-separated waveforms | False |
| `--output` | `-o` | Output directory for HTML files | `.` (current directory) |
| `--show` | `-s` | Display plots in browser | False |

## Examples

### Basic Usage

Generate plots for basic polarizations using `config.json` in current directory:

```bash
dingo-plot
```

### Specify Configuration File

```bash
dingo-plot --config /path/to/my_config.json
```

### Plot Mode-Separated Waveforms

```bash
dingo-plot --modes
```

### Save to Custom Directory

```bash
dingo-plot --output plots/
```

### Display in Browser

```bash
dingo-plot --show
```

### Combine Options

```bash
dingo-plot --config my_config.json --modes --output results/ --show
```

## Configuration File Format

The configuration file uses the same format as [`dingo-verify`](dingo-verify.md):

```json
{
  "domain": {
    "type": "UniformFrequencyDomain",
    "f_min": 20.0,
    "f_max": 1024.0,
    "delta_f": 0.125
  },
  "waveform_generator": {
    "approximant": "IMRPhenomXPHM",
    "f_ref": 20.0,
    "f_start": 20.0
  },
  "waveform_parameters": {
    "mass_1": 36.0,
    "mass_2": 29.0,
    "luminosity_distance": 1000.0,
    "theta_jn": 0.5,
    "phase": 0.0,
    "a_1": 0.3,
    "a_2": 0.2,
    "tilt_1": 0.5,
    "tilt_2": 0.8,
    "phi_12": 1.7,
    "phi_jl": 0.3,
    "geocent_time": 0.0
  }
}
```

### Domain Types

Supported domain types:

- `UniformFrequencyDomain` - Standard frequency domain
- `MultibandedFrequencyDomain` - Adaptive frequency binning
- `TimeDomain` - Time domain

See [Domains](../concepts/domains.md) for details.

## Generated Plots

### Basic Polarizations (without `--modes`)

When plotting basic polarizations, the following HTML files are generated:

#### Frequency Domain
- `polarizations_amplitude.html` - Amplitude plot for h₊ and h×
- `polarizations_both.html` - Combined amplitude and phase plots

#### Time Domain
- `polarizations_time.html` - Time-domain waveforms
- `spectrogram.html` - Spectrogram visualization
- `qtransform.html` - Q-transform visualization

### Mode-Separated Waveforms (with `--modes`)

When using the `--modes` flag, these files are generated:

- `mode_amplitudes.html` - Bar chart comparing mode amplitudes
- `individual_modes.html` - All modes overlaid on single plot
- `mode_comparison.html` - Modes compared to dominant mode
- `modes_grid.html` - Grid layout showing each mode separately

## Interactive Features

All generated plots are interactive Plotly visualizations with:

- **Zoom**: Click and drag to zoom
- **Pan**: Shift + drag to pan
- **Hover**: Hover for data values
- **Legend**: Click to toggle traces
- **Export**: Download as PNG (requires browser)

## Troubleshooting

### Configuration Errors

```
❌ Configuration Error: domain.type is required
```

Ensure your JSON file has all required fields. See configuration format above.

### Mode Generation Errors

```
❌ Configuration Error: Reference mode (2, 2) not found
```

The selected approximant may not support mode-separated waveforms. Use approximants like:
- IMRPhenomXPHM
- SEOBNRv5PHM
- SEOBNRv5HM
- SEOBNRv4PHM

### File Not Found

```
❌ Error: Configuration file not found: config.json
```

Either create a `config.json` in the current directory or specify the path with `--config`.

## See Also

- [dingo-verify](dingo-verify.md) - Verify waveform correctness
- [dingo_generate_dataset](dingo-generate-dataset.md) - Generate waveform datasets
- [Plotting API](../api/plotting.md) - Programmatic plotting interface
