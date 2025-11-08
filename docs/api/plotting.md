# Plotting API

Interactive plotting functions for waveform visualization.

## Polarization Plots

Functions for plotting basic polarizations from `generate_hplus_hcross()`.

### plot_polarizations_frequency

::: dingo_waveform.plotting.plot_polarizations_frequency
    options:
      show_root_heading: true
      show_source: false

### plot_polarizations_time

::: dingo_waveform.plotting.plot_polarizations_time
    options:
      show_root_heading: true
      show_source: false

### plot_polarization_spectrogram

::: dingo_waveform.plotting.plot_polarization_spectrogram
    options:
      show_root_heading: true
      show_source: false

### plot_polarization_qtransform

::: dingo_waveform.plotting.plot_polarization_qtransform
    options:
      show_root_heading: true
      show_source: false

## Mode Plots

Functions for plotting mode-separated waveforms from `generate_hplus_hcross_m()`.

### plot_mode_amplitudes

::: dingo_waveform.plotting.plot_mode_amplitudes
    options:
      show_root_heading: true
      show_source: false

### plot_individual_modes

::: dingo_waveform.plotting.plot_individual_modes
    options:
      show_root_heading: true
      show_source: false

### plot_mode_comparison

::: dingo_waveform.plotting.plot_mode_comparison
    options:
      show_root_heading: true
      show_source: false

### plot_modes_grid

::: dingo_waveform.plotting.plot_modes_grid
    options:
      show_root_heading: true
      show_source: false

### plot_mode_reconstruction

::: dingo_waveform.plotting.plot_mode_reconstruction
    options:
      show_root_heading: true
      show_source: false

## Converter Functions

Advanced functions for converting to gwpy formats.

### polarization_to_gwpy_timeseries

::: dingo_waveform.plotting.polarization_to_gwpy_timeseries
    options:
      show_root_heading: true
      show_source: false

### polarization_to_gwpy_frequencyseries

::: dingo_waveform.plotting.polarization_to_gwpy_frequencyseries
    options:
      show_root_heading: true
      show_source: false

## Usage Examples

### Basic Frequency Domain Plot

```python
from dingo_waveform import WaveformGenerator, UniformFrequencyDomain
from dingo_waveform.plotting import plot_polarizations_frequency

# Generate waveform (see Waveform Generator API)
wfg = WaveformGenerator(...)
pol = wfg.generate_hplus_hcross(params)

# Plot
fig = plot_polarizations_frequency(
    pol,
    wfg.domain,
    plot_type="amplitude",
    log_scale=True
)

# Display
fig.show()

# Or save
fig.write_html("waveform.html")
```

### Mode Amplitude Comparison

```python
from dingo_waveform.plotting import plot_mode_amplitudes

# Generate modes
modes = wfg.generate_hplus_hcross_m(params)

# Plot amplitude comparison
fig = plot_mode_amplitudes(modes, wfg.domain)
fig.show()
```

### All Modes Overlaid

```python
from dingo_waveform.plotting import plot_individual_modes

fig = plot_individual_modes(
    modes,
    wfg.domain,
    domain_type="frequency",
    polarization_type="plus"
)
fig.show()
```

## Plot Customization

All plotting functions return Plotly `Figure` objects that can be further customized:

```python
fig = plot_polarizations_frequency(pol, domain)

# Update layout
fig.update_layout(
    title="My Custom Title",
    width=1200,
    height=600,
    font=dict(size=14)
)

# Update axes
fig.update_xaxes(title_text="Frequency (Hz)")
fig.update_yaxes(title_text="Strain")

fig.show()
```

## See Also

- [CLI: dingo-plot](../cli/dingo-plot.md) - Command-line plotting tool
- [Examples: Plotting](../examples/plotting.md) - More plotting examples
