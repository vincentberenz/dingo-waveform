# Quick Start

This guide will walk you through generating your first gravitational wave waveform with dingo-waveform.

## Basic Waveform Generation

### 1. Import Required Components

```python
from dingo_waveform import (
    WaveformGenerator,
    WaveformParameters,
    Approximant,
    UniformFrequencyDomain
)
```

### 2. Define the Domain

The domain specifies the frequency (or time) sampling for the waveform:

```python
domain = UniformFrequencyDomain(
    f_min=20.0,      # Minimum frequency (Hz)
    f_max=1024.0,    # Maximum frequency (Hz)
    delta_f=0.125    # Frequency resolution (Hz)
)
```

### 3. Create a Waveform Generator

```python
wfg = WaveformGenerator(
    approximant=Approximant("IMRPhenomXPHM"),  # Waveform model
    domain=domain,
    f_ref=20.0,        # Reference frequency
    f_start=20.0       # Starting frequency
)
```

### 4. Set Binary Parameters

```python
params = WaveformParameters(
    mass_1=36.0,                    # Primary mass (solar masses)
    mass_2=29.0,                    # Secondary mass (solar masses)
    luminosity_distance=1000.0,     # Distance (Mpc)
    theta_jn=0.5,                   # Inclination angle
    phase=0.0,                      # Orbital phase
    a_1=0.3,                        # Primary spin magnitude
    a_2=0.2,                        # Secondary spin magnitude
    tilt_1=0.5,                     # Primary tilt angle
    tilt_2=0.8,                     # Secondary tilt angle
    phi_12=1.7,                     # Azimuthal angle between spins
    phi_jl=0.3,                     # Azimuthal angle of L
    geocent_time=0.0                # Coalescence time
)
```

### 5. Generate the Waveform

```python
# Generate h+ and h× polarizations
polarization = wfg.generate_hplus_hcross(params)

print(f"h_plus shape: {polarization.h_plus.shape}")
print(f"h_cross shape: {polarization.h_cross.shape}")
```

## Visualizing the Waveform

### Using the Plotting Module

```python
from dingo_waveform.plotting import plot_polarizations_frequency

# Create interactive plot
fig = plot_polarizations_frequency(
    polarization,
    domain,
    plot_type="amplitude"
)

# Display in browser
fig.show()

# Or save to file
fig.write_html("waveform.html")
```

### Using the Command-Line Tool

Create a configuration file `config.json`:

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
    "f_ref": 20.0
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

Then run:

```bash
# Generate and plot waveform
dingo-plot --config config.json --output plots/
```

## Mode-Separated Waveforms

For approximants that support higher-order modes (like IMRPhenomXPHM):

```python
# Generate mode-separated polarizations
modes = wfg.generate_hplus_hcross_m(params)

print(f"Number of modes: {len(modes)}")
print(f"Available modes: {sorted(modes.keys())}")

# Access individual modes
for m, pol in modes.items():
    print(f"Mode m={m}: h_plus shape = {pol.h_plus.shape}")
```

### Plotting Modes

```python
from dingo_waveform.plotting import (
    plot_mode_amplitudes,
    plot_individual_modes
)

# Bar chart of mode amplitudes
fig1 = plot_mode_amplitudes(modes, domain)
fig1.show()

# All modes on same plot
fig2 = plot_individual_modes(modes, domain, domain_type="frequency")
fig2.show()
```

Or using the CLI:

```bash
dingo-plot --modes --output plots/
```

## Time Domain Waveforms

To generate time-domain waveforms:

```python
from dingo_waveform import TimeDomain

# Define time domain
time_domain = TimeDomain(
    duration=8.0,     # Total duration (seconds)
    delta_t=1/2048.0  # Time step (seconds)
)

# Create generator with time domain
wfg_td = WaveformGenerator(
    approximant=Approximant("SEOBNRv5PHM"),
    domain=time_domain,
    f_ref=20.0
)

# Generate
polarization_td = wfg_td.generate_hplus_hcross(params)
```

## Next Steps

- Learn about [Domains](../concepts/domains.md) for different frequency sampling strategies
- Explore [Approximants](../concepts/approximants.md) to understand different waveform models
- Check [Examples](../examples/basic-waveform.md) for more detailed tutorials
- Read the [API Reference](../api/waveform-generator.md) for complete documentation
