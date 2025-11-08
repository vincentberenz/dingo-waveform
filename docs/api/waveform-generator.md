# Waveform Generator API

The `WaveformGenerator` class is the main interface for generating gravitational wave waveforms.

## WaveformGenerator

::: dingo_waveform.waveform_generator.WaveformGenerator
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - generate_hplus_hcross
        - generate_hplus_hcross_m

## WaveformParameters

::: dingo_waveform.waveform_parameters.WaveformParameters
    options:
      show_root_heading: true
      show_source: false

## Helper Functions

### build_waveform_generator

::: dingo_waveform.waveform_generator.build_waveform_generator
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Basic Generation

```python
from dingo_waveform import (
    WaveformGenerator,
    WaveformParameters,
    Approximant,
    UniformFrequencyDomain
)

# Create domain
domain = UniformFrequencyDomain(
    f_min=20.0,
    f_max=1024.0,
    delta_f=0.125
)

# Create generator
wfg = WaveformGenerator(
    approximant=Approximant("IMRPhenomXPHM"),
    domain=domain,
    f_ref=20.0
)

# Set parameters
params = WaveformParameters(
    mass_1=36.0,
    mass_2=29.0,
    luminosity_distance=1000.0,
    theta_jn=0.5,
    phase=0.0,
    a_1=0.3,
    a_2=0.2,
    tilt_1=0.5,
    tilt_2=0.8,
    phi_12=1.7,
    phi_jl=0.3,
    geocent_time=0.0
)

# Generate
pol = wfg.generate_hplus_hcross(params)
```

### Mode-Separated Generation

```python
# Using same generator and params from above
modes = wfg.generate_hplus_hcross_m(params)

for m, pol in modes.items():
    print(f"Mode m={m}: amplitude = {abs(pol.h_plus[0]):.2e}")
```

### Building from Dictionary

```python
config = {
    "domain": {
        "type": "UniformFrequencyDomain",
        "f_min": 20.0,
        "f_max": 1024.0,
        "delta_f": 0.125
    },
    "waveform_generator": {
        "approximant": "IMRPhenomXPHM",
        "f_ref": 20.0
    }
}

wfg = build_waveform_generator(config)
```

## See Also

- [Domains API](domains.md) - Domain definitions
- [Approximants API](approximants.md) - Waveform approximants
- [Polarizations API](polarizations.md) - Polarization outputs
