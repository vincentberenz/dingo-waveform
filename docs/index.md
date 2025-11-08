# dingo-waveform

**dingo-waveform** is a Python package for generating gravitational wave waveforms, designed as a refactored and optimized component of the [dingo](https://github.com/dingo-gw/dingo) neural posterior estimation framework.

## Overview

This package provides:

- **Fast waveform generation** for gravitational wave inference using LALSimulation
- **Multiple domain support** including uniform and multibanded frequency domains
- **Mode-separated waveforms** for higher-order spherical harmonic modes
- **Interactive plotting** for diagnostics and exploration
- **Dataset generation** tools for machine learning applications
- **Command-line tools** for verification and visualization

## Key Features

### Waveform Approximants

Support for state-of-the-art waveform models including:

- IMRPhenomXPHM - Precessing binary black holes with higher modes
- SEOBNRv5PHM - Effective-one-body with higher modes
- SEOBNRv5HM - Higher modes
- SEOBNRv4PHM - Previous generation with higher modes
- And more via LALSimulation

### Domain Types

- **UniformFrequencyDomain** - Standard uniform frequency sampling
- **MultibandedFrequencyDomain** - Efficient adaptive binning for neural networks
- **TimeDomain** - Time-domain waveform generation

### Visualization

Interactive plotting with Plotly for:

- Polarization waveforms (h₊, h×)
- Mode-separated waveforms
- Time-frequency representations (Q-transforms, spectrograms)
- Mode comparisons and reconstructions

### Command-Line Tools

- `dingo-verify` - Verify correctness against dingo (dingo-gw)
- `dingo-plot` - Generate interactive plots from configuration files
- `dingo_generate_dataset` - Generate waveform datasets for training

## Quick Example

```python
from dingo_waveform import (
    WaveformGenerator,
    WaveformParameters,
    Approximant,
    UniformFrequencyDomain
)

# Define domain
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
    # ... other parameters
)

# Generate waveform
polarization = wfg.generate_hplus_hcross(params)

# Or generate mode-separated waveforms
modes = wfg.generate_hplus_hcross_m(params)
```

## Use Cases

### Gravitational Wave Inference

Generate template waveforms for parameter estimation and matched filtering of gravitational wave signals from LIGO/Virgo/KAGRA detectors.

### Neural Posterior Estimation

Efficiently generate training datasets for neural networks that perform rapid Bayesian inference on gravitational wave events.

### Waveform Diagnostics

Visualize and analyze waveform properties, compare approximants, and validate implementations.

## Related Projects

- [dingo](https://github.com/dingo-gw/dingo) - Neural posterior estimation for gravitational waves
- [LALSuite](https://git.ligo.org/lscsoft/lalsuite) - LIGO Scientific Collaboration Algorithm Library
- [bilby](https://git.ligo.org/lscsoft/bilby) - Bayesian inference library

## Citation

If you use dingo-waveform in your research, please cite:

!!! note "Citation"
    Citation information will be added here.

## Getting Help

- **Documentation**: Browse the sections in the navigation menu
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/dingo-gw/dingo-waveform/issues)
- **Examples**: Check the Examples section for tutorials

## Next Steps

- [Installation Guide](getting-started/installation.md) - Get started with installation
- [Quick Start](getting-started/quickstart.md) - Your first waveform
- [Concepts](concepts/overview.md) - Understand the core concepts
- [API Reference](api/waveform-generator.md) - Detailed API documentation
