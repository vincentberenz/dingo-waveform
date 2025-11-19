# dingo-waveform

A refactored waveform generator for gravitational wave inference, designed for improved modularity, type safety, and performance.

## Overview

`dingo-waveform` is a Python package for generating gravitational waveform polarizations and modes across various frequency and time domains. It provides a clean, well-typed interface to LALSimulation waveform approximants with support for:

- **Multiple waveform approximants**: IMRPhenomXPHM, SEOBNRv4PHM, SEOBNRv5PHM, SEOBNRv5HM, and many more
- **Flexible domains**: UniformFrequencyDomain, MultibandedFrequencyDomain, and TimeDomain
- **Mode-separated waveforms**: Generate individual spherical harmonic mode contributions
- **Dataset generation**: Efficient parallel generation of training datasets for neural posterior estimation
- **Comprehensive verification tools**: Built-in tools to verify correctness against reference implementations

## Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Generate a Waveform

```python
from dingo_waveform import WaveformGenerator
from dingo_waveform.domains import UniformFrequencyDomain

# Define domain
domain = UniformFrequencyDomain(
    f_min=20.0,
    f_max=1024.0,
    delta_f=0.125
)

# Create waveform generator
wfg = WaveformGenerator(
    approximant="IMRPhenomXPHM",
    domain=domain,
    f_ref=20.0
)

# Generate polarizations
parameters = {
    "mass_1": 36.0,
    "mass_2": 29.0,
    "luminosity_distance": 1000.0,
    "theta_jn": 0.5,
    "phase": 1.0,
    "a_1": 0.3,
    "a_2": 0.2,
    "tilt_1": 0.1,
    "tilt_2": 0.2,
    "phi_12": 0.5,
    "phi_jl": 0.7
}

h_plus, h_cross = wfg.generate_hplus_hcross(parameters)
```

### Generate a Dataset

```bash
dingo_generate_dataset --settings_file config.yaml --num_processes 8
```

See `examples/` directory for complete configuration examples.

## Command-Line Tools

### Waveform Verification

Verify waveform generation against reference implementations:

```bash
# Single configuration
dingo-verify --config examples/config_uniform.json

# Batch verification across multiple approximants
dingo-verify-batch

# With auto-generated comprehensive test suite (~36 configs)
dingo-verify-batch --verbose
```

### Dataset Generation

Generate waveform training datasets:

```bash
dingo_generate_dataset --settings_file config.yaml --num_processes 8
```

### Visualization

Interactive waveform plotting:

```bash
dingo-plot --config config.yaml
```

### Benchmarking

Performance benchmarking:

```bash
dingo-benchmark --config config.yaml
```

## Documentation

Comprehensive HTML documentation is available. To build and view locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation with live reload
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

Documentation includes:
- **Getting Started**: Installation and quickstart guides
- **Concepts**: Detailed explanations of domains, approximants, and modes
- **API Reference**: Complete API documentation
- **Examples**: Step-by-step tutorials and use cases

## Examples

Interactive examples are available in the `examples/` directory:

- **`visualize_waveforms.py`** - Interactive marimo notebook for visualizing gravitational waveforms
  - Run with: `marimo edit examples/visualize_waveforms.py`
  - Supports YAML configuration files and interactive parameter controls

- **Configuration files** - Example YAML configurations for various approximants and domains
  - `example_waveform_config.yaml` - Basic waveform configuration
  - See `examples/README.md` for more details

To use the examples, install with optional dependencies:

```bash
pip install -e ".[examples]"
```

## Key Features

### Type Safety

Extensive use of type hints and `TypeAlias` for improved code clarity and IDE support:

```python
from dingo_waveform.types import WaveformParameters, DomainParameters
from dingo_waveform.polarizations import Polarizations
```

### Multiple Domains

- **UniformFrequencyDomain**: Standard frequency-domain with constant delta_f
- **MultibandedFrequencyDomain**: Non-uniform frequency grid with dyadic spacing
- **TimeDomain**: Time-domain generation (in development)

### Mode-Separated Waveforms

Generate individual spherical harmonic mode contributions:

```python
# Get mode-separated polarizations
pol_m = wfg.generate_hplus_hcross_m(parameters)

# Access specific modes
h_22 = pol_m[(2, 2)]  # Dominant (2,2) mode
h_plus_22 = h_22["h_plus"]
h_cross_22 = h_22["h_cross"]
```

### Efficient Dataset Generation

Parallel dataset generation with HDF5 output:

```bash
dingo_generate_dataset \
    --settings_file settings.yaml \
    --num_processes 16 \
    --out_file dataset.hdf5
```

### Verification Tools

Built-in verification against dingo (dingo-gw):

```bash
# Verify single configuration
dingo-verify --config config.yaml --verbose

# Batch verification with comprehensive test suite
dingo-verify-batch
```

Automatically generates ~36 test configurations covering:
- 9 waveform approximants (IMRPhenomXPHM, SEOBNRv4PHM, SEOBNRv5HM, etc.)
- Multiple domains (uniform, multibanded)
- Various parameter combinations (mass ratios, spins, inclinations)

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_waveform_generator.py -v

# Run with coverage
pytest tests/ --cov=dingo_waveform --cov-report=html
```

### Type Checking

```bash
mypy dingo_waveform/
```

### Code Formatting

```bash
black dingo_waveform/ tests/
isort dingo_waveform/ tests/
```

## Requirements

- Python >= 3.8
- lalsuite >= 7.15
- bilby
- numpy
- scipy
- torch
- rich (for CLI formatting)
- matplotlib, plotly (for visualization)

See `pyproject.toml` for complete dependency list.

## Architecture

The package is organized into clear modules:

```
dingo_waveform/
├── waveform_generator.py              # Main WaveformGenerator class
├── domains/                            # Domain definitions
│   ├── frequency_domain.py
│   ├── multibanded_frequency_domain.py
│   └── time_domain.py
├── polarization_functions/             # Polarization generation
├── polarization_modes_functions/       # Mode-separated generation
├── dataset/                            # Dataset generation
│   ├── generate.py
│   └── waveform_dataset.py
├── approximant.py                      # Approximant handling
├── prior.py                            # Prior distributions
├── waveform_parameters.py              # Parameter definitions
├── cli.py                              # Verification CLI
├── cli_batch.py                        # Batch verification CLI
└── dataset/
    └── cli.py                          # Dataset generation CLI (dingo_generate_dataset)
```

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest tests/`
2. Code is formatted: `black .` and `isort .`
3. Type hints are added for new functions
4. Documentation is updated for API changes

## License

MIT License - see LICENSE file for details.

## Related Projects

- **[dingo](https://github.com/dingo-gw/dingo)** - Neural posterior estimation for gravitational wave inference
- **[bilby](https://lscsoft.docs.ligo.org/bilby/)** - Bayesian inference library for gravitational-wave astronomy
- **[LALSuite](https://lscsoft.docs.ligo.org/lalsuite/)** - LIGO Scientific Collaboration Algorithm Library

## Citation

If you use `dingo-waveform` in your research, please cite:

```bibtex
@software{dingo_waveform,
  title = {dingo-waveform: Modular gravitational waveform generator},
  author = {{The dingo team}},
  year = {2024},
  url = {https://github.com/dingo-gw/dingo-waveform}
}
```

## Support

For questions, issues, or contributions:
- **Issues**: https://github.com/dingo-gw/dingo-waveform/issues
- **Documentation**: http://127.0.0.1:8000 (after running `mkdocs serve`)
