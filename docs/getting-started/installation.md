# Installation

## Requirements

- Python ≥ 3.8
- LALSuite ≥ 7.15
- CUDA-capable GPU (optional, for neural network inference in main dingo package)

## Install from Source

### Clone the Repository

```bash
git clone https://github.com/dingo-gw/dingo-waveform.git
cd dingo-waveform
```

### Basic Installation

Install the package in editable mode:

```bash
pip install -e .
```

This installs dingo-waveform with all required dependencies including:

- LALSuite (waveform generation)
- NumPy, SciPy (numerical computing)
- PyTorch (for domain transformations)
- gwpy (time-frequency analysis)
- plotly (interactive visualization)
- bilby (parameter conversions and priors)

### Development Installation

For development work, install with additional tools:

```bash
pip install -e ".[dev]"
```

This includes:

- `black` - Code formatter
- `isort` - Import sorter
- `mypy` - Type checker
- `pylint` - Code linter
- `pytest` - Testing framework
- `marimo` - Interactive notebooks

### Documentation Installation

To build documentation locally:

```bash
pip install -e ".[docs]"
```

This includes:

- `mkdocs` - Documentation generator
- `mkdocs-material` - Material theme
- `mkdocstrings` - API documentation from docstrings

## Verify Installation

After installation, verify that the executables are available:

```bash
# Check version
python -c "import dingo_waveform; print(dingo_waveform.__version__)"

# Test executables
dingo-verify --help
dingo-plot --help
dingo_generate_dataset --help
```

## Troubleshooting

### LALSuite Installation Issues

If you encounter issues with LALSuite, consider using conda:

```bash
conda install -c conda-forge lalsuite
```

### Import Errors

If you see import errors, ensure all dependencies are installed:

```bash
pip install -e . --force-reinstall
```

### GPU Support

While waveform generation is CPU-only, PyTorch with CUDA support may be needed for neural network components:

```bash
# Install PyTorch with CUDA support (adjust version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

Once installed, proceed to the [Quick Start](quickstart.md) guide to generate your first waveform.
