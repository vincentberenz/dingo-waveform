# dingo-waveform

A refactor of dingo-gw's waveform generator, meant to replicate its functionalities and outputs while improving code readability and maintainability.

## Features

- Modular gravitational waveform generation with type-safe interfaces
- Multiple domain support (time, frequency, multibanded frequency)
- **Integrated SVD compression** via `dingo_waveform.svd` subpackage
- Parallel dataset generation with multiprocessing
- CLI tools for verification, benchmarking, and visualization
- Comprehensive test coverage with pytest

## Installation

```bash
pip install -e .
```

or, for development installation:

```bash
pip install -e ".[dev]"
```

## Usage

See the examples folder for examples of scripts and configuration files.

To generate a dataset:

```bash
dingo_generate_dataset --settings_file config.yaml --num_processes 8
```

## Verification

Run `dingo-verify-batch` to run a batch of waveform generation and compare them against dingo (dingo-gw).


