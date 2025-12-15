#!/usr/bin/env python3
"""
Dataset generation example.

Demonstrates how to:
1. Load dataset settings from a configuration file
2. Use the dingo_waveform.dataset API to generate datasets
3. Access and analyze dataset parameters and polarizations
4. Display dataset statistics

Note: For large-scale dataset generation with parallel processing and HDF5 output,
use the dingo_generate_dataset command-line tool instead.

Usage:
    python generate_dataset_example.py
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from dingo_waveform.dataset.dataset_settings import DatasetSettings
from dingo_waveform.dataset.generate import generate_waveform_dataset
from dingo_waveform.dataset.waveform_dataset import WaveformDataset
from dingo_waveform.imports import read_file
from dingo_waveform.logs import set_logging
from dingo_waveform.polarizations import BatchPolarizations

def main() -> None:

    set_logging()
    logger: logging.Logger = logging.getLogger(__name__)

    # Load configuration from YAML file
    config_file: Path = Path(__file__).parent / "dataset_quick_imrphenomd.yaml"
    logger.info(f"Loading configuration from: {config_file.name}")

    # Read config and create DatasetSettings
    config: Dict[str, Any] = read_file(config_file)

    # Override num_samples for this example (original config may have different value)
    config['num_samples'] = 10

    settings: DatasetSettings = DatasetSettings.from_dict(config)

    # Display configuration
    logger.info(f"Domain: {type(settings.domain).__name__}")
    logger.info(f"  f_min: {settings.domain.f_min:.1f} Hz")
    logger.info(f"  f_max: {settings.domain.f_max:.1f} Hz")
    logger.info(f"  delta_f: {settings.domain.delta_f} Hz")

    logger.info(f"Waveform Generator:")
    logger.info(f"  Approximant: {settings.waveform_generator.approximant}")
    logger.info(f"  Reference frequency: {settings.waveform_generator.f_ref} Hz")

    logger.info(f"Prior parameters:")
    for param_name, param_value in settings.intrinsic_prior.__dict__.items():
        if param_value is not None:
            logger.info(f"  {param_name}: {param_value}")

    logger.info(f"Generating {settings.num_samples} waveforms...")

    # Generate dataset using the dataset API
    # num_processes=1 for this small example (use more for production)
    dataset: WaveformDataset = generate_waveform_dataset(settings, num_processes=1)

    logger.info(f"✓ Generated {len(dataset.parameters)} waveforms successfully!")

    # Access dataset components
    parameters: pd.DataFrame = dataset.parameters  # pandas DataFrame with waveform parameters
    polarizations: BatchPolarizations = dataset.polarizations  # BatchPolarizations dataclass

    # Display dataset statistics
    logger.info("Dataset statistics:")

    # Mass statistics
    if 'mass_1' in parameters.columns:
        logger.info(f"  Mass 1: min={parameters['mass_1'].min():.1f}, "
                   f"max={parameters['mass_1'].max():.1f}, "
                   f"mean={parameters['mass_1'].mean():.1f} M☉")
        logger.info(f"  Mass 2: min={parameters['mass_2'].min():.1f}, "
                   f"max={parameters['mass_2'].max():.1f}, "
                   f"mean={parameters['mass_2'].mean():.1f} M☉")

    if 'chirp_mass' in parameters.columns:
        logger.info(f"  Chirp mass: min={parameters['chirp_mass'].min():.1f}, "
                   f"max={parameters['chirp_mass'].max():.1f}, "
                   f"mean={parameters['chirp_mass'].mean():.1f} M☉")

    if 'mass_ratio' in parameters.columns:
        logger.info(f"  Mass ratio: min={parameters['mass_ratio'].min():.2f}, "
                   f"max={parameters['mass_ratio'].max():.2f}, "
                   f"mean={parameters['mass_ratio'].mean():.2f}")

    # Spin statistics
    logger.info(f"  Spin a_1: min={parameters['a_1'].min():.2f}, "
               f"max={parameters['a_1'].max():.2f}, "
               f"mean={parameters['a_1'].mean():.2f}")
    logger.info(f"  Spin a_2: min={parameters['a_2'].min():.2f}, "
               f"max={parameters['a_2'].max():.2f}, "
               f"mean={parameters['a_2'].mean():.2f}")

    # Waveform amplitude statistics
    amplitudes: np.ndarray = np.abs(polarizations.h_plus).max(axis=1)  # shape: (num_samples,)
    logger.info(f"  Max amplitude: min={amplitudes.min():.3e}, "
               f"max={amplitudes.max():.3e}, "
               f"mean={amplitudes.mean():.3e}")

    # Show dataset structure
    logger.info(f"Dataset structure:")
    logger.info(f"  Parameters shape: {parameters.shape}")
    logger.info(f"  h_plus shape: {polarizations.h_plus.shape}")
    logger.info(f"  h_cross shape: {polarizations.h_cross.shape}")

    logger.info("Note: For production-scale dataset generation with parallel processing,")
    logger.info("use the dingo_generate_dataset command-line tool:")
    logger.info(f"  dingo_generate_dataset --settings_file {config_file.name} --num_processes 8")


if __name__ == "__main__":
    main()
