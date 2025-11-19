#!/usr/bin/env python3
"""
Dataset generation example.

Demonstrates how to:
1. Use build_waveform_generator to create a generator from config
2. Sample parameters from a prior distribution
3. Generate multiple waveforms efficiently
4. Display dataset statistics

Note: For large-scale dataset generation with parallel processing and HDF5 output,
use the dingo_generate_dataset command-line tool instead.

Usage:
    python generate_dataset_example.py
"""

import logging
import yaml
import numpy as np
from pathlib import Path

from dingo_waveform.waveform_generator import build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters
from dingo_waveform.prior import build_prior_with_defaults
from dingo_waveform.logs import set_logging

def main():

    set_logging()
    logger = logging.getLogger(__name__)

    # Create waveform generator directly from config file
    config_file = Path(__file__).parent / "dataset_quick_imrphenomd.yaml"
    logger.info(f"Loading configuration from: {config_file.name}")

    wfg = build_waveform_generator(config_file)

    # Access domain and approximant from the generator
    domain = wfg._waveform_gen_params.domain
    approximant = wfg._waveform_gen_params.approximant
    f_ref = wfg._waveform_gen_params.f_ref

    logger.info(f"Domain: {type(domain).__name__}")
    logger.info(f"  Frequency range: {domain.f_min:.1f} - {domain.f_max:.1f} Hz")
    logger.info(f"  Frequency bins: {len(domain)}")

    logger.info(f"Waveform Generator:")
    logger.info(f"  Approximant: {approximant}")
    logger.info(f"  Reference frequency: {f_ref} Hz")

    # Load prior configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Create prior distribution
    logger.info("Creating prior distribution...")
    prior = build_prior_with_defaults(config['intrinsic_prior'])

    logger.info(f"Prior parameters:")
    for param_name in prior.keys():
        param_prior = prior[param_name]
        logger.info(f"  {param_name}: {param_prior}")

    # Generate sample dataset
    num_samples = 10
    logger.info(f"Generating {num_samples} waveforms...")

    waveforms = []
    parameters = []

    for i in range(num_samples):
        # Sample parameters from prior
        sampled_params = prior.sample()

        # Create WaveformParameters object
        # Note: sampled_params may contain chirp_mass/mass_ratio or mass_1/mass_2
        params = WaveformParameters(**sampled_params)

        # Generate waveform
        polarizations = wfg.generate_hplus_hcross(params)

        waveforms.append(polarizations)
        parameters.append(params)

        if (i + 1) % 5 == 0:
            logger.info(f"  Generated {i + 1}/{num_samples} waveforms...")

    logger.info(f"✓ Generated {num_samples} waveforms successfully!")

    # Display dataset statistics
    logger.info("Dataset statistics:")

    # Mass statistics (handle both mass_1/mass_2 and chirp_mass/mass_ratio cases)
    if parameters[0].mass_1 is not None:
        masses_1 = [p.mass_1 for p in parameters]
        masses_2 = [p.mass_2 for p in parameters]
        logger.info(f"  Mass 1: min={min(masses_1):.1f}, max={max(masses_1):.1f}, mean={np.mean(masses_1):.1f} M☉")
        logger.info(f"  Mass 2: min={min(masses_2):.1f}, max={max(masses_2):.1f}, mean={np.mean(masses_2):.1f} M☉")

    if parameters[0].chirp_mass is not None:
        chirp_masses = [p.chirp_mass for p in parameters]
        logger.info(f"  Chirp mass: min={min(chirp_masses):.1f}, max={max(chirp_masses):.1f}, mean={np.mean(chirp_masses):.1f} M☉")

    if parameters[0].mass_ratio is not None:
        mass_ratios = [p.mass_ratio for p in parameters]
        logger.info(f"  Mass ratio: min={min(mass_ratios):.2f}, max={max(mass_ratios):.2f}, mean={np.mean(mass_ratios):.2f}")

    # Spin statistics
    spins_1 = [p.a_1 for p in parameters]
    spins_2 = [p.a_2 for p in parameters]
    logger.info(f"  Spin a_1: min={min(spins_1):.2f}, max={max(spins_1):.2f}, mean={np.mean(spins_1):.2f}")
    logger.info(f"  Spin a_2: min={min(spins_2):.2f}, max={max(spins_2):.2f}, mean={np.mean(spins_2):.2f}")

    # Waveform amplitude statistics
    amplitudes = [np.abs(w.h_plus).max() for w in waveforms]
    logger.info(f"  Max amplitude: min={min(amplitudes):.3e}, max={max(amplitudes):.3e}, mean={np.mean(amplitudes):.3e}")

    logger.info("Note: For production-scale dataset generation with parallel processing,")
    logger.info("use the dingo_generate_dataset command-line tool:")
    logger.info(f"  dingo_generate_dataset --settings_file {config_file.name} --num_processes 8")


if __name__ == "__main__":
    main()
