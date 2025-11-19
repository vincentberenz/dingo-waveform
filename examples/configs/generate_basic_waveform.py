#!/usr/bin/env python3
"""
Basic waveform generation example.

Demonstrates how to:
1. Use build_waveform_generator factory function to create a WaveformGenerator from config
2. Generate waveform polarizations (h+, hx)
3. Access waveform properties

Usage:
    python generate_basic_waveform.py
"""

import logging
import numpy as np
from pathlib import Path

from dingo_waveform.waveform_generator import build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters
from dingo_waveform.logs import set_logging


def main():

    set_logging()
    logger = logging.getLogger(__name__)

    # Create waveform generator directly from config file
    config_file = Path(__file__).parent / "basic_uniform_frequency.yaml"
    logger.info(f"Loading configuration from: {config_file.name}")

    wfg = build_waveform_generator(config_file)

    # Access domain and approximant from the generator
    domain = wfg._waveform_gen_params.domain
    approximant = wfg._waveform_gen_params.approximant
    f_ref = wfg._waveform_gen_params.f_ref

    logger.info(f"Domain: {type(domain).__name__}")
    logger.info(f"  Frequency range: {domain.f_min:.1f} - {domain.f_max:.1f} Hz")
    logger.info(f"  Number of bins: {len(domain)}")

    logger.info(f"Waveform Generator:")
    logger.info(f"  Approximant: {approximant}")
    logger.info(f"  Reference frequency: {f_ref} Hz")

    # Create waveform parameters from the config file
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    params = WaveformParameters(**config['waveform_parameters'])
    logger.info(f"Waveform Parameters:")
    logger.info(f"  Mass 1: {params.mass_1} M☉")
    logger.info(f"  Mass 2: {params.mass_2} M☉")
    logger.info(f"  Total mass: {params.mass_1 + params.mass_2:.1f} M☉")
    logger.info(f"  Luminosity distance: {params.luminosity_distance} Mpc")
    logger.info(f"  Spin magnitudes: a1={params.a_1:.2f}, a2={params.a_2:.2f}")

    # Generate waveform
    logger.info("Generating waveform...")
    polarizations = wfg.generate_hplus_hcross(params)

    # Display results
    logger.info(f"Generated polarizations:")
    logger.info(f"  h_plus shape: {polarizations.h_plus.shape}")
    logger.info(f"  h_cross shape: {polarizations.h_cross.shape}")
    logger.info(f"  h_plus dtype: {polarizations.h_plus.dtype}")

    # Calculate and display some statistics
    h_plus_amp = np.abs(polarizations.h_plus)
    h_cross_amp = np.abs(polarizations.h_cross)

    logger.info(f"Amplitude statistics:")
    logger.info(f"  h_plus  - max: {h_plus_amp.max():.3e}, mean: {h_plus_amp.mean():.3e}")
    logger.info(f"  h_cross - max: {h_cross_amp.max():.3e}, mean: {h_cross_amp.mean():.3e}")

    # Find peak frequency
    peak_idx = np.argmax(h_plus_amp)
    peak_freq = domain.sample_frequencies()[peak_idx]
    logger.info(f"Peak amplitude at frequency: {peak_freq:.1f} Hz")

    logger.info("✓ Waveform generation successful!")


if __name__ == "__main__":
    main()
