#!/usr/bin/env python3
"""
Mode-separated waveform generation example.

Demonstrates how to:
1. Use build_waveform_generator to create a generator from config
2. Generate mode-separated waveforms using generate_hplus_hcross_m()
3. Access individual spherical harmonic modes
4. Analyze mode amplitudes and contributions

Usage:
    python generate_modes.py
"""

import logging
import yaml
import numpy as np
from pathlib import Path

from dingo_waveform.waveform_generator import build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters
from dingo_waveform.logs import set_logging

def main():

    set_logging()
    logger = logging.getLogger(__name__)

    # Create waveform generator directly from config file
    config_file = Path(__file__).parent / "modes_imrphenomxphm.yaml"
    logger.info(f"Loading configuration from: {config_file.name}")

    wfg = build_waveform_generator(config_file)

    # Access domain and approximant from the generator
    domain = wfg._waveform_gen_params.domain
    approximant = wfg._waveform_gen_params.approximant

    logger.info(f"Waveform Generator:")
    logger.info(f"  Approximant: {approximant}")
    logger.info(f"  Domain: {type(domain).__name__}")

    # Load waveform parameters from config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    params = WaveformParameters(**config['waveform_parameters'])
    logger.info(f"Waveform Parameters:")
    logger.info(f"  Masses: {params.mass_1} M☉, {params.mass_2} M☉")
    logger.info(f"  Inclination: {params.theta_jn:.2f} rad")
    logger.info(f"  Spins: a1={params.a_1:.2f}, a2={params.a_2:.2f}")

    # Generate mode-separated waveforms
    logger.info("Generating mode-separated waveforms...")
    modes = wfg.generate_hplus_hcross_m(params)

    # Display available modes
    logger.info(f"Generated {len(modes)} modes:")
    mode_list = sorted(modes.keys())
    logger.info(f"  Modes: {mode_list}")

    # Analyze each mode
    logger.info("Mode analysis:")
    logger.info(f"  {'Mode':<10} {'Max |h+|':<12} {'Max |hx|':<12}")
    logger.info("  " + "-" * 40)

    mode_amplitudes = {}
    for mode in mode_list:
        pol = modes[mode]

        h_plus_max = np.abs(pol.h_plus).max()
        h_cross_max = np.abs(pol.h_cross).max()

        mode_amplitudes[mode] = max(h_plus_max, h_cross_max)

        logger.info(f"  {mode:<10} {h_plus_max:<12.3e} {h_cross_max:<12.3e}")

    # Find dominant mode
    dominant_mode = max(mode_amplitudes, key=mode_amplitudes.get)
    dominant_amplitude = mode_amplitudes[dominant_mode]

    logger.info(f"Dominant mode: {dominant_mode}")
    logger.info(f"  Amplitude: {dominant_amplitude:.3e}")

    # Calculate relative contributions
    logger.info("Relative mode contributions:")
    for mode in mode_list:
        relative_contribution = mode_amplitudes[mode] / dominant_amplitude
        logger.info(f"  {mode}: {relative_contribution:.2%}")

    # Access specific mode (e.g., the dominant mode if it's (2,2))
    if 22 in modes:
        h_22 = modes[22]
        logger.info(f"Accessing dominant mode (22):")
        logger.info(f"  h_plus shape: {h_22.h_plus.shape}")
        logger.info(f"  h_cross shape: {h_22.h_cross.shape}")

    logger.info("✓ Mode-separated waveform generation successful!")


if __name__ == "__main__":
    main()
