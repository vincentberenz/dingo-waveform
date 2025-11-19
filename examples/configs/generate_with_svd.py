#!/usr/bin/env python3
"""
SVD compression example.

Demonstrates how to:
1. Use build_waveform_generator to create a generator from config
2. Generate a training dataset for SVD basis construction
3. Build an SVD basis from training waveforms
4. Compress waveforms using the SVD basis
5. Reconstruct waveforms and measure compression error

Usage:
    python generate_with_svd.py
"""

import logging
import yaml
import numpy as np
from pathlib import Path

from dingo_waveform.waveform_generator import build_waveform_generator
from dingo_waveform.waveform_parameters import WaveformParameters
from dingo_waveform.prior import build_prior_with_defaults
from dingo_waveform.compression.svd import SVDBasis
from dingo_waveform.logs import set_logging

def calculate_mismatch(wf1, wf2):
    """Calculate normalized mismatch between two waveforms."""
    # Inner product
    inner_product = np.vdot(wf1, wf2)
    norm1 = np.sqrt(np.vdot(wf1, wf1))
    norm2 = np.sqrt(np.vdot(wf2, wf2))

    if norm1 == 0 or norm2 == 0:
        return 1.0

    overlap = np.abs(inner_product) / (norm1 * norm2)
    mismatch = 1.0 - overlap.real

    return mismatch


def main():

    set_logging()
    logger = logging.getLogger(__name__)

    # Create waveform generator directly from config file
    config_file = Path(__file__).parent / "svd_compression_small.yaml"
    logger.info(f"Loading configuration from: {config_file.name}")

    wfg = build_waveform_generator(config_file)

    # Access domain from the generator
    domain = wfg._waveform_gen_params.domain
    approximant = wfg._waveform_gen_params.approximant

    logger.info(f"Domain: {type(domain).__name__}")
    logger.info(f"  Frequency range: {domain.f_min:.1f} - {domain.f_max:.1f} Hz")
    logger.info(f"  Frequency bins: {len(domain)} (dimension before compression)")

    logger.info(f"Waveform Generator:")
    logger.info(f"  Approximant: {approximant}")

    # Load configuration for SVD settings and prior
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Get SVD settings
    svd_config = config['svd']
    n_components = svd_config['n_components']
    num_training = svd_config['num_training']
    num_validation = svd_config['num_validation']

    logger.info(f"SVD Configuration:")
    logger.info(f"  Number of components: {n_components}")
    logger.info(f"  Training samples: {num_training}")
    logger.info(f"  Validation samples: {num_validation}")
    logger.info(f"  Compression ratio: {len(domain) / n_components:.1f}x")

    # Create prior
    prior_key = 'prior' if 'prior' in config else 'intrinsic_prior'
    prior = build_prior_with_defaults(config[prior_key])

    # Generate training dataset
    logger.info(f"Generating {num_training} training waveforms...")
    training_waveforms = []

    for i in range(num_training):
        sampled_params = prior.sample()
        params = WaveformParameters(**sampled_params)
        pol = wfg.generate_hplus_hcross(params)

        # Stack h_plus and h_cross into a single array
        stacked = np.concatenate([pol.h_plus, pol.h_cross])
        training_waveforms.append(stacked)

        if (i + 1) % 10 == 0:
            logger.info(f"  Generated {i + 1}/{num_training} waveforms...")

    # Stack into array: shape (num_training, 2*n_freq_bins)
    training_data = np.array(training_waveforms)

    # Build SVD basis
    logger.info(f"Building SVD basis from training data...")
    svd_basis = SVDBasis()
    svd_basis.generate_basis(
        training_data=training_data,
        n_components=n_components,
        method=svd_config.get('method', 'scipy')
    )

    logger.info(f"  ✓ SVD basis created")
    logger.info(f"  Basis shape: {svd_basis.Vh.shape}")
    logger.info(f"  Number of components: {svd_basis.n_components}")

    # Generate validation dataset
    logger.info(f"Generating {num_validation} validation waveforms...")
    validation_waveforms = []

    for i in range(num_validation):
        sampled_params = prior.sample()
        params = WaveformParameters(**sampled_params)
        pol = wfg.generate_hplus_hcross(params)
        validation_waveforms.append(pol)

    # Test compression on validation set
    logger.info(f"Testing compression on validation set...")
    mismatches_plus = []
    mismatches_cross = []

    n_freq_bins = len(domain)
    for i, pol in enumerate(validation_waveforms):
        # Stack h_plus and h_cross
        stacked = np.concatenate([pol.h_plus, pol.h_cross])

        # Compress
        coeffs = svd_basis.compress(stacked)

        # Reconstruct
        reconstructed = svd_basis.decompress(coeffs)

        # Separate back into h_plus and h_cross
        h_plus_reconstructed = reconstructed[:n_freq_bins]
        h_cross_reconstructed = reconstructed[n_freq_bins:]

        # Calculate mismatch
        mismatch_plus = calculate_mismatch(pol.h_plus, h_plus_reconstructed)
        mismatch_cross = calculate_mismatch(pol.h_cross, h_cross_reconstructed)

        mismatches_plus.append(mismatch_plus)
        mismatches_cross.append(mismatch_cross)

    # Display results
    logger.info("=" * 70)
    logger.info("SVD Compression Results")
    logger.info("=" * 70)

    logger.info("Compression Statistics:")
    logger.info(f"  Original dimension: {len(domain)}")
    logger.info(f"  Compressed dimension: {n_components}")
    logger.info(f"  Compression ratio: {len(domain) / n_components:.1f}x")

    logger.info("Reconstruction Error (Mismatch):")
    logger.info("  h_plus:")
    logger.info(f"    Mean:   {np.mean(mismatches_plus):.3e}")
    logger.info(f"    Median: {np.median(mismatches_plus):.3e}")
    logger.info(f"    Max:    {np.max(mismatches_plus):.3e}")
    logger.info(f"    Min:    {np.min(mismatches_plus):.3e}")

    logger.info("  h_cross:")
    logger.info(f"    Mean:   {np.mean(mismatches_cross):.3e}")
    logger.info(f"    Median: {np.median(mismatches_cross):.3e}")
    logger.info(f"    Max:    {np.max(mismatches_cross):.3e}")
    logger.info(f"    Min:    {np.min(mismatches_cross):.3e}")

    logger.info("=" * 70)
    logger.info("✓ SVD compression example complete!")

    # Optional: Save SVD basis
    # output_file = "svd_basis.hdf5"
    # svd_basis.save(output_file)
    # print(f"\nSaved SVD basis to: {output_file}")


if __name__ == "__main__":
    main()
