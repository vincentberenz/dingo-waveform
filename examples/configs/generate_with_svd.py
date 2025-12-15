#!/usr/bin/env python3
"""
SVD compression example.

Demonstrates how to:
1. Load configuration and build waveform generator
2. Use the dataset API to generate training and validation data
3. Build an SVD basis from training waveforms
4. Compress and reconstruct validation waveforms
5. Measure compression error (mismatch)

Usage:
    python generate_with_svd.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from bilby.gw.prior import BBHPriorDict

from dingo_waveform.approximant import Approximant
from dingo_waveform.compression.svd import SVDBasis
from dingo_waveform.dataset.generate import generate_parameters_and_polarizations
from dingo_waveform.domains import Domain
from dingo_waveform.imports import read_file
from dingo_waveform.logs import set_logging
from dingo_waveform.polarizations import BatchPolarizations
from dingo_waveform.prior import build_prior_with_defaults
from dingo_waveform.waveform_generator import WaveformGenerator, build_waveform_generator

def calculate_mismatch(wf1: np.ndarray, wf2: np.ndarray) -> float:
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


def main() -> None:

    set_logging()
    logger: logging.Logger = logging.getLogger(__name__)

    # Load configuration
    config_file: Path = Path(__file__).parent / "svd_compression_small.yaml"
    logger.info(f"Loading configuration from: {config_file.name}")

    config: Dict[str, Any] = read_file(config_file)

    # Create waveform generator
    wfg: WaveformGenerator = build_waveform_generator(config_file)

    # Access domain from the generator
    domain: Domain = wfg._waveform_gen_params.domain
    approximant: Approximant = wfg._waveform_gen_params.approximant

    logger.info(f"Domain: {type(domain).__name__}")
    logger.info(f"  Frequency range: {domain.f_min:.1f} - {domain.f_max:.1f} Hz")
    logger.info(f"  Frequency bins: {len(domain)} (dimension before compression)")

    logger.info(f"Waveform Generator:")
    logger.info(f"  Approximant: {approximant}")

    # Get SVD settings
    svd_config: Dict[str, Any] = config['svd']
    n_components: int = svd_config['n_components']
    num_training: int = svd_config['num_training']
    num_validation: int = svd_config['num_validation']

    logger.info(f"SVD Configuration:")
    logger.info(f"  Number of components: {n_components}")
    logger.info(f"  Training samples: {num_training}")
    logger.info(f"  Validation samples: {num_validation}")
    logger.info(f"  Compression ratio: {len(domain) / n_components:.1f}x")

    # Create prior
    prior_key: str = 'prior' if 'prior' in config else 'intrinsic_prior'
    prior: BBHPriorDict = build_prior_with_defaults(config[prior_key])

    # Generate training dataset using the dataset API
    logger.info(f"Generating {num_training} training waveforms using dataset API...")
    train_parameters: pd.DataFrame
    train_polarizations: BatchPolarizations
    train_parameters, train_polarizations = generate_parameters_and_polarizations(
        waveform_generator=wfg,
        prior=prior,
        num_samples=num_training,
        num_processes=1,  # Use 1 for this small example
    )

    logger.info(f"  ✓ Generated {len(train_parameters)} training waveforms")

    # Build SVD basis
    # Note: We stack h_plus and h_cross into single vectors for each waveform
    logger.info(f"Building SVD basis from training data...")

    # Prepare training data: stack h_plus and h_cross for each waveform
    # Result shape: (num_training, 2*n_freq_bins)
    training_data: np.ndarray = np.concatenate(
        [train_polarizations.h_plus, train_polarizations.h_cross],
        axis=1  # Concatenate along feature dimension
    )

    svd_basis: SVDBasis = SVDBasis()
    svd_basis.generate_basis(
        training_data=training_data,
        n_components=n_components,
        method=svd_config.get('method', 'scipy')
    )

    logger.info(f"  ✓ SVD basis created")
    logger.info(f"  Basis shape: {svd_basis.Vh.shape}")
    logger.info(f"  Number of components: {svd_basis.n_components}")

    # Generate validation dataset using the dataset API
    logger.info(f"Generating {num_validation} validation waveforms using dataset API...")
    val_parameters: pd.DataFrame
    val_polarizations: BatchPolarizations
    val_parameters, val_polarizations = generate_parameters_and_polarizations(
        waveform_generator=wfg,
        prior=prior,
        num_samples=num_validation,
        num_processes=1,
    )

    logger.info(f"  ✓ Generated {len(val_parameters)} validation waveforms")

    # Test compression on validation set
    logger.info(f"Testing compression on validation set...")
    mismatches_plus: List[float] = []
    mismatches_cross: List[float] = []

    n_freq_bins: int = len(domain)
    num_val_samples: int = len(val_parameters)

    # Process each validation waveform
    for i in range(num_val_samples):
        # Get individual waveform from batch
        h_plus_original: np.ndarray = val_polarizations.h_plus[i]  # shape: (n_freq_bins,)
        h_cross_original: np.ndarray = val_polarizations.h_cross[i]  # shape: (n_freq_bins,)

        # Stack h_plus and h_cross
        stacked: np.ndarray = np.concatenate([h_plus_original, h_cross_original])  # shape: (2*n_freq_bins,)

        # Compress
        coeffs: np.ndarray = svd_basis.compress(stacked)  # shape: (n_components,)

        # Reconstruct
        reconstructed: np.ndarray = svd_basis.decompress(coeffs)  # shape: (2*n_freq_bins,)

        # Separate back into h_plus and h_cross
        h_plus_reconstructed: np.ndarray = reconstructed[:n_freq_bins]
        h_cross_reconstructed: np.ndarray = reconstructed[n_freq_bins:]

        # Calculate mismatch
        mismatch_plus: float = calculate_mismatch(h_plus_original, h_plus_reconstructed)
        mismatch_cross: float = calculate_mismatch(h_cross_original, h_cross_reconstructed)

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
