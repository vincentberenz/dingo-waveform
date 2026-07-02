#!/usr/bin/env python3
"""
pSEOBNR waveform generation example.

Demonstrates how to:
1. Configure per-mode fractional deviations of the QNM ringdown frequency
   via ``domega_dict`` on ``BBHWaveformParameters``.
2. Generate a pSEOBNR waveform and compare it against the corresponding GR
   baseline (same source parameters, no deviations) to show the effect of
   ``domega_dict`` on the ringdown.

Usage:
    python generate_pseobnr_domega.py
"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from dingo_waveform.imports import read_file
from dingo_waveform.logs import set_logging
from dingo_waveform.polarizations import Polarization
from dingo_waveform.waveform_generator import WaveformGenerator, build_waveform_generator
from dingo_waveform.waveform_parameters import BBHWaveformParameters


def main() -> None:

    set_logging()
    logger: logging.Logger = logging.getLogger(__name__)

    config_file: Path = Path(__file__).parent / "advanced_pseobnrv5hm_domega.yaml"
    logger.info(f"Loading configuration from: {config_file.name}")

    wfg: WaveformGenerator = build_waveform_generator(config_file)
    config: Dict[str, Any] = read_file(config_file)

    params_pseob: BBHWaveformParameters = BBHWaveformParameters(
        **config["waveform_parameters"]
    )
    # Baseline: identical source parameters but no QNM deviations.
    params_gr: BBHWaveformParameters = replace(params_pseob, domega_dict=None)

    logger.info("pSEOBNR deviations (normalized to (ell, m) tuple keys):")
    for mode, dw in params_pseob.domega_dict.items():
        logger.info(f"  domega{mode} = {dw:+.4f}")

    logger.info("Generating pSEOBNR waveform...")
    pol_pseob: Polarization = wfg.generate_hplus_hcross(params_pseob)

    logger.info("Generating GR baseline waveform (domega_dict = None)...")
    pol_gr: Polarization = wfg.generate_hplus_hcross(params_gr)

    # Difference metric: max fractional deviation of |h+| across the band,
    # measured where the GR amplitude is non-negligible so we don't divide by ~0.
    amp_gr: np.ndarray = np.abs(pol_gr.h_plus)
    amp_pseob: np.ndarray = np.abs(pol_pseob.h_plus)

    mask: np.ndarray = amp_gr > 1e-3 * amp_gr.max()
    frac_diff: np.ndarray = np.abs(amp_pseob[mask] - amp_gr[mask]) / amp_gr[mask]

    logger.info("Waveform comparison (pSEOBNR vs GR baseline):")
    logger.info(f"  |h+| max            GR: {amp_gr.max():.3e}   pSEOBNR: {amp_pseob.max():.3e}")
    logger.info(f"  max fractional |h+| difference: {frac_diff.max():.3e}")
    logger.info(f"  mean fractional |h+| difference: {frac_diff.mean():.3e}")

    logger.info("✓ pSEOBNR waveform generation successful!")


if __name__ == "__main__":
    main()
