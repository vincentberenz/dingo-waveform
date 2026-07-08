#!/usr/bin/env python3
"""
Developer example: generating waveforms with a custom approximant.

This script demonstrates the end-to-end integration of a new waveform
generator (RandomApproximant) into dingo-waveform. It shows how to:

1. Use build_waveform_generator to instantiate a RandomWaveformGenerator from config
2. Generate h+/hx polarizations
3. Generate mode-separated waveforms

For developers adding a new approximant, the integration steps are:

  a) Implement a PolarizationFunction in dingo_waveform/polarization_functions/
     Signature: (WaveformGeneratorParameters, YourWaveformParameters) -> Polarization

  b) Implement a PolarizationModesFunction in dingo_waveform/polarization_modes_functions/
     Signature: (WaveformGeneratorParameters, YourWaveformParameters) -> Dict[Mode, Polarization]

  c) Export both from their respective __init__.py files

  d) Create a WaveformGenerator subclass in dingo_waveform/waveform_generator.py
     that calls your functions in generate_hplus_hcross() and generate_hplus_hcross_m()

  e) Register your approximant in _APPROXIMANT_CLASS_MAP, PolarizationFunctions,
     PolarizationModesFunctions, and polarization_modes_approximants

Usage:
    python generate_random_waveform.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import Domain
from dingo_waveform.imports import read_file
from dingo_waveform.logs import set_logging
from dingo_waveform.polarizations import Polarization
from dingo_waveform.types import Mode
from dingo_waveform.waveform_generator import WaveformGenerator, build_waveform_generator
from dingo_waveform.waveform_parameters import RandomWaveformParameters


def main() -> None:

    set_logging()
    logger: logging.Logger = logging.getLogger(__name__)

    # --- Step 1: Build a waveform generator from config ---
    config_file: Path = Path(__file__).parent / "random_approximant.yaml"
    logger.info(f"Loading configuration from: {config_file.name}")

    wfg: WaveformGenerator = build_waveform_generator(config_file)

    # Access properties from the generator
    domain: Domain = wfg._waveform_gen_params.domain
    approximant: Approximant = wfg._waveform_gen_params.approximant

    logger.info(f"Generator type: {type(wfg).__name__}")
    logger.info(f"Approximant: {approximant}")
    logger.info(f"Domain: {type(domain).__name__} ({domain.f_min:.0f}-{domain.f_max:.0f} Hz)")

    # Load waveform parameters from config
    config: Dict[str, Any] = read_file(config_file)
    params: RandomWaveformParameters = RandomWaveformParameters(**config["waveform_parameters"])

    # --- Step 2: Generate h+/hx polarizations ---
    logger.info("Generating polarizations...")
    polarizations: Polarization = wfg.generate_hplus_hcross(params)

    h_plus_amp: np.ndarray = np.abs(polarizations.h_plus)
    h_cross_amp: np.ndarray = np.abs(polarizations.h_cross)

    logger.info(f"h_plus  shape: {polarizations.h_plus.shape}, max amplitude: {h_plus_amp.max():.3e}")
    logger.info(f"h_cross shape: {polarizations.h_cross.shape}, max amplitude: {h_cross_amp.max():.3e}")

    # --- Step 3: Generate mode-separated waveforms ---
    logger.info("Generating mode-separated waveforms...")
    # Need a non-zero phase for mode generation
    params.phase = 0.5
    modes: Dict[Mode, Polarization] = wfg.generate_hplus_hcross_m(params)

    logger.info(f"Generated {len(modes)} modes: {sorted(modes.keys())}")

    mode_list: List[Mode] = sorted(modes.keys())
    logger.info(f"  {'Mode':<10} {'Max |h+|':<12} {'Max |hx|':<12}")
    logger.info("  " + "-" * 34)
    for mode in mode_list:
        pol: Polarization = modes[mode]
        logger.info(
            f"  {mode:<10} {np.abs(pol.h_plus).max():<12.3e} {np.abs(pol.h_cross).max():<12.3e}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
