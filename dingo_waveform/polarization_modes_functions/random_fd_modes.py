"""
Random frequency-domain mode-separated polarization function for the RandomApproximant.

This module serves as a developer example showing how to implement a
PolarizationModesFunction for integration into dingo-waveform. It generates
synthetic mode-separated waveforms by decomposing the signal into spherical
harmonic contributions.

A PolarizationModesFunction has the signature:
    (WaveformGeneratorParameters, RandomWaveformParameters) -> Dict[Mode, Polarization]

See also:
    dingo_waveform.polarization_modes_functions.lalsimulation_simInspiralChooseFDModes
        for a real implementation using the LALSimulation backend.
"""

from typing import Dict

import numpy as np

from dingo_waveform.domains import BaseFrequencyDomain
from dingo_waveform.polarization_functions.random_fd import _generate_waveform_array
from dingo_waveform.polarizations import Polarization
from dingo_waveform.types import Mode
from dingo_waveform.waveform_generator_parameters import WaveformGeneratorParameters
from dingo_waveform.waveform_parameters import RandomWaveformParameters

# Mode numbers and their relative amplitudes.
# In real waveforms, higher-order modes are sub-dominant.
_MODE_CONFIG = {
    22: 1.0,    # dominant (2,2) mode
    33: 0.3,    # sub-dominant (3,3) mode
    44: 0.1,    # sub-dominant (4,4) mode
}


def random_inspiral_FD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: RandomWaveformParameters,
) -> Dict[Mode, Polarization]:
    """
    Generate synthetic mode-separated frequency-domain polarizations.

    Each mode is generated with a relative amplitude factor and a phase
    rotation of exp(-1j * m * phase), following the convention that
    mode-separated waveforms transform as exp(-1j * m * phase_shift)
    under phase shifts.

    Parameters
    ----------
    waveform_gen_params
        Waveform generation configuration (domain, f_ref, etc.)
    waveform_params
        Waveform parameters (masses, distance, phase, etc.)

    Returns
    -------
    Dictionary mapping mode integer (e.g. 22, 33, 44) to Polarization

    Raises
    ------
    ValueError
        If the domain is not a BaseFrequencyDomain
        If waveform_params.phase is None
    """
    domain = waveform_gen_params.domain
    if not isinstance(domain, BaseFrequencyDomain):
        raise ValueError(
            f"random_inspiral_FD_modes requires a BaseFrequencyDomain, "
            f"got {type(domain).__name__}"
        )

    if waveform_params.phase is None:
        raise ValueError(
            "random_inspiral_FD_modes requires waveform_params.phase to be set"
        )

    # Get frequency array from the domain
    sample_freqs_attr = getattr(domain, "sample_frequencies", None)
    frequencies = sample_freqs_attr() if callable(sample_freqs_attr) else sample_freqs_attr

    # Extract physical parameters
    mass_1 = waveform_params.mass_1
    mass_2 = waveform_params.mass_2
    luminosity_distance = waveform_params.luminosity_distance
    phase = waveform_params.phase

    result: Dict[Mode, Polarization] = {}

    for mode, relative_amplitude in _MODE_CONFIG.items():
        # The "m" quantum number: for mode=22, m=2; for mode=33, m=3; etc.
        m = mode % 10

        # Generate base waveform at phase=0 (mode contribution without phase rotation)
        h_plus_base = _generate_waveform_array(
            frequencies, mass_1, mass_2, luminosity_distance, 0.0, domain.f_min
        )
        h_cross_base = _generate_waveform_array(
            frequencies, mass_1, mass_2, luminosity_distance, np.pi / 2.0, domain.f_min
        )

        # Apply relative amplitude scaling for sub-dominant modes
        h_plus_base *= relative_amplitude
        h_cross_base *= relative_amplitude

        # Apply mode-dependent phase rotation: exp(-1j * m * phase)
        phase_factor = np.exp(-1j * m * phase)
        h_plus_mode = h_plus_base * phase_factor
        h_cross_mode = h_cross_base * phase_factor

        result[mode] = Polarization(h_plus=h_plus_mode, h_cross=h_cross_mode)

    return result
