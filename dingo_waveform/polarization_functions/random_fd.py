"""
Random frequency-domain polarization function for the RandomApproximant.

This module serves as a developer example showing how to implement a
PolarizationFunction for integration into dingo-waveform. It generates
synthetic waveforms with a physically motivated inspiral-like shape,
without calling any LAL or GWSignal backend.

A PolarizationFunction has the signature:
    (WaveformGeneratorParameters, RandomWaveformParameters) -> Polarization

See also:
    dingo_waveform.polarization_functions.lalsimulation_simInspiralFD
        for a real implementation using the LALSimulation backend.
"""

import numpy as np

from dingo_waveform.domains import BaseFrequencyDomain
from dingo_waveform.polarizations import Polarization
from dingo_waveform.waveform_generator_parameters import WaveformGeneratorParameters
from dingo_waveform.waveform_parameters import RandomWaveformParameters

# Solar mass in kg, speed of light, gravitational constant (SI)
_MSUN_SI = 1.989e30
_C_SI = 2.998e8
_G_SI = 6.674e-11


def _compute_f_isco(mass_1: float, mass_2: float) -> float:
    """Compute the ISCO frequency for a binary with given component masses (in solar masses)."""
    total_mass_kg = (mass_1 + mass_2) * _MSUN_SI
    # f_isco = c^3 / (6^(3/2) * pi * G * M)
    return _C_SI**3 / (6**1.5 * np.pi * _G_SI * total_mass_kg)


def _generate_waveform_array(
    frequencies: np.ndarray,
    mass_1: float,
    mass_2: float,
    luminosity_distance: float,
    phase: float,
    f_min: float,
) -> np.ndarray:
    """
    Generate a single complex frequency-domain waveform array.

    The waveform has:
    - Amplitude: f^(-7/6) inspiral envelope, tapered near f_isco
    - Phase: smooth evolution seeded deterministically from the masses
    - Scaling: 1 / luminosity_distance

    Parameters
    ----------
    frequencies
        Frequency array (Hz)
    mass_1, mass_2
        Component masses in solar masses
    luminosity_distance
        Luminosity distance in Mpc
    phase
        Orbital phase (radians)
    f_min
        Minimum frequency (Hz) — below this, output is zero

    Returns
    -------
    Complex frequency-domain strain array
    """
    n = len(frequencies)
    h = np.zeros(n, dtype=np.complex128)

    f_isco = _compute_f_isco(mass_1, mass_2)

    # Mask: only generate signal above f_min and at positive frequencies
    mask = (frequencies >= f_min) & (frequencies > 0)

    if not np.any(mask):
        return h

    f_active = frequencies[mask]

    # Amplitude: inspiral power law with smooth cutoff at f_isco
    amplitude = f_active ** (-7.0 / 6.0)
    # Smooth taper above f_isco using a Fermi function
    exponent = np.clip((f_active - f_isco) / (0.05 * f_isco), -50.0, 50.0)
    taper = 1.0 / (1.0 + np.exp(exponent))
    amplitude *= taper

    # Normalize so max amplitude is O(1e-21) at 1 Mpc
    amplitude *= 1e-21

    # Scale by distance
    if luminosity_distance > 0:
        amplitude /= luminosity_distance

    # Phase: deterministic smooth evolution seeded by the masses
    # Use a chirp-like phase: Psi(f) ~ f^(-5/3) with mass-dependent prefactor
    chirp_mass = (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2
    phase_evolution = -2.0 * np.pi * chirp_mass * (f_active / 100.0) ** (-5.0 / 3.0)
    phase_evolution += phase

    h[mask] = amplitude * np.exp(1j * phase_evolution)

    return h


def random_inspiral_FD(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: RandomWaveformParameters,
) -> Polarization:
    """
    Generate synthetic frequency-domain polarizations for RandomApproximant.

    This function generates waveforms with a physically motivated inspiral shape
    (f^(-7/6) amplitude, chirp-like phase evolution), without calling any
    external waveform generation library. It is deterministic: the same input
    parameters always produce the same output.

    Parameters
    ----------
    waveform_gen_params
        Waveform generation configuration (domain, f_ref, etc.)
    waveform_params
        Waveform parameters (masses, distance, phase, etc.)

    Returns
    -------
    Polarization with h_plus and h_cross arrays

    Raises
    ------
    ValueError
        If the domain is not a BaseFrequencyDomain
    """
    domain = waveform_gen_params.domain
    if not isinstance(domain, BaseFrequencyDomain):
        raise ValueError(
            f"random_inspiral_FD requires a BaseFrequencyDomain, "
            f"got {type(domain).__name__}"
        )

    # Get frequency array from the domain
    sample_freqs_attr = getattr(domain, "sample_frequencies", None)
    frequencies = sample_freqs_attr() if callable(sample_freqs_attr) else sample_freqs_attr

    # Extract physical parameters
    mass_1 = waveform_params.mass_1
    mass_2 = waveform_params.mass_2
    luminosity_distance = waveform_params.luminosity_distance
    phase = waveform_params.phase

    # Generate h_plus
    h_plus = _generate_waveform_array(
        frequencies, mass_1, mass_2, luminosity_distance, phase, domain.f_min
    )

    # Generate h_cross with pi/2 phase offset (circular polarization approximation)
    h_cross = _generate_waveform_array(
        frequencies,
        mass_1,
        mass_2,
        luminosity_distance,
        phase + np.pi / 2.0,
        domain.f_min,
    )

    return Polarization(h_plus=h_plus, h_cross=h_cross)
