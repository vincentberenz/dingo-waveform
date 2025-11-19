"""
Utilities for comparing waveform generation between original dingo and dingo-waveform.

This module provides functions to generate waveforms using both packages and compare results.
It also provides utilities for comparing SVD compression between the two packages.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Original dingo imports
from dingo.gw.domains import MultibandedFrequencyDomain as DingoMFD
from dingo.gw.domains import UniformFrequencyDomain as DingoUFD
from dingo.gw.waveform_generator.waveform_generator import WaveformGenerator as DingoWFG
from dingo.gw.waveform_generator.waveform_generator import NewInterfaceWaveformGenerator as DingoWFG_Gwsignal

# dingo-waveform imports
from dingo_waveform.domains import MultibandedFrequencyDomain as RefactoredMFD
from dingo_waveform.domains import UniformFrequencyDomain as RefactoredUFD
from dingo_waveform.waveform_generator import WaveformGenerator as RefactoredWFG
from dingo_waveform.approximant import Approximant


@dataclass
class WaveformComparisonResult:
    """Results of comparing waveforms from dingo and dingo-waveform."""

    # Shapes
    dingo_shape: tuple
    refactored_shape: tuple
    shapes_match: bool

    # Array comparison (only if shapes match)
    max_diff_h_plus: Optional[float] = None
    max_diff_h_cross: Optional[float] = None
    mean_diff_h_plus: Optional[float] = None
    mean_diff_h_cross: Optional[float] = None

    # Relative differences (only if shapes match)
    max_rel_diff_h_plus: Optional[float] = None
    max_rel_diff_h_cross: Optional[float] = None

    # Domain information
    dingo_domain_type: str = ""
    refactored_domain_type: str = ""
    dingo_domain_len: int = 0
    refactored_domain_len: int = 0
    dingo_f_min: Optional[float] = None
    dingo_f_max: Optional[float] = None
    refactored_f_min: Optional[float] = None
    refactored_f_max: Optional[float] = None

    # Waveforms for further analysis
    dingo_h_plus: Optional[np.ndarray] = None
    dingo_h_cross: Optional[np.ndarray] = None
    refactored_h_plus: Optional[np.ndarray] = None
    refactored_h_cross: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """Human-readable summary of comparison."""
        lines = [
            "=== Waveform Comparison Result ===",
            f"\nDomain Information:",
            f"  Dingo:      {self.dingo_domain_type} (len={self.dingo_domain_len}, "
            f"f_min={self.dingo_f_min:.2f}, f_max={self.dingo_f_max:.2f})",
            f"  Refactored: {self.refactored_domain_type} (len={self.refactored_domain_len}, "
            f"f_min={self.refactored_f_min:.2f}, f_max={self.refactored_f_max:.2f})",
            f"\nShape Comparison:",
            f"  Dingo:      {self.dingo_shape}",
            f"  Refactored: {self.refactored_shape}",
            f"  Match:      {self.shapes_match}",
        ]

        if self.shapes_match and self.max_diff_h_plus is not None:
            lines.extend([
                f"\nAbsolute Differences:",
                f"  h_plus:  max={self.max_diff_h_plus:.2e}, mean={self.mean_diff_h_plus:.2e}",
                f"  h_cross: max={self.max_diff_h_cross:.2e}, mean={self.mean_diff_h_cross:.2e}",
                f"\nRelative Differences:",
                f"  h_plus:  max={self.max_rel_diff_h_plus:.2e}",
                f"  h_cross: max={self.max_rel_diff_h_cross:.2e}",
            ])

        return "\n".join(lines)


def create_uniform_domain_dingo(delta_f: float, f_min: float, f_max: float) -> DingoUFD:
    """Create UniformFrequencyDomain using original dingo."""
    return DingoUFD(f_min=f_min, f_max=f_max, delta_f=delta_f)


def create_uniform_domain_refactored(delta_f: float, f_min: float, f_max: float) -> RefactoredUFD:
    """Create UniformFrequencyDomain using dingo-waveform."""
    return RefactoredUFD(delta_f=delta_f, f_min=f_min, f_max=f_max)


def create_multibanded_domain_dingo(
    nodes: list, delta_f_initial: float, base_delta_f: float
) -> DingoMFD:
    """Create MultibandedFrequencyDomain using original dingo."""
    # Original dingo needs a base_domain
    # Use nodes[0] as f_min (not 0.0) to support mode generation
    # which requires f_min > 0 in LALSimulation
    base_domain = DingoUFD(
        f_min=nodes[0],
        f_max=nodes[-1],
        delta_f=base_delta_f,
    )
    return DingoMFD(
        nodes=nodes,
        delta_f_initial=delta_f_initial,
        base_domain=base_domain,
    )


def create_multibanded_domain_refactored(
    nodes: list, delta_f_initial: float, base_delta_f: float
) -> RefactoredMFD:
    """Create MultibandedFrequencyDomain using dingo-waveform."""
    return RefactoredMFD(
        nodes=nodes,
        delta_f_initial=delta_f_initial,
        base_delta_f=base_delta_f,
    )


def generate_waveform_dingo(
    domain,
    approximant: str,
    parameters: Dict[str, float],
    f_ref: float,
    f_start: float,
    spin_conversion_phase: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate waveform using original dingo.

    Returns
    -------
    Dictionary with keys 'h_plus' and 'h_cross'
    """
    # Use different generator class for gwsignal approximants (SEOBNRv5*)
    if "SEOBNRv5" in approximant:
        WFG_class = DingoWFG_Gwsignal
    else:
        WFG_class = DingoWFG

    wfg = WFG_class(
        approximant=approximant,
        domain=domain,
        f_ref=f_ref,
        f_start=f_start,
        spin_conversion_phase=spin_conversion_phase,
    )
    return wfg.generate_hplus_hcross(parameters)


def generate_waveform_refactored(
    domain,
    approximant: str,
    parameters: Dict[str, float],
    f_ref: float,
    f_start: float,
    spin_conversion_phase: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate waveform using dingo-waveform.

    Returns
    -------
    Dictionary with keys 'h_plus' and 'h_cross' (converted from Polarization)
    """
    from dataclasses import asdict
    from dingo_waveform.waveform_parameters import WaveformParameters

    wfg = RefactoredWFG(
        approximant=Approximant(approximant),
        domain=domain,
        f_ref=f_ref,
        f_start=f_start,
        spin_conversion_phase=spin_conversion_phase,
    )
    # Convert dict to WaveformParameters (it's a dataclass)
    wf_params = WaveformParameters(**parameters)
    pol = wfg.generate_hplus_hcross(wf_params)
    return asdict(pol)


def compare_waveforms(
    domain_type: str,  # "uniform" or "multibanded"
    domain_params: Dict,  # Parameters for domain construction
    approximant: str,
    waveform_params: Dict[str, float],
    f_ref: float,
    f_start: float,
    spin_conversion_phase: Optional[float] = None,
) -> WaveformComparisonResult:
    """
    Generate waveforms using both dingo and dingo-waveform and compare them.

    Parameters
    ----------
    domain_type : str
        Either "uniform" or "multibanded"
    domain_params : dict
        For uniform: {"delta_f": float, "f_min": float, "f_max": float}
        For multibanded: {"nodes": list, "delta_f_initial": float, "base_delta_f": float}
    approximant : str
        Approximant name (e.g., "IMRPhenomXPHM")
    waveform_params : dict
        Waveform parameters (masses, spins, distance, etc.)
    f_ref : float
        Reference frequency
    f_start : float
        Starting frequency
    spin_conversion_phase : float, optional
        Phase for spin conversion

    Returns
    -------
    WaveformComparisonResult
        Detailed comparison of the two waveforms
    """
    # Create domains
    if domain_type == "uniform":
        domain_dingo = create_uniform_domain_dingo(**domain_params)
        domain_refactored = create_uniform_domain_refactored(**domain_params)
    elif domain_type == "multibanded":
        domain_dingo = create_multibanded_domain_dingo(**domain_params)
        domain_refactored = create_multibanded_domain_refactored(**domain_params)
    else:
        raise ValueError(f"Unknown domain_type: {domain_type}")

    # Generate waveforms
    wf_dingo = generate_waveform_dingo(
        domain_dingo, approximant, waveform_params, f_ref, f_start, spin_conversion_phase
    )
    wf_refactored = generate_waveform_refactored(
        domain_refactored, approximant, waveform_params, f_ref, f_start, spin_conversion_phase
    )

    # Extract arrays
    h_plus_dingo = wf_dingo["h_plus"]
    h_cross_dingo = wf_dingo["h_cross"]
    h_plus_refactored = wf_refactored["h_plus"]
    h_cross_refactored = wf_refactored["h_cross"]

    # Compare shapes
    shapes_match = h_plus_dingo.shape == h_plus_refactored.shape

    # Initialize result
    result = WaveformComparisonResult(
        dingo_shape=h_plus_dingo.shape,
        refactored_shape=h_plus_refactored.shape,
        shapes_match=shapes_match,
        dingo_domain_type=type(domain_dingo).__name__,
        refactored_domain_type=type(domain_refactored).__name__,
        dingo_domain_len=len(domain_dingo),
        refactored_domain_len=len(domain_refactored),
        dingo_f_min=domain_dingo.f_min,
        dingo_f_max=domain_dingo.f_max,
        refactored_f_min=domain_refactored.f_min,
        refactored_f_max=domain_refactored.f_max,
        dingo_h_plus=h_plus_dingo,
        dingo_h_cross=h_cross_dingo,
        refactored_h_plus=h_plus_refactored,
        refactored_h_cross=h_cross_refactored,
    )

    # If shapes match, compute differences
    if shapes_match:
        diff_h_plus = np.abs(h_plus_dingo - h_plus_refactored)
        diff_h_cross = np.abs(h_cross_dingo - h_cross_refactored)

        result.max_diff_h_plus = float(np.max(diff_h_plus))
        result.max_diff_h_cross = float(np.max(diff_h_cross))
        result.mean_diff_h_plus = float(np.mean(diff_h_plus))
        result.mean_diff_h_cross = float(np.mean(diff_h_cross))

        # Relative differences (avoid division by zero)
        h_plus_mag = np.abs(h_plus_dingo)
        h_cross_mag = np.abs(h_cross_dingo)

        # Only compute relative diff where magnitude is significant
        threshold = 1e-30
        mask_plus = h_plus_mag > threshold
        mask_cross = h_cross_mag > threshold

        if np.any(mask_plus):
            rel_diff_plus = diff_h_plus[mask_plus] / h_plus_mag[mask_plus]
            result.max_rel_diff_h_plus = float(np.max(rel_diff_plus))

        if np.any(mask_cross):
            rel_diff_cross = diff_h_cross[mask_cross] / h_cross_mag[mask_cross]
            result.max_rel_diff_h_cross = float(np.max(rel_diff_cross))

    return result


def compare_waveforms_modes(
    domain_type: str,
    domain_params: Dict,
    approximant: str,
    waveform_params: Dict[str, float],
    f_ref: float,
    f_start: float,
    spin_conversion_phase: Optional[float] = None,
) -> WaveformComparisonResult:
    """
    Generate mode-separated waveforms using both dingo and dingo-waveform and compare them.

    This function compares the output of generate_hplus_hcross_m() for both packages.

    Parameters
    ----------
    domain_type : str
        Either "uniform" or "multibanded"
    domain_params : dict
        Domain parameters
    approximant : str
        Approximant name (must support modes: *PHM or *HM)
    waveform_params : dict
        Waveform parameters
    f_ref : float
        Reference frequency
    f_start : float
        Starting frequency
    spin_conversion_phase : float, optional
        Phase for spin conversion

    Returns
    -------
    WaveformComparisonResult
        Comparison result aggregated over all modes

    Raises
    ------
    ValueError
        If approximant does not support modes
    """
    # Check if approximant supports modes
    mode_approximants = ["PHM", "HM"]
    if not any(mode_app in approximant for mode_app in mode_approximants):
        raise ValueError(
            f"Approximant '{approximant}' does not support mode generation. "
            f"Must contain 'PHM' or 'HM' (e.g., IMRPhenomXPHM, SEOBNRv5HM)"
        )

    # Create domains
    if domain_type == "uniform":
        domain_dingo = create_uniform_domain_dingo(**domain_params)
        domain_refactored = create_uniform_domain_refactored(**domain_params)
    elif domain_type == "multibanded":
        domain_dingo = create_multibanded_domain_dingo(**domain_params)
        domain_refactored = create_multibanded_domain_refactored(**domain_params)
    else:
        raise ValueError(f"Unknown domain_type: {domain_type}")

    # Create waveform generators
    from dingo.gw.waveform_generator import WaveformGenerator as DingoWFG
    from dingo_waveform.waveform_generator import WaveformGenerator as RefactoredWFG
    from dingo_waveform.approximant import Approximant
    from dingo_waveform.waveform_parameters import WaveformParameters

    wfg_dingo = DingoWFG(
        domain=domain_dingo,
        approximant=approximant,
        f_ref=f_ref,
        f_start=f_start,
        spin_conversion_phase=spin_conversion_phase,
    )

    wfg_refactored = RefactoredWFG(
        approximant=Approximant(approximant),
        domain=domain_refactored,
        f_ref=f_ref,
        f_start=f_start,
        spin_conversion_phase=spin_conversion_phase,
    )

    # Generate mode-separated waveforms
    pol_m_dingo = wfg_dingo.generate_hplus_hcross_m(waveform_params)

    wf_params_obj = WaveformParameters(**waveform_params)
    pol_m_refactored = wfg_refactored.generate_hplus_hcross_m(wf_params_obj)

    # Check mode keys match
    modes_dingo = set(pol_m_dingo.keys())
    modes_refactored = set(pol_m_refactored.keys())

    if modes_dingo != modes_refactored:
        # Mode mismatch - create error result
        return WaveformComparisonResult(
            dingo_shape=(len(modes_dingo),),
            refactored_shape=(len(modes_refactored),),
            shapes_match=False,
            dingo_domain_type=type(domain_dingo).__name__,
            refactored_domain_type=type(domain_refactored).__name__,
            dingo_domain_len=len(domain_dingo),
            refactored_domain_len=len(domain_refactored),
            dingo_f_min=domain_dingo.f_min,
            dingo_f_max=domain_dingo.f_max,
            refactored_f_min=domain_refactored.f_min,
            refactored_f_max=domain_refactored.f_max,
        )

    # Compare each mode and aggregate differences
    max_diff_h_plus_all = 0.0
    max_diff_h_cross_all = 0.0
    mean_diff_h_plus_all = 0.0
    mean_diff_h_cross_all = 0.0
    max_rel_diff_h_plus_all = 0.0
    max_rel_diff_h_cross_all = 0.0

    num_modes = len(modes_dingo)

    for mode in modes_dingo:
        # Get polarizations for this mode
        pol_dingo = pol_m_dingo[mode]
        pol_refactored = pol_m_refactored[mode]

        # Extract arrays (handle different APIs)
        if isinstance(pol_dingo, dict):
            h_plus_dingo = pol_dingo["h_plus"]
            h_cross_dingo = pol_dingo["h_cross"]
        else:
            h_plus_dingo = pol_dingo.h_plus
            h_cross_dingo = pol_dingo.h_cross

        h_plus_refactored = pol_refactored.h_plus
        h_cross_refactored = pol_refactored.h_cross

        # Compute differences
        diff_h_plus = np.abs(h_plus_dingo - h_plus_refactored)
        diff_h_cross = np.abs(h_cross_dingo - h_cross_refactored)

        # Update maximums
        max_diff_h_plus_all = max(max_diff_h_plus_all, float(np.max(diff_h_plus)))
        max_diff_h_cross_all = max(max_diff_h_cross_all, float(np.max(diff_h_cross)))

        # Accumulate means
        mean_diff_h_plus_all += float(np.mean(diff_h_plus))
        mean_diff_h_cross_all += float(np.mean(diff_h_cross))

        # Compute relative differences
        h_plus_mag = np.abs(h_plus_dingo)
        h_cross_mag = np.abs(h_cross_dingo)

        threshold = 1e-30
        mask_plus = h_plus_mag > threshold
        mask_cross = h_cross_mag > threshold

        if np.any(mask_plus):
            rel_diff_plus = diff_h_plus[mask_plus] / h_plus_mag[mask_plus]
            max_rel_diff_h_plus_all = max(
                max_rel_diff_h_plus_all, float(np.max(rel_diff_plus))
            )

        if np.any(mask_cross):
            rel_diff_cross = diff_h_cross[mask_cross] / h_cross_mag[mask_cross]
            max_rel_diff_h_cross_all = max(
                max_rel_diff_h_cross_all, float(np.max(rel_diff_cross))
            )

    # Average the mean differences
    mean_diff_h_plus_all /= num_modes
    mean_diff_h_cross_all /= num_modes

    # Get first mode for shape info
    first_mode = sorted(modes_dingo)[0]
    pol_first = pol_m_dingo[first_mode]
    if isinstance(pol_first, dict):
        h_plus_first = pol_first["h_plus"]
    else:
        h_plus_first = pol_first.h_plus

    # Create result
    result = WaveformComparisonResult(
        dingo_shape=h_plus_first.shape,
        refactored_shape=pol_m_refactored[first_mode].h_plus.shape,
        shapes_match=True,
        max_diff_h_plus=max_diff_h_plus_all,
        max_diff_h_cross=max_diff_h_cross_all,
        mean_diff_h_plus=mean_diff_h_plus_all,
        mean_diff_h_cross=mean_diff_h_cross_all,
        max_rel_diff_h_plus=max_rel_diff_h_plus_all,
        max_rel_diff_h_cross=max_rel_diff_h_cross_all,
        dingo_domain_type=type(domain_dingo).__name__,
        refactored_domain_type=type(domain_refactored).__name__,
        dingo_domain_len=len(domain_dingo),
        refactored_domain_len=len(domain_refactored),
        dingo_f_min=domain_dingo.f_min,
        dingo_f_max=domain_dingo.f_max,
        refactored_f_min=domain_refactored.f_min,
        refactored_f_max=domain_refactored.f_max,
    )

    return result


@dataclass
class SVDComparisonResult:
    """Results of comparing SVD compression between dingo and dingo-waveform."""

    # SVD basis properties
    n_components: int
    dingo_basis_shape: tuple
    refactored_basis_shape: tuple
    shapes_match: bool

    # Basis comparison (V matrices)
    max_diff_V: Optional[float] = None
    mean_diff_V: Optional[float] = None
    max_rel_diff_V: Optional[float] = None

    # Singular values comparison
    max_diff_s: Optional[float] = None
    mean_diff_s: Optional[float] = None
    max_rel_diff_s: Optional[float] = None

    # Reconstruction mismatch comparison
    dingo_mean_mismatch: Optional[float] = None
    refactored_mean_mismatch: Optional[float] = None
    dingo_max_mismatch: Optional[float] = None
    refactored_max_mismatch: Optional[float] = None
    mismatch_difference: Optional[float] = None

    # Cross-system reconstruction test (basis interchangeability)
    cross_system_max_diff: Optional[float] = None
    cross_system_mean_diff: Optional[float] = None

    def __str__(self) -> str:
        """Human-readable summary of comparison."""
        lines = [
            "=== SVD Compression Comparison Result ===",
            f"\nSVD Basis Properties:",
            f"  n_components: {self.n_components}",
            f"  Dingo shape:      {self.dingo_basis_shape}",
            f"  Refactored shape: {self.refactored_basis_shape}",
            f"  Shapes match:     {self.shapes_match}",
        ]

        if self.shapes_match:
            if self.max_diff_V is not None:
                lines.extend([
                    f"\nBasis Matrix (V) Differences:",
                    f"  Max:  {self.max_diff_V:.2e}",
                    f"  Mean: {self.mean_diff_V:.2e}",
                    f"  Max relative: {self.max_rel_diff_V:.2e}",
                ])

            if self.max_diff_s is not None:
                lines.extend([
                    f"\nSingular Values (s) Differences:",
                    f"  Max:  {self.max_diff_s:.2e}",
                    f"  Mean: {self.mean_diff_s:.2e}",
                    f"  Max relative: {self.max_rel_diff_s:.2e}",
                ])

            if self.dingo_mean_mismatch is not None:
                lines.extend([
                    f"\nReconstruction Mismatches:",
                    f"  Dingo mean:      {self.dingo_mean_mismatch:.2e}",
                    f"  Refactored mean: {self.refactored_mean_mismatch:.2e}",
                    f"  Dingo max:       {self.dingo_max_mismatch:.2e}",
                    f"  Refactored max:  {self.refactored_max_mismatch:.2e}",
                    f"  Difference:      {self.mismatch_difference:.2e}",
                ])

            if self.cross_system_max_diff is not None:
                lines.extend([
                    f"\nCross-System Reconstruction Test:",
                    f"  Max difference:  {self.cross_system_max_diff:.2e}",
                    f"  Mean difference: {self.cross_system_mean_diff:.2e}",
                ])

        return "\n".join(lines)


def compare_svd_compression(
    domain_type: str,
    domain_params: Dict,
    approximant: str,
    waveform_params_list: list[Dict[str, float]],
    f_ref: float,
    f_start: float,
    n_components: int,
    num_training: int,
    num_validation: int,
    spin_conversion_phase: Optional[float] = None,
    svd_method: str = "scipy",
) -> SVDComparisonResult:
    """
    Compare SVD compression between dingo and dingo-waveform.

    This function:
    1. Generates training waveforms using both systems
    2. Trains SVD basis in both systems
    3. Validates reconstruction quality on test waveforms
    4. Tests cross-system compatibility (basis interchangeability)

    Parameters
    ----------
    domain_type : str
        Either "uniform" or "multibanded"
    domain_params : dict
        Domain parameters
    approximant : str
        Approximant name (e.g., "IMRPhenomXPHM")
    waveform_params_list : list[dict]
        List of waveform parameters for training and validation.
        First num_training used for training, next num_validation for validation.
    f_ref : float
        Reference frequency
    f_start : float
        Starting frequency
    n_components : int
        Number of SVD basis components to keep
    num_training : int
        Number of waveforms to use for training SVD
    num_validation : int
        Number of waveforms to use for validation
    spin_conversion_phase : float, optional
        Phase for spin conversion
    svd_method : str
        SVD method to use ("scipy" or "randomized")

    Returns
    -------
    SVDComparisonResult
        Detailed comparison of SVD compression in both systems

    Raises
    ------
    ValueError
        If insufficient waveform parameters provided
    """
    # Import SVD classes
    from dingo.gw.SVD import SVDBasis as DingoSVD
    from dingo_waveform.compression.svd import SVDBasis as RefactoredSVD

    if len(waveform_params_list) < num_training + num_validation:
        raise ValueError(
            f"Insufficient waveform parameters: need {num_training + num_validation}, "
            f"got {len(waveform_params_list)}"
        )

    # Create domains
    if domain_type == "uniform":
        domain_dingo = create_uniform_domain_dingo(**domain_params)
        domain_refactored = create_uniform_domain_refactored(**domain_params)
    elif domain_type == "multibanded":
        domain_dingo = create_multibanded_domain_dingo(**domain_params)
        domain_refactored = create_multibanded_domain_refactored(**domain_params)
    else:
        raise ValueError(f"Unknown domain_type: {domain_type}")

    print(f"\nGenerating {num_training} training waveforms...")

    # Generate training waveforms
    training_data_dingo = []
    training_data_refactored = []

    for i, params in enumerate(waveform_params_list[:num_training]):
        # Generate waveforms
        wf_dingo = generate_waveform_dingo(
            domain_dingo, approximant, params, f_ref, f_start, spin_conversion_phase
        )
        wf_refactored = generate_waveform_refactored(
            domain_refactored, approximant, params, f_ref, f_start, spin_conversion_phase
        )

        # Concatenate h_plus and h_cross (standard dingo format)
        data_dingo = np.concatenate([wf_dingo["h_plus"], wf_dingo["h_cross"]])
        data_refactored = np.concatenate([wf_refactored["h_plus"], wf_refactored["h_cross"]])

        training_data_dingo.append(data_dingo)
        training_data_refactored.append(data_refactored)

    training_data_dingo = np.array(training_data_dingo)
    training_data_refactored = np.array(training_data_refactored)

    print(f"Training data shape: {training_data_dingo.shape}")

    # Train SVD basis in both systems
    print(f"\nTraining SVD basis with {n_components} components...")
    print("  - Training dingo SVD...")
    svd_dingo = DingoSVD()
    svd_dingo.generate_basis(training_data_dingo, n=n_components, method=svd_method)

    print("  - Training dingo-waveform SVD...")
    svd_refactored = RefactoredSVD()
    svd_refactored.generate_basis(
        training_data_refactored, n_components=n_components, method=svd_method
    )

    # Compare basis shapes
    shapes_match = svd_dingo.V.shape == svd_refactored.V.shape

    result = SVDComparisonResult(
        n_components=n_components,
        dingo_basis_shape=svd_dingo.V.shape,
        refactored_basis_shape=svd_refactored.V.shape,
        shapes_match=shapes_match,
    )

    if not shapes_match:
        return result

    # Compare basis matrices (V)
    diff_V = np.abs(svd_dingo.V - svd_refactored.V)
    result.max_diff_V = float(np.max(diff_V))
    result.mean_diff_V = float(np.mean(diff_V))

    V_mag = np.abs(svd_dingo.V)
    threshold = 1e-30
    mask = V_mag > threshold
    if np.any(mask):
        rel_diff_V = diff_V[mask] / V_mag[mask]
        result.max_rel_diff_V = float(np.max(rel_diff_V))

    # Compare singular values (s)
    diff_s = np.abs(svd_dingo.s[:n_components] - svd_refactored.s)
    result.max_diff_s = float(np.max(diff_s))
    result.mean_diff_s = float(np.mean(diff_s))

    s_mag = np.abs(svd_dingo.s[:n_components])
    mask_s = s_mag > threshold
    if np.any(mask_s):
        rel_diff_s = diff_s[mask_s] / s_mag[mask_s]
        result.max_rel_diff_s = float(np.max(rel_diff_s))

    # Generate validation waveforms and compute mismatches
    if num_validation > 0:
        print(f"\nGenerating {num_validation} validation waveforms...")

        validation_data_dingo = []
        validation_data_refactored = []
        validation_params = waveform_params_list[num_training:num_training + num_validation]

        for params in validation_params:
            wf_dingo = generate_waveform_dingo(
                domain_dingo, approximant, params, f_ref, f_start, spin_conversion_phase
            )
            wf_refactored = generate_waveform_refactored(
                domain_refactored, approximant, params, f_ref, f_start, spin_conversion_phase
            )

            data_dingo = np.concatenate([wf_dingo["h_plus"], wf_dingo["h_cross"]])
            data_refactored = np.concatenate([wf_refactored["h_plus"], wf_refactored["h_cross"]])

            validation_data_dingo.append(data_dingo)
            validation_data_refactored.append(data_refactored)

        validation_data_dingo = np.array(validation_data_dingo)
        validation_data_refactored = np.array(validation_data_refactored)

        print(f"Validation data shape: {validation_data_dingo.shape}")

        # Compute mismatches in dingo
        print("\nComputing reconstruction mismatches...")
        print("  - Dingo...")
        mismatches_dingo = []
        for data in validation_data_dingo:
            compressed = svd_dingo.compress(data)
            reconstructed = svd_dingo.decompress(compressed)
            norm1 = np.sqrt(np.sum(np.abs(data) ** 2))
            norm2 = np.sqrt(np.sum(np.abs(reconstructed) ** 2))
            inner = np.sum(data.conj() * reconstructed).real
            mismatch = 1.0 - inner / (norm1 * norm2)
            mismatches_dingo.append(mismatch)

        # Compute mismatches in dingo-waveform
        print("  - Dingo-waveform...")
        mismatches_refactored = []
        for data in validation_data_refactored:
            compressed = svd_refactored.compress(data)
            reconstructed = svd_refactored.decompress(compressed)
            norm1 = np.sqrt(np.sum(np.abs(data) ** 2))
            norm2 = np.sqrt(np.sum(np.abs(reconstructed) ** 2))
            inner = np.sum(data.conj() * reconstructed).real
            mismatch = 1.0 - inner / (norm1 * norm2)
            mismatches_refactored.append(mismatch)

        mismatches_dingo = np.array(mismatches_dingo)
        mismatches_refactored = np.array(mismatches_refactored)

        result.dingo_mean_mismatch = float(np.mean(mismatches_dingo))
        result.refactored_mean_mismatch = float(np.mean(mismatches_refactored))
        result.dingo_max_mismatch = float(np.max(mismatches_dingo))
        result.refactored_max_mismatch = float(np.max(mismatches_refactored))
        result.mismatch_difference = float(np.abs(
            result.dingo_mean_mismatch - result.refactored_mean_mismatch
        ))

        # Test cross-system compatibility (basis interchangeability)
        print("\nTesting basis interchangeability...")
        print("  - Compressing with dingo, decompressing with dingo-waveform...")

        # Use first validation waveform for cross-system test
        test_data = validation_data_dingo[0]

        # Compress with dingo, decompress with refactored
        compressed_dingo = svd_dingo.compress(test_data)
        reconstructed_cross = svd_refactored.decompress(compressed_dingo)

        # Decompress with dingo for comparison
        reconstructed_dingo = svd_dingo.decompress(compressed_dingo)

        # Compare reconstructions
        cross_diff = np.abs(reconstructed_dingo - reconstructed_cross)
        result.cross_system_max_diff = float(np.max(cross_diff))
        result.cross_system_mean_diff = float(np.mean(cross_diff))

    return result
