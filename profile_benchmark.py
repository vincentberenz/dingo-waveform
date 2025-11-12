#!/usr/bin/env python3
"""
Profile waveform generation to identify performance bottlenecks.

Uses cProfile to analyze where time is spent in both dingo-gw and dingo-waveform.
"""

import cProfile
import pstats
import io
from pathlib import Path
import numpy as np
import yaml

# dingo-gw imports
from dingo.gw.domains import build_domain as dingo_build_domain
from dingo.gw.waveform_generator import WaveformGenerator as DingoWFG

# dingo-waveform imports
from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import UniformFrequencyDomain
from dingo_waveform.prior import build_prior_with_defaults
from dingo_waveform.waveform_generator import WaveformGenerator as RefactoredWFG
from dingo_waveform.waveform_parameters import WaveformParameters


def profile_dingo_gw(config_path: str, num_waveforms: int, seed: int = 42):
    """Profile dingo-gw waveform generation."""
    np.random.seed(seed)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    domain_params = config["domain"]
    wfg_config = config["waveform_generator"]
    prior_config = config.get("prior") or config.get("intrinsic_prior")

    # Setup
    dingo_domain = dingo_build_domain(domain_params)
    dingo_wfg = DingoWFG(
        approximant=wfg_config["approximant"],
        domain=dingo_domain,
        f_ref=wfg_config.get("f_ref", 20.0),
        f_start=wfg_config.get("f_start", 20.0),
        spin_conversion_phase=wfg_config.get("spin_conversion_phase", 0.0),
    )

    # Sample parameters
    prior = build_prior_with_defaults(prior_config)
    params_list = [prior.sample() for _ in range(num_waveforms)]

    # Generate waveforms (this is what we profile)
    for params in params_list:
        _ = dingo_wfg.generate_hplus_hcross(params)


def profile_dingo_waveform(config_path: str, num_waveforms: int, seed: int = 42):
    """Profile dingo-waveform generation."""
    np.random.seed(seed)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    domain_params = config["domain"]
    wfg_config = config["waveform_generator"]
    prior_config = config.get("prior") or config.get("intrinsic_prior")

    # Setup
    domain_params_clean = {k: v for k, v in domain_params.items() if k != "type"}
    refactored_domain = UniformFrequencyDomain(**domain_params_clean)

    refactored_wfg = RefactoredWFG(
        approximant=Approximant(wfg_config["approximant"]),
        domain=refactored_domain,
        f_ref=wfg_config.get("f_ref", 20.0),
        f_start=wfg_config.get("f_start", 20.0),
        spin_conversion_phase=wfg_config.get("spin_conversion_phase", 0.0),
    )

    # Sample parameters
    prior = build_prior_with_defaults(prior_config)
    params_list = [prior.sample() for _ in range(num_waveforms)]

    # Generate waveforms (this is what we profile)
    for params_dict in params_list:
        wf_params = WaveformParameters(**params_dict)
        _ = refactored_wfg.generate_hplus_hcross(wf_params)


def print_profile_stats(profiler, title: str, top_n: int = 30):
    """Print formatted profile statistics."""
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    # Sort by cumulative time
    print("\n--- Top functions by CUMULATIVE time ---")
    ps.sort_stats('cumulative')
    ps.print_stats(top_n)

    # Sort by total time (time spent in function itself, not including subcalls)
    print("\n--- Top functions by TOTAL time (excluding subcalls) ---")
    ps.sort_stats('tottime')
    ps.print_stats(top_n)

    print(s.getvalue())


def main():
    config_path = "examples/benchmark_quick.yaml"
    num_waveforms = 50
    seed = 42

    print(f"Profiling waveform generation:")
    print(f"  Config: {config_path}")
    print(f"  Waveforms: {num_waveforms}")
    print(f"  Seed: {seed}")

    # Profile dingo-gw
    print("\nProfiling dingo-gw...")
    profiler_dingo = cProfile.Profile()
    profiler_dingo.enable()
    profile_dingo_gw(config_path, num_waveforms, seed)
    profiler_dingo.disable()

    print_profile_stats(profiler_dingo, "DINGO-GW PROFILE", top_n=30)

    # Profile dingo-waveform
    print("\nProfiling dingo-waveform...")
    profiler_refactored = cProfile.Profile()
    profiler_refactored.enable()
    profile_dingo_waveform(config_path, num_waveforms, seed)
    profiler_refactored.disable()

    print_profile_stats(profiler_refactored, "DINGO-WAVEFORM PROFILE", top_n=30)

    # Save detailed stats to files for further analysis
    profiler_dingo.dump_stats('/tmp/dingo_gw_profile.prof')
    profiler_refactored.dump_stats('/tmp/dingo_waveform_profile.prof')

    print("\n" + "=" * 80)
    print("Profile data saved to:")
    print("  /tmp/dingo_gw_profile.prof")
    print("  /tmp/dingo_waveform_profile.prof")
    print("\nAnalyze with: python -m pstats /tmp/dingo_gw_profile.prof")
    print("=" * 80)


if __name__ == "__main__":
    main()
