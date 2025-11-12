#!/usr/bin/env python3
"""
Detailed profiling of hot path to identify exact overhead sources.
"""
import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
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


def profile_dingo_gw_single_call(wfg, params):
    """Profile a single dingo-gw call."""
    profiler = cProfile.Profile()
    profiler.enable()

    _ = wfg.generate_hplus_hcross(params)

    profiler.disable()
    return profiler


def profile_dingo_waveform_single_call(wfg, params):
    """Profile a single dingo-waveform call."""
    profiler = cProfile.Profile()
    profiler.enable()

    wf_params = WaveformParameters(**params)
    _ = wfg.generate_hplus_hcross(wf_params)

    profiler.disable()
    return profiler


def main():
    print("=" * 80)
    print("DETAILED HOT PATH PROFILING")
    print("=" * 80)

    # Load config
    with open("examples/benchmark_quick.yaml", "r") as f:
        config = yaml.safe_load(f)

    domain_params = config["domain"]
    wfg_config = config["waveform_generator"]
    prior_config = config.get("prior") or config.get("intrinsic_prior")

    # Sample one parameter set
    prior = build_prior_with_defaults(prior_config)
    params = prior.sample()

    # Setup dingo-gw
    print("\nSetting up dingo-gw...")
    dingo_domain = dingo_build_domain(domain_params)
    dingo_wfg = DingoWFG(
        approximant=wfg_config["approximant"],
        domain=dingo_domain,
        f_ref=wfg_config.get("f_ref", 20.0),
        f_start=wfg_config.get("f_start", 20.0),
        spin_conversion_phase=wfg_config.get("spin_conversion_phase", 0.0),
    )

    # Setup dingo-waveform
    print("Setting up dingo-waveform...")
    domain_params_clean = {k: v for k, v in domain_params.items() if k != "type"}
    refactored_domain = UniformFrequencyDomain(**domain_params_clean)
    refactored_wfg = RefactoredWFG(
        approximant=Approximant(wfg_config["approximant"]),
        domain=refactored_domain,
        f_ref=wfg_config.get("f_ref", 20.0),
        f_start=wfg_config.get("f_start", 20.0),
        spin_conversion_phase=wfg_config.get("spin_conversion_phase", 0.0),
    )

    # Warm up
    print("\nWarming up...")
    for _ in range(10):
        _ = dingo_wfg.generate_hplus_hcross(params)
        wf_params = WaveformParameters(**params)
        _ = refactored_wfg.generate_hplus_hcross(wf_params)

    # Profile single calls
    print("\n" + "=" * 80)
    print("PROFILING SINGLE WAVEFORM GENERATION")
    print("=" * 80)

    print("\n--- dingo-gw (single call) ---")
    dingo_profiler = profile_dingo_gw_single_call(dingo_wfg, params)

    s = StringIO()
    ps = pstats.Stats(dingo_profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    print("\n--- dingo-waveform (single call) ---")
    refactored_profiler = profile_dingo_waveform_single_call(refactored_wfg, params)

    s = StringIO()
    ps = pstats.Stats(refactored_profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    # Time 100 calls for accuracy
    print("\n" + "=" * 80)
    print("TIMING 100 WAVEFORMS")
    print("=" * 80)

    n = 100
    params_list = [prior.sample() for _ in range(n)]

    # dingo-gw
    t0 = time.perf_counter()
    for p in params_list:
        _ = dingo_wfg.generate_hplus_hcross(p)
    dingo_time = time.perf_counter() - t0

    # dingo-waveform
    t0 = time.perf_counter()
    for p in params_list:
        wf_p = WaveformParameters(**p)
        _ = refactored_wfg.generate_hplus_hcross(wf_p)
    refactored_time = time.perf_counter() - t0

    print(f"\ndingo-gw: {dingo_time:.4f}s ({dingo_time/n*1000:.3f}ms per waveform)")
    print(f"dingo-waveform: {refactored_time:.4f}s ({refactored_time/n*1000:.3f}ms per waveform)")
    print(f"Overhead: {(refactored_time - dingo_time)/n*1000:.3f}ms per waveform ({(refactored_time/dingo_time - 1)*100:.1f}%)")

    # Profile 100 calls
    print("\n" + "=" * 80)
    print("PROFILING 100 WAVEFORMS")
    print("=" * 80)

    print("\n--- dingo-gw (100 calls) ---")
    profiler = cProfile.Profile()
    profiler.enable()
    for p in params_list:
        _ = dingo_wfg.generate_hplus_hcross(p)
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(40)
    print(s.getvalue())

    print("\n--- dingo-waveform (100 calls) ---")
    profiler = cProfile.Profile()
    profiler.enable()
    for p in params_list:
        wf_p = WaveformParameters(**p)
        _ = refactored_wfg.generate_hplus_hcross(wf_p)
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(40)
    print(s.getvalue())


if __name__ == "__main__":
    main()
