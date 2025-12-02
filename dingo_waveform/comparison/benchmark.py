#!/usr/bin/env python3
"""
Benchmark waveform generation performance: dingo-waveform vs dingo-gw.

This script compares the time to generate waveform datasets using both packages,
ensuring identical parameters and domain settings for a fair comparison.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

# dingo-gw imports
from dingo.gw.domains import build_domain as dingo_build_domain
from dingo.gw.waveform_generator import WaveformGenerator as DingoWFG
from dingo.gw.waveform_generator import generate_waveforms_parallel as dingo_generate_parallel

# dingo-waveform imports
from dingo_waveform.approximant import Approximant
from dingo_waveform.domains import (
    MultibandedFrequencyDomain,
    UniformFrequencyDomain,
)
from dingo_waveform.prior import build_prior_with_defaults
from dingo_waveform.waveform_generator import WaveformGenerator as RefactoredWFG
from dingo_waveform.waveform_parameters import WaveformParameters

# Set up logger
logger = logging.getLogger(__name__)
console = Console()


class BenchmarkResult:
    """Store and display benchmark results."""

    def __init__(
        self,
        num_waveforms: int,
        domain_type: str,
        approximant: str,
        domain_size: int,
    ):
        self.num_waveforms = num_waveforms
        self.domain_type = domain_type
        self.approximant = approximant
        self.domain_size = domain_size

        # Timing results
        self.dingo_setup_time: float = 0.0
        self.refactored_setup_time: float = 0.0
        self.dingo_generation_time: float = 0.0
        self.refactored_generation_time: float = 0.0

        # Per-waveform statistics
        self.dingo_times: List[float] = []
        self.refactored_times: List[float] = []

    @property
    def dingo_total_time(self) -> float:
        return self.dingo_setup_time + self.dingo_generation_time

    @property
    def refactored_total_time(self) -> float:
        return self.refactored_setup_time + self.refactored_generation_time

    @property
    def speedup(self) -> float:
        """Speedup factor (>1 means dingo-waveform is faster)."""
        if self.refactored_total_time == 0:
            return float("inf")
        return self.dingo_total_time / self.refactored_total_time

    @property
    def generation_speedup(self) -> float:
        """Speedup for generation only (excluding setup)."""
        if self.refactored_generation_time == 0:
            return float("inf")
        return self.dingo_generation_time / self.refactored_generation_time

    def to_dict(self) -> Dict:
        """Convert results to dictionary for JSON serialization."""
        return {
            "configuration": {
                "num_waveforms": self.num_waveforms,
                "domain_type": self.domain_type,
                "approximant": self.approximant,
                "domain_size": self.domain_size,
            },
            "dingo_gw": {
                "setup_time_s": self.dingo_setup_time,
                "generation_time_s": self.dingo_generation_time,
                "total_time_s": self.dingo_total_time,
                "time_per_waveform_s": self.dingo_generation_time / self.num_waveforms,
                "waveforms_per_second": self.num_waveforms / self.dingo_generation_time
                if self.dingo_generation_time > 0
                else 0,
            },
            "dingo_waveform": {
                "setup_time_s": self.refactored_setup_time,
                "generation_time_s": self.refactored_generation_time,
                "total_time_s": self.refactored_total_time,
                "time_per_waveform_s": self.refactored_generation_time
                / self.num_waveforms,
                "waveforms_per_second": self.num_waveforms
                / self.refactored_generation_time
                if self.refactored_generation_time > 0
                else 0,
            },
            "comparison": {
                "total_speedup": self.speedup,
                "generation_speedup": self.generation_speedup,
                "time_difference_s": self.dingo_total_time - self.refactored_total_time,
                "percent_faster": (self.speedup - 1.0) * 100,
            },
        }

    def print_summary(self, verbose: bool = False):
        """Print formatted benchmark results using rich tables."""
        console.print()

        # Configuration table
        config_table = Table(title="Benchmark Configuration", show_header=False)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="white")
        config_table.add_row("Waveforms", str(self.num_waveforms))
        config_table.add_row("Domain", f"{self.domain_type} ({self.domain_size} bins)")
        config_table.add_row("Approximant", self.approximant)
        console.print(config_table)
        console.print()

        # Results table
        dingo_wf_per_sec = (
            self.num_waveforms / self.dingo_generation_time
            if self.dingo_generation_time > 0
            else 0
        )
        refactored_wf_per_sec = (
            self.num_waveforms / self.refactored_generation_time
            if self.refactored_generation_time > 0
            else 0
        )

        results_table = Table(title="Benchmark Results")
        results_table.add_column("Package", style="cyan", no_wrap=True)
        results_table.add_column("Setup (s)", justify="right", style="magenta")
        results_table.add_column("Generation (s)", justify="right", style="magenta")
        results_table.add_column("Total (s)", justify="right", style="yellow")
        results_table.add_column("Waveforms/s", justify="right", style="green")

        results_table.add_row(
            "dingo-gw",
            f"{self.dingo_setup_time:.4f}",
            f"{self.dingo_generation_time:.4f}",
            f"{self.dingo_total_time:.4f}",
            f"{dingo_wf_per_sec:.2f}",
        )
        results_table.add_row(
            "dingo-waveform",
            f"{self.refactored_setup_time:.4f}",
            f"{self.refactored_generation_time:.4f}",
            f"{self.refactored_total_time:.4f}",
            f"{refactored_wf_per_sec:.2f}",
        )

        console.print(results_table)
        console.print()

        # Comparison table
        comparison_table = Table(title="Performance Comparison", show_header=False)
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Value", style="white")

        comparison_table.add_row("Total speedup", f"{self.speedup:.3f}x")
        comparison_table.add_row("Generation speedup", f"{self.generation_speedup:.3f}x")
        comparison_table.add_row(
            "Time difference",
            f"{self.dingo_total_time - self.refactored_total_time:.3f}s"
        )

        if self.speedup > 1.0:
            status_color = "green"
            status_msg = f"✅ dingo-waveform is {(self.speedup - 1.0) * 100:.1f}% FASTER"
        elif self.speedup < 1.0:
            status_color = "yellow"
            status_msg = f"⚠️  dingo-waveform is {(1.0 - self.speedup) * 100:.1f}% slower"
        else:
            status_color = "yellow"
            status_msg = "⚠️  Performance is identical"

        comparison_table.add_row("Status", f"[{status_color}]{status_msg}[/{status_color}]")
        console.print(comparison_table)

        if verbose and len(self.dingo_times) > 0:
            console.print()
            stats_table = Table(title="Per-Waveform Statistics")
            stats_table.add_column("Package", style="cyan")
            stats_table.add_column("Mean (s)", justify="right", style="magenta")
            stats_table.add_column("Std Dev (s)", justify="right", style="magenta")

            stats_table.add_row(
                "dingo-gw",
                f"{np.mean(self.dingo_times):.4f}",
                f"{np.std(self.dingo_times):.4f}",
            )
            stats_table.add_row(
                "dingo-waveform",
                f"{np.mean(self.refactored_times):.4f}",
                f"{np.std(self.refactored_times):.4f}",
            )
            console.print(stats_table)

        console.print()


def benchmark_waveform_generation(
    config_path: str,
    num_waveforms: int,
    seed: Optional[int] = None,
    per_waveform_timing: bool = False,
    verbose: bool = False,
    num_processes: int = 1,
) -> BenchmarkResult:
    """
    Benchmark waveform generation for both dingo-gw and dingo-waveform.

    Parameters
    ----------
    config_path : str
        Path to JSON configuration file with domain, approximant, and prior settings
    num_waveforms : int
        Number of waveforms to generate
    seed : int, optional
        Random seed for reproducibility
    per_waveform_timing : bool
        If True, time each waveform individually (slower but more detailed)
    verbose : bool
        If True, print progress messages
    num_processes : int
        Number of parallel processes to use (Priority 7 optimization).
        If 1, runs sequentially. If > 1, uses parallel generation with worker initialization.

    Returns
    -------
    BenchmarkResult
        Benchmark results with timing information
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        logger.debug(f"Random seed set to: {seed}")

    # Load configuration
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.debug(f"Loading configuration from: {config_path}")
    with open(config_file, "r") as f:
        if config_file.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif config_file.suffix == ".json":
            config = json.load(f)
        else:
            # Try YAML by default
            config = yaml.safe_load(f)

    domain_params = config["domain"]
    wfg_config = config["waveform_generator"]
    # Support both 'prior' (dingo-waveform) and 'intrinsic_prior' (dingo-gw)
    prior_config = config.get("prior") or config.get("intrinsic_prior")

    # Determine domain type
    domain_type = domain_params["type"]
    if "multibanded" in domain_type.lower():
        domain_type_str = "MultibandedFrequencyDomain"
    else:
        domain_type_str = "UniformFrequencyDomain"

    logger.info(f"Configuration loaded:")
    logger.info(f"  Domain: {domain_type_str}")
    logger.info(f"  Approximant: {wfg_config['approximant']}")
    logger.info(f"  Waveforms: {num_waveforms}")

    # Sample parameters from prior
    logger.debug(f"Sampling {num_waveforms} parameter sets from prior...")

    prior = build_prior_with_defaults(prior_config)
    params_list = []
    for _ in range(num_waveforms):
        params_list.append(prior.sample())

    # =========================================================================
    # BENCHMARK: dingo-gw
    # =========================================================================
    logger.info("=" * 80)
    logger.info("BENCHMARKING: dingo-gw")
    logger.info("=" * 80)

    # Setup phase
    logger.debug("Setting up dingo-gw waveform generator...")
    t0 = time.perf_counter()
    dingo_domain = dingo_build_domain(domain_params)
    dingo_wfg = DingoWFG(
        approximant=wfg_config["approximant"],
        domain=dingo_domain,
        f_ref=wfg_config.get("f_ref", 20.0),
        f_start=wfg_config.get("f_start", 20.0),
        spin_conversion_phase=wfg_config.get("spin_conversion_phase", 0.0),
    )
    dingo_setup_time = time.perf_counter() - t0

    logger.info(f"dingo-gw setup time: {dingo_setup_time:.4f}s")

    if num_processes > 1:
        logger.debug(f"Generating {num_waveforms} waveforms with dingo-gw (parallel, {num_processes} processes)...")
    else:
        logger.debug(f"Generating {num_waveforms} waveforms with dingo-gw (sequential)...")

    # Generation phase
    dingo_times = []
    t0 = time.perf_counter()

    if num_processes > 1:
        # Parallel generation using dingo-gw's native parallel support
        from multiprocessing import Pool
        from threadpoolctl import threadpool_limits

        # Convert params_list to DataFrame (dingo-gw expects DataFrame)
        params_df = pd.DataFrame(params_list)

        # Use dingo-gw's parallel generation with thread limits
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                _ = dingo_generate_parallel(dingo_wfg, params_df, pool)

        # Note: per_waveform_timing not supported in parallel mode
        if per_waveform_timing:
            logger.warning("per_waveform_timing not supported in parallel mode for dingo-gw")
    else:
        # Sequential generation
        for i, params in enumerate(params_list):
            if per_waveform_timing:
                t_wf = time.perf_counter()

            _ = dingo_wfg.generate_hplus_hcross(params)

            if per_waveform_timing:
                dingo_times.append(time.perf_counter() - t_wf)

            if (i + 1) % max(1, num_waveforms // 10) == 0:
                logger.debug(f"  dingo-gw progress: {i + 1}/{num_waveforms}")

    dingo_generation_time = time.perf_counter() - t0

    logger.info(f"dingo-gw generation time: {dingo_generation_time:.4f}s")
    logger.debug(f"dingo-gw time per waveform: {dingo_generation_time / num_waveforms:.4f}s")

    # =========================================================================
    # BENCHMARK: dingo-waveform
    # =========================================================================
    logger.info("=" * 80)
    logger.info("BENCHMARKING: dingo-waveform")
    logger.info("=" * 80)

    # Setup phase
    logger.debug("Setting up dingo-waveform generator...")
    t0 = time.perf_counter()

    # Build domain
    domain_params_clean = {k: v for k, v in domain_params.items() if k != "type"}
    refactored_domain: Union[UniformFrequencyDomain, MultibandedFrequencyDomain]
    if "multibanded" in domain_type.lower():
        refactored_domain = MultibandedFrequencyDomain(**domain_params_clean)
    else:
        refactored_domain = UniformFrequencyDomain(**domain_params_clean)

    refactored_wfg = RefactoredWFG(
        approximant=Approximant(wfg_config["approximant"]),
        domain=refactored_domain,
        f_ref=wfg_config.get("f_ref", 20.0),
        f_start=wfg_config.get("f_start", 20.0),
        spin_conversion_phase=wfg_config.get("spin_conversion_phase", 0.0),
    )
    refactored_setup_time = time.perf_counter() - t0

    logger.info(f"dingo-waveform setup time: {refactored_setup_time:.4f}s")

    if num_processes > 1:
        logger.debug(f"Generating {num_waveforms} waveforms with dingo-waveform (parallel, {num_processes} processes)...")
    else:
        logger.debug(f"Generating {num_waveforms} waveforms with dingo-waveform (sequential)...")

    # Generation phase
    refactored_times = []
    t0 = time.perf_counter()

    if num_processes > 1:
        # Use optimized parallel generation (Priority 7)
        from dingo_waveform.dataset.generate import generate_waveforms_parallel_optimized

        # Convert params_list to DataFrame
        params_df = pd.DataFrame(params_list)

        # Generate in parallel
        _ = generate_waveforms_parallel_optimized(
            refactored_wfg,
            params_df,
            num_processes=num_processes
        )

        # Note: per_waveform_timing not supported in parallel mode
        if per_waveform_timing:
            logger.warning("per_waveform_timing not supported in parallel mode")
    else:
        # Sequential generation
        for i, params_dict in enumerate(params_list):
            if per_waveform_timing:
                t_wf = time.perf_counter()

            # Convert dict to WaveformParameters
            wf_params = WaveformParameters(**params_dict)
            _ = refactored_wfg.generate_hplus_hcross(wf_params)

            if per_waveform_timing:
                refactored_times.append(time.perf_counter() - t_wf)

            if (i + 1) % max(1, num_waveforms // 10) == 0:
                logger.debug(f"  dingo-waveform progress: {i + 1}/{num_waveforms}")

    refactored_generation_time = time.perf_counter() - t0

    logger.info(f"dingo-waveform generation time: {refactored_generation_time:.4f}s")
    logger.debug(f"dingo-waveform time per waveform: {refactored_generation_time / num_waveforms:.4f}s")

    # =========================================================================
    # RESULTS
    # =========================================================================
    result = BenchmarkResult(
        num_waveforms=num_waveforms,
        domain_type=domain_type_str,
        approximant=wfg_config["approximant"],
        domain_size=len(dingo_domain),
    )

    result.dingo_setup_time = dingo_setup_time
    result.dingo_generation_time = dingo_generation_time
    result.dingo_times = dingo_times

    result.refactored_setup_time = refactored_setup_time
    result.refactored_generation_time = refactored_generation_time
    result.refactored_times = refactored_times

    return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark waveform generation: dingo-waveform vs dingo-gw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with 100 waveforms
  dingo-benchmark --config examples/benchmark_quick.yaml --num-waveforms 100

  # Larger benchmark with detailed timing
  dingo-benchmark --config examples/benchmark_production.yaml --num-waveforms 1000 --verbose

  # Save results to file
  dingo-benchmark --config config.yaml -n 500 --output benchmark_results.json

  # Per-waveform timing (slower but more detailed)
  dingo-benchmark --config config.yaml -n 100 --per-waveform-timing

Configuration File Format:
  YAML format compatible with dingo-gw. Requires domain, waveform_generator,
  and intrinsic_prior (or prior) sections.
  See examples/benchmark_quick.yaml for reference.
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="examples/benchmark_quick.yaml",
        help="Path to YAML configuration file (default: examples/benchmark_quick.yaml)",
    )
    parser.add_argument(
        "--num-waveforms",
        "-n",
        type=int,
        default=100,
        help="Number of waveforms to generate (default: 100)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for JSON results (default: print to stdout only)",
    )
    parser.add_argument(
        "--per-waveform-timing",
        action="store_true",
        help="Time each waveform individually (slower but provides statistics)",
    )
    parser.add_argument(
        "--num-processes",
        "-p",
        type=int,
        default=1,
        help="Number of parallel processes to use (default: 1 = sequential). Priority 7 optimization.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # Ensure our module logger uses DEBUG level
    logger.setLevel(logging.DEBUG)

    try:
        # Run benchmark
        result = benchmark_waveform_generation(
            config_path=args.config,
            num_waveforms=args.num_waveforms,
            seed=args.seed,
            per_waveform_timing=args.per_waveform_timing,
            verbose=args.verbose,
            num_processes=args.num_processes,
        )

        # Print results
        result.print_summary(verbose=args.per_waveform_timing)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Results saved to: {output_path}")

        # Exit code based on performance
        if result.speedup >= 0.95:  # Allow 5% margin
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Performance regression

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
