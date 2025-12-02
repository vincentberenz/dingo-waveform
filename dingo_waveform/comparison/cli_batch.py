#!/usr/bin/env python3
"""
Batch verification CLI for comparing dingo-waveform against dingo (dingo-gw).

This tool runs verification across multiple configurations and displays a summary report.
"""
import argparse
import logging
import sys
import time
import tempfile
import uuid
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Note: logging not yet configured here, will warn after set_logging() is called

from .cli import load_config, validate_config, validate_svd_config
from .core import compare_waveforms, compare_waveforms_modes, compare_svd_compression
from ..logs import set_logging


@dataclass
class VerificationResult:
    """Result of a single verification run."""
    config_file: str
    approximant: str
    domain_type: str
    status: str  # 'pass', 'fail', 'error', 'acceptable'
    max_diff: Optional[float]
    duration: float
    error_msg: Optional[str] = None

    @property
    def status_symbol(self) -> str:
        """Get status symbol for display."""
        symbols = {
            'pass': '✅',
            'acceptable': '⚠️',
            'fail': '❌',
            'error': '💥'
        }
        return symbols.get(self.status, '?')

    @property
    def status_color(self) -> str:
        """Get status color for rich display."""
        colors = {
            'pass': 'green',
            'acceptable': 'yellow',
            'fail': 'red',
            'error': 'magenta'
        }
        return colors.get(self.status, 'white')


def generate_test_configurations(output_dir: Path) -> List[Path]:
    """
    Generate a comprehensive suite of ~50 test configurations.

    Parameters
    ----------
    output_dir : Path
        Directory to write configuration files

    Returns
    -------
    List[Path]
        List of generated configuration file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base parameter sets
    base_phm = {
        'mass_1': 36.0,
        'mass_2': 29.0,
        'luminosity_distance': 1000.0,
        'theta_jn': 0.5,
        'phase': 1.0,
        'a_1': 0.3,
        'a_2': 0.2,
        'tilt_1': 0.1,
        'tilt_2': 0.2,
        'phi_12': 0.5,
        'phi_jl': 0.7,
        'geocent_time': 0.0,
    }

    base_aligned = {
        'mass_1': 36.0,
        'mass_2': 29.0,
        'luminosity_distance': 1000.0,
        'theta_jn': 0.5,
        'phase': 1.0,
        'chi_1': 0.3,
        'chi_2': 0.2,
        'geocent_time': 0.0,
    }

    # Domain configurations
    uniform_std = {
        'type': 'UniformFrequencyDomain',
        'delta_f': 0.125,
        'f_min': 20.0,
        'f_max': 1024.0,
    }

    uniform_high_res = {
        'type': 'UniformFrequencyDomain',
        'delta_f': 0.0625,
        'f_min': 20.0,
        'f_max': 1024.0,
    }

    uniform_extended = {
        'type': 'UniformFrequencyDomain',
        'delta_f': 0.125,
        'f_min': 20.0,
        'f_max': 2048.0,
    }

    mfd_std = {
        'type': 'MultibandedFrequencyDomain',
        'nodes': [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        'delta_f_initial': 0.0625,
        'base_delta_f': 0.0625,
    }

    mfd_coarse = {
        'type': 'MultibandedFrequencyDomain',
        'nodes': [20.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
        'delta_f_initial': 0.125,
        'base_delta_f': 0.125,
    }

    # Waveform generator settings
    wfg_std = {'f_ref': 20.0, 'f_start': 20.0, 'spin_conversion_phase': 0.0}

    # Mass variations
    masses_equal = {'mass_1': 30.0, 'mass_2': 30.0}
    masses_q2 = {'mass_1': 40.0, 'mass_2': 20.0}
    masses_q4 = {'mass_1': 40.0, 'mass_2': 10.0}
    masses_q8 = {'mass_1': 40.0, 'mass_2': 5.0}
    masses_heavy = {'mass_1': 60.0, 'mass_2': 50.0}
    masses_light = {'mass_1': 15.0, 'mass_2': 12.0}

    # Spin variations (aligned)
    spins_zero = {'chi_1': 0.0, 'chi_2': 0.0}
    spins_low = {'chi_1': 0.1, 'chi_2': 0.1}
    spins_high = {'chi_1': 0.8, 'chi_2': 0.6}
    spins_anti = {'chi_1': 0.5, 'chi_2': -0.5}

    # Spin variations (precessing)
    spins_p_low = {'a_1': 0.1, 'a_2': 0.1, 'tilt_1': 0.3, 'tilt_2': 0.3}
    spins_p_high = {'a_1': 0.8, 'a_2': 0.6, 'tilt_1': 1.0, 'tilt_2': 1.2}
    spins_p_aligned = {'a_1': 0.5, 'a_2': 0.5, 'tilt_1': 0.0, 'tilt_2': 0.0}

    # Inclination variations
    incl_face_on = {'theta_jn': 0.0}
    incl_edge_on = {'theta_jn': np.pi/2}

    test_configs = []

    # IMRPhenomXPHM - extensive coverage (FD, precessing)
    for domain_name, domain in [('uniform', uniform_std), ('mfd', mfd_std)]:
        test_configs.append((f'IMRPhenomXPHM_{domain_name}', 'IMRPhenomXPHM', domain, base_phm))

    # IMRPhenomXPHM mass variations
    for suffix, masses in [('q1', masses_equal), ('q2', masses_q2), ('q4', masses_q4)]:
        test_configs.append((f'IMRPhenomXPHM_uniform_{suffix}', 'IMRPhenomXPHM', uniform_std,
                           {**base_phm, **masses}))

    # IMRPhenomXPHM spin variations
    for suffix, spins in [('low_spin', spins_p_low), ('high_spin', spins_p_high), ('aligned', spins_p_aligned)]:
        test_configs.append((f'IMRPhenomXPHM_uniform_{suffix}', 'IMRPhenomXPHM', uniform_std,
                           {**base_phm, **spins}))

    # IMRPhenomXPHM inclination variations
    for suffix, incl in [('face_on', incl_face_on), ('edge_on', incl_edge_on)]:
        test_configs.append((f'IMRPhenomXPHM_uniform_{suffix}', 'IMRPhenomXPHM', uniform_std,
                           {**base_phm, **incl}))

    # IMRPhenomD - aligned spins (FD)
    for domain_name, domain in [('uniform', uniform_std), ('mfd', mfd_std), ('uniform_hires', uniform_high_res)]:
        test_configs.append((f'IMRPhenomD_{domain_name}', 'IMRPhenomD', domain, base_aligned))

    # IMRPhenomD variations
    test_configs.append((f'IMRPhenomD_uniform_q4', 'IMRPhenomD', uniform_std,
                       {**base_aligned, **masses_q4}))
    test_configs.append((f'IMRPhenomD_uniform_high_spin', 'IMRPhenomD', uniform_std,
                       {**base_aligned, **spins_high}))
    test_configs.append((f'IMRPhenomD_uniform_zero_spin', 'IMRPhenomD', uniform_std,
                       {**base_aligned, **spins_zero}))

    # SEOBNRv4HM - aligned spins, higher modes (TD)
    for suffix, masses in [('std', {}), ('q4', masses_q4), ('heavy', masses_heavy)]:
        test_configs.append((f'SEOBNRv4HM_uniform_{suffix}', 'SEOBNRv4HM', uniform_std,
                           {**base_aligned, **masses}))

    test_configs.append((f'SEOBNRv4HM_uniform_high_spin', 'SEOBNRv4HM', uniform_std,
                       {**base_aligned, **spins_high}))

    # SEOBNRv5HM - aligned spins, higher modes (TD)
    for suffix, masses in [('std', {}), ('q2', masses_q2), ('light', masses_light)]:
        test_configs.append((f'SEOBNRv5HM_uniform_{suffix}', 'SEOBNRv5HM', uniform_std,
                           {**base_aligned, **masses}))

    test_configs.append((f'SEOBNRv5HM_uniform_anti_aligned', 'SEOBNRv5HM', uniform_std,
                       {**base_aligned, **spins_anti}))

    # SEOBNRv4PHM - precessing, higher modes (TD)
    for suffix, masses in [('std', {}), ('q2', masses_q2)]:
        test_configs.append((f'SEOBNRv4PHM_uniform_{suffix}', 'SEOBNRv4PHM', uniform_std,
                           {**base_phm, **masses}))

    test_configs.append((f'SEOBNRv4PHM_uniform_high_spin', 'SEOBNRv4PHM', uniform_std,
                       {**base_phm, **spins_p_high}))

    # SEOBNRv5PHM - precessing, higher modes (TD)
    for suffix, masses in [('std', {}), ('equal', masses_equal)]:
        test_configs.append((f'SEOBNRv5PHM_uniform_{suffix}', 'SEOBNRv5PHM', uniform_std,
                           {**base_phm, **masses}))

    test_configs.append((f'SEOBNRv5PHM_uniform_edge_on', 'SEOBNRv5PHM', uniform_std,
                       {**base_phm, **incl_edge_on}))

    # Additional approximants if available
    # IMRPhenomPv2 - precessing, dominant mode (FD)
    test_configs.append((f'IMRPhenomPv2_uniform', 'IMRPhenomPv2', uniform_std, base_phm))
    test_configs.append((f'IMRPhenomPv2_mfd', 'IMRPhenomPv2', mfd_std, base_phm))

    # IMRPhenomHM - aligned, higher modes (FD)
    test_configs.append((f'IMRPhenomHM_uniform', 'IMRPhenomHM', uniform_std, base_aligned))
    test_configs.append((f'IMRPhenomHM_mfd', 'IMRPhenomHM', mfd_std, base_aligned))

    # SEOBNRv4 - aligned, dominant mode (TD)
    test_configs.append((f'SEOBNRv4_uniform', 'SEOBNRv4', uniform_std, base_aligned))
    test_configs.append((f'SEOBNRv4_uniform_q4', 'SEOBNRv4', uniform_std,
                       {**base_aligned, **masses_q4}))

    # Generate config files
    generated_files = []
    for name, approximant, domain, params in test_configs:
        config = {
            'domain': domain.copy(),
            'waveform_generator': {
                'approximant': approximant,
                **wfg_std
            },
            'waveform_parameters': params.copy()
        }

        config_file = output_dir / f"{name}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        generated_files.append(config_file)

    return generated_files


def find_config_files(path: Path, pattern: str = "*.{yaml,yml,json}") -> List[Path]:
    """
    Find all configuration files in a directory.

    Parameters
    ----------
    path : Path
        Directory to search or single config file
    pattern : str
        Glob pattern for config files

    Returns
    -------
    List[Path]
        List of configuration file paths
    """
    if path.is_file():
        return [path]

    configs: list[Path] = []
    # Search for yaml, yml, and json files
    for ext in ['yaml', 'yml', 'json']:
        configs.extend(path.glob(f"*.{ext}"))
        configs.extend(path.glob(f"**/*.{ext}"))

    return sorted(set(configs))


def run_single_verification(
    config_path: Path,
    modes: bool = False,
    svd: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False
) -> VerificationResult:
    """
    Run verification for a single configuration.

    Parameters
    ----------
    config_path : Path
        Path to configuration file
    modes : bool
        Compare mode-separated waveforms
    svd : bool
        Compare SVD compression
    seed : int, optional
        Random seed
    verbose : bool
        Verbose output

    Returns
    -------
    VerificationResult
        Verification result
    """
    start_time = time.time()

    try:
        # Load and validate configuration
        config = load_config(str(config_path))

        if not svd:
            validate_config(config)
        else:
            validate_svd_config(config)

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Extract configuration
        domain_params = config['domain']
        wfg_config = config['waveform_generator']

        # Determine domain type
        domain_type = domain_params['type'].lower()
        if 'multibanded' in domain_type:
            domain_type = 'multibanded'
        else:
            domain_type = 'uniform'

        approximant = wfg_config['approximant']

        # Prepare domain parameters
        domain_params_clean = {k: v for k, v in domain_params.items() if k != 'type'}

        # Run comparison
        if svd:
            # SVD comparison requires waveform_params_list, skip for now
            # TODO: Refactor SVD comparison to work with new API
            from rich.console import Console
            Console().print(f"[yellow]SVD comparison not yet supported in batch mode[/yellow]")
            return None  # type: ignore[return-value]
        elif modes:
            result = compare_waveforms_modes(
                domain_type=domain_type,
                domain_params=domain_params_clean,
                approximant=approximant,
                waveform_params=config.get('waveform_parameters', {}),
                f_ref=wfg_config.get('f_ref', 20.0),
                f_start=wfg_config.get('f_start', 20.0),
                spin_conversion_phase=wfg_config.get('spin_conversion_phase', 0.0),
            )
            max_diff = max(result.max_diff_h_plus, result.max_diff_h_cross) if result.max_diff_h_plus is not None else None
        else:
            result = compare_waveforms(
                domain_type=domain_type,
                domain_params=domain_params_clean,
                approximant=approximant,
                waveform_params=config.get('waveform_parameters', {}),
                f_ref=wfg_config.get('f_ref', 20.0),
                f_start=wfg_config.get('f_start', 20.0),
                spin_conversion_phase=wfg_config.get('spin_conversion_phase', 0.0),
            )
            max_diff = max(result.max_diff_h_plus, result.max_diff_h_cross) if result.max_diff_h_plus is not None else None

        # Determine status
        if not result.shapes_match:
            status = 'fail'
        elif max_diff is None:
            status = 'error'
        elif max_diff < 1e-15:
            status = 'pass'
        elif max_diff < 1e-10:
            status = 'acceptable'
        else:
            status = 'fail'

        duration = time.time() - start_time

        return VerificationResult(
            config_file=config_path.name,
            approximant=approximant,
            domain_type=domain_type,
            status=status,
            max_diff=max_diff,
            duration=duration,
        )

    except Exception as e:
        duration = time.time() - start_time
        return VerificationResult(
            config_file=config_path.name,
            approximant='unknown',
            domain_type='unknown',
            status='error',
            max_diff=None,
            duration=duration,
            error_msg=str(e),
        )


def display_results_rich(results: List[VerificationResult], total_duration: float):
    """Display results using rich formatting."""
    console = Console()

    # Summary statistics
    passed = sum(1 for r in results if r.status == 'pass')
    acceptable = sum(1 for r in results if r.status == 'acceptable')
    failed = sum(1 for r in results if r.status == 'fail')
    errors = sum(1 for r in results if r.status == 'error')
    total = len(results)

    # Create summary panel
    summary_text = Text()
    summary_text.append(f"Total: {total}  ", style="bold")
    summary_text.append(f"✅ Passed: {passed}  ", style="bold green")
    summary_text.append(f"⚠️  Acceptable: {acceptable}  ", style="bold yellow")
    summary_text.append(f"❌ Failed: {failed}  ", style="bold red")
    summary_text.append(f"💥 Errors: {errors}", style="bold magenta")

    console.print()
    console.print(Panel(
        summary_text,
        title="[bold]BATCH VERIFICATION SUMMARY[/bold]",
        border_style="blue"
    ))

    # Create results table
    table = Table(title="\nDetailed Results", show_header=True, header_style="bold cyan")
    table.add_column("Status", style="bold", width=6)
    table.add_column("Config File", style="cyan", width=30)
    table.add_column("Approximant", width=20)
    table.add_column("Domain", width=12)
    table.add_column("Max Diff", justify="right", width=12)
    table.add_column("Time (s)", justify="right", width=10)

    for result in results:
        max_diff_str = f"{result.max_diff:.2e}" if result.max_diff is not None else "N/A"

        table.add_row(
            result.status_symbol,
            result.config_file,
            result.approximant,
            result.domain_type,
            max_diff_str,
            f"{result.duration:.2f}",
            style=result.status_color if result.status == 'fail' else None
        )

    console.print(table)

    # Error details if any
    errors_found = [r for r in results if r.status == 'error']
    if errors_found:
        console.print("\n[bold red]Error Details:[/bold red]")
        for result in errors_found:
            error_msg = result.error_msg if result.error_msg else "Unknown error (max_diff is None)"
            console.print(f"  [cyan]{result.config_file}[/cyan]: [red]{error_msg}[/red]")

    # Final summary
    console.print(f"\n[bold]Total Duration:[/bold] {total_duration:.2f}s")

    if failed == 0 and errors == 0:
        console.print("\n[bold green]🎉 ALL VERIFICATIONS PASSED! 🎉[/bold green]")
    elif failed > 0:
        console.print(f"\n[bold red]⚠️  {failed} verification(s) failed[/bold red]")
        return 1
    else:
        console.print(f"\n[bold yellow]⚠️  {errors} error(s) occurred[/bold yellow]")
        return 2

    return 0


def display_results_basic(results: List[VerificationResult], total_duration: float):
    """Display results using basic formatting (fallback when rich not available)."""
    # Summary statistics
    passed = sum(1 for r in results if r.status == 'pass')
    acceptable = sum(1 for r in results if r.status == 'acceptable')
    failed = sum(1 for r in results if r.status == 'fail')
    errors = sum(1 for r in results if r.status == 'error')
    total = len(results)

    print("\n" + "=" * 100)
    print("BATCH VERIFICATION SUMMARY".center(100))
    print("=" * 100)
    print(f"Total: {total}  |  ✅ Passed: {passed}  |  ⚠️  Acceptable: {acceptable}  |  ❌ Failed: {failed}  |  💥 Errors: {errors}")
    print("=" * 100)

    # Results table
    print(f"\n{'Status':<8} {'Config File':<32} {'Approximant':<22} {'Domain':<14} {'Max Diff':<14} {'Time (s)':<10}")
    print("-" * 100)

    for result in results:
        max_diff_str = f"{result.max_diff:.2e}" if result.max_diff is not None else "N/A"
        print(f"{result.status_symbol:<8} {result.config_file:<32} {result.approximant:<22} {result.domain_type:<14} {max_diff_str:<14} {result.duration:<10.2f}")

    # Error details if any
    errors_found = [r for r in results if r.status == 'error']
    if errors_found:
        print("\nError Details:")
        for result in errors_found:
            error_msg = result.error_msg if result.error_msg else "Unknown error (max_diff is None)"
            print(f"  {result.config_file}: {error_msg}")

    print(f"\nTotal Duration: {total_duration:.2f}s")
    print("=" * 100)

    if failed == 0 and errors == 0:
        print("\n🎉 ALL VERIFICATIONS PASSED! 🎉")
        return 0
    elif failed > 0:
        print(f"\n⚠️  {failed} verification(s) failed")
        return 1
    else:
        print(f"\n⚠️  {errors} error(s) occurred")
        return 2


def main():
    """Main entry point for batch verification CLI."""
    parser = argparse.ArgumentParser(
        description="Batch verification of dingo-waveform against dingo (dingo-gw)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and verify ~50 configs (default, configs preserved)
  dingo-verify-batch

  # Generate and cleanup after
  dingo-verify-batch --cleanup

  # Verify all configs in specific directory
  dingo-verify-batch --dir examples/

  # Verify specific config files
  dingo-verify-batch --configs config1.yaml config2.yaml config3.yaml

  # Compare mode-separated waveforms
  dingo-verify-batch --modes

  # Use specific random seed
  dingo-verify-batch --seed 42
        """
    )

    parser.add_argument(
        '--dir', '-d',
        type=Path,
        default=None,
        help='Directory containing config files (if not specified, generates test suite)'
    )
    parser.add_argument(
        '--configs', '-c',
        nargs='+',
        type=Path,
        help='Specific config files to verify'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Delete generated config files after run (default: keep them)'
    )
    parser.add_argument(
        '--modes', '-m',
        action='store_true',
        help='Compare mode-separated waveforms'
    )
    parser.add_argument(
        '--svd',
        action='store_true',
        help='Compare SVD compression'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for parameter sampling'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Display detailed output for each verification'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='*.{yaml,yml,json}',
        help='Glob pattern for config files (default: *.{yaml,yml,json})'
    )

    args = parser.parse_args()

    # Set up logging
    set_logging()
    logger = logging.getLogger(__name__)

    # Determine config files source
    generated_dir = None
    if args.configs:
        # Use specific config files
        config_files = args.configs
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"[cyan]Using {len(config_files)} specified config file(s)[/cyan]")
        else:
            logger.info(f"Using {len(config_files)} specified config file(s)")
    elif args.dir is not None:
        # Use configs from specified directory
        config_files = find_config_files(args.dir, args.pattern)
        if not config_files:
            logger.error(f"No configuration files found in {args.dir}")
            return 1
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"[cyan]Found {len(config_files)} config file(s) in {args.dir}[/cyan]")
        else:
            logger.info(f"Found {len(config_files)} config file(s) in {args.dir}")
    else:
        # Generate comprehensive test suite
        session_id = uuid.uuid4().hex[:8]
        generated_dir = Path(tempfile.gettempdir()) / f"dingo-verify-{session_id}"

        if RICH_AVAILABLE:
            console = Console()
            console.print(Panel(
                f"[cyan]Generating comprehensive test suite (~50 configs)...[/cyan]\n"
                f"[bold]Config directory:[/bold] {generated_dir}\n"
                f"[dim](Configs will be preserved. Use --cleanup to delete after run)[/dim]",
                title="[bold]Auto-Generated Test Suite[/bold]",
                border_style="cyan"
            ))
        else:
            logger.info("=" * 80)
            logger.info("AUTO-GENERATED TEST SUITE")
            logger.info("=" * 80)
            logger.info(f"Config directory: {generated_dir}")
            logger.info("(Configs will be preserved. Use --cleanup to delete after run)")
            logger.info("=" * 80)

        config_files = generate_test_configurations(generated_dir)

        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Generated {len(config_files)} test configurations")
        else:
            logger.info(f"✓ Generated {len(config_files)} test configurations")

    # Run verifications
    results = []
    total_start = time.time()

    if RICH_AVAILABLE:
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Verifying configurations...", total=len(config_files))

            for config_file in config_files:
                progress.update(task, description=f"[cyan]Verifying {config_file.name}...")
                result = run_single_verification(
                    config_file,
                    modes=args.modes,
                    svd=args.svd,
                    seed=args.seed,
                    verbose=args.verbose
                )
                results.append(result)
                progress.advance(task)
    else:
        for i, config_file in enumerate(config_files, 1):
            logger.info(f"[{i}/{len(config_files)}] Verifying {config_file.name}...")
            result = run_single_verification(
                config_file,
                modes=args.modes,
                svd=args.svd,
                seed=args.seed,
                verbose=args.verbose
            )
            results.append(result)
            logger.info(f"  {result.status_symbol} ({result.duration:.2f}s)")

    total_duration = time.time() - total_start

    # Display results
    if RICH_AVAILABLE:
        exit_code = display_results_rich(results, total_duration)
    else:
        exit_code = display_results_basic(results, total_duration)

    # Cleanup or preserve generated configs
    if generated_dir is not None:
        if args.cleanup:
            import shutil
            try:
                shutil.rmtree(generated_dir)
                if RICH_AVAILABLE:
                    console = Console()
                    console.print(f"[dim]Cleaned up temporary configs[/dim]")
            except Exception as e:
                if RICH_AVAILABLE:
                    console = Console()
                    console.print(f"[yellow]Warning: Could not clean up {generated_dir}: {e}[/yellow]")
                else:
                    logger.warning(f"Could not clean up {generated_dir}: {e}")
        else:
            if RICH_AVAILABLE:
                console = Console()
                console.print(f"\n[cyan]Config files preserved at:[/cyan] [bold]{generated_dir}[/bold]")
            else:
                logger.info(f"Config files preserved at: {generated_dir}")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
