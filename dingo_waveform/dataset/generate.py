"""Core functions for generating waveform datasets."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..domains import Domain, build_domain
from ..prior import build_prior_with_defaults
from ..waveform_generator import WaveformGenerator, build_waveform_generator
from ..waveform_parameters import WaveformParameters
from .polarizations import Polarizations
from .dataset_settings import DatasetSettings
from .waveform_dataset import WaveformDataset

_logger = logging.getLogger(__name__)


def _generate_single_waveform(
    parameters_dict: Dict, waveform_generator_params: Dict, domain_params: Dict
) -> Dict[str, np.ndarray]:
    """
    Generate a single waveform for given parameters.

    This function is designed to be called in parallel via ProcessPoolExecutor.
    It reconstructs the waveform generator from parameters to avoid pickling issues.

    Parameters
    ----------
    parameters_dict
        Dictionary of waveform parameters.
    waveform_generator_params
        Dictionary of waveform generator configuration.
    domain_params
        Dictionary of domain configuration.

    Returns
    -------
    Dict with 'h_plus' and 'h_cross' arrays, or None values if generation failed.
    """
    try:
        # Rebuild domain and generator (avoids pickling large objects)
        domain = build_domain(domain_params)
        wfg = build_waveform_generator(waveform_generator_params, domain)

        # Convert dict to WaveformParameters
        wf_params = WaveformParameters(**parameters_dict)

        # Generate waveform
        polarization = wfg.generate_hplus_hcross(wf_params)

        return {
            "h_plus": polarization.h_plus,
            "h_cross": polarization.h_cross,
        }
    except Exception as e:
        _logger.warning(
            f"Failed to generate waveform for parameters {parameters_dict}: {e}"
        )
        return {"h_plus": None, "h_cross": None}


def generate_waveforms_sequential(
    waveform_generator: WaveformGenerator,
    parameters: pd.DataFrame,
) -> Polarizations:
    """
    Generate waveforms sequentially (single process).

    Parameters
    ----------
    waveform_generator
        Configured waveform generator.
    parameters
        DataFrame of waveform parameters.

    Returns
    -------
    Polarizations with h_plus and h_cross arrays of shape (num_samples, frequency_bins).
    """
    h_plus_list = []
    h_cross_list = []

    _logger.info(f"Generating {len(parameters)} waveforms sequentially...")

    for idx, row in parameters.iterrows():
        try:
            wf_params = WaveformParameters(**row.to_dict())
            polarization = waveform_generator.generate_hplus_hcross(wf_params)
            h_plus_list.append(polarization.h_plus)
            h_cross_list.append(polarization.h_cross)
        except Exception as e:
            _logger.warning(f"Failed to generate waveform {idx}: {e}")
            # Append NaN arrays for failed waveforms
            domain_length = len(waveform_generator._waveform_gen_params.domain)
            h_plus_list.append(np.full(domain_length, np.nan, dtype=complex))
            h_cross_list.append(np.full(domain_length, np.nan, dtype=complex))

    return Polarizations(
        h_plus=np.array(h_plus_list),
        h_cross=np.array(h_cross_list),
    )


def generate_waveforms_parallel(
    waveform_generator: WaveformGenerator,
    parameters: pd.DataFrame,
    num_processes: int = 4,
) -> Polarizations:
    """
    Generate waveforms in parallel using ProcessPoolExecutor.

    Parameters
    ----------
    waveform_generator
        Configured waveform generator.
    parameters
        DataFrame of waveform parameters.
    num_processes
        Number of parallel processes to use.

    Returns
    -------
    Polarizations with h_plus and h_cross arrays of shape (num_samples, frequency_bins).
    """
    if num_processes == 1:
        return generate_waveforms_sequential(waveform_generator, parameters)

    _logger.info(
        f"Generating {len(parameters)} waveforms with {num_processes} processes..."
    )

    # Extract configuration for passing to workers
    wfg_params = waveform_generator._waveform_gen_params
    domain_params = wfg_params.domain.get_parameters()
    wfg_config = {
        "approximant": str(wfg_params.approximant),
        "f_ref": wfg_params.f_ref,
        "spin_conversion_phase": wfg_params.spin_conversion_phase,
    }

    # Submit all tasks
    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(
                _generate_single_waveform,
                row.to_dict(),
                wfg_config,
                domain_params.__dict__,
            ): idx
            for idx, row in parameters.iterrows()
        }

        # Collect results as they complete
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                _logger.error(f"Worker failed for waveform {idx}: {e}")
                # Store None for failed cases
                results[idx] = {"h_plus": None, "h_cross": None}

    # Aggregate results in order
    h_plus_list = []
    h_cross_list = []
    for idx in sorted(results.keys()):
        h_plus_list.append(results[idx]["h_plus"])
        h_cross_list.append(results[idx]["h_cross"])

    return Polarizations(
        h_plus=np.array(h_plus_list),
        h_cross=np.array(h_cross_list),
    )


def generate_parameters_and_polarizations(
    waveform_generator: WaveformGenerator,
    prior: Dict,
    num_samples: int,
    num_processes: int = 1,
) -> Tuple[pd.DataFrame, Polarizations]:
    """
    Generate dataset of waveforms based on parameters drawn from prior.

    Parameters
    ----------
    waveform_generator
        Configured waveform generator.
    prior
        Prior dictionary (bilby BBHPriorDict).
    num_samples
        Number of samples to generate.
    num_processes
        Number of parallel processes to use.

    Returns
    -------
    Tuple of (parameters DataFrame, Polarizations dataclass).
    If some waveforms fail, only successful ones are returned.
    """
    _logger.info(f"Generating dataset of size {num_samples}")

    # Sample parameters from prior
    parameters = pd.DataFrame(prior.sample(num_samples))

    # Generate waveforms
    if num_processes > 1:
        polarizations = generate_waveforms_parallel(
            waveform_generator, parameters, num_processes
        )
    else:
        polarizations = generate_waveforms_sequential(waveform_generator, parameters)

    # Find cases where waveform generation failed
    wf_failed = np.any(np.isnan(polarizations.h_plus), axis=1)
    if wf_failed.any():
        idx_failed = np.where(wf_failed)[0]
        idx_ok = np.where(~wf_failed)[0]
        polarizations_ok = Polarizations(
            h_plus=polarizations.h_plus[idx_ok],
            h_cross=polarizations.h_cross[idx_ok],
        )
        parameters_ok = parameters.iloc[idx_ok].reset_index(drop=True)
        failed_percent = 100 * len(idx_failed) / len(parameters)
        _logger.warning(
            f"{len(idx_failed)} out of {len(parameters)} configurations "
            f"({failed_percent:.1f}%) failed to generate."
        )
        _logger.info(
            f"Returning {len(idx_ok)} successfully generated configurations."
        )
        return parameters_ok, polarizations_ok

    return parameters, polarizations


def generate_waveform_dataset(
    settings: DatasetSettings, num_processes: int = 1
) -> WaveformDataset:
    """
    Generate a waveform dataset based on settings.

    Parameters
    ----------
    settings
        Dataset generation settings.
    num_processes
        Number of parallel processes to use.

    Returns
    -------
    WaveformDataset containing parameters and polarizations.
    """
    # Validate settings
    settings.validate()

    # Build components
    _logger.info("Building domain, prior, and waveform generator...")
    domain = build_domain(settings.domain)
    prior = build_prior_with_defaults(settings.intrinsic_prior)
    # Convert WaveformGeneratorSettings to dict for build_waveform_generator
    wfg_dict = settings.waveform_generator.to_dict()
    waveform_generator = build_waveform_generator(wfg_dict, domain)

    # Generate waveforms
    parameters, polarizations = generate_parameters_and_polarizations(
        waveform_generator, prior, settings.num_samples, num_processes
    )

    # Create dataset
    dataset = WaveformDataset(
        parameters=parameters,
        polarizations=polarizations,
        settings=settings,
    )

    _logger.info(f"Dataset generated successfully with {len(parameters)} samples.")
    return dataset
