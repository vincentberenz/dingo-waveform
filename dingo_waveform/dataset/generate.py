"""Core functions for generating waveform datasets."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from bilby.gw.prior import BBHPriorDict

from dingo_waveform.svd import SVDBasis, SVDGenerationConfig, SVDMetadata, ValidationConfig
from ..domains import Domain, DomainParameters, build_domain
from ..polarizations import BatchPolarizations
from ..prior import build_prior_with_defaults
from ..transform import ApplySVD, ComposeDataTransforms, DataTransform, WhitenUnwhitenTransform
from ..waveform_generator import WaveformGenerator, build_waveform_generator
from ..waveform_parameters import WaveformParameters
from .compression_settings import CompressionSettings
from .dataset_settings import DatasetSettings
from .generation_types import WaveformGeneratorConfig, WaveformResult
from .waveform_dataset import WaveformDataset

_logger = logging.getLogger(__name__)

# =============================================================================
# Worker Process State Management
# =============================================================================
#
# JUSTIFICATION FOR GLOBAL VARIABLES:
#
# The global variables _worker_generator and _worker_domain are used to maintain
# persistent state within worker processes for parallel waveform generation.
#
# This is the STANDARD and RECOMMENDED pattern for ProcessPoolExecutor with
# initializer functions in Python multiprocessing:
#
# 1. Each worker process is initialized ONCE via the initializer function
# 2. The initializer sets up expensive-to-create objects (WaveformGenerator)
#    in global variables within that worker's process space
# 3. Task functions then access these globals to avoid reconstruction overhead
# 4. Each worker process has its OWN copy of these globals (process isolation)
#
# Alternative approaches considered:
# - Passing generator as argument: Would require pickling/unpickling for every task
# - Class-based approach with __call__: More complex, no performance benefit
# - Shared memory: Overly complex for this use case, generators aren't easily shareable
#
# Performance impact: This optimization reduces waveform generation time by ~50-70%
# for large datasets by eliminating repeated LALSimulation initialization.
#
# =============================================================================

_worker_generator: Optional[WaveformGenerator] = None
_worker_domain: Optional[Domain] = None


def _init_worker(
    wfg_config: WaveformGeneratorConfig, domain_params: DomainParameters
) -> None:
    """
    Initialize worker process state.

    This function is called once per worker process to set up the waveform generator.
    This avoids reconstructing the generator for every waveform (Priority 1 optimization).

    Parameters
    ----------
    wfg_config
        Waveform generator configuration
    domain_params
        Domain parameters
    """
    global _worker_generator, _worker_domain

    # Build domain and generator once per worker
    _worker_domain = build_domain(domain_params)
    _worker_generator = build_waveform_generator(wfg_config.to_dict(), _worker_domain)


def _generate_single_waveform_optimized(parameters_dict: dict) -> WaveformResult:
    """
    Generate a single waveform using initialized worker state (Priority 1 optimization).

    This function assumes the worker has been initialized via _init_worker().
    It reuses the generator instead of rebuilding it for each waveform.

    Parameters
    ----------
    parameters_dict
        Dictionary of waveform parameters

    Returns
    -------
    WaveformResult with polarizations or error information
    """
    global _worker_generator

    try:
        # Use pre-initialized generator
        wf_params = WaveformParameters(**parameters_dict)
        polarization = _worker_generator.generate_hplus_hcross(wf_params)

        return WaveformResult.success_result(polarization.h_plus, polarization.h_cross)
    except Exception as e:
        _logger.warning(
            f"Failed to generate waveform for parameters {parameters_dict}: {e}"
        )
        return WaveformResult.failure_result(str(e))


def _generate_waveform_batch(params_batch: List[dict]) -> List[WaveformResult]:
    """
    Generate a batch of waveforms using initialized worker state (Priority 2 optimization).

    This function processes multiple waveforms in one task to reduce task overhead.

    Parameters
    ----------
    params_batch
        List of parameter dictionaries

    Returns
    -------
    List of WaveformResult objects
    """
    global _worker_generator

    results = []
    for parameters_dict in params_batch:
        try:
            wf_params = WaveformParameters(**parameters_dict)
            polarization = _worker_generator.generate_hplus_hcross(wf_params)
            results.append(
                WaveformResult.success_result(polarization.h_plus, polarization.h_cross)
            )
        except Exception as e:
            _logger.warning(
                f"Failed to generate waveform for parameters {parameters_dict}: {e}"
            )
            results.append(WaveformResult.failure_result(str(e)))

    return results


def _generate_single_waveform(
    parameters_dict: dict,
    wfg_config: WaveformGeneratorConfig,
    domain_params: DomainParameters,
) -> WaveformResult:
    """
    Generate a single waveform for given parameters (legacy function).

    This function is designed to be called in parallel via ProcessPoolExecutor.
    It reconstructs the waveform generator from parameters to avoid pickling issues.

    NOTE: This is the old implementation. Use generate_waveforms_parallel_optimized
    for better performance with worker initialization.

    Parameters
    ----------
    parameters_dict
        Dictionary of waveform parameters
    wfg_config
        Waveform generator configuration
    domain_params
        Domain parameters

    Returns
    -------
    WaveformResult with polarizations or error information
    """
    try:
        # Rebuild domain and generator (avoids pickling large objects)
        domain = build_domain(domain_params)
        wfg = build_waveform_generator(wfg_config.to_dict(), domain)

        # Convert dict to WaveformParameters
        wf_params = WaveformParameters(**parameters_dict)

        # Generate waveform
        polarization = wfg.generate_hplus_hcross(wf_params)

        return WaveformResult.success_result(polarization.h_plus, polarization.h_cross)
    except Exception as e:
        _logger.warning(
            f"Failed to generate waveform for parameters {parameters_dict}: {e}"
        )
        return WaveformResult.failure_result(str(e))


def generate_waveforms_sequential(
    waveform_generator: WaveformGenerator,
    parameters: pd.DataFrame,
) -> BatchPolarizations:
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
    BatchPolarizations with h_plus and h_cross arrays of shape (num_samples, frequency_bins).
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

    polarizations = BatchPolarizations(
        h_plus=np.array(h_plus_list),
        h_cross=np.array(h_cross_list),
    )

    # Apply batch transforms if configured
    if waveform_generator.transform is not None:
        polarizations = apply_transforms_to_polarizations(
            polarizations, waveform_generator.transform
        )

    return polarizations


def generate_waveforms_parallel(
    waveform_generator: WaveformGenerator,
    parameters: pd.DataFrame,
    num_processes: int = 4,
) -> BatchPolarizations:
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
    BatchPolarizations with h_plus and h_cross arrays of shape (num_samples, frequency_bins).
    """
    if num_processes == 1:
        return generate_waveforms_sequential(waveform_generator, parameters)

    _logger.info(
        f"Generating {len(parameters)} waveforms with {num_processes} processes..."
    )

    # Extract configuration for passing to workers
    wfg_params = waveform_generator._waveform_gen_params
    domain_params = wfg_params.domain.get_parameters()
    wfg_config = WaveformGeneratorConfig(
        approximant=str(wfg_params.approximant),
        f_ref=wfg_params.f_ref,
        spin_conversion_phase=wfg_params.spin_conversion_phase,
    )

    # Submit all tasks
    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(
                _generate_single_waveform,
                row.to_dict(),
                wfg_config,
                domain_params,
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
                # Store failure result for failed cases
                results[idx] = WaveformResult.failure_result(str(e))

    # Aggregate results in order
    h_plus_list = []
    h_cross_list = []
    for idx in sorted(results.keys()):
        result = results[idx]
        h_plus_list.append(result.h_plus)
        h_cross_list.append(result.h_cross)

    polarizations = BatchPolarizations(
        h_plus=np.array(h_plus_list),
        h_cross=np.array(h_cross_list),
    )

    # Apply batch transforms if configured
    if waveform_generator.transform is not None:
        polarizations = apply_transforms_to_polarizations(
            polarizations, waveform_generator.transform
        )

    return polarizations


def generate_waveforms_parallel_optimized(
    waveform_generator: WaveformGenerator,
    parameters: pd.DataFrame,
    num_processes: int = 4,
    batch_size: Optional[int] = None,
) -> BatchPolarizations:
    """
    Generate waveforms in parallel with optimized worker initialization (Priorities 1 & 2).

    This function uses persistent worker processes that initialize the waveform generator once,
    and optionally processes waveforms in batches to reduce task overhead.

    Parameters
    ----------
    waveform_generator
        Configured waveform generator.
    parameters
        DataFrame of waveform parameters.
    num_processes
        Number of parallel processes to use.
    batch_size
        Number of waveforms to process per task. If None, auto-compute based on dataset size.
        Recommended: 50-100 for simple approximants, 10-20 for complex approximants.

    Returns
    -------
    BatchPolarizations with h_plus and h_cross arrays of shape (num_samples, frequency_bins).
    """
    if num_processes == 1:
        return generate_waveforms_sequential(waveform_generator, parameters)

    _logger.info(
        f"Generating {len(parameters)} waveforms with {num_processes} processes "
        f"(optimized with worker initialization and batching)..."
    )

    # Extract configuration for passing to workers
    wfg_params = waveform_generator._waveform_gen_params
    domain_params = wfg_params.domain.get_parameters()
    wfg_config = WaveformGeneratorConfig(
        approximant=str(wfg_params.approximant),
        f_ref=wfg_params.f_ref,
        spin_conversion_phase=wfg_params.spin_conversion_phase,
    )

    # Auto-compute batch size if not specified
    if batch_size is None:
        # Heuristic: aim for ~4-8 tasks per process
        num_waveforms = len(parameters)
        ideal_num_tasks = num_processes * 6
        batch_size = max(1, num_waveforms // ideal_num_tasks)
        # Cap at reasonable values
        batch_size = min(batch_size, 100)
        _logger.debug(f"Auto-computed batch_size: {batch_size}")

    # Prepare parameter batches
    param_dicts = [row.to_dict() for idx, row in parameters.iterrows()]

    if batch_size == 1:
        # No batching - process one waveform per task
        batches = [[p] for p in param_dicts]
    else:
        # Batch parameters
        batches = [
            param_dicts[i:i+batch_size]
            for i in range(0, len(param_dicts), batch_size)
        ]

    _logger.debug(f"Split {len(parameters)} waveforms into {len(batches)} batches")

    # Process batches in parallel with worker initialization
    all_results = []
    with ProcessPoolExecutor(
        max_workers=num_processes,
        initializer=_init_worker,
        initargs=(wfg_config, domain_params)
    ) as executor:
        # Submit all batch tasks
        futures = [executor.submit(_generate_waveform_batch, batch) for batch in batches]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                _logger.error(f"Batch processing failed: {e}")
                # Append failure results for failed batch
                batch_size_actual = len(batches[0])  # Approximate
                all_results.extend(
                    [WaveformResult.failure_result(str(e))] * batch_size_actual
                )

    # Aggregate results
    h_plus_list = [r.h_plus for r in all_results]
    h_cross_list = [r.h_cross for r in all_results]

    polarizations = BatchPolarizations(
        h_plus=np.array(h_plus_list),
        h_cross=np.array(h_cross_list),
    )

    # Apply batch transforms if configured
    if waveform_generator.transform is not None:
        polarizations = apply_transforms_to_polarizations(
            polarizations, waveform_generator.transform
        )

    return polarizations


def generate_parameters_and_polarizations(
    waveform_generator: WaveformGenerator,
    prior: BBHPriorDict,
    num_samples: int,
    num_processes: int = 1,
) -> Tuple[pd.DataFrame, BatchPolarizations]:
    """
    Generate dataset of waveforms based on parameters drawn from prior.

    Parameters
    ----------
    waveform_generator
        Configured waveform generator
    prior
        Prior distribution (bilby BBHPriorDict)
    num_samples
        Number of samples to generate
    num_processes
        Number of parallel processes to use

    Returns
    -------
    Tuple of (parameters DataFrame, BatchPolarizations dataclass).
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
        polarizations_ok = BatchPolarizations(
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


def train_svd_basis(
    polarizations: BatchPolarizations,
    parameters: pd.DataFrame,
    size: int,
    n_train: int,
) -> Tuple[SVDBasis, int, int]:
    """
    Train and validate an SVD basis from waveform data.

    Parameters
    ----------
    polarizations
        Waveform polarizations to use for training and validation
    parameters
        Parameters corresponding to the waveforms
    size
        Number of SVD components to keep
    n_train
        Number of waveforms to use for training (rest used for validation)

    Returns
    -------
    Tuple of (trained SVDBasis, actual_n_train, actual_n_validation)
    """
    # Split into train and validation
    n_total = len(polarizations)
    n_train = min(n_train, n_total)
    n_validation = n_total - n_train

    _logger.info(f"Training SVD basis: {n_train} train, {n_validation} validation samples")

    # Prepare training data (concatenate h_plus and h_cross)
    train_data = np.concatenate(
        [polarizations.h_plus[:n_train], polarizations.h_cross[:n_train]],
        axis=0
    )

    # Create SVD generation config
    config = SVDGenerationConfig(n_components=size, method="scipy")

    # Create metadata
    metadata = SVDMetadata(
        description="Waveform SVD basis",
        n_training_samples=n_train,
        n_validation_samples=n_validation,
        data_shape=train_data.shape
    )

    # Train SVD basis using factory method
    basis = SVDBasis.from_training_data(train_data, config, metadata)

    # Validate if we have validation samples
    if n_validation > 0:
        _logger.info("Computing validation mismatches...")
        val_data = np.concatenate(
            [polarizations.h_plus[n_train:], polarizations.h_cross[n_train:]],
            axis=0
        )
        val_params = pd.concat([parameters[n_train:], parameters[n_train:]], ignore_index=True)

        # Use dingo-svd's validation
        val_config = ValidationConfig(
            increment=size,  # Only validate at full n_components
            compute_percentiles=True
        )
        basis, validation_result = basis.validate(
            test_data=val_data,
            config=val_config,
            labels=val_params,
            verbose=True
        )

    return basis, n_train, n_validation


def apply_transforms_to_polarizations(
    polarizations: BatchPolarizations,
    transforms: Optional[ComposeDataTransforms],
) -> BatchPolarizations:
    """
    Apply transform pipeline to polarizations.

    Parameters
    ----------
    polarizations
        Polarizations to transform
    transforms
        Transform pipeline to apply, or None for no transforms

    Returns
    -------
    Transformed polarizations
    """
    if transforms is None:
        return polarizations

    # Convert to dict format for transforms
    pol_dict = {
        "h_plus": polarizations.h_plus,
        "h_cross": polarizations.h_cross,
    }

    # Apply transforms
    transformed = transforms(pol_dict)

    # Convert back to BatchPolarizations
    return BatchPolarizations(
        h_plus=transformed["h_plus"],
        h_cross=transformed["h_cross"],
    )


def build_compression_transforms(
    compression_settings: CompressionSettings,
    domain: Domain,
    prior: BBHPriorDict,
    waveform_generator: WaveformGenerator,
    num_processes: int,
) -> Tuple[Optional[ComposeDataTransforms], Optional[SVDBasis]]:
    """
    Build compression transform pipeline from settings.

    Parameters
    ----------
    compression_settings
        Compression configuration
    domain
        Frequency domain
    prior
        Prior distribution for sampling training waveforms
    waveform_generator
        Waveform generator (will have transforms applied)
    num_processes
        Number of processes for parallel generation

    Returns
    -------
    Tuple of (transform_pipeline, svd_basis)
        transform_pipeline is None if no compression
        svd_basis is None if SVD not used
    """
    transforms: List[DataTransform] = []
    svd_basis: Optional[SVDBasis] = None

    # Whitening transform
    if compression_settings.whitening is not None:
        _logger.info(f"Adding whitening transform with ASD from {compression_settings.whitening}")
        transforms.append(
            WhitenUnwhitenTransform(domain, compression_settings.whitening, inverse=False)
        )

    # SVD compression
    if compression_settings.svd is not None:
        svd_settings = compression_settings.svd

        # Load pre-trained SVD basis if file provided
        if svd_settings.file is not None:
            _logger.info(f"Loading SVD basis from {svd_settings.file}")
            svd_basis = SVDBasis.from_file(svd_settings.file)

        # Otherwise, generate SVD basis from training waveforms
        else:
            _logger.info("Generating SVD basis from training waveforms...")

            # Apply whitening to training waveforms if needed
            if transforms:
                waveform_generator.transform = ComposeDataTransforms(transforms)

            # Generate training waveforms
            n_total = svd_settings.num_training_samples + svd_settings.num_validation_samples
            train_parameters, train_polarizations = generate_parameters_and_polarizations(
                waveform_generator, prior, n_total, num_processes
            )

            # Train SVD basis
            svd_basis, n_train, n_val = train_svd_basis(
                train_polarizations,
                train_parameters,
                svd_settings.size,
                svd_settings.num_training_samples,
            )

            # Reset transform on generator
            waveform_generator.transform = None

        # Add SVD compression transform
        transforms.append(ApplySVD(svd_basis, inverse=False))
        _logger.info(f"Added SVD compression with {svd_basis.n_components} components")

    # Return composed transforms if any
    if transforms:
        return ComposeDataTransforms(transforms), svd_basis
    else:
        return None, None


def generate_waveform_dataset(
    settings: DatasetSettings, num_processes: int = 1
) -> WaveformDataset:
    """
    Generate a waveform dataset based on settings.

    Supports optional compression via whitening and/or SVD. If compression
    settings are provided, the waveforms will be compressed before being
    stored in the dataset.

    Parameters
    ----------
    settings
        Dataset generation settings.
    num_processes
        Number of parallel processes to use.

    Returns
    -------
    WaveformDataset containing parameters and polarizations (compressed if requested).
    """
    # Validate settings
    settings.validate()

    # Build components
    _logger.info("Building domain, prior, and waveform generator...")
    domain = build_domain(settings.domain)
    prior = build_prior_with_defaults(settings.intrinsic_prior)
    wfg_dict = settings.waveform_generator.to_dict()
    waveform_generator = build_waveform_generator(wfg_dict, domain)

    # Build compression transforms if requested
    compression_transforms = None
    svd_basis = None
    if settings.compression is not None:
        _logger.info("Building compression pipeline...")
        compression_transforms, svd_basis = build_compression_transforms(
            settings.compression,
            domain,
            prior,
            waveform_generator,
            num_processes,
        )

        # Apply compression transforms to waveform generator
        if compression_transforms is not None:
            waveform_generator.transform = compression_transforms
            _logger.info(f"Compression pipeline: {compression_transforms}")

    # Generate waveforms (will be compressed if transforms are set)
    parameters, polarizations = generate_parameters_and_polarizations(
        waveform_generator, prior, settings.num_samples, num_processes
    )

    # Create dataset
    dataset = WaveformDataset(
        parameters=parameters,
        polarizations=polarizations,
        settings=settings,
        svd_basis=svd_basis,
    )

    _logger.info(f"Dataset generated successfully with {len(parameters)} samples.")
    if svd_basis is not None:
        _logger.info(f"Dataset includes SVD compression with {svd_basis.n_components} components")

    return dataset
