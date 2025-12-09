"""Factory functions for building transform chains.

This module provides factory functions that replace the Transform class
orchestrator pattern. Each factory function builds a complete transform chain
as a TransformCompose object with all dependencies passed as explicit arguments.

Key design principles:
- NO imports from dingo (dingo-gw) or parent dingo_waveform package
- All external objects passed as arguments (duck-typed interfaces)
- Factory functions only import from within transform package
- Users responsible for loading/building ASDDataset, priors, domains, etc.
"""

from typing import Any, Callable, Dict, List, Optional, Union

# Import type infrastructure from types.py for precise type hints
from .types import (
    # Protocols for external objects (duck-typed interfaces)
    ASDDatasetLike,
    DomainProtocol,
    ExtrinsicPriorLike,
    InterferometerListLike,
    # TypedDicts for configuration dictionaries
    StandardizationDict,
    GNPETimeShiftsDict,
    RandomStrainCroppingDict,
    DomainUpdateDict,
    # Type aliases for semantic clarity
    DetectorList,
    ParameterNames,
)
from .compose import TransformCompose


def build_training_transform(
    ifo_list: InterferometerListLike,
    domain: DomainProtocol,
    ref_time: float,
    asd_dataset: ASDDatasetLike,
    extrinsic_prior: ExtrinsicPriorLike,
    inference_parameters: ParameterNames,
    context_parameters: Optional[ParameterNames] = None,
    standardization: Optional[StandardizationDict] = None,
    random_strain_cropping: Optional[RandomStrainCroppingDict] = None,
    gnpe_time_shifts: Optional[GNPETimeShiftsDict] = None,
    zero_noise: bool = False,
) -> TransformCompose:
    """
    Build training transform chain (11 transforms).

    Transform chain:
    1. SampleExtrinsicParameters - Sample sky location, orientation, etc.
    2. GetDetectorTimes - Calculate arrival times at each detector
    3. (optional) GNPECoalescenceTimes - Apply GNPE time shifts
    4. ProjectOntoDetectors - Project waveform onto detector responses
    5. SampleNoiseASD - Sample noise amplitude spectral densities
    6. WhitenAndScaleStrain - Whiten strains using ASDs
    7. (optional) AddWhiteNoiseComplex - Add white noise to strains
    8. SelectStandardizeRepackageParameters - Select and standardize parameters
    9. RepackageStrainsAndASDS - Repackage strains/ASDs for network format
    10. (optional) CropMaskStrainRandom - Randomly crop and mask strains
    11. UnpackDict - Convert dict to list/tuple for DataLoader

    Parameters
    ----------
    ifo_list : Any
        Interferometer list object (must be iterable, each element has .name attribute).
        Typically bilby.gw.detector.InterferometerList.
    domain : Any
        Domain object (must have `time_translate_data` and `noise_std` attributes).
        Typically UniformFrequencyDomain or MultibandedFrequencyDomain.
    ref_time : float
        Reference GPS time for waveform generation (seconds).
    asd_dataset : Any
        ASD dataset object (must have `sample_random_asds(n)` method returning dict).
        Typically dingo.gw.noise.asd_dataset.ASDDataset.
    extrinsic_prior : Any
        Extrinsic prior object (must have `sample(n)` method returning dict).
        Typically from dingo.gw.prior or bilby.core.prior.
    inference_parameters : List[str]
        List of parameter names to use for inference (e.g., ['mass_1', 'mass_2', ...]).
    context_parameters : Optional[List[str]], optional
        List of parameter names to use as context (conditioning), by default None.
    standardization : Optional[Dict[str, Dict[str, float]]], optional
        Standardization statistics with 'mean' and 'std' dicts, by default None.
        Example: {'mean': {'mass_1': 35.0, ...}, 'std': {'mass_1': 10.0, ...}}
    random_strain_cropping : Optional[Dict[str, Any]], optional
        Random cropping configuration dict, by default None.
        Example: {'min_length': 1000, 'max_length': 2000, ...}
    gnpe_time_shifts : Optional[Dict[str, Any]], optional
        GNPE configuration dict with 'kernel' and 'exact_equiv' keys, by default None.
    zero_noise : bool, optional
        If True, skip AddWhiteNoiseComplex transform (noiseless training), by default False.

    Returns
    -------
    TransformCompose
        Composed transform chain for training.

    Examples
    --------
    >>> from dingo.gw.noise.asd_dataset import ASDDataset
    >>> from dingo.gw.prior import BBHExtrinsicPriorDict
    >>> from bilby.gw.detector import InterferometerList
    >>>
    >>> ifo_list = InterferometerList(['H1', 'L1'])
    >>> asd_dataset = ASDDataset('asd.hdf5')
    >>> prior = BBHExtrinsicPriorDict({'ra': ..., 'dec': ...})
    >>>
    >>> transform = build_training_transform(
    ...     ifo_list=ifo_list,
    ...     domain=domain,
    ...     ref_time=1234567890.0,
    ...     asd_dataset=asd_dataset,
    ...     extrinsic_prior=prior,
    ...     inference_parameters=['mass_1', 'mass_2', ...],
    ...     standardization={'mean': {...}, 'std': {...}},
    ... )
    """
    # Import transform classes from current package
    from . import (
        AddWhiteNoiseComplex,
        CropMaskStrainRandom,
        GetDetectorTimes,
        GNPECoalescenceTimes,
        ProjectOntoDetectors,
        RepackageStrainsAndASDS,
        SampleExtrinsicParameters,
        SampleNoiseASD,
        SelectStandardizeRepackageParameters,
        UnpackDict,
        WhitenAndScaleStrain,
    )

    # Default context_parameters to empty list
    if context_parameters is None:
        context_parameters = []

    # Validate required attributes via duck typing
    if not hasattr(domain, "time_translate_data"):
        raise TypeError(
            "domain must have time_translate_data attribute "
            "(expected Domain-like object)"
        )
    if not hasattr(domain, "noise_std"):
        raise TypeError("domain must have noise_std attribute (expected Domain-like object)")
    if not hasattr(asd_dataset, "sample_random_asds"):
        raise TypeError(
            "asd_dataset must have sample_random_asds() method "
            "(expected ASDDataset-like object)"
        )
    if not hasattr(extrinsic_prior, "sample"):
        raise TypeError(
            "extrinsic_prior must have sample() method (expected Prior-like object)"
        )

    # Extract detector names
    detectors = [ifo.name for ifo in ifo_list]

    # Build transform chain
    transforms: List[Callable] = [
        SampleExtrinsicParameters(extrinsic_prior),
        GetDetectorTimes(ifo_list, ref_time),
    ]

    # Add GNPE if configured
    if gnpe_time_shifts is not None:
        transforms.append(
            GNPECoalescenceTimes(
                ifo_list,
                gnpe_time_shifts["kernel"],
                gnpe_time_shifts["exact_equiv"],
                inference=False,
            )
        )

    # Continue building transform chain
    transforms.extend(
        [
            ProjectOntoDetectors(ifo_list, domain, ref_time),
            SampleNoiseASD(asd_dataset),
            WhitenAndScaleStrain(domain.noise_std),
        ]
    )

    # Add noise unless zero_noise is True
    if not zero_noise:
        transforms.append(AddWhiteNoiseComplex())

    # Add parameter standardization and repackaging
    transforms.extend(
        [
            SelectStandardizeRepackageParameters(
                {
                    "inference_parameters": inference_parameters,
                    "context_parameters": context_parameters,
                },
                standardization or {"mean": {}, "std": {}},
            ),
            RepackageStrainsAndASDS(
                detectors,
                first_index=domain.min_idx,
            ),
        ]
    )

    # Add random strain cropping if configured
    if random_strain_cropping is not None:
        transforms.append(
            CropMaskStrainRandom(
                domain,
                **random_strain_cropping,
            )
        )

    # Determine selected keys for UnpackDict
    if context_parameters:
        selected_keys = ["inference_parameters", "waveform", "context_parameters"]
    else:
        selected_keys = ["inference_parameters", "waveform"]

    transforms.append(UnpackDict(selected_keys=selected_keys))

    return TransformCompose(transforms)


def build_svd_transform(
    ifo_list: InterferometerListLike,
    domain: DomainProtocol,
    ref_time: float,
    asd_dataset: ASDDatasetLike,
    extrinsic_prior: ExtrinsicPriorLike,
    gnpe_time_shifts: Optional[GNPETimeShiftsDict] = None,
) -> TransformCompose:
    """
    Build SVD transform chain (6 transforms, no noise/repackaging).

    Transform chain:
    1. SampleExtrinsicParameters - Sample sky location, orientation, etc.
    2. GetDetectorTimes - Calculate arrival times at each detector
    3. (optional) GNPECoalescenceTimes - Apply GNPE time shifts
    4. ProjectOntoDetectors - Project waveform onto detector responses
    5. SampleNoiseASD - Sample noise amplitude spectral densities
    6. WhitenAndScaleStrain - Whiten strains using ASDs

    This chain is used for generating SVD basis (no noise addition, parameter
    repackaging, or final unpacking).

    Parameters
    ----------
    ifo_list : Any
        Interferometer list object.
    domain : Any
        Domain object with time_translate_data and noise_std attributes.
    ref_time : float
        Reference GPS time.
    asd_dataset : Any
        ASD dataset object with sample_random_asds(n) method.
    extrinsic_prior : Any
        Extrinsic prior object with sample(n) method.
    gnpe_time_shifts : Optional[Dict[str, Any]], optional
        GNPE configuration dict, by default None.

    Returns
    -------
    TransformCompose
        Composed transform chain for SVD generation.
    """
    # Import transform classes from current package
    from . import (
        GetDetectorTimes,
        GNPECoalescenceTimes,
        ProjectOntoDetectors,
        SampleExtrinsicParameters,
        SampleNoiseASD,
        WhitenAndScaleStrain,
    )

    # Validate required attributes via duck typing
    if not hasattr(domain, "time_translate_data"):
        raise TypeError(
            "domain must have time_translate_data attribute "
            "(expected Domain-like object)"
        )
    if not hasattr(domain, "noise_std"):
        raise TypeError("domain must have noise_std attribute (expected Domain-like object)")
    if not hasattr(asd_dataset, "sample_random_asds"):
        raise TypeError(
            "asd_dataset must have sample_random_asds() method "
            "(expected ASDDataset-like object)"
        )
    if not hasattr(extrinsic_prior, "sample"):
        raise TypeError(
            "extrinsic_prior must have sample() method (expected Prior-like object)"
        )

    # Build transform chain (similar to training but without noise/repackaging)
    transforms: List[Callable] = [
        SampleExtrinsicParameters(extrinsic_prior),
        GetDetectorTimes(ifo_list, ref_time),
    ]

    # Add GNPE if configured
    if gnpe_time_shifts is not None:
        transforms.append(
            GNPECoalescenceTimes(
                ifo_list,
                gnpe_time_shifts["kernel"],
                gnpe_time_shifts["exact_equiv"],
                inference=False,
            )
        )

    # Continue building transform chain (stop before noise/parameter repackaging)
    transforms.extend(
        [
            ProjectOntoDetectors(ifo_list, domain, ref_time),
            SampleNoiseASD(asd_dataset),
            WhitenAndScaleStrain(domain.noise_std),
        ]
    )

    return TransformCompose(transforms)


def build_inference_transform_pre(
    domain: DomainProtocol,
    detectors: DetectorList,
    standardization: Optional[StandardizationDict] = None,
    domain_update: Optional[DomainUpdateDict] = None,
) -> TransformCompose:
    """
    Build inference pre-transform chain (5 transforms).

    Transform chain:
    1. (optional) DecimateWaveformsAndASDS - Decimate if MultibandedFrequencyDomain
    2. WhitenAndScaleStrain - Whiten strains using provided ASDs
    3. (optional) MaskDataForFrequencyRangeUpdate - Mask frequency ranges
    4. RepackageStrainsAndASDS - Repackage for network format
    5. ToTorch - Convert numpy arrays to torch tensors

    This chain prepares real detector data for inference (whitening, masking, etc.).

    Parameters
    ----------
    domain : Any
        Domain object. If has `decimate()` method, DecimateWaveformsAndASDS is added.
    detectors : List[str]
        List of detector names (e.g., ['H1', 'L1']).
    standardization : Optional[Dict[str, Dict[str, float]]], optional
        Unused in pre-transform but kept for API consistency, by default None.
    domain_update : Optional[Dict[str, float]], optional
        Dict with optional 'f_min' and 'f_max' keys for frequency masking, by default None.
        Example: {'f_min': 20.0, 'f_max': 1024.0}

    Returns
    -------
    TransformCompose
        Composed transform chain for inference pre-processing.
    """
    from . import (
        DecimateWaveformsAndASDS,
        MaskDataForFrequencyRangeUpdate,
        RepackageStrainsAndASDS,
        ToTorch,
        WhitenAndScaleStrain,
    )

    # Validate required attributes via duck typing
    if not hasattr(domain, "noise_std"):
        raise TypeError("domain must have noise_std attribute (expected Domain-like object)")
    if not hasattr(domain, "min_idx"):
        raise TypeError("domain must have min_idx attribute (expected Domain-like object)")

    transforms: List[Callable] = []

    # Add decimation if using MultibandedFrequencyDomain (duck typing)
    if hasattr(domain, "decimate"):
        transforms.append(
            DecimateWaveformsAndASDS(
                domain,
                decimation_mode="whitened",
            )
        )

    # Always whiten and scale
    transforms.append(WhitenAndScaleStrain(domain.noise_std))

    # Add frequency range masking if domain_update is specified
    if domain_update is not None:
        minimum_frequency = domain_update.get("f_min")
        maximum_frequency = domain_update.get("f_max")
        if minimum_frequency is not None or maximum_frequency is not None:
            transforms.append(
                MaskDataForFrequencyRangeUpdate(
                    domain,
                    minimum_frequency=minimum_frequency,
                    maximum_frequency=maximum_frequency,
                )
            )

    # Repackage strains and ASDs
    transforms.append(
        RepackageStrainsAndASDS(
            detectors,
            first_index=domain.min_idx,
        )
    )

    # Convert to torch tensors
    transforms.append(ToTorch(device="cpu"))

    return TransformCompose(transforms)


def build_inference_transform_post(
    inference_parameters: ParameterNames,
    standardization: StandardizationDict,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Build inference post-transform (1 transform).

    Transform chain:
    1. SelectStandardizeRepackageParameters(inverse=True) - De-standardize parameters

    This single transform converts standardized network outputs back to physical parameters.

    Parameters
    ----------
    inference_parameters : List[str]
        List of parameter names for inference.
    standardization : Dict[str, Dict[str, float]]
        Standardization statistics with 'mean' and 'std' dicts.

    Returns
    -------
    Callable
        Single inverse transform for de-standardization.
    """
    from . import SelectStandardizeRepackageParameters

    # Create inverse transform for de-standardization
    return SelectStandardizeRepackageParameters(
        {"inference_parameters": inference_parameters},
        {
            "mean": standardization["mean"],
            "std": standardization["std"],
        },
        inverse=True,
        as_type="dict",
    )
