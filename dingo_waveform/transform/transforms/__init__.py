"""
Individual transform modules organized by category.

This package contains all waveform transforms for the gravitational wave
inference pipeline. Each transform has:

1. A frozen dataclass config (subclass of WaveformTransformConfig)
2. A transform class (subclass of WaveformTransform[ConfigT])
3. Type-safe field definitions using typing.Literal, TypedDict, and Protocols

Type Safety
-----------
All transforms use strict type hints to catch errors early:

- **String constants** use Literal types (e.g., Literal["whitened", "unwhitened"])
- **Structured dicts** use TypedDict for key enforcement (e.g., StandardizationDict)
- **Domain objects** use DomainProtocol for structural typing (avoids circular imports)
- **Generic parameters** maintain ConfigT type safety via Generic[ConfigT]

The type system ensures:
- Mypy can catch typos and invalid values at type-check time
- IDEs provide better autocomplete and inline documentation
- Self-documenting code with explicit type contracts

Common Type Imports
-------------------
Import shared types from dingo_waveform.transform.types:

>>> from dingo_waveform.transform.types import (
...     DecimationMode,           # Literal["whitened", "unwhitened"]
...     OutputFormat,             # Literal["dict", "pandas"]
...     Device,                   # "cpu" | "cuda" | "cuda:N"
...     GroupOperator,            # Literal["+", "x"]
...     StandardizationDict,      # TypedDict with 'mean', 'std'
...     DomainProtocol,           # Structural type for Domain objects
...     ParameterValue,           # float | ndarray | Tensor
...     ExtrinsicPriorDict,       # Dict[str, ExtrinsicPriorSpecDict]
...     NestedDict,               # Dict[str, Any] alias
... )

Transform Categories
--------------------
**general/**
    Generic utilities
    - UnpackDict: Extract selected keys as list (for DataLoader)

**waveform/**
    Waveform processing
    - DecimateWaveformsAndASDS: Decimate to multibanded domain (whitened/unwhitened)
    - DecimateAll: Recursive decimation of nested arrays
    - CropMaskStrainRandom: Random frequency cropping
    - MaskDataForFrequencyRangeUpdate: Mask waveform/ASD outside range

**detector/**
    Detector operations
    - ProjectOntoDetectors: Project polarizations onto detector network
    - GetDetectorTimes: Compute detector-specific times
    - TimeShiftStrain: Apply time shifts to strains
    - ApplyCalibrationUncertainty: Apply calibration uncertainty

**noise/**
    Noise and whitening
    - WhitenStrain: Whiten strain with ASD
    - WhitenAndScaleStrain: Whiten and scale strain data
    - WhitenFixedASD: Whiten with fixed ASD from file
    - SampleNoiseASD: Sample noise ASD from dataset
    - AddWhiteNoiseComplex: Add white noise to complex data
    - RepackageStrainsAndASDS: Repackage for inference pipeline

**parameters/**
    Parameter handling
    - SampleExtrinsicParameters: Sample extrinsic parameters from prior
    - StandardizeParameters: Z-score normalization (x - mu) / std
    - SelectStandardizeRepackageParameters: Select, normalize, repackage

**inference/**
    Inference pipeline
    - ToTorch: Convert numpy arrays to PyTorch tensors
    - ExpandStrain: Expand strain for batch processing
    - PostCorrectGeocentTime: Post-correct geocentric time
    - CopyToExtrinsicParameters: Copy parameters to extrinsic dict
    - ResetSample: Reset sample state

**gnpe/**
    Group-Equivariant Neural Posterior Estimation
    - GNPEBase: Abstract base for GNPE transforms (group algebra)
    - GNPECoalescenceTimes: GNPE for coalescence time perturbations

Examples
--------
Using transforms with type safety:

>>> from dingo_waveform.transform.transforms.waveform.decimate_waveforms_and_asds import (
...     DecimateWaveformsAndASDS,
...     DecimateWaveformsAndASDSConfig,
... )
>>> from dingo_waveform.transform.types import DecimationMode
>>>
>>> # Mypy enforces valid decimation modes
>>> config = DecimateWaveformsAndASDSConfig(
...     multibanded_frequency_domain=mfd,
...     decimation_mode="whitened"  # Type-safe: must be "whitened" or "unwhitened"
... )
>>> transform = DecimateWaveformsAndASDS.from_config(config)

Backward Compatibility
----------------------
All type improvements are annotation-only; no runtime behavior changes.
Existing code continues to work without modification.

Type Checking
-------------
Run mypy to catch type errors:

```bash
mypy dingo_waveform/transform/transforms/ --check-untyped-defs
```

Notes
-----
- All transforms implement WaveformTransform[ConfigT] generic interface
- All configs are frozen dataclasses (immutable)
- Most transforms return Dict[str, Any], except UnpackDict returns List[Any]
- Transforms can be composed using torchvision.transforms.Compose
"""

# Transforms will be imported here as they are implemented
