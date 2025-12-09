"""
Dingo Transform Package

Provides factory functions for building gravitational wave transform chains.

Key components:
- TransformCompose: Compose multiple transforms into a pipeline
- Factory functions: build_training_transform, build_svd_transform, etc.
- Individual transform classes: SampleNoiseASD, ProjectOntoDetectors, etc.
"""

from .compose import TransformCompose
from .factory import (
    build_training_transform,
    build_svd_transform,
    build_inference_transform_pre,
    build_inference_transform_post,
)
from .base import TransformProtocol

# Type infrastructure for user type hints
from .types import (
    # Protocols for external objects
    ASDDatasetLike,
    DomainProtocol,
    ExtrinsicPriorLike,
    InterferometerLike,
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

# Configuration types
from .config_types import (
    StandardizationConfig,
    RandomStrainCroppingConfig,
    GNPETimeShiftsConfig,
    DomainUpdateConfig,
    ExtrinsicPriorConfig,
)

# Generic data transform framework
from .data_transform import DataTransform, ComposeDataTransforms
from .svd_transform import ApplySVD
from .whitening_transform import WhitenUnwhitenTransform

# Export all transform classes for direct import (from new transforms/ subdirectories)
from .transforms.detector import (
    GetDetectorTimes,
    ProjectOntoDetectors,
    TimeShiftStrain,
    ApplyCalibrationUncertainty,
)
from .transforms.noise import (
    SampleNoiseASD,
    WhitenStrain,
    WhitenFixedASD,
    WhitenAndScaleStrain,
    AddWhiteNoiseComplex,
    RepackageStrainsAndASDS,
)
from .transforms.parameters import (
    SampleExtrinsicParameters,
    SelectStandardizeRepackageParameters,
    StandardizeParameters,
)
from .transforms.waveform import (
    DecimateAll,
    DecimateWaveformsAndASDS,
    CropMaskStrainRandom,
    MaskDataForFrequencyRangeUpdate,
)
from .transforms.general import UnpackDict
from .transforms.gnpe import GNPEBase, GNPECoalescenceTimes
from .transforms.inference import (
    PostCorrectGeocentTime,
    CopyToExtrinsicParameters,
    ExpandStrain,
    ToTorch,
    ResetSample,
)
from .utils import get_batch_size_of_input_sample

__all__ = [
    # Main API - Transform composition and factory functions
    "TransformCompose",
    "build_training_transform",
    "build_svd_transform",
    "build_inference_transform_pre",
    "build_inference_transform_post",
    "TransformProtocol",
    # Type infrastructure (Protocols, TypedDicts, TypeAliases)
    "ASDDatasetLike",
    "DomainProtocol",
    "ExtrinsicPriorLike",
    "InterferometerLike",
    "InterferometerListLike",
    "StandardizationDict",
    "GNPETimeShiftsDict",
    "RandomStrainCroppingDict",
    "DomainUpdateDict",
    "DetectorList",
    "ParameterNames",
    # Configuration types
    "StandardizationConfig",
    "RandomStrainCroppingConfig",
    "GNPETimeShiftsConfig",
    "DomainUpdateConfig",
    "ExtrinsicPriorConfig",
    # Generic data transform framework
    "DataTransform",
    "ComposeDataTransforms",
    "ApplySVD",
    "WhitenUnwhitenTransform",
    # Detector transforms
    "GetDetectorTimes",
    "ProjectOntoDetectors",
    "TimeShiftStrain",
    "ApplyCalibrationUncertainty",
    # Noise transforms
    "SampleNoiseASD",
    "WhitenStrain",
    "WhitenFixedASD",
    "WhitenAndScaleStrain",
    "AddWhiteNoiseComplex",
    "RepackageStrainsAndASDS",
    # Parameter transforms
    "SampleExtrinsicParameters",
    "SelectStandardizeRepackageParameters",
    "StandardizeParameters",
    # Waveform transforms
    "DecimateAll",
    "DecimateWaveformsAndASDS",
    "CropMaskStrainRandom",
    "MaskDataForFrequencyRangeUpdate",
    # General transforms
    "UnpackDict",
    # GNPE transforms
    "GNPEBase",
    "GNPECoalescenceTimes",
    # Inference transforms
    "PostCorrectGeocentTime",
    "CopyToExtrinsicParameters",
    "ExpandStrain",
    "ToTorch",
    "ResetSample",
    # Utilities
    "get_batch_size_of_input_sample",
]
