"""
Dingo Transform Package

Provides unified Transform class for managing gravitational wave data transforms.
"""

from .transform import Transform, TransformConfig
from .base import TransformProtocol

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

# Export all transform classes for direct import
from .detector import (
    GetDetectorTimes,
    ProjectOntoDetectors,
    TimeShiftStrain,
    ApplyCalibrationUncertainty,
)
from .noise import (
    SampleNoiseASD,
    WhitenStrain,
    WhitenFixedASD,
    WhitenAndScaleStrain,
    AddWhiteNoiseComplex,
    RepackageStrainsAndASDS,
)
from .parameters import (
    SampleExtrinsicParameters,
    SelectStandardizeRepackageParameters,
    StandardizeParameters,
)
from .waveform import (
    DecimateAll,
    DecimateWaveformsAndASDS,
    CropMaskStrainRandom,
    MaskDataForFrequencyRangeUpdate,
)
from .general import UnpackDict
from .gnpe import GNPEBase, GNPECoalescenceTimes
from .inference import (
    PostCorrectGeocentTime,
    CopyToExtrinsicParameters,
    ExpandStrain,
    ToTorch,
    ResetSample,
)
from .utils import get_batch_size_of_input_sample

__all__ = [
    # Main API
    "Transform",
    "TransformConfig",
    "TransformProtocol",
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
