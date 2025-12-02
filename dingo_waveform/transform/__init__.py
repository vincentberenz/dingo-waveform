"""
Dingo Transform Package

Provides unified Transform class for managing gravitational wave data transforms.
"""

from .transform import Transform, TransformConfig
from .base import TransformProtocol

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
