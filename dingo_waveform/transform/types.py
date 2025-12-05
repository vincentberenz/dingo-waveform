"""
Type definitions for the transform subpackage.

This module provides type hints, TypedDicts, Literals, and Protocols
for improved type safety across all transforms.

The types defined here avoid circular imports and enable better IDE support,
earlier error detection via mypy, and self-documenting code.
"""

from typing import (
    TypedDict,
    Literal,
    Union,
    Dict,
    Any,
    Protocol,
    runtime_checkable,
    TypeAlias,
    List,
)
import numpy as np
import torch
from numpy.typing import NDArray

# ============================================================================
# Literal Types (String Constants)
# ============================================================================

DecimationMode: TypeAlias = Literal["whitened", "unwhitened"]
"""
Decimation strategy for multibanded frequency domain.

- "whitened": Whiten data first, then decimate (better signal preservation)
- "unwhitened": Decimate data first, then whiten
"""

OutputFormat: TypeAlias = Literal["dict", "pandas"]
"""
Output format for inverse parameter transforms.

- "dict": Return parameters as dictionary
- "pandas": Return parameters as pandas DataFrame
"""

GroupOperator: TypeAlias = Literal["+", "x"]
"""
Group multiplication operator for GNPE (Group-Equivariant Neural Posterior Estimation).

- "+": Additive group (a + b)
- "x": Multiplicative group (a * b)
"""

Device: TypeAlias = Union[Literal["cpu"], str]
"""
Device specification for PyTorch tensors.

Can be:
- "cpu": CPU device
- "cuda": Default CUDA device
- "cuda:N": Specific CUDA device (e.g., "cuda:0", "cuda:1")

Note: We use Union[Literal["cpu"], str] to allow the "cuda:N" pattern
while still providing type safety for the common "cpu" case.
"""


# ============================================================================
# TypedDict Definitions (Structured Dicts)
# ============================================================================


class StandardizationDict(TypedDict):
    """
    Dictionary structure for parameter standardization (z-score normalization).

    Used for normalizing parameters with: (param - mean) / std

    Attributes
    ----------
    mean : Dict[str, float]
        Mean values for each parameter
    std : Dict[str, float]
        Standard deviation values for each parameter

    Notes
    -----
    Keys in 'mean' and 'std' must match.
    """

    mean: Dict[str, float]
    std: Dict[str, float]


class PolarizationDict(TypedDict, total=False):
    """
    Dictionary structure for gravitational wave polarizations.

    This TypedDict uses total=False, meaning all keys are optional.
    This allows it to represent either:
    - Unprojected polarizations: {'h_plus': ..., 'h_cross': ...}
    - Detector-projected strains: {'H1': ..., 'L1': ..., 'V1': ...}

    Attributes
    ----------
    h_plus : NDArray[np.complex128]
        Plus polarization
    h_cross : NDArray[np.complex128]
        Cross polarization

    Notes
    -----
    Detector keys (H1, L1, V1, etc.) are dynamic and cannot be enumerated
    in TypedDict. This TypedDict provides partial type safety for the
    polarization case.
    """

    h_plus: NDArray[np.complex128]
    h_cross: NDArray[np.complex128]


class ExtrinsicPriorSpecDict(TypedDict, total=False):
    """
    Specification for a single extrinsic parameter prior (bilby format).

    Used within extrinsic_prior_dict to specify each parameter's distribution.

    Attributes
    ----------
    type : str
        Prior distribution type (e.g., 'Uniform', 'Cosine', 'UniformSourceFrame')
    minimum : float
        Lower bound for bounded distributions
    maximum : float
        Upper bound for bounded distributions

    Notes
    -----
    Additional bilby-specific fields may be present depending on the prior type.
    This TypedDict captures the most common fields.
    """

    type: str
    minimum: float
    maximum: float


# Full extrinsic_prior_dict is Dict[str, ExtrinsicPriorSpecDict]
ExtrinsicPriorDict: TypeAlias = Dict[str, ExtrinsicPriorSpecDict]
"""
Dictionary mapping parameter names to their prior specifications.

Example
-------
>>> extrinsic_prior_dict = {
...     'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28},
...     'dec': {'type': 'Cosine', 'minimum': -1.57, 'maximum': 1.57},
...     'psi': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 3.14}
... }
"""


class WaveformSampleDict(TypedDict, total=False):
    """
    Structure of a sample dict flowing through transform pipeline.

    Keys are progressively added as transforms are applied in the pipeline.

    Attributes
    ----------
    parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Intrinsic waveform parameters (mass, spin, etc.)
    extrinsic_parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Extrinsic parameters (sky location, distance, time)
        Added by SampleExtrinsicParameters
    waveform : Union[PolarizationDict, Dict[str, NDArray[np.complex128]]]
        Waveform data - either polarizations or detector strains
        Changes from polarizations to detectors after ProjectOntoDetectors
    asds : Dict[str, NDArray[np.float64]]
        Amplitude spectral density for each detector
        Added by WhitenAndScale
    inference_parameters : NDArray[np.float32]
        Standardized and repackaged parameters for inference
        Added by SelectStandardizeRepackageParameters
    log_prob : float
        Log probability (for inverse transforms)

    Notes
    -----
    All keys are optional (total=False) because different transforms
    work with different subsets of the full structure.
    """

    parameters: Dict[str, Union[float, NDArray[np.float32]]]
    extrinsic_parameters: Dict[str, Union[float, NDArray[np.float32]]]
    waveform: Union[PolarizationDict, Dict[str, NDArray[np.complex128]]]
    asds: Dict[str, NDArray[np.float64]]
    inference_parameters: NDArray[np.float32]
    log_prob: float


# ============================================================================
# Pipeline Stage TypedDicts (Dictionary Structure Evolution)
# ============================================================================


class PolarizationWaveformDict(TypedDict, total=False):
    """
    Waveform dict with polarization keys (h_plus, h_cross).

    Used before ProjectOntoDetectors transform projects onto detector network.

    Attributes
    ----------
    h_plus : NDArray[np.complex128]
        Plus polarization waveform in frequency domain
    h_cross : NDArray[np.complex128]
        Cross polarization waveform in frequency domain
    """

    h_plus: NDArray[np.complex128]
    h_cross: NDArray[np.complex128]


class PolarizationSample(TypedDict, total=False):
    """
    Sample after waveform generation with polarizations.

    This is the initial stage of the transform pipeline, containing
    the raw waveform polarizations before extrinsic parameter sampling
    or detector projection.

    Attributes
    ----------
    parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Intrinsic waveform parameters (mass, spin, etc.)
    waveform : PolarizationWaveformDict
        Polarization waveforms (h_plus, h_cross)

    Notes
    -----
    All keys are optional (total=False) to match actual usage patterns.
    """

    parameters: Dict[str, Union[float, NDArray[np.float32]]]
    waveform: PolarizationWaveformDict


class ExtrinsicSample(TypedDict, total=False):
    """
    Sample with extrinsic parameters added.

    Stage after SampleExtrinsicParameters has added sky location,
    distance, and time parameters.

    Attributes
    ----------
    parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Intrinsic waveform parameters
    extrinsic_parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Extrinsic parameters (ra, dec, psi, luminosity_distance, geocent_time,
        detector times)
    waveform : PolarizationWaveformDict
        Polarization waveforms (h_plus, h_cross)
    """

    parameters: Dict[str, Union[float, NDArray[np.float32]]]
    extrinsic_parameters: Dict[str, Union[float, NDArray[np.float32]]]
    waveform: PolarizationWaveformDict


class DetectorWaveformDict(TypedDict, total=False):
    """
    Waveform dict with detector keys (H1, L1, V1, etc.).

    Used after ProjectOntoDetectors transform. Keys are detector names,
    values are projected strains. This is a critical semantic change from
    polarizations (h_plus, h_cross) to detector-specific strains.

    Attributes
    ----------
    H1 : NDArray[np.complex128]
        LIGO Hanford strain
    L1 : NDArray[np.complex128]
        LIGO Livingston strain
    V1 : NDArray[np.complex128]
        Virgo strain

    Notes
    -----
    Additional detectors (K1, etc.) can be present. All keys are optional
    since the detector network configuration is dynamic.
    """

    H1: NDArray[np.complex128]
    L1: NDArray[np.complex128]
    V1: NDArray[np.complex128]


class DetectorStrainSample(TypedDict, total=False):
    """
    Sample after projection to detectors.

    This is a critical transition point: waveform dict changes from
    polarizations (h_plus, h_cross) to detector strains (H1, L1, V1).
    Extrinsic parameters are also consolidated into parameters dict.

    Attributes
    ----------
    parameters : Dict[str, Union[float, NDArray[np.float32]]]
        All parameters (intrinsic + consolidated extrinsics)
    extrinsic_parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Remaining unprocessed extrinsic parameters (if any)
    waveform : DetectorWaveformDict
        Detector-projected strains (H1, L1, V1, etc.)

    Notes
    -----
    After ProjectOntoDetectors, many extrinsic parameters are moved from
    extrinsic_parameters → parameters for consolidation.
    """

    parameters: Dict[str, Union[float, NDArray[np.float32]]]
    extrinsic_parameters: Dict[str, Union[float, NDArray[np.float32]]]
    waveform: DetectorWaveformDict


class NoiseASDSample(TypedDict, total=False):
    """
    Sample with ASDs (Amplitude Spectral Densities) added.

    Stage after SampleNoiseASD or similar transforms have added
    noise amplitude spectral densities for each detector.

    Attributes
    ----------
    parameters : Dict[str, Union[float, NDArray[np.float32]]]
        All parameters
    extrinsic_parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Remaining extrinsic parameters
    waveform : DetectorWaveformDict
        Detector strains
    asds : Dict[str, NDArray[np.float64]]
        Amplitude spectral densities (keys match waveform keys)

    Notes
    -----
    The asds dict keys should match waveform dict keys (H1, L1, V1, etc.).
    """

    parameters: Dict[str, Union[float, NDArray[np.float32]]]
    extrinsic_parameters: Dict[str, Union[float, NDArray[np.float32]]]
    waveform: DetectorWaveformDict
    asds: Dict[str, NDArray[np.float64]]


class InferenceSample(TypedDict, total=False):
    """
    Sample with standardized inference parameters.

    Stage after SelectStandardizeRepackageParameters has extracted,
    standardized (z-score), and packed parameters into an array for
    neural network input.

    Attributes
    ----------
    parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Original parameters dict (preserved)
    extrinsic_parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Original extrinsic parameters (preserved)
    waveform : DetectorWaveformDict
        Detector strains
    asds : Dict[str, NDArray[np.float64]]
        Amplitude spectral densities
    inference_parameters : NDArray[np.float32]
        Standardized parameter array: (param - mean) / std

    Notes
    -----
    The inference_parameters array is the primary input to neural networks.
    Original parameter dicts are kept for reference and inverse transforms.
    """

    parameters: Dict[str, Union[float, NDArray[np.float32]]]
    extrinsic_parameters: Dict[str, Union[float, NDArray[np.float32]]]
    waveform: DetectorWaveformDict
    asds: Dict[str, NDArray[np.float64]]
    inference_parameters: NDArray[np.float32]


class TensorPackedSample(TypedDict, total=False):
    """
    Sample with waveform repackaged as tensor.

    Stage after RepackageStrainsAndASDS has converted the nested waveform
    and asds dicts into a single tensor with shape [num_ifos, 3, num_bins].

    Attributes
    ----------
    parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Original parameters dict
    inference_parameters : NDArray[np.float32]
        Standardized parameters array
    waveform : NDArray[np.float32]
        Repackaged tensor with shape [num_ifos, 3, num_bins]
        - [:, 0, :] = strain.real
        - [:, 1, :] = strain.imag
        - [:, 2, :] = 1 / (asd * scale_factor)
    asds : Dict[str, NDArray[np.float64]]
        Original ASD dicts (preserved)

    Notes
    -----
    The waveform tensor format is optimized for neural network input,
    combining strain (real/imag) and inverse ASD in a single array.
    """

    parameters: Dict[str, Union[float, NDArray[np.float32]]]
    inference_parameters: NDArray[np.float32]
    waveform: NDArray[np.float32]
    asds: Dict[str, NDArray[np.float64]]


class TorchSample(TypedDict, total=False):
    """
    Sample with numpy arrays converted to torch tensors.

    Final stage after ToTorch has converted top-level numpy arrays
    to PyTorch tensors for GPU computation.

    Attributes
    ----------
    parameters : Dict[str, Union[float, NDArray[np.float32]]]
        Original parameters dict (NOT converted - nested structure)
    inference_parameters : torch.Tensor
        Standardized parameters as torch tensor
    waveform : torch.Tensor
        Repackaged waveform as torch tensor
    asds : Dict[str, NDArray[np.float64]]
        Original ASDs (NOT converted - nested dict structure)

    Notes
    -----
    ToTorch only converts top-level numpy arrays, not nested dicts.
    Use RepackageStrainsAndASDS before ToTorch to flatten waveform/asd dicts.
    """

    parameters: Dict[str, Union[float, NDArray[np.float32]]]
    inference_parameters: torch.Tensor
    waveform: torch.Tensor
    asds: Dict[str, NDArray[np.float64]]


# ============================================================================
# Protocols (Structural Subtyping)
# ============================================================================


@runtime_checkable
class DomainProtocol(Protocol):
    """
    Minimal protocol for Domain objects used in transforms.

    This protocol avoids circular imports by not importing the actual Domain
    classes from dingo_waveform.domains. Instead, it defines the minimal
    interface that transforms actually use.

    Methods
    -------
    time_translate_data(data, dt)
        Time-shift data by dt
    decimate(data)
        Decimate uniform-frequency data to multibanded representation

    Attributes
    ----------
    domain_dict : Dict[str, Any]
        Dictionary representation for serialization
    """

    def time_translate_data(
        self, data: NDArray[np.complex128], dt: Union[float, NDArray[np.float64]]
    ) -> NDArray[np.complex128]:
        """
        Time-shift data by dt.

        Parameters
        ----------
        data : NDArray[np.complex128]
            Frequency-domain data
        dt : Union[float, NDArray[np.float64]]
            Time shift in seconds

        Returns
        -------
        NDArray[np.complex128]
            Time-shifted data
        """
        ...

    def decimate(self, data: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Decimate uniform-frequency data to multibanded representation.

        Parameters
        ----------
        data : NDArray[np.complex128]
            Uniform-frequency data

        Returns
        -------
        NDArray[np.complex128]
            Decimated data
        """
        ...

    @property
    def domain_dict(self) -> Dict[str, Any]:
        """
        Dictionary representation for serialization.

        Returns
        -------
        Dict[str, Any]
            Domain parameters as dictionary
        """
        ...


@runtime_checkable
class InvertibleTransform(Protocol):
    """
    Protocol for transforms that support inverse operations.

    Only a few transforms are invertible:
    - SelectStandardizeRepackageParameters (de-standardization)
    - StandardizeParameters (de-standardization)
    - GNPEBase (has inverse group operation, but not a transform inverse)

    Methods
    -------
    inverse(sample, **kwargs)
        Apply inverse transform to sample
    """

    def inverse(self, sample: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Apply inverse transform to sample.

        Parameters
        ----------
        sample : Dict[str, Any]
            Transformed sample to invert
        **kwargs : Any
            Transform-specific arguments (e.g., as_type for parameter transforms)

        Returns
        -------
        Dict[str, Any]
            Inverse-transformed sample
        """
        ...


@runtime_checkable
class BatchAwareTransform(Protocol):
    """
    Protocol for transforms that handle batched inputs differently.

    Many transforms detect batch dimensions and process accordingly:
    - SampleExtrinsicParameters (samples batch_size parameter sets)
    - ProjectOntoDetectors (vectorized antenna patterns)
    - SelectStandardizeRepackageParameters (batched normalization)

    Methods
    -------
    __call__(input_sample)
        Apply transform, handling both single and batched inputs
    """

    def __call__(self, input_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transform, handling both single and batched inputs.

        Implementation should detect batch dimension from array shapes
        and process appropriately.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample (may be batched or single)

        Returns
        -------
        Dict[str, Any]
            Transformed sample
        """
        ...


# ============================================================================
# Type Aliases for Common Patterns
# ============================================================================

ParameterValue: TypeAlias = Union[float, np.ndarray, torch.Tensor]
"""
Type for parameter values that can be scalar (training) or array (batched).

Used extensively in GNPE transforms where parameters can be:
- float: Single parameter value (training mode)
- np.ndarray: Batch of parameter values (numpy)
- torch.Tensor: Batch of parameter values (PyTorch)
"""

NestedDict: TypeAlias = Dict[str, Any]
"""
Generic nested dictionary type (used pervasively in transforms).

This is an alias for Dict[str, Any] that provides semantic clarity
when a dict contains nested structure but we cannot be more specific
about the type without circular imports or excessive complexity.
"""

# Union of all pipeline stage types for generic handling
TransformSample: TypeAlias = Union[
    PolarizationSample,
    ExtrinsicSample,
    DetectorStrainSample,
    NoiseASDSample,
    InferenceSample,
    TensorPackedSample,
    TorchSample,
    NestedDict,  # Fallback for complex cases
]
"""
Union of all pipeline stage TypedDicts.

This type alias allows transforms to accept any pipeline stage when
the specific stage is not known or the transform works across multiple stages.

Use specific stage types (e.g., NoiseASDSample) when possible for better
type safety. Use TransformSample as a fallback for generic transforms.
"""
