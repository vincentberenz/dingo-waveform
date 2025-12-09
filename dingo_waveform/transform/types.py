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
    Sequence,
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


class GNPETimeShiftsDict(TypedDict):
    """
    Configuration for GNPE (Group Equivariant NPE) time shifts.

    Used by GNPECoalescenceTimes transform to enforce time translation
    equivariance in neural posterior estimation. This enables the network
    to learn symmetries in the gravitational wave parameter space.

    Attributes
    ----------
    kernel : str
        Bilby prior specification string for time shift distribution.
        Example: "Uniform(minimum=-0.1, maximum=0.1)"
        The kernel defines the random time shifts applied to enforce equivariance.
    exact_equiv : bool
        If True, enforce exact global time translation symmetry by applying
        the same time shift to all detectors. If False, apply independent
        shifts per detector.

    Examples
    --------
    >>> gnpe_config: GNPETimeShiftsDict = {
    ...     'kernel': 'Uniform(minimum=-0.1, maximum=0.1)',
    ...     'exact_equiv': True
    ... }

    Notes
    -----
    GNPE (Group-equivariant Neural Posterior Estimation) leverages symmetries
    in the parameter space to improve sample efficiency. Time translation
    equivariance means the posterior is invariant to global time shifts.
    """

    kernel: str
    exact_equiv: bool


class RandomStrainCroppingDict(TypedDict, total=False):
    """
    Configuration for random frequency-domain cropping.

    Used by CropMaskStrainRandom transform to randomly mask waveforms
    outside sampled frequency bounds. Supports both stochastic (random)
    and deterministic (fixed) cropping modes.

    All keys are optional (total=False) since the transform has defaults.

    Attributes
    ----------
    f_min_upper : Optional[float]
        Upper bound for random f_min sampling: f_min ~ Uniform[domain.f_min, f_min_upper].
        Example: If domain.f_min=20 Hz and f_min_upper=50 Hz, each sample gets
        a random f_min in [20, 50] Hz.
    f_max_lower : Optional[float]
        Lower bound for random f_max sampling: f_max ~ Uniform[f_max_lower, domain.f_max].
        Example: If f_max_lower=512 Hz and domain.f_max=1024 Hz, f_max ~ [512, 1024] Hz.
    cropping_probability : float
        Probability of applying crop to a sample. Default 1.0 (always crop).
        Example: 0.8 means 80% of samples are cropped, 20% are unchanged.
    independent_detectors : bool
        If True, sample crop bounds independently per detector (H1, L1 get different bounds).
        If False, use same bounds for all detectors. Default True.
    independent_lower_upper : bool
        If True, apply cropping_probability independently to lower and upper bounds.
        If False, apply probability once to both bounds. Default True.
    deterministic_fmin_fmax : Optional[Union[List[float], List[List[float]]]]
        Fixed [f_min, f_max] bounds (disables random sampling).
        Can be:
        - Single pair [f_min, f_max]: Same bounds for all detectors
        - List of pairs [[f_min1, f_max1], [f_min2, f_max2], ...]: Per-detector bounds
        When set, f_min_upper/f_max_lower are ignored and cropping_probability must be 1.0.

    Examples
    --------
    >>> # Stochastic mode: random bounds
    >>> stochastic_config: RandomStrainCroppingDict = {
    ...     'f_min_upper': 50.0,
    ...     'f_max_lower': 512.0,
    ...     'cropping_probability': 0.8,
    ...     'independent_detectors': True,
    ...     'independent_lower_upper': True
    ... }

    >>> # Deterministic mode: fixed bounds
    >>> deterministic_config: RandomStrainCroppingDict = {
    ...     'deterministic_fmin_fmax': [30.0, 600.0]
    ... }

    >>> # Per-detector deterministic bounds
    >>> per_det_config: RandomStrainCroppingDict = {
    ...     'deterministic_fmin_fmax': [[30.0, 600.0], [25.0, 700.0]]
    ... }

    Notes
    -----
    This transform helps neural networks learn robustness to different
    frequency ranges, simulating varying detector sensitivities.

    Waveform is set to zero outside the sampled/fixed bounds.
    """

    f_min_upper: float
    f_max_lower: float
    cropping_probability: float
    independent_detectors: bool
    independent_lower_upper: bool
    deterministic_fmin_fmax: Union[List[float], List[List[float]]]


class DomainUpdateDict(TypedDict, total=False):
    """
    Configuration for frequency range masking in inference.

    Used by MaskDataForFrequencyRangeUpdate transform to mask data
    outside specified frequency bounds. This allows narrowing the
    frequency range at inference time compared to training.

    All keys are optional (total=False).

    Attributes
    ----------
    f_min : Union[float, Dict[str, float]]
        Minimum frequency (Hz) for masking.
        Can be:
        - float: Same f_min for all detectors
        - Dict[str, float]: Per-detector f_min {'H1': 30.0, 'L1': 25.0}
        Data below f_min is masked (waveform → 0, ASD → 1).
    f_max : Union[float, Dict[str, float]]
        Maximum frequency (Hz) for masking.
        Can be:
        - float: Same f_max for all detectors
        - Dict[str, float]: Per-detector f_max {'H1': 512.0, 'L1': 600.0}
        Data above f_max is masked (waveform → 0, ASD → 1).

    Examples
    --------
    >>> # Global bounds (all detectors)
    >>> global_config: DomainUpdateDict = {
    ...     'f_min': 30.0,
    ...     'f_max': 512.0
    ... }

    >>> # Per-detector bounds
    >>> per_det_config: DomainUpdateDict = {
    ...     'f_min': {'H1': 30.0, 'L1': 25.0, 'V1': 35.0},
    ...     'f_max': {'H1': 512.0, 'L1': 600.0, 'V1': 500.0}
    ... }

    >>> # Only mask lower bound
    >>> lower_only_config: DomainUpdateDict = {
    ...     'f_min': 30.0
    ... }

    Notes
    -----
    This masking operation modifies the sample in-place:
    - waveform[f < f_min or f > f_max] = 0.0
    - asd[f < f_min or f > f_max] = 1.0

    Setting ASD to 1.0 outside the range tells the neural network to
    ignore those frequencies (unit noise variance = no information).
    """

    f_min: Union[float, Dict[str, float]]
    f_max: Union[float, Dict[str, float]]


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
    Protocol for Domain objects used in transform chains.

    This protocol captures the duck-typed interface for Domain objects
    (UniformFrequencyDomain, MultibandedFrequencyDomain, TimeDomain) without
    importing them directly, avoiding circular dependencies.

    Typical implementations:
    - dingo_waveform.domains.UniformFrequencyDomain
    - dingo_waveform.domains.MultibandedFrequencyDomain
    - dingo_waveform.domains.TimeDomain

    Required Attributes/Methods (used by all transforms)
    -----------------------------------------------------
    time_translate_data(data, dt)
        Time-shift waveform data by dt seconds
    noise_std : Union[float, NDArray[np.float64]]
        Standard deviation of noise for whitening
    min_idx : int
        Minimum index in frequency array
    max_idx : int
        Maximum index in frequency array
    domain_dict : Dict[str, Any]
        Dictionary representation for serialization
    sample_frequencies : NDArray[np.float64]
        Array of sample frequencies (Hz)
    __call__()
        Return frequency array (callable interface)

    Optional Attributes (domain-specific)
    --------------------------------------
    f_min : float
        Minimum frequency (Hz) - for frequency domains
    f_max : float
        Maximum frequency (Hz) - for frequency domains
    delta_f : float
        Frequency spacing (Hz) - for uniform frequency domains only
    decimate(data)
        Decimate uniform-frequency data to multibanded - for MultibandedFrequencyDomain only

    Examples
    --------
    >>> from dingo_waveform.domains import UniformFrequencyDomain
    >>> domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
    >>> domain.sample_frequencies.shape
    (8032,)
    >>> domain.noise_std
    1.0
    >>> data_shifted = domain.time_translate_data(data, dt=0.1)

    Notes
    -----
    Not all domains have all attributes. For example:
    - Only MultibandedFrequencyDomain has decimate() method
    - Only frequency domains have f_min, f_max attributes
    - TimeDomain has different attributes

    Factory functions check for required attributes with hasattr() before use.
    """

    # REQUIRED: Methods used by multiple transforms
    def time_translate_data(
        self,
        data: Union[NDArray[np.complex128], Any],
        dt: Union[float, NDArray[np.float64]]
    ) -> Union[NDArray[np.complex128], Any]:
        """
        Time-translate waveform data by dt seconds.

        Parameters
        ----------
        data : Union[NDArray[np.complex128], torch.Tensor]
            Frequency-domain data
        dt : Union[float, NDArray[np.float64]]
            Time shift in seconds (can be batched)

        Returns
        -------
        Union[NDArray[np.complex128], torch.Tensor]
            Time-shifted data
        """
        ...

    def update_data(
        self,
        data: NDArray[np.float64],
        low_value: float = 1e-22
    ) -> NDArray[np.float64]:
        """
        Update low-frequency bins to avoid numerical issues.

        Used by WhitenFixedASD to avoid division by very small ASD values.

        Parameters
        ----------
        data : NDArray[np.float64]
            Data array to update
        low_value : float
            Value to use for low-frequency bins

        Returns
        -------
        NDArray[np.float64]
            Updated data with low-frequency bins set to low_value
        """
        ...

    def __call__(self) -> NDArray[np.float64]:
        """
        Return frequency array (callable interface).

        Returns
        -------
        NDArray[np.float64]
            Array of frequencies (Hz)
        """
        ...

    # REQUIRED: Properties used by multiple transforms
    @property
    def noise_std(self) -> Union[float, NDArray[np.float64]]:
        """
        Standard deviation of noise for whitening.

        Used by WhitenAndScaleStrain to compute scale factors.

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Noise standard deviation
        """
        ...

    @property
    def min_idx(self) -> int:
        """
        Minimum index in frequency array.

        Returns
        -------
        int
            Index of first valid frequency bin
        """
        ...

    @property
    def max_idx(self) -> int:
        """
        Maximum index in frequency array.

        Returns
        -------
        int
            Index of last valid frequency bin
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

    @property
    def sample_frequencies(self) -> NDArray[np.float64]:
        """
        Array of sample frequencies (Hz).

        Returns
        -------
        NDArray[np.float64]
            Frequency array
        """
        ...

    @property
    def frequency_mask(self) -> NDArray[np.bool_]:
        """
        Boolean mask for valid frequency bins.

        Used by ApplyCalibrationUncertainty to select calibration frequencies.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask (True for valid bins)
        """
        ...

    # OPTIONAL: Frequency domain attributes (not present in TimeDomain)
    @property
    def f_min(self) -> float:
        """
        Minimum frequency (Hz).

        Only present in frequency domains (UniformFrequencyDomain,
        MultibandedFrequencyDomain).

        Returns
        -------
        float
            Minimum frequency
        """
        ...

    @property
    def f_max(self) -> float:
        """
        Maximum frequency (Hz).

        Only present in frequency domains.

        Returns
        -------
        float
            Maximum frequency
        """
        ...

    @property
    def delta_f(self) -> float:
        """
        Frequency spacing (Hz).

        Only present in UniformFrequencyDomain.

        Returns
        -------
        float
            Frequency spacing
        """
        ...

    # OPTIONAL: MultibandedFrequencyDomain-specific method
    def decimate(self, data: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Decimate uniform-frequency data to multibanded representation.

        Only present in MultibandedFrequencyDomain.

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


@runtime_checkable
class ASDDatasetLike(Protocol):
    """
    Protocol for ASD dataset objects.

    This protocol captures the duck-typed interface for ASD (Amplitude Spectral Density)
    dataset objects used in transform factories. Objects conforming to this protocol
    can sample random ASD realizations for noise simulation.

    Typical implementations:
    - dingo.gw.noise.asd_dataset.ASDDataset

    Methods
    -------
    sample_random_asds(n)
        Sample n random ASD realizations from the dataset

    Examples
    --------
    >>> from dingo.gw.noise.asd_dataset import ASDDataset
    >>> asd_dataset = ASDDataset('/path/to/asd_dataset.hdf5')
    >>> # Sample single ASD
    >>> asds = asd_dataset.sample_random_asds(None)
    >>> # asds = {'H1': array(...), 'L1': array(...)}
    >>> # Sample batch of 10 ASDs
    >>> asds_batch = asd_dataset.sample_random_asds(10)
    >>> # asds_batch = {'H1': array(10, D), 'L1': array(10, D)}

    Notes
    -----
    The return type uses Dict[str, NDArray[np.float64]] where keys are detector
    names ('H1', 'L1', 'V1', etc.) and values are ASD arrays.

    When n=None, returns single sample without batch dimension: shape (D,)
    When n is int, returns batched samples with batch dimension: shape (n, D)
    """

    def sample_random_asds(
        self, n: Any = None
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Sample n random ASD realizations (or single if n=None).

        Parameters
        ----------
        n : Optional[int]
            Number of samples. If None, returns single sample without batch dim.

        Returns
        -------
        Dict[str, NDArray[np.float64]]
            Mapping of detector names to ASD arrays.
            Shape: (n, D) if batched, (D,) if single.
        """
        ...


@runtime_checkable
class ExtrinsicPriorLike(Protocol):
    """
    Protocol for extrinsic parameter prior objects.

    This protocol captures the duck-typed interface for prior objects that sample
    extrinsic parameters (sky location, orientation, distance, time, etc.) for
    gravitational wave events.

    Typical implementations:
    - dingo.gw.prior.BBHExtrinsicPriorDict
    - bilby.core.prior.PriorDict
    - bilby.gw.prior.BBHPriorDict

    Methods
    -------
    sample(n)
        Sample n extrinsic parameter sets from the prior

    Examples
    --------
    >>> from dingo.gw.prior import BBHExtrinsicPriorDict
    >>> prior_dict = {
    ...     'ra': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 6.28},
    ...     'dec': {'type': 'Cosine', 'minimum': -1.57, 'maximum': 1.57},
    ...     'psi': {'type': 'Uniform', 'minimum': 0.0, 'maximum': 3.14},
    ...     'luminosity_distance': {'type': 'UniformSourceFrame', 'minimum': 100, 'maximum': 5000}
    ... }
    >>> prior = BBHExtrinsicPriorDict(prior_dict)
    >>> # Sample single parameter set
    >>> params = prior.sample(None)
    >>> # params = {'ra': 1.23, 'dec': 0.45, 'psi': 2.1, 'luminosity_distance': 1200.0}
    >>> # Sample batch of 10 parameter sets
    >>> params_batch = prior.sample(10)
    >>> # params_batch = {'ra': array([...]), 'dec': array([...]), ...}

    Notes
    -----
    The return type uses Dict[str, Union[float, NDArray[np.float32]]] where:
    - When n=None: values are floats (single parameter set)
    - When n is int: values are arrays (batch of parameter sets)

    Extrinsic parameters typically include: ra, dec, psi, luminosity_distance,
    geocent_time, and potentially detector-specific times.
    """

    def sample(
        self, n: Any = None
    ) -> Dict[str, Union[float, NDArray[np.float32]]]:
        """
        Sample n extrinsic parameter sets.

        Parameters
        ----------
        n : Optional[int]
            Number of samples. If None, returns single sample (floats).

        Returns
        -------
        Dict[str, Union[float, NDArray[np.float32]]]
            Mapping of parameter names to sampled values.
            Values are floats if n=None, arrays if batched.
        """
        ...


@runtime_checkable
class InterferometerLike(Protocol):
    """
    Protocol for interferometer objects.

    This protocol captures the minimal duck-typed interface for interferometer
    objects used in transform factories. Objects conforming to this protocol
    represent gravitational wave detectors (LIGO Hanford, LIGO Livingston,
    Virgo, KAGRA, etc.).

    Typical implementations:
    - bilby.gw.detector.Interferometer

    Attributes
    ----------
    name : str
        Detector identifier (e.g., 'H1', 'L1', 'V1', 'K1')

    Examples
    --------
    >>> from bilby.gw.detector import Interferometer
    >>> ifo = Interferometer('H1')
    >>> ifo.name
    'H1'
    >>> # Often used in lists/iterables:
    >>> from bilby.gw.detector import InterferometerList
    >>> ifo_list = InterferometerList(['H1', 'L1'])
    >>> for ifo in ifo_list:
    ...     print(ifo.name)
    'H1'
    'L1'

    Notes
    -----
    This protocol only specifies the 'name' attribute since that's the minimal
    interface used by factory functions. Actual Interferometer objects have
    many more attributes (antenna_response, strain_data, etc.) but those are
    accessed via specific methods, not required by the Protocol.
    """

    name: str


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

# Type aliases for factory function parameters
InterferometerListLike: TypeAlias = Union[
    Sequence[InterferometerLike],
    Sequence[str]
]
"""
Type alias for interferometer list parameter in factory functions.

Accepts either:
- Sequence of InterferometerLike objects (e.g., bilby.gw.detector.InterferometerList)
- Sequence of detector name strings (e.g., ['H1', 'L1', 'V1'])

Examples
--------
>>> from bilby.gw.detector import InterferometerList
>>> # Full interferometer objects
>>> ifo_list: InterferometerListLike = InterferometerList(['H1', 'L1'])
>>>
>>> # Simple list of detector names
>>> ifo_list: InterferometerListLike = ['H1', 'L1', 'V1']

Notes
-----
Factory functions iterate over the list and access the .name attribute
(or use the string directly if it's a list of strings). This flexibility
allows users to pass either full Interferometer objects or simple name lists.
"""

DetectorList: TypeAlias = List[str]
"""
Type alias for detector name lists.

Used in factory functions that require explicit detector names
(e.g., for inference transforms that don't need full Interferometer objects).

Examples
--------
>>> detectors: DetectorList = ['H1', 'L1']
>>> detectors: DetectorList = ['H1', 'L1', 'V1', 'K1']

Notes
-----
Detector names follow LIGO/Virgo/KAGRA conventions:
- H1: LIGO Hanford
- L1: LIGO Livingston
- V1: Virgo
- K1: KAGRA
"""

ParameterNames: TypeAlias = List[str]
"""
Type alias for parameter name lists.

Used in factory functions to specify which parameters are used for
inference, context, or standardization.

Examples
--------
>>> inference_params: ParameterNames = [
...     'mass_1', 'mass_2', 'luminosity_distance',
...     'theta_jn', 'phase', 'a_1', 'a_2'
... ]
>>> context_params: ParameterNames = ['geocent_time']

Notes
-----
Common parameter names include:
- Intrinsic: mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
- Extrinsic: ra, dec, psi, luminosity_distance, geocent_time, theta_jn, phase

The parameter names must match keys in the parameter dictionaries
flowing through the transform pipeline.
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
