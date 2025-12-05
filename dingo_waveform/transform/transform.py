"""
Main Transform class and configuration for gravitational wave data transforms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
import torch.utils.data
from torchvision.transforms import Compose
from bilby.gw.detector import InterferometerList

# Import proper type definitions
from .config_types import (
    StandardizationConfig,
    RandomStrainCroppingConfig,
    GNPETimeShiftsConfig,
    DomainUpdateConfig,
    ExtrinsicPriorConfig,
)


@dataclass(frozen=True)
class TransformConfig:
    """
    Immutable configuration for transforms with proper type annotations.

    This configuration uses dataclasses instead of opaque dictionaries,
    following dingo-waveform's design principle of type safety and clarity.
    All attributes must be initialized (no None values except where explicitly Optional).
    """

    detectors: List[str]
    domain: Any  # Domain type from dingo.gw.domains (UniformFrequencyDomain, MultibandedFrequencyDomain, etc.)
    ref_time: float
    asd_dataset_path: Optional[str]
    extrinsic_prior: ExtrinsicPriorConfig
    inference_parameters: List[str]
    context_parameters: List[str] = field(default_factory=list)
    standardization: StandardizationConfig = field(default_factory=lambda: StandardizationConfig())
    random_strain_cropping: Optional[RandomStrainCroppingConfig] = None
    gnpe_time_shifts: Optional[GNPETimeShiftsConfig] = None
    zero_noise: bool = False
    domain_update: Optional[DomainUpdateConfig] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        object.__setattr__(self, "_validated", True)
        self._validate()

    def _validate(self) -> None:
        """Private validation helper."""
        # Validate detectors
        if not self.detectors:
            raise ValueError("detectors list cannot be empty")

        # Validate domain
        if self.domain is None:
            raise ValueError("domain cannot be None")

        # Validate ref_time
        if self.ref_time <= 0:
            raise ValueError(f"ref_time must be positive, got {self.ref_time}")

        # Validate inference_parameters
        if not self.inference_parameters:
            raise ValueError("inference_parameters list cannot be empty")

        # Validate context_parameters is a list (can be empty)
        if not isinstance(self.context_parameters, list):
            raise ValueError("context_parameters must be a list")

        # Validate extrinsic_prior is proper type
        if not isinstance(self.extrinsic_prior, ExtrinsicPriorConfig):
            raise ValueError(
                f"extrinsic_prior must be ExtrinsicPriorConfig, got {type(self.extrinsic_prior)}"
            )

        # Validate standardization is proper type
        if not isinstance(self.standardization, StandardizationConfig):
            raise ValueError(
                f"standardization must be StandardizationConfig, got {type(self.standardization)}"
            )

        # Validate optional configs are proper types if not None
        if self.random_strain_cropping is not None:
            if not isinstance(self.random_strain_cropping, RandomStrainCroppingConfig):
                raise ValueError(
                    f"random_strain_cropping must be RandomStrainCroppingConfig, got {type(self.random_strain_cropping)}"
                )

        if self.gnpe_time_shifts is not None:
            if not isinstance(self.gnpe_time_shifts, GNPETimeShiftsConfig):
                raise ValueError(
                    f"gnpe_time_shifts must be GNPETimeShiftsConfig, got {type(self.gnpe_time_shifts)}"
                )

        if self.domain_update is not None:
            if not isinstance(self.domain_update, DomainUpdateConfig):
                raise ValueError(
                    f"domain_update must be DomainUpdateConfig, got {type(self.domain_update)}"
                )

        # Validate zero_noise is bool
        if not isinstance(self.zero_noise, bool):
            raise ValueError("zero_noise must be a boolean")


class Transform:
    """
    Main transform manager class.
    Encapsulates transform chain construction and provides convenient iterator methods.
    """

    def __init__(
        self,
        detectors: List[str],
        domain: Any,
        ref_time: float,
        asd_dataset_path: Optional[str],
        extrinsic_prior: Union[ExtrinsicPriorConfig, Dict[str, Any]],
        inference_parameters: List[str],
        context_parameters: List[str] = [],
        standardization: Optional[Union[StandardizationConfig, Dict[str, Dict[str, float]]]] = None,
        random_strain_cropping: Optional[Union[RandomStrainCroppingConfig, Dict[str, Any]]] = None,
        gnpe_time_shifts: Optional[Union[GNPETimeShiftsConfig, Dict[str, Any]]] = None,
        zero_noise: bool = False,
        domain_update: Optional[Union[DomainUpdateConfig, Dict[str, float]]] = None,
    ) -> None:
        """
        Initialize Transform with explicit parameters.

        This constructor accepts both proper dataclass types (ExtrinsicPriorConfig, etc.)
        and legacy dictionary formats for backward compatibility.

        Parameters
        ----------
        detectors : List[str]
            List of detector names (e.g., ["H1", "L1"])
        domain : Domain
            Frequency domain object (UniformFrequencyDomain, MultibandedFrequencyDomain, etc.)
        ref_time : float
            Reference GPS time
        asd_dataset_path : Optional[str]
            Path to ASD dataset (None for inference with fixed ASDs)
        extrinsic_prior : Union[ExtrinsicPriorConfig, Dict[str, Any]]
            Extrinsic parameter prior configuration or legacy dict
        inference_parameters : List[str]
            Parameters for inference
        context_parameters : List[str]
            GNPE context parameters (empty list if not used)
        standardization : Optional[Union[StandardizationConfig, Dict[str, Dict[str, float]]]]
            Parameter standardization config or legacy dict (mean/std)
        random_strain_cropping : Optional[Union[RandomStrainCroppingConfig, Dict[str, Any]]]
            Random cropping config or legacy dict (None if not used)
        gnpe_time_shifts : Optional[Union[GNPETimeShiftsConfig, Dict[str, Any]]]
            GNPE time shift config or legacy dict (None if not used)
        zero_noise : bool
            Whether to omit noise addition (default False)
        domain_update : Optional[Union[DomainUpdateConfig, Dict[str, float]]]
            Domain update config or legacy dict (None if not needed)
        """
        # Convert legacy dict formats to proper types
        extrinsic_prior_config = self._convert_extrinsic_prior(extrinsic_prior)
        standardization_config = self._convert_standardization(standardization)
        random_strain_cropping_config = self._convert_random_strain_cropping(random_strain_cropping)
        gnpe_time_shifts_config = self._convert_gnpe_time_shifts(gnpe_time_shifts)
        domain_update_config = self._convert_domain_update(domain_update)

        # Create config (this will validate via __post_init__)
        self._config = TransformConfig(
            detectors=detectors,
            domain=domain,
            ref_time=ref_time,
            asd_dataset_path=asd_dataset_path,
            extrinsic_prior=extrinsic_prior_config,
            inference_parameters=inference_parameters,
            context_parameters=context_parameters,
            standardization=standardization_config,
            random_strain_cropping=random_strain_cropping_config,
            gnpe_time_shifts=gnpe_time_shifts_config,
            zero_noise=zero_noise,
            domain_update=domain_update_config,
        )

        # Initialize cached attributes (lazy initialization)
        self._ifo_list: Optional[InterferometerList] = None
        self._training_transform: Optional[Compose] = None
        self._svd_transform: Optional[Compose] = None
        self._inference_transform_pre: Optional[Compose] = None
        self._inference_transform_post: Optional[Callable] = None

    @staticmethod
    def _convert_extrinsic_prior(
        extrinsic_prior: Union[ExtrinsicPriorConfig, Dict[str, Any]]
    ) -> ExtrinsicPriorConfig:
        """Convert extrinsic_prior to ExtrinsicPriorConfig if needed."""
        if isinstance(extrinsic_prior, ExtrinsicPriorConfig):
            return extrinsic_prior
        return ExtrinsicPriorConfig.from_dict(extrinsic_prior)

    @staticmethod
    def _convert_standardization(
        standardization: Optional[Union[StandardizationConfig, Dict[str, Dict[str, float]]]]
    ) -> StandardizationConfig:
        """Convert standardization to StandardizationConfig if needed."""
        if standardization is None:
            return StandardizationConfig()
        if isinstance(standardization, StandardizationConfig):
            return standardization
        # Handle legacy dict format
        if not standardization:
            return StandardizationConfig()
        return StandardizationConfig(
            mean=standardization.get("mean", {}),
            std=standardization.get("std", {})
        )

    @staticmethod
    def _convert_random_strain_cropping(
        random_strain_cropping: Optional[Union[RandomStrainCroppingConfig, Dict[str, Any]]]
    ) -> Optional[RandomStrainCroppingConfig]:
        """Convert random_strain_cropping to RandomStrainCroppingConfig if needed."""
        if random_strain_cropping is None:
            return None
        if isinstance(random_strain_cropping, RandomStrainCroppingConfig):
            return random_strain_cropping
        return RandomStrainCroppingConfig.from_dict(random_strain_cropping)

    @staticmethod
    def _convert_gnpe_time_shifts(
        gnpe_time_shifts: Optional[Union[GNPETimeShiftsConfig, Dict[str, Any]]]
    ) -> Optional[GNPETimeShiftsConfig]:
        """Convert gnpe_time_shifts to GNPETimeShiftsConfig if needed."""
        if gnpe_time_shifts is None:
            return None
        if isinstance(gnpe_time_shifts, GNPETimeShiftsConfig):
            return gnpe_time_shifts
        return GNPETimeShiftsConfig.from_dict(gnpe_time_shifts)

    @staticmethod
    def _convert_domain_update(
        domain_update: Optional[Union[DomainUpdateConfig, Dict[str, float]]]
    ) -> Optional[DomainUpdateConfig]:
        """Convert domain_update to DomainUpdateConfig if needed."""
        if domain_update is None:
            return None
        if isinstance(domain_update, DomainUpdateConfig):
            return domain_update
        return DomainUpdateConfig.from_dict(domain_update)

    # Factory methods
    @classmethod
    def for_training(
        cls,
        data_settings: Dict[str, Any],
        waveform_dataset: Any,
        asd_dataset_path: str,
    ) -> "Transform":
        """
        Build Transform for training from data_settings dictionary.
        Extracts all needed parameters from data_settings and waveform_dataset.

        Parameters
        ----------
        data_settings : Dict[str, Any]
            Training data settings containing detectors, priors, parameters, etc.
        waveform_dataset : WaveformDataset
            Waveform dataset with domain and parameters
        asd_dataset_path : str
            Path to ASD dataset

        Returns
        -------
        Transform
            Configured Transform instance for training
        """
        from dingo.gw.prior import default_inference_parameters

        # Extract parameters from data_settings
        detectors = data_settings["detectors"]
        ref_time = data_settings["ref_time"]
        extrinsic_prior = data_settings["extrinsic_prior"]

        # Handle default inference parameters
        if data_settings.get("inference_parameters") == "default":
            inference_parameters = default_inference_parameters
        else:
            inference_parameters = data_settings["inference_parameters"]

        # Get context parameters (empty list if not present)
        context_parameters = data_settings.get("context_parameters", [])

        # Get standardization (will be calculated if not present)
        standardization = data_settings.get("standardization", None)

        # Get optional settings
        random_strain_cropping = data_settings.get("random_strain_cropping", None)
        gnpe_time_shifts = data_settings.get("gnpe_time_shifts", None)
        zero_noise = data_settings.get("zero_noise", False)
        domain_update = data_settings.get("domain_update", None)

        # Create transform instance
        return cls(
            detectors=detectors,
            domain=waveform_dataset.domain,
            ref_time=ref_time,
            asd_dataset_path=asd_dataset_path,
            extrinsic_prior=extrinsic_prior,
            inference_parameters=inference_parameters,
            context_parameters=context_parameters,
            standardization=standardization,
            random_strain_cropping=random_strain_cropping,
            gnpe_time_shifts=gnpe_time_shifts,
            zero_noise=zero_noise,
            domain_update=domain_update,
        )

    @classmethod
    def for_svd(
        cls,
        data_settings: Dict[str, Any],
        waveform_dataset: Any,
        asd_dataset_path: str,
    ) -> "Transform":
        """
        Build Transform for SVD generation (no noise, no repackaging).

        Parameters
        ----------
        data_settings : Dict[str, Any]
            Training data settings
        waveform_dataset : WaveformDataset
            Waveform dataset
        asd_dataset_path : str
            Path to ASD dataset

        Returns
        -------
        Transform
            Configured Transform instance for SVD generation
        """
        # For SVD, we use the same settings as training but will build different transform chain
        # The for_training method extracts everything we need
        return cls.for_training(data_settings, waveform_dataset, asd_dataset_path)

    @classmethod
    def for_inference(
        cls,
        model_metadata: Dict[str, Any],
        detectors: List[str],
        ref_time: float,
    ) -> "Transform":
        """
        Build Transform for inference from model metadata.
        Uses different transform chain suitable for inference.

        Parameters
        ----------
        model_metadata : Dict[str, Any]
            Model metadata containing domain, standardization, etc.
        detectors : List[str]
            List of detector names
        ref_time : float
            Reference GPS time for event

        Returns
        -------
        Transform
            Configured Transform instance for inference
        """
        from dingo.gw.domains import build_domain

        # Extract domain from model metadata
        domain = build_domain(model_metadata["train_settings"]["data"]["domain"])

        # Extract inference parameters and standardization
        data_settings = model_metadata["train_settings"]["data"]
        inference_parameters = data_settings["inference_parameters"]
        standardization = data_settings["standardization"]

        # Context parameters (empty for standard inference)
        context_parameters = data_settings.get("context_parameters", [])

        # For inference, we don't use extrinsic prior, asd_dataset_path
        # Create minimal transform for inference
        return cls(
            detectors=detectors,
            domain=domain,
            ref_time=ref_time,
            asd_dataset_path=None,  # Not needed for inference
            extrinsic_prior={},  # Not needed for inference
            inference_parameters=inference_parameters,
            context_parameters=context_parameters,
            standardization=standardization,
            random_strain_cropping=None,
            gnpe_time_shifts=None,
            zero_noise=True,  # No noise in inference
            domain_update=data_settings.get("domain_update", None),
        )

    # Public iterator methods
    def get_training_iterator(
        self,
        waveform_dataset: Any,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
    ) -> torch.utils.data.DataLoader:
        """
        Return DataLoader configured for training.
        Sets waveform_dataset.transform to training transform chain.

        Parameters
        ----------
        waveform_dataset : WaveformDataset
            Dataset to iterate over
        batch_size : int
            Batch size for DataLoader
        num_workers : int
            Number of worker processes
        shuffle : bool
            Whether to shuffle data
        pin_memory : bool
            Whether to pin memory for faster GPU transfer

        Returns
        -------
        torch.utils.data.DataLoader
            Configured DataLoader for training
        """
        from dingo.core.utils.torchutils import fix_random_seeds

        # Build training transform if not already built
        if self._training_transform is None:
            self._training_transform = self._build_training_transform()

        # Set the transform on the dataset
        waveform_dataset.transform = self._training_transform

        # Create and return DataLoader
        return torch.utils.data.DataLoader(
            waveform_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            worker_init_fn=fix_random_seeds,
        )

    def get_svd_iterator(
        self,
        waveform_dataset: Any,
        batch_size: int,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        """
        Return DataLoader configured for SVD generation.
        Sets waveform_dataset.transform to SVD transform chain (no noise/repackaging).

        Parameters
        ----------
        waveform_dataset : WaveformDataset
            Dataset to iterate over
        batch_size : int
            Batch size for DataLoader
        num_workers : int
            Number of worker processes

        Returns
        -------
        torch.utils.data.DataLoader
            Configured DataLoader for SVD generation
        """
        from dingo.core.utils.torchutils import fix_random_seeds

        # Build SVD transform if not already built
        if self._svd_transform is None:
            self._svd_transform = self._build_svd_transform()

        # Set the transform on the dataset
        waveform_dataset.transform = self._svd_transform

        # Create and return DataLoader (no shuffling for SVD)
        return torch.utils.data.DataLoader(
            waveform_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=fix_random_seeds,
        )

    def get_inference_transform_pre(self) -> Compose:
        """
        Return transform chain applied before sampling (data preparation).
        Used in GWSampler for preparing event data.

        Returns
        -------
        Compose
            Transform chain for pre-sampling data preparation
        """
        # Build inference transform pre if not already built
        if self._inference_transform_pre is None:
            self._inference_transform_pre = self._build_inference_transform_pre()

        return self._inference_transform_pre

    def get_inference_transform_post(self) -> Callable:
        """
        Return transform applied after sampling (post-processing).
        Used in GWSampler for de-standardization and corrections.

        Returns
        -------
        Callable
            Transform for post-sampling processing
        """
        # Build inference transform post if not already built
        if self._inference_transform_post is None:
            self._inference_transform_post = self._build_inference_transform_post()

        return self._inference_transform_post

    # Public properties (read-only access to config)
    @property
    def detectors(self) -> List[str]:
        """List of detector names."""
        return self._config.detectors

    @property
    def domain(self) -> Any:
        """Frequency domain object."""
        return self._config.domain

    @property
    def ref_time(self) -> float:
        """Reference GPS time."""
        return self._config.ref_time

    @property
    def inference_parameters(self) -> List[str]:
        """List of inference parameter names."""
        return self._config.inference_parameters

    @property
    def context_parameters(self) -> List[str]:
        """List of context parameter names (for GNPE)."""
        return self._config.context_parameters

    @property
    def standardization(self) -> Dict[str, Dict[str, float]]:
        """
        Parameter standardization dictionary (mean/std).

        Returns dict format for backward compatibility with existing code.
        """
        return {
            "mean": self._config.standardization.mean,
            "std": self._config.standardization.std
        }

    @property
    def config(self) -> TransformConfig:
        """Return immutable config object."""
        return self._config

    # Private methods (transform chain builders)
    def _build_training_transform(self) -> Compose:
        """
        Build full training transform chain:
        1. SampleExtrinsicParameters
        2. GetDetectorTimes
        3. (optional) GNPECoalescenceTimes
        4. ProjectOntoDetectors
        5. SampleNoiseASD
        6. WhitenAndScaleStrain
        7. (optional) AddWhiteNoiseComplex
        8. SelectStandardizeRepackageParameters
        9. RepackageStrainsAndASDS
        10. (optional) CropMaskStrainRandom
        11. UnpackDict

        Returns
        -------
        Compose
            Training transform chain
        """
        # Import transform classes from current package
        from . import (
            SampleExtrinsicParameters,
            GetDetectorTimes,
            GNPECoalescenceTimes,
            ProjectOntoDetectors,
            SampleNoiseASD,
            WhitenAndScaleStrain,
            AddWhiteNoiseComplex,
            SelectStandardizeRepackageParameters,
            RepackageStrainsAndASDS,
            CropMaskStrainRandom,
            UnpackDict,
        )
        from dingo.gw.prior import get_extrinsic_prior_dict

        # Get interferometer list
        ifo_list = self._get_ifo_list()

        # Load ASD dataset
        asd_dataset = self._load_asd_dataset()

        # Get extrinsic prior (convert to dict for compatibility with dingo.gw.prior)
        extrinsic_prior_dict = get_extrinsic_prior_dict(self._config.extrinsic_prior.to_dict())

        # Build transform chain
        transforms = [
            SampleExtrinsicParameters(extrinsic_prior_dict),
            GetDetectorTimes(ifo_list, self._config.ref_time),
        ]

        # Add GNPE if configured
        if self._config.gnpe_time_shifts is not None:
            gnpe_config = self._config.gnpe_time_shifts
            transforms.append(
                GNPECoalescenceTimes(
                    ifo_list,
                    gnpe_config.kernel,
                    gnpe_config.exact_equiv,
                    inference=False,
                )
            )

        # Continue building transform chain
        transforms.extend([
            ProjectOntoDetectors(ifo_list, self._config.domain, self._config.ref_time),
            SampleNoiseASD(asd_dataset),
            WhitenAndScaleStrain(self._config.domain.noise_std),
        ])

        # Add noise unless zero_noise is True
        if not self._config.zero_noise:
            transforms.append(AddWhiteNoiseComplex())

        # Add parameter standardization and repackaging
        transforms.extend([
            SelectStandardizeRepackageParameters(
                {
                    "inference_parameters": self._config.inference_parameters,
                    "context_parameters": self._config.context_parameters,
                },
                {
                    "mean": self._config.standardization.mean,
                    "std": self._config.standardization.std,
                },
            ),
            RepackageStrainsAndASDS(
                self._config.detectors,
                first_index=self._config.domain.min_idx,
            ),
        ])

        # Add random strain cropping if configured
        if self._config.random_strain_cropping is not None:
            transforms.append(
                CropMaskStrainRandom(
                    self._config.domain,
                    **self._config.random_strain_cropping.to_dict(),
                )
            )

        # Determine selected keys for UnpackDict
        if self._config.context_parameters:
            selected_keys = ["inference_parameters", "waveform", "context_parameters"]
        else:
            selected_keys = ["inference_parameters", "waveform"]

        transforms.append(UnpackDict(selected_keys=selected_keys))

        return Compose(transforms)

    def _build_svd_transform(self) -> Compose:
        """
        Build SVD transform chain (omits noise, repackaging, parameter transforms):
        1. SampleExtrinsicParameters
        2. GetDetectorTimes
        3. (optional) GNPECoalescenceTimes
        4. ProjectOntoDetectors
        5. SampleNoiseASD
        6. WhitenAndScaleStrain

        Returns
        -------
        Compose
            SVD transform chain
        """
        # Import transform classes from current package
        from . import (
            SampleExtrinsicParameters,
            GetDetectorTimes,
            GNPECoalescenceTimes,
            ProjectOntoDetectors,
            SampleNoiseASD,
            WhitenAndScaleStrain,
        )
        from dingo.gw.prior import get_extrinsic_prior_dict

        # Get interferometer list
        ifo_list = self._get_ifo_list()

        # Load ASD dataset
        asd_dataset = self._load_asd_dataset()

        # Get extrinsic prior (convert to dict for compatibility with dingo.gw.prior)
        extrinsic_prior_dict = get_extrinsic_prior_dict(self._config.extrinsic_prior.to_dict())

        # Build transform chain (similar to training but without noise/repackaging)
        transforms = [
            SampleExtrinsicParameters(extrinsic_prior_dict),
            GetDetectorTimes(ifo_list, self._config.ref_time),
        ]

        # Add GNPE if configured
        if self._config.gnpe_time_shifts is not None:
            gnpe_config = self._config.gnpe_time_shifts
            transforms.append(
                GNPECoalescenceTimes(
                    ifo_list,
                    gnpe_config.kernel,
                    gnpe_config.exact_equiv,
                    inference=False,
                )
            )

        # Continue building transform chain (stop before noise/parameter repackaging)
        transforms.extend([
            ProjectOntoDetectors(ifo_list, self._config.domain, self._config.ref_time),
            SampleNoiseASD(asd_dataset),
            WhitenAndScaleStrain(self._config.domain.noise_std),
        ])

        return Compose(transforms)

    def _build_inference_transform_pre(self) -> Compose:
        """
        Build inference pre-transform chain (data preparation):
        1. (optional) DecimateWaveformsAndASDS
        2. WhitenAndScaleStrain
        3. (optional) MaskDataForFrequencyRangeUpdate
        4. RepackageStrainsAndASDS
        5. ToTorch

        Returns
        -------
        Compose
            Inference pre-transform chain
        """
        from . import (
            DecimateWaveformsAndASDS,
            WhitenAndScaleStrain,
            MaskDataForFrequencyRangeUpdate,
            RepackageStrainsAndASDS,
            ToTorch,
        )
        from dingo.gw.domains import MultibandedFrequencyDomain

        transforms = []

        # Add decimation if using MultibandedFrequencyDomain
        if isinstance(self._config.domain, MultibandedFrequencyDomain):
            transforms.append(
                DecimateWaveformsAndASDS(
                    self._config.domain,
                    decimation_mode="whitened",
                )
            )

        # Always whiten and scale
        transforms.append(WhitenAndScaleStrain(self._config.domain.noise_std))

        # Add frequency range masking if domain_update is specified
        if self._config.domain_update is not None:
            domain_update_config = self._config.domain_update
            minimum_frequency = domain_update_config.f_min
            maximum_frequency = domain_update_config.f_max
            if minimum_frequency is not None or maximum_frequency is not None:
                transforms.append(
                    MaskDataForFrequencyRangeUpdate(
                        self._config.domain,
                        minimum_frequency=minimum_frequency,
                        maximum_frequency=maximum_frequency,
                    )
                )

        # Repackage strains and ASDs
        transforms.append(
            RepackageStrainsAndASDS(
                self._config.detectors,
                first_index=self._config.domain.min_idx,
            )
        )

        # Convert to torch tensors
        transforms.append(ToTorch(device="cpu"))

        return Compose(transforms)

    def _build_inference_transform_post(self) -> Callable:
        """
        Build inference post-transform (de-standardization):
        - SelectStandardizeRepackageParameters(inverse=True)

        Returns
        -------
        Callable
            Inference post-transform
        """
        from . import SelectStandardizeRepackageParameters

        # Create inverse transform for de-standardization
        return SelectStandardizeRepackageParameters(
            {"inference_parameters": self._config.inference_parameters},
            {
                "mean": self._config.standardization.mean,
                "std": self._config.standardization.std,
            },
            inverse=True,
            as_type="dict",
        )

    def _get_ifo_list(self) -> InterferometerList:
        """
        Build and cache InterferometerList from detectors.

        Returns
        -------
        InterferometerList
            List of interferometer objects
        """
        if self._ifo_list is None:
            self._ifo_list = InterferometerList(self._config.detectors)
        return self._ifo_list

    def _load_asd_dataset(self) -> Any:
        """
        Load ASD dataset from path with proper domain matching.

        Returns
        -------
        ASDDataset
            Loaded ASD dataset
        """
        from dingo.gw.noise.asd_dataset import ASDDataset

        if self._config.asd_dataset_path is None:
            raise ValueError("asd_dataset_path is None, cannot load ASD dataset")

        # Load ASD dataset with domain matching
        asd_dataset = ASDDataset(
            self._config.asd_dataset_path,
            ifos=self._config.detectors,
            precision="single",
            domain_update=self._config.domain.domain_dict,
        )

        # Verify domain matches
        if asd_dataset.domain != self._config.domain:
            raise ValueError(
                f"ASD dataset domain {asd_dataset.domain} does not match "
                f"transform domain {self._config.domain}"
            )

        return asd_dataset
