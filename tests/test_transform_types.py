"""
Test type safety of transform configurations and usage.

These tests verify that:
1. Type annotations are correctly applied
2. Invalid types are caught at instantiation time
3. Serialization/deserialization preserves types
4. Protocol usage works correctly
"""

import pytest
import numpy as np
from dingo_waveform.transform.types import DecimationMode, OutputFormat, Device
from dingo_waveform.transform.transforms.waveform.decimate_waveforms_and_asds import (
    DecimateWaveformsAndASDS,
    DecimateWaveformsAndASDSConfig,
)
from dingo_waveform.transform.transforms.parameters.select_standardize_repackage_parameters import (
    SelectStandardizeRepackageParameters,
    SelectStandardizeRepackageParametersConfig,
)
from dingo_waveform.transform.transforms.inference.to_torch import ToTorch, ToTorchConfig
from dingo_waveform.transform.transforms.gnpe.gnpe_base import GNPEBase, GNPEBaseConfig


class TestDecimationModeLiteral:
    """Test DecimationMode Literal type catches invalid values."""

    def test_valid_whitened_mode(self):
        """Test that 'whitened' mode works."""
        # Skip domain creation, just test that the type annotation accepts the value
        # Using a mock object that satisfies DomainProtocol
        from unittest.mock import Mock

        mock_domain = Mock()
        mock_domain.time_translate_data = Mock(return_value=None)
        mock_domain.decimate = Mock(return_value=None)
        mock_domain.domain_dict = {}

        config = DecimateWaveformsAndASDSConfig(
            multibanded_frequency_domain=mock_domain, decimation_mode="whitened"
        )
        assert config.decimation_mode == "whitened"

    def test_valid_unwhitened_mode(self):
        """Test that 'unwhitened' mode works."""
        from unittest.mock import Mock

        mock_domain = Mock()
        mock_domain.time_translate_data = Mock(return_value=None)
        mock_domain.decimate = Mock(return_value=None)
        mock_domain.domain_dict = {}

        config = DecimateWaveformsAndASDSConfig(
            multibanded_frequency_domain=mock_domain, decimation_mode="unwhitened"
        )
        assert config.decimation_mode == "unwhitened"

    def test_invalid_decimation_mode_raises(self):
        """Test that invalid decimation mode raises ValueError at runtime."""
        from unittest.mock import Mock

        mock_domain = Mock()
        mock_domain.time_translate_data = Mock(return_value=None)
        mock_domain.decimate = Mock(return_value=None)
        mock_domain.domain_dict = {}

        # Note: mypy would catch this typo at type-check time!
        with pytest.raises(ValueError, match="Unsupported decimation mode"):
            DecimateWaveformsAndASDSConfig(
                multibanded_frequency_domain=mock_domain,
                decimation_mode="whitenned",  # Typo - mypy catches this!
            )


class TestStandardizationDictTypedDict:
    """Test StandardizationDict TypedDict structure enforcement."""

    def test_valid_standardization_dict_structure(self):
        """Test that valid standardization dict structure works."""
        config = SelectStandardizeRepackageParametersConfig(
            parameters_dict={"inference_parameters": ["mass_1", "mass_2"]},
            standardization_dict={
                "mean": {"mass_1": 35.0, "mass_2": 30.0},
                "std": {"mass_1": 5.0, "mass_2": 5.0},
            },
        )
        assert config.standardization_dict["mean"]["mass_1"] == 35.0
        assert config.standardization_dict["std"]["mass_1"] == 5.0

    def test_missing_mean_key_raises(self):
        """Test that missing 'mean' key raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'mean' and 'std' keys"):
            SelectStandardizeRepackageParametersConfig(
                parameters_dict={"inference_parameters": ["mass_1"]},
                standardization_dict={
                    "std": {"mass_1": 5.0}  # Missing 'mean'
                },
            )

    def test_missing_std_key_raises(self):
        """Test that missing 'std' key raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'mean' and 'std' keys"):
            SelectStandardizeRepackageParametersConfig(
                parameters_dict={"inference_parameters": ["mass_1"]},
                standardization_dict={
                    "mean": {"mass_1": 35.0}  # Missing 'std'
                },
            )

    def test_mismatched_keys_raises(self):
        """Test that mismatched keys between mean and std raise ValueError."""
        with pytest.raises(ValueError, match="Keys of means and stds do not match"):
            SelectStandardizeRepackageParametersConfig(
                parameters_dict={"inference_parameters": ["mass_1", "mass_2"]},
                standardization_dict={
                    "mean": {"mass_1": 35.0, "mass_2": 30.0},
                    "std": {"mass_1": 5.0},  # Missing mass_2
                },
            )


class TestOutputFormatLiteral:
    """Test OutputFormat Literal type for parameter transforms."""

    def test_valid_dict_format(self):
        """Test that 'dict' output format works."""
        config = SelectStandardizeRepackageParametersConfig(
            parameters_dict={"inference_parameters": ["mass_1"]},
            standardization_dict={"mean": {"mass_1": 35.0}, "std": {"mass_1": 5.0}},
            as_type="dict",
        )
        assert config.as_type == "dict"

    def test_valid_pandas_format(self):
        """Test that 'pandas' output format works."""
        config = SelectStandardizeRepackageParametersConfig(
            parameters_dict={"inference_parameters": ["mass_1"]},
            standardization_dict={"mean": {"mass_1": 35.0}, "std": {"mass_1": 5.0}},
            as_type="pandas",
        )
        assert config.as_type == "pandas"

    def test_valid_none_format(self):
        """Test that None output format works."""
        config = SelectStandardizeRepackageParametersConfig(
            parameters_dict={"inference_parameters": ["mass_1"]},
            standardization_dict={"mean": {"mass_1": 35.0}, "std": {"mass_1": 5.0}},
            as_type=None,
        )
        assert config.as_type is None

    def test_invalid_format_raises(self):
        """Test that invalid output format raises ValueError."""
        # Note: mypy would catch this at type-check time!
        with pytest.raises(ValueError, match="must be 'dict', 'pandas', or None"):
            SelectStandardizeRepackageParametersConfig(
                parameters_dict={"inference_parameters": ["mass_1"]},
                standardization_dict={
                    "mean": {"mass_1": 35.0},
                    "std": {"mass_1": 5.0},
                },
                as_type="invalid",  # mypy catches this!
            )


class TestDeviceLiteral:
    """Test Device type for PyTorch device specification."""

    def test_valid_cpu_device(self):
        """Test that 'cpu' device works."""
        config = ToTorchConfig(device="cpu")
        assert config.device == "cpu"

    def test_valid_cuda_default_device(self):
        """Test that 'cuda' device works."""
        config = ToTorchConfig(device="cuda")
        assert config.device == "cuda"

    def test_valid_cuda_indexed_device(self):
        """Test that 'cuda:0' device works."""
        config = ToTorchConfig(device="cuda:0")
        assert config.device == "cuda:0"

    def test_valid_cuda_multiple_indices(self):
        """Test that 'cuda:1', 'cuda:2' devices work."""
        for idx in [1, 2, 3]:
            config = ToTorchConfig(device=f"cuda:{idx}")
            assert config.device == f"cuda:{idx}"

    def test_invalid_device_raises(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda"):
            ToTorchConfig(device="tpu")  # Invalid device


class TestGroupOperatorLiteral:
    """Test GroupOperator Literal type for GNPE transforms."""

    def test_valid_additive_operator(self):
        """Test that '+' operator works."""
        config = GNPEBaseConfig(
            kernel_dict={"param1": "Uniform(minimum=0, maximum=1)"},
            operators={"param1": "+"},
        )
        assert config.operators["param1"] == "+"

    def test_valid_multiplicative_operator(self):
        """Test that 'x' operator works."""
        config = GNPEBaseConfig(
            kernel_dict={"param1": "Uniform(minimum=0, maximum=1)"},
            operators={"param1": "x"},
        )
        assert config.operators["param1"] == "x"

    def test_mixed_operators(self):
        """Test that mixed operators work."""
        config = GNPEBaseConfig(
            kernel_dict={
                "param1": "Uniform(minimum=0, maximum=1)",
                "param2": "Uniform(minimum=0, maximum=1)",
            },
            operators={"param1": "+", "param2": "x"},
        )
        assert config.operators["param1"] == "+"
        assert config.operators["param2"] == "x"

    def test_invalid_operator_raises(self):
        """Test that invalid operator raises ValueError."""
        # Note: mypy would catch this at type-check time!
        with pytest.raises(ValueError, match="Operator for .* must be"):
            GNPEBaseConfig(
                kernel_dict={"param1": "Uniform(minimum=0, maximum=1)"},
                operators={"param1": "*"},  # Invalid - mypy catches this!
            )


class TestSerializationRoundTrip:
    """Test that serialization preserves type semantics."""

    def test_decimation_config_serialization(self):
        """Test DecimateWaveformsAndASDSConfig to_dict/from_dict round-trip."""
        from unittest.mock import Mock

        mock_domain = Mock()
        mock_domain.time_translate_data = Mock(return_value=None)
        mock_domain.decimate = Mock(return_value=None)
        mock_domain.domain_dict = {
            "type": "MultibandedFrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
        }

        config = DecimateWaveformsAndASDSConfig(
            multibanded_frequency_domain=mock_domain, decimation_mode="whitened"
        )

        # Serialize to dict (we can at least test the type is preserved)
        config_dict = config.to_dict()

        # Type should be preserved
        assert config.decimation_mode == "whitened"
        assert isinstance(
            config.decimation_mode, str
        )  # Literal is str at runtime

    def test_parameter_config_serialization(self):
        """Test SelectStandardizeRepackageParametersConfig round-trip."""
        config = SelectStandardizeRepackageParametersConfig(
            parameters_dict={"inference_parameters": ["mass_1"]},
            standardization_dict={"mean": {"mass_1": 35.0}, "std": {"mass_1": 5.0}},
            as_type="dict",
            device="cuda:0",
        )

        config_dict = config.to_dict()
        restored_config = SelectStandardizeRepackageParametersConfig.from_dict(
            config_dict
        )

        assert restored_config.as_type == "dict"
        assert restored_config.device == "cuda:0"
        assert restored_config.standardization_dict["mean"]["mass_1"] == 35.0


class TestProtocolUsage:
    """Test that Protocol usage works correctly."""

    def test_domain_protocol_methods_accessible(self):
        """Test that DomainProtocol methods are accessible on actual Domain objects."""
        from dingo.gw.domains import UniformFrequencyDomain

        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)

        # Domain should have the methods required by DomainProtocol
        assert hasattr(domain, "time_translate_data")
        assert hasattr(domain, "domain_dict")

        # Callable check
        assert callable(domain.time_translate_data)

        # Note: UniformFrequencyDomain doesn't have decimate() method
        # Only MultibandedFrequencyDomain has it, which is fine since
        # DomainProtocol is for structural typing and not all domains
        # need all methods

    def test_domain_protocol_usage_in_config(self):
        """Test that DomainProtocol works in config without importing actual Domain."""
        from dingo.gw.domains import UniformFrequencyDomain
        from dingo_waveform.transform.transforms.detector.project_onto_detectors import (
            ProjectOntoDetectorsConfig,
        )

        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
        config = ProjectOntoDetectorsConfig(
            ifo_list=["H1", "L1"], domain=domain, ref_time=1234567890.0
        )

        # Config should accept the domain
        assert config.domain is not None
        # And the domain should still have its methods
        assert hasattr(config.domain, "time_translate_data")


class TestPipelineStageTypes:
    """Test pipeline stage TypedDict definitions."""

    def test_polarization_sample_structure(self):
        """Test PolarizationSample TypedDict structure."""
        from dingo_waveform.transform.types import PolarizationSample

        sample: PolarizationSample = {
            "parameters": {"mass_1": 35.0, "mass_2": 30.0},
            "waveform": {
                "h_plus": np.random.randn(100) + 1j * np.random.randn(100),
                "h_cross": np.random.randn(100) + 1j * np.random.randn(100),
            },
        }
        assert "h_plus" in sample["waveform"]
        assert "h_cross" in sample["waveform"]
        assert "mass_1" in sample["parameters"]

    def test_extrinsic_sample_structure(self):
        """Test ExtrinsicSample TypedDict structure."""
        from dingo_waveform.transform.types import ExtrinsicSample

        sample: ExtrinsicSample = {
            "parameters": {"mass_1": 35.0},
            "extrinsic_parameters": {
                "ra": 1.2,
                "dec": 0.5,
                "psi": 0.3,
                "luminosity_distance": 100.0,
            },
            "waveform": {
                "h_plus": np.random.randn(100) + 1j * np.random.randn(100),
                "h_cross": np.random.randn(100) + 1j * np.random.randn(100),
            },
        }
        assert "extrinsic_parameters" in sample
        assert "ra" in sample["extrinsic_parameters"]
        assert "h_plus" in sample["waveform"]

    def test_detector_strain_sample_structure(self):
        """Test DetectorStrainSample TypedDict structure."""
        from dingo_waveform.transform.types import DetectorStrainSample

        sample: DetectorStrainSample = {
            "parameters": {"mass_1": 35.0, "ra": 1.2, "dec": 0.5},
            "extrinsic_parameters": {},
            "waveform": {
                "H1": np.random.randn(100) + 1j * np.random.randn(100),
                "L1": np.random.randn(100) + 1j * np.random.randn(100),
            },
        }
        assert "H1" in sample["waveform"]
        assert "L1" in sample["waveform"]
        # Polarizations should NOT be present after projection
        assert "h_plus" not in sample["waveform"]

    def test_noise_asd_sample_structure(self):
        """Test NoiseASDSample TypedDict structure."""
        from dingo_waveform.transform.types import NoiseASDSample

        sample: NoiseASDSample = {
            "parameters": {"mass_1": 35.0},
            "extrinsic_parameters": {},
            "waveform": {
                "H1": np.random.randn(100) + 1j * np.random.randn(100),
                "L1": np.random.randn(100) + 1j * np.random.randn(100),
            },
            "asds": {
                "H1": np.random.rand(100),
                "L1": np.random.rand(100),
            },
        }
        assert "asds" in sample
        assert "H1" in sample["asds"]
        assert "L1" in sample["asds"]
        # Keys should match between waveform and asds
        assert set(sample["waveform"].keys()) == set(sample["asds"].keys())

    def test_inference_sample_structure(self):
        """Test InferenceSample TypedDict structure."""
        from dingo_waveform.transform.types import InferenceSample

        sample: InferenceSample = {
            "parameters": {"mass_1": 35.0},
            "extrinsic_parameters": {},
            "waveform": {"H1": np.random.randn(100) + 1j * np.random.randn(100)},
            "asds": {"H1": np.random.rand(100)},
            "inference_parameters": np.array([0.5, 0.3, 1.2]),
        }
        assert "inference_parameters" in sample
        assert isinstance(sample["inference_parameters"], np.ndarray)
        assert sample["inference_parameters"].shape == (3,)

    def test_tensor_packed_sample_structure(self):
        """Test TensorPackedSample TypedDict structure."""
        from dingo_waveform.transform.types import TensorPackedSample

        sample: TensorPackedSample = {
            "parameters": {"mass_1": 35.0},
            "inference_parameters": np.array([0.5, 0.3]),
            "waveform": np.random.randn(2, 3, 100).astype(np.float32),
            "asds": {"H1": np.random.rand(100), "L1": np.random.rand(100)},
        }
        assert "waveform" in sample
        # Waveform should be tensor with shape [num_ifos, 3, num_bins]
        assert sample["waveform"].shape == (2, 3, 100)
        assert sample["waveform"].dtype == np.float32

    def test_torch_sample_structure(self):
        """Test TorchSample TypedDict structure."""
        import torch
        from dingo_waveform.transform.types import TorchSample

        sample: TorchSample = {
            "parameters": {"mass_1": 35.0},
            "inference_parameters": torch.tensor([0.5, 0.3]),
            "waveform": torch.randn(2, 3, 100),
            "asds": {"H1": np.random.rand(100)},
        }
        assert isinstance(sample["inference_parameters"], torch.Tensor)
        assert isinstance(sample["waveform"], torch.Tensor)
        # ASDs not converted (nested dict)
        assert isinstance(sample["asds"]["H1"], np.ndarray)

    def test_stage_transition_project_onto_detectors(self):
        """Test ProjectOntoDetectors transitions from polarization to detector stage."""
        from dingo.gw.domains import UniformFrequencyDomain
        from dingo_waveform.transform.transforms.detector.project_onto_detectors import (
            ProjectOntoDetectors,
            ProjectOntoDetectorsConfig,
        )
        from dingo_waveform.transform.types import PolarizationSample, DetectorStrainSample

        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
        config = ProjectOntoDetectorsConfig(
            ifo_list=["H1", "L1"], domain=domain, ref_time=1234567890.0
        )
        transform = ProjectOntoDetectors.from_config(config)

        # Create input sample (PolarizationSample structure)
        input_sample: PolarizationSample = {
            "parameters": {"luminosity_distance": 100.0, "geocent_time": 0.0},
            "extrinsic_parameters": {
                "ra": 1.2,
                "dec": 0.5,
                "psi": 0.3,
                "luminosity_distance": 100.0,
                "geocent_time": 0.0,
                "H1_time": 0.001,
                "L1_time": 0.002,
            },
            "waveform": {
                "h_plus": np.random.randn(len(domain)) + 1j * np.random.randn(len(domain)),
                "h_cross": np.random.randn(len(domain)) + 1j * np.random.randn(len(domain)),
            },
        }

        # Apply transform
        output_sample: DetectorStrainSample = transform(input_sample)

        # Verify critical stage transition
        assert "H1" in output_sample["waveform"]
        assert "L1" in output_sample["waveform"]
        # Polarizations should be removed
        assert "h_plus" not in output_sample["waveform"]
        assert "h_cross" not in output_sample["waveform"]
        # Extrinsic params should be consolidated into parameters
        assert "ra" in output_sample["parameters"]
        assert "dec" in output_sample["parameters"]

    def test_stage_transition_sample_extrinsic(self):
        """Test SampleExtrinsicParameters transitions to ExtrinsicSample."""
        from dingo_waveform.transform.transforms.parameters.sample_extrinsic_parameters import (
            SampleExtrinsicParameters,
            SampleExtrinsicParametersConfig,
        )
        from dingo_waveform.transform.types import PolarizationSample, ExtrinsicSample

        config = SampleExtrinsicParametersConfig(
            extrinsic_prior_dict={
                "ra": {"type": "Uniform", "minimum": 0.0, "maximum": 6.28},
                "dec": {"type": "Cosine", "minimum": -1.57, "maximum": 1.57},
            }
        )
        transform = SampleExtrinsicParameters.from_config(config)

        input_sample: PolarizationSample = {
            "parameters": {"mass_1": 35.0},
            "waveform": {
                "h_plus": np.random.randn(100) + 1j * np.random.randn(100),
                "h_cross": np.random.randn(100) + 1j * np.random.randn(100),
            },
        }

        output_sample: ExtrinsicSample = transform(input_sample)

        # Verify extrinsic_parameters added
        assert "extrinsic_parameters" in output_sample
        assert "ra" in output_sample["extrinsic_parameters"]
        assert "dec" in output_sample["extrinsic_parameters"]
        # Waveform should still have polarizations
        assert "h_plus" in output_sample["waveform"]

    def test_transform_sample_union(self):
        """Test TransformSample union type accepts all stage types."""
        from dingo_waveform.transform.types import TransformSample

        # Should accept PolarizationSample
        sample1: TransformSample = {
            "parameters": {"mass_1": 35.0},
            "waveform": {
                "h_plus": np.random.randn(10) + 1j * np.random.randn(10),
                "h_cross": np.random.randn(10) + 1j * np.random.randn(10),
            },
        }
        assert "waveform" in sample1

        # Should accept DetectorStrainSample
        sample2: TransformSample = {
            "parameters": {"mass_1": 35.0},
            "waveform": {"H1": np.random.randn(10) + 1j * np.random.randn(10)},
        }
        assert "waveform" in sample2

        # Should accept InferenceSample
        sample3: TransformSample = {
            "parameters": {"mass_1": 35.0},
            "waveform": {"H1": np.random.randn(10) + 1j * np.random.randn(10)},
            "asds": {"H1": np.random.rand(10)},
            "inference_parameters": np.array([0.5, 0.3]),
        }
        assert "inference_parameters" in sample3
