"""Dataset generation and management for gravitational waveforms."""

from ..polarizations import BatchPolarizations
from .compression_settings import CompressionSettings, SVDSettings
from .dataset_settings import DatasetSettings
from .generate import generate_waveform_dataset
from .sampler import Sampler
from .waveform_dataset import WaveformDataset
from .waveform_generator_settings import WaveformGeneratorSettings

__all__ = [
    "BatchPolarizations",
    "CompressionSettings",
    "DatasetSettings",
    "Sampler",
    "SVDSettings",
    "WaveformDataset",
    "WaveformGeneratorSettings",
    "generate_waveform_dataset",
]
