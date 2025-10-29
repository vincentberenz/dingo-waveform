"""Dataset generation and management for gravitational waveforms."""

from .compression_settings import CompressionSettings, SVDSettings
from .dataset_settings import DatasetSettings
from .generate import generate_waveform_dataset
from .polarizations import Polarizations
from .waveform_dataset import WaveformDataset
from .waveform_generator_settings import WaveformGeneratorSettings

__all__ = [
    "CompressionSettings",
    "DatasetSettings",
    "Polarizations",
    "SVDSettings",
    "WaveformDataset",
    "WaveformGeneratorSettings",
    "generate_waveform_dataset",
]
