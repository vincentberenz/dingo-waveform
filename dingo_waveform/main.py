import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .domains import DomainParameters
from .prior import IntrinsicPriors
from .waveform_generator import WaveformGeneratorParameters


@dataclass
class DatasetParameters:
    domain: DomainParameters
    waveform_generator: WaveformGeneratorParameters
    intrinsic_priors: IntrinsicPriors
    num_samples: int = 1000
    compression: Optional[Compression] = None


@dataclass
class DatasetArgs:
    settings_file: Path
    num_processes: int
    outfile: Path


def _parse_args():

    parser = (
        argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Generate a waveform dataset based on a settings file.",
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel waveform generation",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="waveform_dataset.hdf5",
        help="Name of file for storing dataset.",
    )

    settings_file = Path(parser.setting_files)
    if not settings_file.is_file():
        raise FileNotFoundError(
            f"Can not generate waveform from file {settings_file}: file not found"
        )

    outfile = Path.cwd() / parser.out_file

    return DatasetArgs(settings_file, parser.num_processes, parser.out_file, outfile)
