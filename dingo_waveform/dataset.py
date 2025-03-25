from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from pathlib import Path
import tomli
import yaml
from .domains import build_domain, Domain
from .prior import IntrinsicPriors
from .waveform_generator import WaveformGenerator, build_waveform_generator


class Compression(Enum):
    NO_COMPRESSION = 0
    SVD = 1
    WHITENING = 2


@dataclass
class DatasetParameters:
    waveform_generator: WaveformGenerator
    intrinsic_priors: IntrinsicPriors
    num_samples: int
    compression: Compression
    
    @classmethod
    def build(
        cls, config: Mapping, default_num_samples: int = 1000
    )->"DatasetParameters":
        required_keys = ('domain', 'waveform_generator', 'intrinsic_priors')
        optional_keys = ('num_samples', 'compression')
        
        # Check for required keys
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in config: {', '.join(missing_keys)}"
            )
            
        # Check for unknown keys
        allowed_keys = set(required_keys + optional_keys)
        unknown_keys = [key for key in config if key not in allowed_keys]
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config: {', '.join(unknown_keys)}. "
                f"Allowed keys are: {', '.join(allowed_keys)}"
            )

        domain: Domain = build_domain(config['domain'])
        
        waveform_generator: WaveformGenerator = build_waveform_generator(
            config['waveform_generator'],domain
        )
                       
        intrinsic_priors = IntrinsicPriors(**config['intrinsic_priors'])

        if 'compression' in config.keys():
            try:
                compression = Compression[config['compression'].lower()]
            except KeyError as e:
                raise ValueError(
                    f"Invalid compression type '{config['compression']}'. "
                    f"Must be one of: {', '.join(c.name for c in Compression)}"
                ) from e
        else:
            compression = Compression.NO_COMPRESSION

        try:
            num_samples = config['num_samples']
        except KeyError:
            num_samples = default_num_samples

        return cls(
            waveform_generator=waveform_generator,
            intrinsic_priors=intrinsic_priors,
            num_samples=num_samples,
            compression=compression
        )  
        
    @classmethod
    def build_from_file(cls, filepath: Path) -> 'DatasetParameters':
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        if filepath.suffix.lower in ('.toml','tml'):
            with open(filepath, 'rb') as f:
                config = tomli.load(f)
        elif filepath.suffix.lower in ('.yaml', '.yml'):
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(
                "Unsupported file extension: "
                f"{filepath.suffix}. Must be .toml, .tml, .yaml or .yml"
            )
    
        return cls.build(config)     


