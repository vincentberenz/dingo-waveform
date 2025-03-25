from dataclasses import dataclass
from enum import Enum

from dingo_waveform.domains import DomainParameters


class Compression(Enum):
    svd = 0
    whitening = 1

@dataclass
class DatasetSettings:
    ...
