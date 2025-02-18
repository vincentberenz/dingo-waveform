from dataclasses import dataclass
from typing import NewType, Tuple, TypeAlias

import lalsimulation as LS
import numpy as np
from nptyping import Complex128, NDArray, Shape

Iota = NewType("Iota", float)

# numpy array of shape (n,) and dtype complex 128
FrequencySeries: TypeAlias = NDArray[Shape["*"], Complex128]
Mode: TypeAlias = Tuple[int, int]


