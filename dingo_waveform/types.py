from dataclasses import dataclass
from typing import NewType, Tuple, TypeAlias, Union

import lalsimulation as LS
import numpy as np
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.models import (
    gwsignal_get_waveform_generator,
    pyseobnr_model,
)
from nptyping import Complex128, NDArray, Shape

Iota = NewType("Iota", float)
"""
Type for iota
"""

F_ref = NewType("F_ref", float)
"""
Type for frequency reference
"""

FrequencySeries: TypeAlias = NDArray[Shape["*"], Complex128]
"""
Waveform frequency series, i.e. a one dimentional numpy array of complex type.
"""

Mode = NewType("Mode", int)
"""
Gravitational wave more
"""

Modes: TypeAlias = Tuple[Mode, Mode]
"""
Tuple of two modes.
"""

GWSignalsGenerators = Union[
    pyseobnr_model.SEOBNRv5HM,
    pyseobnr_model.SEOBNRv5EHM,
    pyseobnr_model.SEOBNRv5PHM,
    waveform.LALCompactBinaryCoalescenceGenerator,
]


class WaveformGenerationError(Exception): ...
