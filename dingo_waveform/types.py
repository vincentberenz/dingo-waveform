from typing import Callable, Dict, NewType, Tuple, TypeAlias, Union

from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.models import pyseobnr_model
from nptyping import Complex128, NDArray, Shape

Iota = NewType("Iota", float)
"""
Type for iota (inclination angle between the binary's orbital angular momentum and the line of sight)
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
Gravitational wave mode
"""

Modes: TypeAlias = Tuple[Mode, Mode]
"""
Tuple of two modes (the degree of the spherical harmonic mode and the order of the spherical harmonic mode)
"""

GWSignalsGenerators = Union[
    pyseobnr_model.SEOBNRv5HM,
    pyseobnr_model.SEOBNRv5EHM,
    pyseobnr_model.SEOBNRv5PHM,
    waveform.LALCompactBinaryCoalescenceGenerator,
]
"""
Return type of the lalsimulation method gwsignal_get_waveform_generator
"""


class WaveformGenerationError(Exception):
    """
    To be raised when generation of gravitational waveform fails.
    """

    ...
