# To be used in NewInterfaceWaveformGenerator.generate_hplus_hcross
# (it does *not* override the WaveformGenerator method, but the convert_parameters
# method is different)


import logging
from dataclasses import asdict, dataclass
from math import isclose
from typing import Optional, Union

import astropy.units
import numpy as np
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations
from lalsimulation.gwsignal.models import (
    gwsignal_get_waveform_generator,
    pyseobnr_model,
)

from .approximant import Approximant, is_gwsignal_approximant
from .binary_black_holes import BinaryBlackHoleParameters
from .domains import Domain, DomainParameters, FrequencyDomain
from .gw_signals import GwSignalParameters
from .logging import TableStr
from .polarizations import Polarization
from .spins import Spins
from .types import Iota, WaveformGenerationError

_logger = logging.getLogger(__name__)

_Generators = Union[
    pyseobnr_model.SEOBNRv5HM,
    pyseobnr_model.SEOBNRv5EHM,
    pyseobnr_model.SEOBNRv5PHM,
    waveform.LALCompactBinaryCoalescenceGenerator,
]


@dataclass
class GenerateFDModesParameters(GwSignalParameters):
    """Dataclass for storing parameters for
    lal simulation's GenerateFDModes function
    via a generator (lalsimulation 'new interface')
    """

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "GenerateFDModesParameters":

        # note: "from_waveform_parameters" is implemented by the superclass
        # GwSignalParameters

        gw_signal_params: (
            GwSignalParameters
        ) = super().from_binary_black_hole_parameters(
            bbh_params,
            domain_params,
            spin_conversion_phase,
            f_start,
        )

        return cls(**asdict(gw_signal_params))

    def apply(
        self, domain: FrequencyDomain, approximant: Approximant, ref_tol: float = 1e-6
    ) -> Polarization:
        """
        Generate Fourier domain GW polarizations (h_plus, h_cross).

        Wrapper over lalsimulation generator and
        GenerateFDModes.
        """

        if not is_gwsignal_approximant(approximant):
            raise ValueError(
                f"Approximant {approximant} not supported for GenerateFDModesParameters "
                "(not implemented in lalsimulation GWSignal)"
            )

        generator: _Generators = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        hpc: GravitationalWavePolarizations = waveform.GenerateFDModes(
            params, generator
        )

        hp = hpc.hp
        hc = hpc.hc

        # Ensure that the waveform agrees with the frequency grid defined in the domain.
        if not isclose(self.deltaF, hp.df.value, rel_tol=ref_tol):
            raise WaveformGenerationError(
                f"Waveform delta_f is inconsistent with domain: {hp.df.value} vs {self.deltaF}!"
                f"To avoid this, ensure that f_max = {self.f_max} is a power of two"
                "when you are using a native time-domain waveform model."
            )

        h_plus = np.zeros_like((len(domain),), dtype=complex)
        h_cross = np.zeros_like((len(domain),), dtype=complex)

        # Ensure that length of wf agrees with length of domain. Enforce by truncating frequencies beyond f_max
        if len(hp) > len(domain):
            _logger.warning(
                "GWSignal waveform longer than domain's `frequency_array`"
                f"({len(hp)} vs {len(domain)}). Truncating gwsignal array."
            )
            h_plus = hp[: len(h_plus)].value
            h_cross = hc[: len(h_cross)].value
        else:
            h_plus = hp.value
            h_cross = hc.value

        dt = 1 / hp.df.value + hp.epoch.value
        time_shift = np.exp(-1j * 2 * np.pi * dt * domain())
        h_plus *= time_shift
        h_cross *= time_shift

        return Polarization(h_cross=h_cross, h_plus=h_plus)
