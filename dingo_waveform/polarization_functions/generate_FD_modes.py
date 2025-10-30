import logging
import sys
from dataclasses import asdict, dataclass
from math import isclose
from typing import Optional, Union, cast

import numpy as np
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations
from lalsimulation.gwsignal.models import (
    gwsignal_get_waveform_generator,
    pyseobnr_model,
)

from ..approximant import Approximant
from ..binary_black_holes_parameters import BinaryBlackHoleParameters
from ..domains import DomainParameters, BaseFrequencyDomain
from ..gw_signals_parameters import GwSignalParameters
from ..polarizations import Polarization
from ..types import WaveformGenerationError
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)

_Generators = Union[
    pyseobnr_model.SEOBNRv5HM,
    pyseobnr_model.SEOBNRv5EHM,
    pyseobnr_model.SEOBNRv5PHM,
    waveform.LALCompactBinaryCoalescenceGenerator,
]


@dataclass
class _GenerateFDModesParameters(GwSignalParameters):
    # For the list of Fields, see the superclass GwSignalParameter.
    #
    # This class is private to this module, see generate_FD_modes
    # at the bottom if this file.

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GenerateFDModesParameters":

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
        self, domain: BaseFrequencyDomain, approximant: Approximant, ref_tol
    ) -> Polarization:

        self.lmax_nyquist = 2

        _logger.info(
            self.to_table(
                "generating polarization using "
                "lalsimulation.gwsignal.core.waveform.GenerateFDWaveform"
            )
        )

        if not "pyseobnr" in sys.modules:
            import pyseobnr

        generator: _Generators = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        hpc: GravitationalWavePolarizations = waveform.GenerateFDWaveform(
            params, generator
        )

        hp = hpc.hp
        hc = hpc.hc

        # Ensure that the waveform agrees with the frequency grid defined in the domain.
        if not isclose(self.deltaF.value, hp.df.value, rel_tol=ref_tol):
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


def generate_FD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: WaveformParameters,
    ref_tol: float = 1e-6,
) -> Polarization:
    """
    Wrapper over lalsimulation.gwsignal.core.waveform.GenerateFDModes

    Arguments
    ---------
    waveform_gen_params
      waveform generation configuration
    waveform_params
      waveform configuration

    Returns
    -------
    Polarizations

    Raises
    ------
    ValueError
      if the domain is not a frequency domain or the
      approximant is not supported by lalsimulation GWSignal.
    WaveformGeneratorError
      if the generatate waveform does not 'agree' with the frequency
      grid defined in the domain
    """

    approximant = waveform_gen_params.approximant

    if not isinstance(waveform_gen_params.domain, BaseFrequencyDomain):
        raise ValueError(
            "generate_FD_modes can only be applied using on a BaseFrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    # note: from_waveform_parameters is implemented by the superclass
    #  of _GenerateFDModesParameters (GwSignalParameters).
    instance = cast(
        _GenerateFDModesParameters,
        _GenerateFDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            spin_conversion_phase=waveform_gen_params.spin_conversion_phase,
            f_start=waveform_gen_params.f_start,
        ),
    )

    return instance.apply(waveform_gen_params.domain, approximant, ref_tol)
