import logging
from dataclasses import asdict, dataclass
from typing import Optional, cast

import pyseobnr
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

from ..approximant import Approximant
from ..binary_black_holes_parameters import BinaryBlackHoleParameters
from ..domains import DomainParameters
from ..gw_signals_parameters import GwSignalParameters
from ..polarizations import Polarization
from ..types import GWSignalsGenerators
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class _GWSignals_GenerateTDModesParameters(GwSignalParameters):
    # To see the list of fiels: see the superclass GwSignalParameters.
    # The superclass also implement the from_waveform_parameters
    # method.
    # This class is private to this module, see the method
    # generate_TD_modes at the bottom of this file.

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GWSignals_GenerateTDModesParameters":

        gw_signal_params: (
            GwSignalParameters
        ) = super().from_binary_black_hole_parameters(
            bbh_params,
            domain_params,
            spin_conversion_phase,
            f_start,
        )

        return cls(**asdict(gw_signal_params))

    def apply(self, approximant: Approximant) -> Polarization:

        _logger.debug(
            self.to_table(
                "generating polarization using "
                "lalsimulation.gwsignal.core.waveform.GenerateFDModes"
            )
        )

        generator: GWSignalsGenerators = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        hpc = waveform.GenerateTDWaveform(params, generator)
        return Polarization(h_cross=hpc.hp.value, h_plus=hpc.hp.value)


def gwsignals_generate_TD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: WaveformParameters,
) -> Polarization:
    """
    Wrapper over lalsimulation.gwsignal.core.waveform.GenerateTDWaveform

    Arguments
    ---------
    waveform_gen_params
      waveform generation configuration
    waveform_params
      waveform configuration

    Returns
    -------
    Polarizations
    """

    # note: from_waveform_parameters is implemented by the superclass
    #  of _GenerateTDModesParameters (GwSignalParameters).
    instance = cast(
        _GWSignals_GenerateTDModesParameters,
        _GWSignals_GenerateTDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            f_start=waveform_gen_params.f_start,
            spin_conversion_phase=waveform_gen_params.spin_conversion_phase,
        ),
    )

    return instance.apply(waveform_gen_params.approximant)
