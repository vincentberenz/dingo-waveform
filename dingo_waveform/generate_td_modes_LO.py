# To be used in NewInterfaceWaveformGenerator.generate_hplus_hcross
# (it does *not* override the WaveformGenerator method, but the convert_parameters
# method is different)


import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union, cast

import astropy.units
import gwpy
import gwpy.frequencyseries
import lal
import lalsimulation
import numpy as np
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.core.gw import (
    GravitationalWaveModes,
    SpinWeightedSphericalHarmonicMode,
)
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

from .approximant import Approximant
from .binary_black_holes import BinaryBlackHoleParameters
from .domains import DomainParameters, FrequencyDomain
from .gw_signals import GwSignalParameters
from .polarizations import Polarization, get_polarizations_from_fd_modes_m
from .types import FrequencySeries, GWSignalsGenerators, Iota, Mode, Modes
from .waveform_generator_parameters import WaveformGeneratorParameters
from .waveform_parameters import WaveformParameters
from .wfg_utils import taper_td_modes_in_place, td_modes_to_fd_modes

_logger = logging.getLogger(__name__)


@dataclass
class _GenerateTDModesLO(GwSignalParameters):

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GenerateTDModesLO":

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
        self, approximant: Approximant, domain: FrequencyDomain, phase: float
    ) -> Dict[Mode, Polarization]:

        generator: GWSignalsGenerators = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}

        key: SpinWeightedSphericalHarmonicMode
        value: Union[gwpy.timeseries.TimeSeries, gwpy.frequencyseries.FrequencySeries]
        hlm_td: GravitationalWaveModes = waveform.GenerateFDModes(params, generator)

        hlms_lal: Dict[Modes, lal.CreateCOMPLEX16TimeSeries] = {}

        for key, value in hlm_td.items():
            if type(key) != str:
                hlm_lal = lal.CreateCOMPLEX16TimeSeries(
                    "hplus",
                    value.epoch.value,
                    0,
                    value.dt.value,
                    lal.DimensionlessUnit,
                    len(value),
                )
                hlm_lal.data.data = value.value
                hlms_lal[key] = hlm_lal

        taper_td_modes_in_place(hlm_td)

        h: Dict[Modes, FrequencySeries] = td_modes_to_fd_modes(hlm_td, domain)

        return get_polarizations_from_fd_modes_m(h, Iota(self.inclination), phase)


def generate_TD_modes_LO(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: WaveformParameters,
    approximant: Approximant,
) -> Dict[Mode, Polarization]:

    if not isinstance(waveform_gen_params.domain, FrequencyDomain):
        raise ValueError(
            "generate_FD_modes can only be applied using on a FrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    if waveform_params.phase is None:
        raise ValueError(f"generate_FD_modes_LO: phase parameter should not be None")

    instance = cast(
        _GenerateTDModesLO,
        _GenerateTDModesLO.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            waveform_gen_params.f_start,
            waveform_gen_params.convert_to_SI,
        ),
    )

    return instance.apply(
        approximant, waveform_gen_params.domain, waveform_params.phase
    )
