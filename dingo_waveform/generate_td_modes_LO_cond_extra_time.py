# To be used in NewInterfaceWaveformGenerator.generate_hplus_hcross
# (it does *not* override the WaveformGenerator method, but the convert_parameters
# method is different)


import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union

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
from .types import FrequencySeries, GWSignalsGenerators, Modes
from .wfg_utils import td_modes_to_fd_modes

_logger = logging.getLogger(__name__)


@dataclass
class SEOBRNRv5Conditioning:
    f_min: float
    new_f_start: float
    t_extra: float
    original_f_min: float
    f_isco: float

    def taper_td_modes_for_SEOBRNRv5_extra_time(
        self,
        series: Union[gwpy.timeseries.TimeSeries, gwpy.frequencyseries.FrequencySeries],
    ) -> lal.CreateCOMPLEX16TimeSeries:

        # Split in real and imaginary parts, since LAL conditioning routines are for real timeseries
        h_tapered_re = lal.CreateREAL8TimeSeries(
            "h_tapered", series.epoch.value, 0, series.dt.value, None, len(series)
        )
        h_tapered_re.data.data = series.value.copy().real

        h_tapered_im = lal.CreateREAL8TimeSeries(
            "h_tapered_im", series.epoch.value, 0, series.dt.value, None, len(series)
        )
        h_tapered_im.data.data = series.value.copy().imag

        # condition the time domain waveform by tapering in the extra time at the beginning and high-pass filtering above original f_min
        lalsimulation.SimInspiralTDConditionStage1(
            h_tapered_re, h_tapered_im, self.t_extra, self.original_f_min
        )
        # final tapering at the beginning and at the end to remove filter transients
        # waveform should terminate at a frequency >= Schwarzschild ISCO
        # so taper one cycle at this frequency at the end; should not make
        # any difference to IMR waveforms */
        lalsimulation.SimInspiralTDConditionStage2(
            h_tapered_re, h_tapered_im, self.f_min, self.f_isco
        )

        # Construct complex timeseries
        h_return = lal.CreateCOMPLEX16TimeSeries(
            "h_return",
            h_tapered_re.epoch,
            0,
            h_tapered_re.deltaT,
            None,
            h_tapered_re.data.length,
        )

        h_return.data.data = h_tapered_re.data.data + 1j * h_tapered_im.data.data

        # return timeseries
        return h_return


@dataclass
class GenerateTDModesLOConditionalExtraTimeParameters(GwSignalParameters):

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "GenerateTDModesLOConditionalExtraTimeParameters":

        gw_signal_params: (
            GwSignalParameters
        ) = super().from_binary_black_hole_parameters(
            bbh_params,
            domain_params,
            spin_conversion_phase,
            f_start,
        )

        return cls(**asdict(gw_signal_params))

    def get_starting_frequency_for_SEOBRNRv5_conditioning(
        self, extra_time_fraction: float = 0.1, extra_cycles: float = 3.0
    ) -> SEOBRNRv5Conditioning:

        f_min = self.f22_start
        m1 = self.mass1
        m2 = self.mass2
        S1z = self.spin1z
        S2z = self.spin2z
        original_f_min = f_min

        f_isco = 1.0 / (pow(9.0, 1.5) * np.pi * (m1 + m2) * lalsimulation.MTSUN_SI)
        if f_min > f_isco:
            f_min = f_isco

        # upper bound on the chirp time starting at f_min
        tchirp = lalsimulation.SimInspiralChirpTimeBound(
            f_min, m1 * lalsimulation.MSUN_SI, m2 * lalsimulation.MSUN_SI, S1z, S2z
        )
        # upper bound on the final black hole spin */
        spinkerr = lalsimulation.SimInspiralFinalBlackHoleSpinBound(S1z, S2z)
        # upper bound on the final plunge, merger, and ringdown time */
        tmerge = lalsimulation.SimInspiralMergeTimeBound(
            m1 * lalsimulation.MSUN_SI, m2 * lalsimulation.MSUN_SI
        ) + lalsimulation.SimInspiralRingdownTimeBound(
            (m1 + m2) * lalsimulation.MSUN_SI, spinkerr
        )

        # extra time to include for all waveforms to take care of situations where
        # the frequency is close to merger (and is sweeping rapidly):
        # this is a few cycles at the low frequency
        textra = extra_cycles / f_min
        # compute a new lower frequency
        f_start = lalsimulation.SimInspiralChirpStartFrequencyBound(
            (1.0 + extra_time_fraction) * tchirp + tmerge + textra,
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
        )

        f_isco = 1.0 / (pow(6.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)

        return SEOBRNRv5Conditioning(
            f_min=f_min,
            new_f_start=f_start,
            t_extra=extra_time_fraction * tchirp + textra,
            original_f_min=original_f_min,
            f_isco=f_isco,
        )

    def apply(
        self, approximant: Approximant, domain: FrequencyDomain
    ) -> Dict[Modes, FrequencySeries]:
        SEOBRNRv5_conditioning: SEOBRNRv5Conditioning = (
            self.get_starting_frequency_for_SEOBRNRv5_conditioning()
        )
        generator: GWSignalsGenerators = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        params["f22_start"] = SEOBRNRv5_conditioning.new_f_start * astropy.units.Hz
        hlm_td: GravitationalWaveModes = waveform.GenerateFDModes(params, generator)

        key: SpinWeightedSphericalHarmonicMode
        value: Union[gwpy.timeseries.TimeSeries, gwpy.frequencyseries.FrequencySeries]
        hlms_lal: Dict[Modes, lal.CreateCOMPLEX16TimeSeries] = {}

        for key, value in hlm_td.items():
            if type(key) != str:
                modes: Modes = (key.l, key.m)
                hlm_lal: lal.CreateCOMPLEX16TimeSeries = (
                    SEOBRNRv5_conditioning.taper_td_modes_for_SEOBRNRv5_extra_time(
                        value
                    )
                )
                hlms_lal[modes] = hlm_lal

        return td_modes_to_fd_modes(hlm_lal, domain)
