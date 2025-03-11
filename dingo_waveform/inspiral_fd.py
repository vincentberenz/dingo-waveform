import logging
from copy import deepcopy
from dataclasses import asdict, astuple, dataclass
from math import isclose
from typing import Dict, Optional, Tuple

import lal
import lalsimulation as LS
import numpy as np
from nptyping import Float32, NDArray, Shape

from . import wfg_utils
from .approximant import Approximant
from .binary_black_holes import BinaryBlackHoleParameters
from .domains import DomainParameters, FrequencyDomain
from .inspiral_choose_fd_modes import InspiralChooseFDModesParameters
from .logging import TableStr
from .polarizations import Polarization
from .spins import Spins
from .types import FrequencySeries, Iota, Mode
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class InspiralFDParameters(TableStr):
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    r: float
    iota: Iota
    phase: float
    longAscNode: float
    eccentricity: float
    meanPerAno: float
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    lal_params: Optional[lal.Dict]
    approximant: Approximant

    def get_spins(self) -> Spins:
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> "InspiralFDParameters":
        # InspiralFDParameters are the same as InspiralChooseFDModesParameters,
        # but with extra ecc attributes all set to zero.
        inspiral_choose_fd_modes_parameters = (
            InspiralChooseFDModesParameters.from_binary_black_hole_parameters(
                bbh_parameters,
                domain_params,
                spin_conversion_phase,
                lal_params,
                approximant,
            )
        )
        d = asdict(inspiral_choose_fd_modes_parameters)
        ecc_attrs = ("longAscNode", "eccentricity", "meanPerAno")
        for attr in ecc_attrs:
            d[attr] = 0
        instance = cls(**d)
        _logger.debug(instance.to_table("generated inspiral fd parameters"))
        return instance

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_parameters: WaveformParameters,
        f_ref: float,
        convert_to_SI: bool,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Optional[Approximant],
    ) -> "InspiralFDParameters":
        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_parameters, f_ref, convert_to_SI
        )
        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            domain_params,
            spin_conversion_phase,
            lal_params,
            approximant,
        )

    def _turn_off_multibanding(
        self,
        hp: lal.COMPLEX16FrequencySeries,
        hc: lal.COMPLEX16FrequencySeries,
        threshold: float,
    ) -> Tuple[lal.COMPLEX16FrequencySeries, lal.COMPLEX16FrequencySeries, bool]:
        """
        Returns either the hp and hc as passed as argument (if no numerical unstability)
        or the 'fixed' hp and hc (calling SimInspiralWaveformParamsInsertPhenomXHMThresholdMband and
        SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband).
        Also returns a boolean, which is:
        - True if the returned value are numerically stable
        - False if this failed and the values are still not numerically stable
        """
        if max(np.max(np.abs(hp.data.data)), np.max(np.abs(hc.data.data))) <= threshold:
            return hp, hc, True
        lal_params = (
            self.lal_params if self.lal_params is not None else lal.CreateDict()
        )
        LS.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lal_params, 0)
        LS.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lal_params, 0)
        params: "InspiralFDParameters" = deepcopy(self)
        params.lal_params = lal_params
        arguments = list(astuple(params))
        hp, hc = LS.SimInspiralFD(*arguments)
        if max(np.max(np.abs(hp.data.data)), np.max(np.abs(hc.data.data))) <= threshold:
            return hp, hc, True
        else:
            return hp, hc, False

    def apply(
        self,
        frequency_array: NDArray[Shape["*"], Float32],
        auto_turn_off_multibanding: bool = True,
        raise_error_on_numerical_unstability: bool = False,
        stability_threshold: float = 1e-20,
        delta_f_tolerance: float = 1e-6,
    ) -> Polarization:

        hp: lal.COMPLEX16FrequencySeries
        hc: lal.COMPLEX16FrequencySeries

        arguments = list(astuple(self))
        hp, hc = LS.SimInspiralFD(*arguments)

        if auto_turn_off_multibanding:
            hp, hc, success = self._turn_off_multibanding(hp, hc, stability_threshold)

        if not success:
            error_message = str(
                f"SimInspiralFD failed to reach numerical stability (threshold {stability_threshold}), "
                "attempted to apply SimInspiralWaveformParamsInsertPhenomXHMThresholdMband and "
                "SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband, but no success"
            )
            if raise_error_on_numerical_unstability and not success:
                raise RuntimeError(error_message)
            else:
                _logger.error(error_message)

        if not isclose(self.delta_f, hp.deltaF, rel_tol=delta_f_tolerance):
            raise ValueError(
                f"Waveform delta_f is inconsistent with domain: {hp.deltaF} vs {self.delta_f}!"
                f"To avoid this, ensure that f_max = {self.f_max} is a power of two"
                "when you are using a native time-domain waveform model."
            )

        h_plus = np.zeros_like(frequency_array, dtype=complex)
        h_cross = np.zeros_like(frequency_array, dtype=complex)

        # Ensure that length of wf agrees with length of domain. Enforce by truncating frequencies beyond f_max
        if len(hp.data.data) > len(frequency_array):
            _logger.warning(
                "LALsimulation waveform longer than domain's `frequency_array`"
                f"({len(hp.data.data)} vs {len(frequency_array)}). Truncating lalsim array."
            )
            h_plus = hp.data.data[: len(h_plus)]
            h_cross = hc.data.data[: len(h_cross)]
        else:
            h_plus[: len(hp.data.data)] = hp.data.data
            h_cross[: len(hc.data.data)] = hc.data.data

        # Undo the time shift done in SimInspiralFD to the waveform
        dt = 1 / hp.deltaF + (hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array)
        h_plus *= time_shift
        h_cross *= time_shift
        return Polarization(h_plus=h_plus, h_cross=h_cross)
