# To be used in NewInterfaceWaveformGenerator.generate_hplus_hcross_m


import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, cast

import lal
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

from ..approximant import Approximant
from ..binary_black_holes import BinaryBlackHoleParameters
from ..domains import DomainParameters
from ..gw_signals_parameters import GwSignalParameters
from ..polarizations import Polarization, get_polarizations_from_fd_modes_m
from ..spins import Spins
from ..types import FrequencySeries, GWSignalsGenerators, Iota, Mode, Modes
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import WaveformParameters
from .polarization_modes_utils import linked_list_modes_to_dict_modes

_logger = logging.getLogger(__name__)


_SupportedApproximant = Approximant("IMRPhenomXPHM")


@dataclass
class _GenerateFDModesLOParameters(GwSignalParameters):

    # For fields and source of the method "from_waveform_parameters"
    # see the code of the superclass GwSignalParameters

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GenerateFDModesLOParameters":

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
        self,
        spin_conversion_phase: float,
        phase: float,
    ) -> Dict[Mode, Polarization]:

        # only approximant supported.
        # (so no need to pass it as argument)
        approximant = Approximant("IMRPhenomXPHM")

        _logger.debug(
            self.to_table("generating polarization using waveform.GenerateFDModes")
        )

        generator: GWSignalsGenerators = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        hlm_fd: GravitationalWavePolarizations = waveform.GenerateFDModes(
            params, generator
        )

        hlms_lal = {}
        for key, value in hlm_fd.items():
            if type(key) != str:
                hlm_lal = lal.CreateCOMPLEX16TimeSeries(
                    key,
                    value.epoch.value,
                    0,
                    value.dt.value,
                    lal.DimensionlessUnit,
                    len(value),
                )
                hlm_lal.data.data = value.value
                hlms_lal[key] = hlm_lal

        hlm_fd_: Dict[Modes, lal.COMPLEX16FrequencySeries] = (
            linked_list_modes_to_dict_modes(hlms_lal)
        )
        hlm_fd__: Dict[Modes, FrequencySeries] = {
            k: v.data.data for k, v in hlm_fd_.items()
        }
        # For the waveform models considered here (e.g., IMRPhenomXPHM), the modes
        # are returned in the J frame (where the observer is at inclination=theta_JN,
        # azimuth=0). In this frame, the dependence on the reference phase enters
        # via the modes themselves. We need to convert to the L0 frame so that the
        # dependence on phase enters via the spherical harmonics.
        spins = Spins(
            self.inclination,
            self.spin1x,
            self.spin1y,
            self.spin1z,
            self.spin2x,
            self.spin2y,
            self.spin2z,
        )
        convert_to_SI = True
        hlm_fd___: Dict[Modes, FrequencySeries] = spins.convert_J_to_L0_frame(
            hlm_fd__,
            self.mass1,
            self.mass2,
            convert_to_SI,
            self.f22_ref,
            spin_conversion_phase,
        )

        return get_polarizations_from_fd_modes_m(
            hlm_fd___, Iota(self.inclination), phase
        )


def generate_FD_modes_LO(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: WaveformParameters,
) -> Dict[Mode, Polarization]:
    """
    Wrapper over lalsimulation.gwsignal.core.waveform.GenerateFDModes
    using the IMRPhenomXPHM approximant.

    Arguments
    ---------
    waveform_gen_params
      waveform generation configuration
    waveform_params
      waveform configuration

    Returns
    -------
    Dictionary mode / polarizations

    Raises
    ------
    ValueError
      if spin_conversion_phase or phase is not specified in the configuration
    """

    if waveform_gen_params.spin_conversion_phase is None:
        raise ValueError(
            f"generate_FD_modes_LO: spin_conversion_phase parameter should not be None"
        )

    if waveform_params.phase is None:
        raise ValueError(f"generate_FD_modes_LO: phase parameter should not be None")

    instance = cast(
        _GenerateFDModesLOParameters,
        _GenerateFDModesLOParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            waveform_gen_params.f_start,
            waveform_gen_params.convert_to_SI,
        ),
    )

    return instance.apply(
        waveform_gen_params.spin_conversion_phase, waveform_params.phase
    )
