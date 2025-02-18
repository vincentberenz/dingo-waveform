from copy import deepcopy
from dataclasses import astuple
from typing import Dict, List, Optional, Tuple

import lal
import lalsimulation as LS

import dingo_waveform.wfg_utils as wfg_utils
from dingo_waveform.approximant import Approximant, get_approximant
from dingo_waveform.domains import Domain, FrequencyDomain
from dingo_waveform.inspiral_choose_fd_modes import InspiralChooseFDModesParameters
from dingo_waveform.lal_params import get_lal_params
from dingo_waveform.types import FrequencySeries, Mode, Iota
from dingo_waveform.waveform import WaveformParams


def sim_inspiral_choose_FD_modes(
    params: WaveformParams,
    f_ref: float,
    convert_to_SI: bool,
    domain: Domain,
    approximant: Approximant,
    mode_list: List[Mode],
    spin_conversion_phase: Optional[float],
    lal_params: Optional[lal.Dict],
)->Tuple[Dict[Mode,FrequencySeries], Iota]:
    supported_approximants: Tuple[Optional[Approximant], ...] = (Approximant(101),)
    if approximant not in supported_approximants:
        raise ValueError(
            "the 'LS.SimInspiralChooseFDModes' supports only the approximents: "
            f"{','.join([str(ap) for ap in supported_approximants])} ({approximant} not supported)"
        )

    # generating the frequencies series
    params_ = deepcopy(params)
    params_.f_ref = f_ref
    bbh_parameters = params_.to_binary_black_hole_parameters(convert_to_SI)
    lal_params = get_lal_params(mode_list)
    inspiral_choose_fd_modes_params: InspiralChooseFDModesParameters = (
        bbh_parameters.to_InspiralChooseFDModesParameters(
            domain, spin_conversion_phase, lal_params, approximant
        )
    )
    hlm_fd___: LS.SphHarmFrequencySeries = LS.SimInspiralChooseFDModes(
        list(astuple(inspiral_choose_fd_modes_params))
    )
    hlm_fd__: Dict[Mode, lal.COMPLEX16FrequencySeries] = (
        wfg_utils.linked_list_modes_to_dict_modes(hlm_fd___)
    )
    hlm_fd_: Dict[Mode, FrequencySeries] = {k: v.data.data for k, v in hlm_fd__.items()}

    hlm_fd: Dict[Mode, FrequencySeries] = inspiral_choose_fd_modes_params.convert_J_to_L0_frame(
        hlm_fd, spin_conversion_phase
    )

    return hlm_fd, inspiral_choose_fd_modes_params.iota


def generate_hplus_hcross_m(
        domain: FrequencyDomain,
        waveform_parameter: WaveformParams
    self, parameters: Dict[str, float]
) -> Dict[tuple, Dict[str, np.ndarray]]:



    if not isinstance(parameters, dict):
        raise ValueError("parameters should be a dictionary, but got", parameters)
    elif not isinstance(list(parameters.values())[0], float):
        raise ValueError("parameters dictionary must contain floats", parameters)

    if isinstance(self.domain, FrequencyDomain):
        # Generate FD modes in for frequencies [-f_max, ..., 0, ..., f_max].
        if LS.SimInspiralImplementedFDApproximants(self.approximant):
            # Step 1: generate waveform modes in L0 frame in native domain of
            # approximant (here: FD)
            hlm_fd, iota = self.generate_FD_modes_LO(parameters)

            # Step 2: Transform modes to target domain.
            # Not required here, as approximant domain and target domain are both FD.

        else:
            assert LS.SimInspiralImplementedTDApproximants(self.approximant)
            # Step 1: generate waveform modes in L0 frame in native domain of
            # approximant (here: TD)
            hlm_td, iota = self.generate_TD_modes_L0(parameters)

            # Step 2: Transform modes to target domain.
            # This requires tapering of TD modes, and FFT to transform to FD.
            wfg_utils.taper_td_modes_in_place(hlm_td)
            hlm_fd = wfg_utils.td_modes_to_fd_modes(hlm_td, self.domain)

        # Step 3: Separate negative and positive frequency parts of the modes,
        # and add contributions according to their transformation behavior under
        # phase shifts.
        pol_m = wfg_utils.get_polarizations_from_fd_modes_m(
            hlm_fd, iota, parameters["phase"]
        )

    else:
        raise NotImplementedError(
            f"Target domain of type {type(self.domain)} not yet implemented."
        )

    return pol_m


class WaveformGenerator:
    """Generate polarizations using LALSimulation routines in the specified domain for a
    single GW coalescence given a set of waveform parameters.
    """

    def __init__(
        self,
        approximant: str,
        domain: Domain,
        f_ref: float,
        f_start: Optional[float] = None,
        mode_list: Optional[List[Mode]] = None,
        transform=None,
        spin_conversion_phase: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        approximant : str
            Waveform "approximant" string understood by lalsimulation
            This is defines which waveform model is used.
        domain : Domain
            Domain object that specifies on which physical domain the
            waveform polarizations will be generated, e.g. Fourier
            domain, time domain.
        f_ref : float
            Reference frequency for the waveforms
        f_start : float
            Starting frequency for waveform generation. This is optional, and if not
            included, the starting frequency will be set to f_min. This exists so that
            EOB waveforms can be generated starting from a lower frequency than f_min.
        mode_list : List[Tuple]
            A list of waveform (ell, m) modes to include when generating
            the polarizations.
        spin_conversion_phase : float = None
            Value for phiRef when computing cartesian spins from bilby spins via
            bilby_to_lalsimulation_spins. The common convention is to use the value of
            the phase parameter here, which is also used in the spherical harmonics
            when combining the different modes. If spin_conversion_phase = None,
            this default behavior is adapted.
            For dingo, this convention for the phase parameter makes it impossible to
            treat the phase as an extrinsic parameter, since we can only account for
            the change of phase in the spherical harmonics when changing the phase (in
            order to also change the cartesian spins -- specifically, to rotate the spins
            by phase in the sx-sy plane -- one would need to recompute the modes,
            which is expensive).
            By setting spin_conversion_phase != None, we impose the convention to always
            use phase = spin_conversion_phase when computing the cartesian spins.
        """
        self._approximant_str = approximant
        self._lal_params: Optional[lal.Dict] = None
        self._approximant: Optional[Approximant] = None

        if "SEOBNRv5" not in approximant:
            self._approximant = get_approximant(approximant)
            if mode_list is not None:
                self._lal_params = get_lal_params(mode_list)

        self._domain = domain
        self._f_ref = f_ref
        self._f_start = f_start
        self._transform = transform
        self._spin_conversion_phase = spin_conversion_phase

    def generate_hplus_hcross_m(
        self, parameters: Dict[str, float]
    ) -> Dict[tuple, Dict[str, np.ndarray]]:
        """
        Generate GW polarizations (h_plus, h_cross), separated into contributions from
        the different modes. This method is identical to self.generate_hplus_hcross,
        except that it generates the individual contributions of the modes to the
        polarizations and sorts these according to their transformation behavior (see
        below), instead of returning the overall sum.

        This is useful in order to treat the phase as an extrinsic parameter. Instead of
        {"h_plus": hp, "h_cross": hc}, this method returns a dict in the form of
        {m: {"h_plus": hp_m, "h_cross": hc_m} for m in [-l_max,...,0,...,l_max]}. Each
        key m contains the contribution to the polarization that transforms according
        to exp(-1j * m * phase) under phase transformations (due to the spherical
        harmonics).

        Note:
            - pol_m[m] contains contributions of the m modes *and* and the -m modes.
              This is because the frequency domain (FD) modes have a positive frequency
              part which transforms as exp(-1j * m * phase), while the negative
              frequency part transforms as exp(+1j * m * phase). Typically, one of these
              dominates [e.g., the (2,2) mode is dominated by the negative frequency
              part and the (-2,2) mode is dominated by the positive frequency part]
              such that the sum of (l,|m|) and (l,-|m|) modes transforms approximately as
              exp(1j * |m| * phase), which is e.g. used for phase marginalization in
              bilby/lalinference. However, this is not exact. In this method we account
              for this effect, such that each contribution pol_m[m] transforms
              *exactly* as exp(-1j * m * phase).
            - Phase shifts contribute in two ways: Firstly via the spherical harmonics,
              which we account for with the exp(-1j * m * phase) transformation.
              Secondly, the phase determines how the PE spins transform to cartesian
              spins, by rotating (sx,sy) by phase. This is *not* accounted for in this
              function. Instead, the phase for computing the cartesian spins is fixed
              to self.spin_conversion_phase (if not None). This effectively changes the
              PE parameters {phi_jl, phi_12} to parameters {phi_jl_prime, phi_12_prime}.
              For parameter estimation, a postprocessing operation can be applied to
              account for this, {phi_jl_prime, phi_12_prime} -> {phi_jl, phi_12}.
              See also documentation of __init__ method for more information on
              self.spin_conversion_phase.

        Differences to self.generate_hplus_hcross:
        - We don't catch errors yet TODO
        - We don't apply transforms yet TODO

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters for the waveform.
            For details see see self.generate_hplus_hcross.

        Returns
        -------
        pol_m: dict
            Dictionary with contributions to h_plus and h_cross, sorted by their
            transformation behaviour under phase shifts:
            {m: {"h_plus": hp_m, "h_cross": hc_m} for m in [-l_max,...,0,...,l_max]}
            Each contribution h_m transforms as exp(-1j * m * phase) under phase shifts
            (for fixed self.spin_conversion_phase, see above).
        """
        if not isinstance(parameters, dict):
            raise ValueError("parameters should be a dictionary, but got", parameters)
        elif not isinstance(list(parameters.values())[0], float):
            raise ValueError("parameters dictionary must contain floats", parameters)

        if isinstance(self.domain, FrequencyDomain):
            # Generate FD modes in for frequencies [-f_max, ..., 0, ..., f_max].
            if LS.SimInspiralImplementedFDApproximants(self.approximant):
                # Step 1: generate waveform modes in L0 frame in native domain of
                # approximant (here: FD)
                hlm_fd, iota = self.generate_FD_modes_LO(parameters)

                # Step 2: Transform modes to target domain.
                # Not required here, as approximant domain and target domain are both FD.

            else:
                assert LS.SimInspiralImplementedTDApproximants(self.approximant)
                # Step 1: generate waveform modes in L0 frame in native domain of
                # approximant (here: TD)
                hlm_td, iota = self.generate_TD_modes_L0(parameters)

                # Step 2: Transform modes to target domain.
                # This requires tapering of TD modes, and FFT to transform to FD.
                wfg_utils.taper_td_modes_in_place(hlm_td)
                hlm_fd = wfg_utils.td_modes_to_fd_modes(hlm_td, self.domain)

            # Step 3: Separate negative and positive frequency parts of the modes,
            # and add contributions according to their transformation behavior under
            # phase shifts.
            pol_m = wfg_utils.get_polarizations_from_fd_modes_m(
                hlm_fd, iota, parameters["phase"]
            )

        else:
            raise NotImplementedError(
                f"Target domain of type {type(self.domain)} not yet implemented."
            )

        return pol_m
