import logging
from copy import deepcopy
from dataclasses import asdict, astuple, dataclass, fields
from math import isclose
from typing import Optional, Tuple, Union

import lal
import lalsimulation as LS
import numpy as np
from nptyping import Float32, NDArray, Shape

from ..approximant import Approximant
from ..binary_black_holes_parameters import BinaryBlackHoleParameters
from ..domains import DomainParameters, BaseFrequencyDomain, MultibandedFrequencyDomain
from ..logs import TableStr
from ..polarization_modes_functions.inspiral_choose_FD_modes import (
    _InspiralChooseFDModesParameters,
)
from ..polarizations import Polarization
from ..spins import Spins
from ..types import Iota, WaveformGenerationError
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class _InspiralFDParameters(TableStr):
    """Dataclass for storing parameters for
    lal simulation's SimInspiralFD function.
    """

    # Order matters ! The arguments to SimInspiralFD will
    # be these attributes, in the order they are defined here:
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
    approximant: int

    def get_spins(self) -> Spins:
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    def to_tuple(self) -> Tuple[Union[float, Optional[lal.Dict]]]:
        # This instance is casted to a tuple. It differs from:
        #
        #   p = InspiralFDParameters()
        #   t = astuple(p)
        #
        # Above, t contains deep copies of fields, which is not
        # supported by instances of lal.Dict; i.e. if lal_params is not None
        # `astuple(p)` will raise a runtime error. But in the following case:
        #
        #   t = p.to_tuple()
        #
        # t will contain a reference to lal_params and no error will be raised.

        return tuple(getattr(self, f.name) for f in fields(self))

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        f_start: Optional[float],
    ) -> "_InspiralFDParameters":

        inspiral_choose_fd_modes_parameters = (
            _InspiralChooseFDModesParameters.from_binary_black_hole_parameters(
                bbh_parameters,
                domain_params,
                spin_conversion_phase,
                lal_params,
                approximant,
                f_start,
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
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        f_start: Optional[float],
    ) -> "_InspiralFDParameters":

        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_parameters, f_ref
        )
        # Ensure domain parameters passed to LAL are scalar-typed where required.
        # In multibanded domains, delta_f may be None; prefer delta_f_initial or base_delta_f
        dp_dict = asdict(domain_params)
        df = dp_dict.get("delta_f")
        if df is None:
            df = dp_dict.get("delta_f_initial")
        if df is None:
            df = dp_dict.get("base_delta_f")
        # Construct a sanitized DomainParameters copy
        sanitized = DomainParameters(**dp_dict)
        sanitized.delta_f = df

        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            sanitized,
            spin_conversion_phase,
            lal_params,
            approximant,
            f_start,
        )

    def _turn_off_multibanding(
        self,
        hp: lal.COMPLEX16FrequencySeries,
        hc: lal.COMPLEX16FrequencySeries,
        threshold: float,
    ) -> Tuple[lal.COMPLEX16FrequencySeries, lal.COMPLEX16FrequencySeries, bool]:
        # Returns either the hp and hc as passed as argument (if no numerical unstability)
        # or the 'fixed' hp and hc (calling SimInspiralWaveformParamsInsertPhenomXHMThresholdMband and
        # SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband).
        # Also returns a boolean, which is:
        # - True if the returned value are numerically stable
        # - False if this failed and the values are still not numerically stable

        if max(np.max(np.abs(hp.data.data)), np.max(np.abs(hc.data.data))) <= threshold:
            return hp, hc, True

        _logger.debug("unstability detected, attempting to turning of multibanding")

        lal_params = (
            self.lal_params if self.lal_params is not None else lal.CreateDict()
        )
        LS.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lal_params, 0)
        LS.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lal_params, 0)
        params: "_InspiralFDParameters" = deepcopy(self)
        params.mass_1 *= lal.MSUN_SI
        params.mass_2 *= lal.MSUN_SI
        params.r *= 1e6 * lal.PC_SI
        params.lal_params = lal_params
        # arguments = list(astuple(params))
        #  The above would not always work, because lal.Dict can not be deep copied.
        #  (a RuntimeError is raised).
        #  So we do instead:
        arguments = list(params.to_tuple())

        _logger.debug(params.to_table("calling LS.SimInspiralFD with parameters"))

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

        # SimInspiralFD requires kg, converting here
        params: "_InspiralFDParameters" = deepcopy(self)
        params.mass_1 *= lal.MSUN_SI
        params.mass_2 *= lal.MSUN_SI
        params.r *= 1e6 * lal.PC_SI

        _logger.debug(
            params.to_table("generating polarization using lalsimulation.SimInspiralFD")
        )

        hp: lal.COMPLEX16FrequencySeries
        hc: lal.COMPLEX16FrequencySeries

        arguments = list(astuple(params))
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
            raise WaveformGenerationError(
                f"Waveform delta_f is inconsistent with domain: {hp.deltaF} vs {self.delta_f}!"
                f"To avoid this, ensure that f_max = {self.f_max} is a power of two"
                "when you are using a native time-domain waveform model."
            )

        h_plus = np.zeros_like(frequency_array, dtype=complex)
        h_cross = np.zeros_like(frequency_array, dtype=complex)

        # Ensure that length of wf agrees with length of domain. Enforce by truncating frequencies beyond f_max
        # Map LAL uniform frequency grid to the requested frequency_array (supports multibanded domains)
        lal_delta_f = hp.deltaF
        # Some LAL series may have a starting frequency offset; try to obtain it, default 0.0
        try:
            f0 = hp.f0
        except AttributeError:
            f0 = 0.0
        # Prefer exact bin-aligned copy when frequency_array is uniform and matches LAL deltaF
        diffs = np.diff(frequency_array)
        is_uniform = np.allclose(diffs, diffs[0])
        if is_uniform and np.isclose(diffs[0], lal_delta_f, rtol=0, atol=1e-12):
            start_idx = int(np.rint((float(frequency_array[0]) - float(f0)) / float(lal_delta_f)))
            end_idx = start_idx + len(frequency_array)
            if start_idx < 0 or end_idx > len(hp.data.data):
                # fallback to safe clipping
                start_idx = max(start_idx, 0)
                end_idx = min(end_idx, len(hp.data.data))
            # Fill overlapping region
            n_copy = end_idx - start_idx
            if n_copy > 0:
                h_plus[:n_copy] = hp.data.data[start_idx:end_idx]
                h_cross[:n_copy] = hc.data.data[start_idx:end_idx]
            # Remaining entries (if any) stay zero
        else:
            # Interpolate LAL uniform frequency grid to the requested frequency_array (supports multibanded domains)
            N = len(hp.data.data)
            lal_freqs = f0 + np.arange(N, dtype=frequency_array.dtype) * lal_delta_f
            # Determine valid interpolation range
            fmin = lal_freqs[0]
            fmax = lal_freqs[-1]
            valid = (frequency_array >= fmin) & (frequency_array <= fmax)
            if not np.all(valid):
                n_out = int(np.count_nonzero(~valid))
                _logger.warning(
                    f"{n_out} frequencies are outside the LAL waveform range and will be left as zeros."
                )
            # Use linear interpolation on real and imaginary parts separately
            if np.any(valid):
                hp_real = np.interp(frequency_array[valid].astype(float), lal_freqs.astype(float), hp.data.data.real.astype(float))
                hp_imag = np.interp(frequency_array[valid].astype(float), lal_freqs.astype(float), hp.data.data.imag.astype(float))
                hc_real = np.interp(frequency_array[valid].astype(float), lal_freqs.astype(float), hc.data.data.real.astype(float))
                hc_imag = np.interp(frequency_array[valid].astype(float), lal_freqs.astype(float), hc.data.data.imag.astype(float))
                h_plus[valid] = hp_real + 1j * hp_imag
                h_cross[valid] = hc_real + 1j * hc_imag

        _logger.debug("undoing the time shift done in SimInspiralFD to the waveform")

        dt = 1 / hp.deltaF + (hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array)
        h_plus *= time_shift
        h_cross *= time_shift

        return Polarization(h_plus=h_plus, h_cross=h_cross)


def inspiral_FD(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: WaveformParameters,
) -> Polarization:
    """
    Wrapper over lalsimulation.SimInspiralFD

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
      if the domain is not an instance of UniformFrequencyDomain
    WaveformGenerationError
      if an numerical instability that could not be solved by
      turning off the multibanding is detected
    """

    if not isinstance(waveform_gen_params.domain, BaseFrequencyDomain):
        raise ValueError(
            "inspiral_fd can only be applied using on a BaseFrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    inspiral_fd_params = _InspiralFDParameters.from_waveform_parameters(
        waveform_params,
        waveform_gen_params.f_ref,
        waveform_gen_params.domain.get_parameters(),
        waveform_gen_params.spin_conversion_phase,
        waveform_gen_params.lal_params,
        waveform_gen_params.approximant,
        waveform_gen_params.f_start,
    )

    # Support both method and property access for sample frequencies across domains
    domain = waveform_gen_params.domain

    # For multibanded domains, generate on the base uniform grid
    # The domain's waveform_transform() method will handle decimation
    if isinstance(domain, MultibandedFrequencyDomain):
        # Create a base uniform domain for waveform generation
        # Start from f_min=0 to capture all frequencies the waveform generator will produce
        from dingo_waveform.domains import UniformFrequencyDomain
        base = UniformFrequencyDomain(
            f_min=0.0,
            f_max=domain.f_max,
            delta_f=domain.base_delta_f,
            window_factor=domain.window_factor,
        )
        # Handle both method and property for uniform domain
        base_sf_attr = getattr(base, "sample_frequencies", None)
        base_freqs = base_sf_attr() if callable(base_sf_attr) else base_sf_attr
        return inspiral_fd_params.apply(base_freqs.astype(np.float32))

    # Otherwise, use the domain's own frequency sampling
    sample_freqs_attr = getattr(domain, "sample_frequencies", None)
    frequency_array = sample_freqs_attr() if callable(sample_freqs_attr) else sample_freqs_attr

    return inspiral_fd_params.apply(frequency_array)
