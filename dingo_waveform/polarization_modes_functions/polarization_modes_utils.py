from typing import Dict

import lal
import lalsimulation as LS
import numpy as np

from ..domains.frequency_domain import FrequencyDomain
from ..types import FrequencySeries, Modes


def linked_list_modes_to_dict_modes(
    hlm_ll: LS.SphHarmFrequencySeries,
) -> Dict[Modes, lal.COMPLEX16FrequencySeries]:
    """Convert linked list of modes into dictionary with keys (l,m)."""
    hlm_dict = {}

    mode = hlm_ll.this
    while mode is not None:
        l, m = mode.l, mode.m
        hlm_dict[(l, m)] = mode.mode
        mode = mode.next

    return hlm_dict


def _get_tapering_window_for_complex_time_series(h, tapering_flag: int = 1):
    """
    Get window for tapering of a complex time series from the lal backend. This is done
    by tapering the time series with lal, and dividing tapered output by untapered
    input. lal does not support tapering of complex time series objects, so as a
    workaround we taper only the real part of the array and extract the window based on
    this.

    Parameters
    ----------
    h:
        complex lal time series object
    tapering_flag: int = 1
        Flag for tapering. See e.g. lines 2773-2777 in
            https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
            _l_a_l_sim_inspiral_waveform_taper_8c_source.html#l00222
        tapering_flag = 1 corresponds to LAL_SIM_INSPIRAL_TAPER_START

    Returns
    -------
    window: np.ndarray
        Array of length h.data.length, with the window used for tapering.
    """
    h_tapered = lal.CreateREAL8TimeSeries(
        "h_tapered", h.epoch, 0, h.deltaT, None, h.data.length
    )
    h_tapered.data.data = h.data.data.copy().real
    LS.SimInspiralREAL8WaveTaper(h_tapered.data, tapering_flag)
    eps = 1e-20 * np.max(np.abs(h.data.data))
    window = (np.abs(h_tapered.data.data) + eps) / (np.abs(h.data.data.real) + eps)
    # FIXME: using eps for numerical stability is not really robust here
    return window


def taper_td_modes_in_place(
    hlm_td: Dict[Modes, lal.COMPLEX16FrequencySeries], tapering_flag: int = 1
) -> None:
    """
    Taper the time domain modes in place.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex lal time series objects for the
        corresponding modes.
    tapering_flag: int = 1
        Flag for tapering. See e.g. lines 2773-2777 in
            https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
            _l_a_l_sim_inspiral_waveform_taper_8c_source.html#l00222
        tapering_flag = 1 corresponds to LAL_SIM_INSPIRAL_TAPER_START
    """
    for _, h in hlm_td.items():
        window = _get_tapering_window_for_complex_time_series(h, tapering_flag)
        h.data.data *= window


def td_modes_to_fd_modes(
    hlm_td: Dict[Modes, lal.COMPLEX16FrequencySeries],
    domain: FrequencyDomain,
) -> Dict[Modes, FrequencySeries]:
    """
    Transform dict of td modes to dict of fd modes via FFT. The td modes are expected
    to be tapered.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex lal time series objects for the
        corresponding tapered modes.
    domain: dingo.gw.domains.FrequencyDomain
        Target domain after FFT.

    Returns
    -------
    hlm_fd: dict
        Dictionary with (l,m) keys and numpy arrays with the corresponding modes as
        values.
    """
    hlm_fd: Dict[Modes, FrequencySeries] = {}

    domain_params = domain.get_parameters()

    if domain_params.delta_f is None:
        raise ValueError(
            "td_modes_to_fd_modes: frequency domain delta_f should not be None"
        )

    if domain_params.f_max is None:
        raise ValueError(
            "td_modes_to_fd_modes: frequency domain f_max should not be None"
        )

    delta_f = domain_params.delta_f
    delta_t = 0.5 / domain_params.f_max
    f_nyquist = domain_params.f_max  # use f_max as f_nyquist
    n = round(f_nyquist / delta_f)
    if (n & (n - 1)) != 0:
        raise NotImplementedError("f_nyquist not a power of two of delta_f.")
    chirplen = int(2 * f_nyquist / delta_f)
    # sample frequencies, -f_max,...,-f_min,...0,...,f_min,...,f_max
    sample_frequencies = domain.sample_frequencies()
    freqs = np.concatenate((-sample_frequencies[::-1], sample_frequencies[1:]), axis=0)
    # For even chirplength, we get chirplen + 1 output frequencies. However, the f_max
    # and -f_max bins are redundant, so we have chirplen unique bins.
    assert len(freqs) == chirplen + 1

    lal_fft_plan = lal.CreateForwardCOMPLEX16FFTPlan(chirplen, 0)
    for lm, h_td in hlm_td.items():
        assert np.abs(h_td.deltaT - delta_t) < 1e-12

        # resize data to chirplen by zero-padding or truncating
        # if chirplen < h_td.data.length:
        #     print(
        #         f"Specified frequency interval of {delta_f} Hz is too large "
        #         f"for a chirp of duration {h_td.data.length * delta_t} s with "
        #         f"Nyquist frequency {f_nyquist} Hz. The inspiral will be "
        #         f"truncated."
        #     )
        lal.ResizeCOMPLEX16TimeSeries(h_td, h_td.data.length - chirplen, chirplen)

        # Initialize a lal frequency series. We choose length chirplen + 1, while h_td is
        # only of length chirplen. This means, that the last bin h_fd.data.data[-1]
        # will not be modified by the lal FFT, and we have to copy over h_fd.data.data[0]
        # to h_fd.data.data[-1]. This corresponds to setting h(-f_max) = h(f_max).
        h_fd = lal.CreateCOMPLEX16FrequencySeries(
            "h_fd", h_td.epoch, 0, delta_f, None, chirplen + 1
        )
        # apply FFT
        lal.COMPLEX16TimeFreqFFT(h_fd, h_td, lal_fft_plan)
        assert np.abs(h_fd.deltaF - delta_f) < 1e-10
        assert np.abs(h_fd.f0 + domain_params.f_max) < 1e-6

        # time shift
        dt = (
            1.0 / h_fd.deltaF + h_fd.epoch.gpsSeconds + h_fd.epoch.gpsNanoSeconds * 1e-9
        )
        hlm_fd[lm] = h_fd.data.data * np.exp(-1j * 2 * np.pi * dt * freqs)
        # Set h(-f_max) = h(f_max), see above
        hlm_fd[lm][-1] = hlm_fd[lm][0]

    return hlm_fd
