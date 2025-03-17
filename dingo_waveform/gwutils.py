from dataclasses import dataclass
from typing import Optional

import numpy as np
from bilby.gw.detector import PowerSpectralDensity
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

from dingo_waveform.domains import FrequencyDomain
from dingo_waveform.polarizations import Polarization


def get_tukey_window_factor(T: float, f_s: int, roll_off: float):
    """Compute window factor. If window is not provided as array or tensor but as
    window_kwargs, first build the window."""
    alpha = 2 * roll_off / T
    w = tukey(int(T * f_s), alpha)
    return np.sum(w**2) / len(w)


def get_mismatch(
    a: Polarization,
    b: Polarization,
    domain: FrequencyDomain,
    asd_file: Optional[str] = None,
) -> float:
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.

    Parameters
    ----------
    a
    b
    domain
    asd_file

    Returns
    -------

    """
    if asd_file is not None:
        # whiten a and b, such that we can use flat-spectrum inner products below
        psd = PowerSpectralDensity(asd_file=asd_file)
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        asd_array = asd_interp(domain.sample_frequencies())
        a = a / asd_array
        b = b / asd_array
    min_idx = domain.min_idx
    inner_ab = np.sum((a.conj() * b)[..., min_idx:], axis=-1).real
    inner_aa = np.sum((a.conj() * a)[..., min_idx:], axis=-1).real
    inner_bb = np.sum((b.conj() * b)[..., min_idx:], axis=-1).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap
