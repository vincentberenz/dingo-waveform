"""
Detector-related transform classes.
"""

from typing import Dict, Any, Union, Optional
import math
import numpy as np
import torch
import pandas as pd
from bilby.gw.detector.interferometer import Interferometer
from lal import GreenwichMeanSiderealTime
from bilby.gw.detector import calibration
from bilby.gw.prior import CalibrationPriorDict

# Try to import from bilby_pipe, fallback to default mapping if unavailable
try:
    from bilby_pipe.utils import CALIBRATION_CORRECTION_TYPE_LOOKUP
except ImportError:
    # Fallback for newer bilby_pipe versions where this constant was removed
    CALIBRATION_CORRECTION_TYPE_LOOKUP = {
        "H1": "template",
        "L1": "template",
        "V1": "template",
    }

CC = 299792458.0


def time_delay_from_geocenter(
    ifo: Interferometer,
    ra: Union[float, np.ndarray, torch.Tensor],
    dec: Union[float, np.ndarray, torch.Tensor],
    time: float,
):
    """
    Calculate time delay between ifo and geocenter. Identical to method
    ifo.time_delay_from_geocenter(ra, dec, time), but the present implementation allows
    for batched computation, i.e., it also accepts arrays and tensors for ra and dec.

    Implementation analogous to bilby-cython implementation
    https://git.ligo.org/colm.talbot/bilby-cython/-/blob/main/bilby_cython/geometry.pyx,
    which is in turn based on XLALArrivaTimeDiff in TimeDelay.c.

    Parameters
    ----------
    ifo: bilby.gw.detector.interferometer.Interferometer
        bilby interferometer object.
    ra: Union[float, np.array, torch.Tensor]
        Right ascension of the source in radians. Either float, or float array/tensor.
    dec: Union[float, np.array, torch.Tensor]
        Declination of the source in radians. Either float, or float array/tensor.
    time: float
        GPS time in the geocentric frame.

    Returns
    -------
    float: Time delay between the two detectors in the geocentric frame
    """
    # check that ra and dec are of same type and length
    if type(ra) != type(dec):
        raise ValueError(
            f"ra type ({type(ra)}) and dec type ({type(dec)}) don't match."
        )
    if isinstance(ra, (np.ndarray, torch.Tensor)):
        if len(ra.shape) != 1:
            raise ValueError(f"Only one axis expected for ra and dec, got multiple.")
        if ra.shape != dec.shape:
            raise ValueError(
                f"Shapes of ra ({ra.shape}) and dec ({dec.shape}) don't match."
            )

    if isinstance(ra, (float, np.float32, np.float64)):
        return ifo.time_delay_from_geocenter(ra, dec, time)

    elif isinstance(ra, (np.ndarray, torch.Tensor)) and len(ra) == 1:
        return ifo.time_delay_from_geocenter(ra[0], dec[0], time)

    else:
        if isinstance(ra, np.ndarray):
            sin = np.sin
            cos = np.cos
        elif isinstance(ra, torch.Tensor):
            sin = torch.sin
            cos = torch.cos
        else:
            raise NotImplementedError(
                "ra, dec must be either float, np.ndarray, or torch.Tensor."
            )

        gmst = math.fmod(GreenwichMeanSiderealTime(float(time)), 2 * np.pi)
        phi = ra - gmst
        theta = np.pi / 2 - dec
        sintheta = sin(theta)
        costheta = cos(theta)
        sinphi = sin(phi)
        cosphi = cos(phi)
        detector_1 = ifo.vertex
        detector_2 = np.zeros(3)
        return (
            (detector_2[0] - detector_1[0]) * sintheta * cosphi
            + (detector_2[1] - detector_1[1]) * sintheta * sinphi
            + (detector_2[2] - detector_1[2]) * costheta
        ) / CC
