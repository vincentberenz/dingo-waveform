"""
Converters between dingo-waveform data structures and gwpy objects.

This module provides an isolation layer for gwpy, allowing the rest of the
plotting package to use gwpy's excellent time-frequency analysis tools
without exposing gwpy objects in the public API.
"""

from typing import Dict, Tuple

import gwpy.frequencyseries
import gwpy.timeseries
import numpy as np

from ..domains import Domain, TimeDomain, BaseFrequencyDomain
from ..polarizations import Polarization
from ..types import Mode


def polarization_to_gwpy_timeseries(
    polarization: Polarization, domain: TimeDomain
) -> Tuple[gwpy.timeseries.TimeSeries, gwpy.timeseries.TimeSeries]:
    """
    Convert dingo-waveform Polarization to gwpy TimeSeries objects.

    Parameters
    ----------
    polarization : Polarization
        Polarization with h_plus and h_cross arrays
    domain : TimeDomain
        Time domain specification

    Returns
    -------
    Tuple[gwpy.timeseries.TimeSeries, gwpy.timeseries.TimeSeries]
        (h_plus_ts, h_cross_ts) as gwpy TimeSeries objects
    """
    # Get time array and sampling info
    times = domain()
    dt = domain.delta_t

    # Create TimeSeries objects
    # gwpy expects times to start from t0, with spacing dt
    h_plus_ts = gwpy.timeseries.TimeSeries(
        polarization.h_plus,
        times=times,
        name="h_plus",
        unit="strain",
    )

    h_cross_ts = gwpy.timeseries.TimeSeries(
        polarization.h_cross,
        times=times,
        name="h_cross",
        unit="strain",
    )

    return h_plus_ts, h_cross_ts


def polarization_to_gwpy_frequencyseries(
    polarization: Polarization, domain: BaseFrequencyDomain
) -> Tuple[gwpy.frequencyseries.FrequencySeries, gwpy.frequencyseries.FrequencySeries]:
    """
    Convert dingo-waveform Polarization to gwpy FrequencySeries objects.

    Parameters
    ----------
    polarization : Polarization
        Polarization with h_plus and h_cross frequency-domain arrays
    domain : BaseFrequencyDomain
        Frequency domain specification (UniformFrequencyDomain or MultibandedFrequencyDomain)

    Returns
    -------
    Tuple[gwpy.frequencyseries.FrequencySeries, gwpy.frequencyseries.FrequencySeries]
        (h_plus_fs, h_cross_fs) as gwpy FrequencySeries objects
    """
    # Get frequency array
    frequencies = domain()

    # Create FrequencySeries objects
    h_plus_fs = gwpy.frequencyseries.FrequencySeries(
        polarization.h_plus,
        frequencies=frequencies,
        name="h_plus",
        unit="strain",
    )

    h_cross_fs = gwpy.frequencyseries.FrequencySeries(
        polarization.h_cross,
        frequencies=frequencies,
        name="h_cross",
        unit="strain",
    )

    return h_plus_fs, h_cross_fs


def modes_to_gwpy_dict_timeseries(
    modes: Dict[Mode, Polarization], domain: TimeDomain
) -> Dict[Mode, Tuple[gwpy.timeseries.TimeSeries, gwpy.timeseries.TimeSeries]]:
    """
    Convert mode-separated polarizations to gwpy TimeSeries objects.

    Parameters
    ----------
    modes : Dict[Mode, Polarization]
        Mode-separated polarizations from generate_hplus_hcross_m
    domain : TimeDomain
        Time domain specification

    Returns
    -------
    Dict[Mode, Tuple[gwpy.timeseries.TimeSeries, gwpy.timeseries.TimeSeries]]
        Dictionary mapping each mode to (h_plus_ts, h_cross_ts)
    """
    result = {}
    for mode, polarization in modes.items():
        h_plus_ts, h_cross_ts = polarization_to_gwpy_timeseries(polarization, domain)
        # Add mode label to name
        h_plus_ts.name = f"h_plus_({mode[0]},{mode[1]})"
        h_cross_ts.name = f"h_cross_({mode[0]},{mode[1]})"
        result[mode] = (h_plus_ts, h_cross_ts)
    return result


def modes_to_gwpy_dict_frequencyseries(
    modes: Dict[Mode, Polarization], domain: BaseFrequencyDomain
) -> Dict[Mode, Tuple[gwpy.frequencyseries.FrequencySeries, gwpy.frequencyseries.FrequencySeries]]:
    """
    Convert mode-separated polarizations to gwpy FrequencySeries objects.

    Parameters
    ----------
    modes : Dict[Mode, Polarization]
        Mode-separated polarizations from generate_hplus_hcross_m
    domain : BaseFrequencyDomain
        Frequency domain specification

    Returns
    -------
    Dict[Mode, Tuple[gwpy.frequencyseries.FrequencySeries, gwpy.frequencyseries.FrequencySeries]]
        Dictionary mapping each mode to (h_plus_fs, h_cross_fs)
    """
    result = {}
    for mode, polarization in modes.items():
        h_plus_fs, h_cross_fs = polarization_to_gwpy_frequencyseries(
            polarization, domain
        )
        # Add mode label to name
        h_plus_fs.name = f"h_plus_({mode[0]},{mode[1]})"
        h_cross_fs.name = f"h_cross_({mode[0]},{mode[1]})"
        result[mode] = (h_plus_fs, h_cross_fs)
    return result


def gwpy_to_polarization(
    ts_plus: gwpy.timeseries.TimeSeries, ts_cross: gwpy.timeseries.TimeSeries
) -> Polarization:
    """
    Convert gwpy TimeSeries objects back to dingo-waveform Polarization.

    This is mainly useful if you've processed waveforms with gwpy and want
    to convert back to dingo-waveform format.

    Parameters
    ----------
    ts_plus : gwpy.timeseries.TimeSeries
        h_plus time series
    ts_cross : gwpy.timeseries.TimeSeries
        h_cross time series

    Returns
    -------
    Polarization
        Reconstructed polarization object
    """
    return Polarization(h_plus=ts_plus.value, h_cross=ts_cross.value)
