"""
Define pulse functions for use in initializing other experiments
"""

import logging

import numpy as np

MIN_LENGTH = 3
MAX_DAC_GAIN = 32766
DAC_SAMPLES_PER_CLOCK = 16
logger = logging.getLogger(__name__)


def chop(length=MIN_LENGTH, maxv=MAX_DAC_GAIN, samps_per_clk=DAC_SAMPLES_PER_CLOCK):
    """Defines a chop waveform"""
    amp = maxv / 2
    pulse_length = int(length * samps_per_clk / 2)
    y1 = amp * np.ones(pulse_length)
    y2 = -1 * amp * np.ones(pulse_length)
    y = np.concatenate((y1, y2))
    return np.around(y, 0)


def baseband(length=MIN_LENGTH, maxv=MAX_DAC_GAIN, samps_per_clk=DAC_SAMPLES_PER_CLOCK):
    """Minimum constant pulse array, length=3 (used by baseband)."""
    length = length * samps_per_clk  # must be a multiple of dac_samples_per_clk
    return maxv * np.ones(length)


def baseband_short_pulse_seq(
    peakv: int,
    minv: int,
    peaktime: np.ndarray,
    mintime: np.ndarray,
    cycles_per_us: int = 430,
    dac_samps_per_cycle: int = 16,
):
    """create a train of pulses which are individually smaller than the minimum pulse envelope of 3 fabric clock cycles"""
    waveform = np.array([])
    for idx, _ in enumerate(peaktime):
        peak_length = peaktime[idx] * dac_samps_per_cycle * cycles_per_us
        peak_amp = peakv
        min_length = mintime[idx] * dac_samps_per_cycle * cycles_per_us
        min_amp = minv
        wf_temp = np.concatenate(
            (np.ones(int(peak_length)) * peak_amp, np.ones(int(min_length)) * min_amp)
        )
        waveform = np.concatenate((waveform, wf_temp))
    time_axis = np.arange(0, len(waveform)) / (dac_samps_per_cycle * cycles_per_us)
    return time_axis, waveform


def ramp(
    ramp_length: int,
    startv: int,
    stopv: int,
    samps_per_clk: int = DAC_SAMPLES_PER_CLOCK,
):
    """Ramp to and from measurement window
    :param ramp_length: length of ramp in cycles
    """

    length = ramp_length * samps_per_clk
    slope = (stopv - startv) / length
    array = np.around(slope * np.arange(length) + startv, 1)
    return array.astype(int)
