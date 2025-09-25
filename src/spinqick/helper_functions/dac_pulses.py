"""
Define pulse functions for use in initializing other experiments
"""

import logging
import numpy as np
from scipy import signal
from qick import QickConfig
from spinqick.settings import filter_settings

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
    :param ramp_length: length of ramp in fabric cycles
    """

    length = ramp_length * samps_per_clk
    slope = (stopv - startv) / length
    array = np.around(slope * np.arange(length) + startv, 1)
    return array.astype(int)


### the following code was developed when converting spinqick to tprocv2 ###


def generate_baseband(
    gain: float,
    length: float,
    gen: int,
    soccfg: QickConfig,
    gain_2: float | None = None,
    min_pulse: bool = False,
    pad_pulse: bool = False,
) -> np.ndarray:
    """Produces a numpy array of the baseband waveform
    :param gain: gain in range (-1, 1)
    :param length: time of baseband pulse in microseconds
    :param gen: generator channel
    :param soccfg: soc config object
    :param gain_2: optionally set a second pulse gain to pad end of pulse with instead of zeros
    :param min_pulse: if true, this ignores the length parameter and sets the pulse length to three dac clock cycles
    :param pad_pulse: if false, this rounds the pulse length up to the next dac fabric clock cycle
    """
    cycles_per_clk = soccfg["gens"][gen]["samps_per_clk"]
    dac_cycles = us2daccycles(length, gen, soccfg)

    if min_pulse:
        # make the shortest possible length of pulse
        min_samples = cycles_per_clk * 3
        pulse_final = gain * np.ones(min_samples)
    else:
        if pad_pulse:
            # pad pulses with gain_2 sample values
            if dac_cycles // cycles_per_clk < 3:
                extra_samples = 3 * cycles_per_clk - dac_cycles
                arr1 = gain * np.ones(dac_cycles)
                if gain_2 is None:
                    arr2 = np.zeros(extra_samples)
                else:
                    arr2 = gain_2 * np.ones(extra_samples)
                pulse_final = np.append(arr1, arr2)
            else:
                remainder = dac_cycles % cycles_per_clk
                arr1 = gain * np.ones(dac_cycles)
                extra_samples = cycles_per_clk - remainder
                if extra_samples == 0:
                    pulse_final = arr1
                else:
                    # arr2 = np.zeros(cycles_per_clk - remainder)
                    if gain_2 is None:
                        arr2 = np.zeros(extra_samples)
                    else:
                        arr2 = gain_2 * np.ones(extra_samples)
                    pulse_final = np.append(arr1, arr2)
        else:
            samples = np.ceil(dac_cycles / cycles_per_clk)
            if samples < 3:
                pulse_final = gain * np.ones(int(3 * cycles_per_clk))
            else:
                pulse_final = gain * np.ones(int(samples * cycles_per_clk))

    return np.around(pulse_final * MAX_DAC_GAIN)


def baseband_pulse_train(
    pulse_gain: float,
    idle_gain: float,
    pulse_length: float,
    idle_length: float,
    n_pulses: int,
    gen: int,
    soccfg,
) -> np.ndarray:
    """create a train of pulses, used for angle calibrations"""
    cycles_per_clk = soccfg["gens"][gen]["samps_per_clk"]
    interp = soccfg["gens"][gen]["interpolation"]
    dac_cycles = us2daccycles(pulse_length, gen, soccfg)
    samples = int(dac_cycles / interp)
    idle_dac_cycles = us2daccycles(idle_length, gen, soccfg)
    idle_samples = int(idle_dac_cycles / interp)
    arr1 = pulse_gain * np.ones(samples)
    arr2 = idle_gain * np.ones(idle_samples)
    array = np.append(arr1, arr2)
    array_full = array
    nloops = n_pulses - 1
    for n in range(nloops):
        array_full = np.append(array_full, array)
    samples_per_clock = cycles_per_clk / interp
    # pad with zeros, in the future we may need to pad with something other than zero
    if len(array_full) // cycles_per_clk < 3:
        extra_samples = 3 * samples_per_clock - len(array_full)
    else:
        remainder = len(array_full) % samples_per_clock
        extra_samples = samples_per_clock - remainder
    arr2 = np.zeros(int(extra_samples))
    pulse_final = np.append(array_full, arr2)
    return pulse_final * MAX_DAC_GAIN


def baseband_centered(
    pulse_gain: float,
    idle_gain: float,
    pulse_length: float,
    idle_length: float,
    gen: int,
    soccfg,
) -> np.ndarray:
    """create a baseband pulse with idle time/2 before and idle time/2 after"""
    cycles_per_clk = soccfg["gens"][gen]["samps_per_clk"]
    dac_cycles = us2daccycles(pulse_length, gen, soccfg)
    idle_dac_cycles = us2daccycles(idle_length / 2, gen, soccfg)
    arr1 = pulse_gain * np.ones(dac_cycles)
    arr2 = idle_gain * np.ones(idle_dac_cycles)
    array_full = np.concatenate((arr2, arr1, arr2))
    if len(array_full) // cycles_per_clk < 3:
        extra_samples = 3 * cycles_per_clk - len(array_full)
    else:
        remainder = len(array_full) % cycles_per_clk
        extra_samples = cycles_per_clk - remainder
    arr3 = np.ones(int(extra_samples)) * idle_gain
    pulse_final = np.append(array_full, arr3)
    return pulse_final * MAX_DAC_GAIN


def us2daccycles(time: float, gen: int, soccfg):
    fs = soccfg["gens"][gen][
        "f_dds"
    ]  # this may be the wrong number to use, seems to work for now
    cycles = int(time * fs)
    return cycles


def generate_ramp(start_gain, stop_gain, time, gen, soccfg) -> np.ndarray:
    """generate a ramp envelope"""
    cycles_per_clk = soccfg["gens"][gen]["samps_per_clk"]
    interp = soccfg["gens"][gen]["interpolation"]
    dac_cycles = us2daccycles(time, gen, soccfg)
    samples = dac_cycles / interp
    ramp_pts = (
        samples - samples % cycles_per_clk
    )  # make sure its a multiple of samps_per_clk.  Check this math for different generators!
    slope = (stop_gain - start_gain) * MAX_DAC_GAIN / ramp_pts
    array = np.around(slope * np.arange(ramp_pts) + start_gain * MAX_DAC_GAIN, 1)
    return array.astype(int)


def calculate_ramp_sweep(
    start_sweep, end_sweep, ramp_ampl, ramp_time, sweep_points, gen, soccfg
) -> tuple:
    """generate a ramp waveform for sweeping ramps"""
    slope = ramp_ampl / ramp_time
    dac_incr = (end_sweep - start_sweep) / sweep_points
    time_incr = dac_incr / slope
    envelope_incr = soccfg.us2cycles(time_incr, gen_ch=gen)
    stop_r_final = np.abs(ramp_ampl) + end_sweep  # end of programmed ramp
    length_full = (
        stop_r_final - start_sweep
    ) / slope  # length of full ramp in microseconds
    ramp = generate_ramp(start_sweep, stop_r_final, np.abs(length_full), gen, soccfg)
    return ramp, envelope_incr


def add_wf_filter(wf):
    """apply filters to the waveforms"""
    filter_mode = filter_settings.apply_filter
    if filter_mode is None:
        wf_filt = wf
    if filter_mode == "iir_1" or filter_mode == "both":
        if filter_settings.iir_taps is not None:
            iir_b, iir_a = filter_settings.iir_taps
        else:
            raise Exception(" no iir_1 taps specified ")
        wf_filt_1 = signal.lfilter(iir_b, iir_a, wf)
        wf_filt = wf_filt_1
    if filter_mode == "both":
        if filter_settings.iir_2_taps is not None:
            iir_b, iir_a = filter_settings.iir_2_taps
        else:
            raise Exception(" no iir_2 taps specified ")
        wf_filt = signal.lfilter(iir_b, iir_a, wf_filt_1)
    return wf_filt


def offset_sine_wave(gen, sine_ampl, sine_freq, step_ampl, soccfg):
    """produce a sine wave with a DC offset"""
    cycles_per_clk = soccfg["gens"][gen]["samps_per_clk"]
    fs = soccfg["gens"][gen]["f_dds"]
    cycle_time = 1 / fs
    sine_period = 1 / sine_freq
    steps = sine_period // cycle_time
    min_pulse_length = 3 * cycles_per_clk
    # TODO need to handle high frequency pulses better
    # if steps<min_pulse_length:
    #     remainder_1 = min_pulse_length % steps
    #     steps = min_pulse_length - remainder_1 + cycles_per_clk
    remainder = steps % cycles_per_clk
    steps -= remainder
    if steps < min_pulse_length:
        steps = min_pulse_length
    time_array = np.linspace(0, steps * cycle_time, int(steps))
    pulse = sine_ampl * np.sin(2 * np.pi * sine_freq * time_array) + step_ampl
    return pulse * MAX_DAC_GAIN
