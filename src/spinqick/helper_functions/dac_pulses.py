"""Define pulse functions for use in initializing other experiments."""

import logging

import numpy as np
from qick import QickConfig
from scipy import signal

from spinqick.settings import filter_settings

MIN_LENGTH = 3
MAX_DAC_GAIN = 32766
DAC_SAMPLES_PER_CLOCK = 16
logger = logging.getLogger(__name__)


def chop(
    length: int = MIN_LENGTH,
    maxv: int = MAX_DAC_GAIN,
    samps_per_clk: int = DAC_SAMPLES_PER_CLOCK,
):
    """Defines a periodic square pulse waveform."""
    amp = maxv / 2
    pulse_length = int(length * samps_per_clk / 2)
    y1 = amp * np.ones(pulse_length)
    y2 = -1 * amp * np.ones(pulse_length)
    y = np.concatenate((y1, y2))
    return np.around(y, 0)


def baseband(
    length: int = MIN_LENGTH,
    maxv: int = MAX_DAC_GAIN,
    samps_per_clk: int = DAC_SAMPLES_PER_CLOCK,
):
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
    """Create a train of pulses which are individually smaller than the minimum pulse envelope of 3
    fabric clock cycles."""
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
    """Ramp to and from measurement window.

    :param start_v: gain at start of ramp
    :param stop_v: gain at end of ramp
    :param ramp_length: length of ramp in fabric cycles.
    """

    length = ramp_length * samps_per_clk
    slope = (stopv - startv) / length
    array = np.around(slope * np.arange(length) + startv, 1)
    return array.astype(int)


def generate_baseband(
    gain: float,
    length: float,
    gen: int,
    soccfg: QickConfig,
    gain_2: float | None = None,
    min_pulse: bool = False,
    pad_pulse: bool = False,
) -> np.ndarray:
    """Produces a numpy array of the baseband waveform.

    :param gain: gain in range (-1, 1)
    :param length: time of baseband pulse in microseconds
    :param gen: generator channel
    :param soccfg: soc config object
    :param gain_2: optionally set a second pulse gain to pad end of pulse with instead of zeros
    :param min_pulse: if true, this ignores the length parameter and sets the pulse length to three
        dac clock cycles
    :param pad_pulse: if false, this rounds the pulse length up to the next dac fabric clock cycle.
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
    soccfg: QickConfig,
) -> np.ndarray:
    """Create a train of pulses, each of the n_pulses includes a pulse and an idle. Before first
    pulse and after last pulse, idles for idle time divided by two.

    :param pulse_length: duration of pulse in microseconds
    :param idle_length: duration of idle in microseconds
    :param n_pulses: number of pulses in train
    :param gen: generator channel
    :param soccfg: qickconfig object describing the generators
    """
    cycles_per_clk = soccfg["gens"][gen]["samps_per_clk"]
    interp = soccfg["gens"][gen]["interpolation"]
    dac_cycles = us2daccycles(pulse_length, gen, soccfg)
    samples = int(dac_cycles / interp)
    idle_dac_cycles = us2daccycles(idle_length, gen, soccfg)
    idle_samples = int(idle_dac_cycles / interp)
    arr_idle_sandwich = idle_gain * np.ones(idle_samples // 2)
    arr1 = pulse_gain * np.ones(samples)
    arr2 = idle_gain * np.ones(idle_samples)
    array = np.concatenate((arr_idle_sandwich, arr1, arr_idle_sandwich))
    array_full = array.copy()
    nloops = n_pulses - 1
    for n in range(nloops):
        array_full = np.append(array_full, array)
    samples_per_clock = cycles_per_clk / interp
    # pad with idle_gain at the end
    if len(array_full) // cycles_per_clk < 3:
        extra_samples = 3 * samples_per_clock - len(array_full)
    else:
        remainder = len(array_full) % samples_per_clock
        extra_samples = samples_per_clock - remainder
    arr2 = np.ones(int(extra_samples)) * idle_gain
    pulse_final = np.append(array_full, arr2)

    return pulse_final * MAX_DAC_GAIN


def baseband_centered(
    pulse_gain: float,
    idle_gain: float,
    pulse_length: float,
    idle_length: float,
    gen: int,
    soccfg: QickConfig,
) -> np.ndarray:
    """Create a baseband pulse with idle time/2 before and idle time/2 after.

    :param pulse_length: duration of pulse in microseconds
    :param idle_length: duration of idle in microseconds
    """
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
    """Convert time in microseconds to dac clock cycles."""
    fs = soccfg["gens"][gen]["f_dds"]  # this may be the wrong number to use, seems to work for now
    cycles = int(time * fs)
    return cycles


def generate_ramp(
    start_gain: float, stop_gain: float, time: float, gen: int, soccfg: QickConfig
) -> np.ndarray:
    """Generate a ramp envelope.

    :param time: ramp duration in microseconds
    """
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
    start_sweep: float,
    end_sweep: float,
    ramp_ampl: float,
    ramp_time: float,
    sweep_points: int,
    gen: int,
    soccfg: QickConfig,
) -> tuple:
    """Generate a ramp waveform for sweeping ramps."""
    slope = ramp_ampl / ramp_time
    dac_incr = (end_sweep - start_sweep) / sweep_points
    time_incr = dac_incr / slope
    envelope_incr = soccfg.us2cycles(time_incr, gen_ch=gen)
    stop_r_final = np.abs(ramp_ampl) + end_sweep  # end of programmed ramp
    length_full = (stop_r_final - start_sweep) / slope  # length of full ramp in microseconds
    ramp = generate_ramp(start_sweep, stop_r_final, np.abs(length_full), gen, soccfg)
    return ramp, envelope_incr


def add_wf_filter(
    wf: np.ndarray,
    pre: int | None = None,
    post: int | None = None,
    iir_butter: float = 1e9,
    fir_butter: float = 80e6,
):
    """Apply filters to the waveforms as specified in file_settings.

    :param wf: waveform to apply filter to
    :param pre: adds a specified number of identical waveforms before the waveform being filtered,
        and then applies the filter. The script then removes the extra waveforms.
    :param post: adds a specified number of identical waveforms after the waveform being filtered,
        and then applies the filter. The script then removes the extra waveforms.
    :param iir_butter: apply an additional butterworth filter to the iir filtered data to round the
        baseband pulse edges
    :param fir_butter: apply an additional butterworth filter to the fir filtered data to round the
        baseband pulse edges
    """
    wf_full = wf
    if pre is not None:
        for n in range(pre):
            wf_full = np.concatenate((wf, wf_full))
        if post is not None:
            for n in range(post):
                wf_full = np.concatenate((wf_full, wf))
    elif post is not None:
        for n in range(post):
            wf_full = np.concatenate((wf_full, wf))
    filter_mode = filter_settings.apply_filter
    if filter_mode is None:
        wf_filt = wf_full
    elif filter_mode == "iir_1" or filter_mode == "both":
        wf_rounded = wf_full
        if filter_settings.iir_taps is not None:
            iir_b, iir_a = filter_settings.iir_taps
        else:
            raise Exception(" no iir_1 taps specified ")
        wf_filt_1 = signal.lfilter(iir_b, iir_a, wf_rounded)
        b, a = signal.butter(1, iir_butter, fs=6.881e9)  # apply a lpf to round the pulse edges
        wf_filt = signal.filtfilt(b, a, wf_filt_1)
        if filter_settings.fir_taps is not None:
            wf_filt = signal.convolve(wf_filt_1, filter_settings.fir_taps, mode="same")
    elif filter_mode == "both":
        if filter_settings.iir_2_taps is not None:
            iir_b, iir_a = filter_settings.iir_2_taps
        else:
            raise Exception(" no iir_2 taps specified ")
        wf_filt = signal.lfilter(iir_b, iir_a, wf_full)
    elif filter_mode == "fir":
        if filter_settings.fir_taps is not None:
            b, a = signal.butter(1, fir_butter, fs=6.881e9)
            wf_rounded = signal.lfilter(b, a, wf_full)
            wf_filt = signal.convolve(wf_rounded, filter_settings.fir_taps, mode="same")
        else:
            raise Exception(" no fir taps specified ")
    if pre is not None:
        pre_length = len(wf) * pre
    else:
        pre_length = 0
    if post is not None:
        post_length = -1 * len(wf) * post
        wf_final = wf_filt[pre_length:post_length]
    else:
        wf_final = wf_filt[pre_length:]

    return wf_final


def offset_sine_wave(
    gen: int, sine_ampl: float, sine_freq: float, step_ampl: float, soccfg: QickConfig
):
    """Produces a sine wave with a DC offset.

    :param gen: DAC generator number
    :param sine_ampl: amplitude of sine wave in gain units
    :param sine_freq: frequency of sine wave in MHz
    :param step_ampl: amplitude of DC offset in gain units
    :param soccfg: QickConfig object describing the generators
    """
    cycles_per_clk = soccfg["gens"][gen]["samps_per_clk"]
    fs = soccfg["gens"][gen]["f_dds"]
    cycle_time = 1 / fs
    sine_period = 1 / sine_freq
    steps = sine_period // cycle_time
    min_pulse_length = 3 * cycles_per_clk
    # TODO need to handle high frequency pulses better
    remainder = steps % cycles_per_clk
    steps -= remainder
    if steps < min_pulse_length:
        steps = min_pulse_length
    time_array = np.linspace(0, steps * cycle_time, int(steps))
    pulse = sine_ampl * np.sin(2 * np.pi * sine_freq * time_array) + step_ampl
    return pulse * MAX_DAC_GAIN
