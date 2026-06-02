"""Defines the MeasureNoise class used to perform general noise measurements with QICK."""

import logging
import time
from typing import Literal, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal

from spinqick.core import dot_experiment, spinqick_data
from spinqick.helper_functions import (
    analysis,
    hardware_manager,
    plot_tools,
    spinqick_enums,
)
from spinqick.models import experiment_models
from spinqick.qick_code_v2 import (
    measure_noise_programs_v2,
    tune_electrostatics_programs_v2,
)

logger = logging.getLogger(__name__)

# TODO create FFT plotting function


def _charge_noise_psd(m_dot_slope, side_psd, peak_psd, off_psd, side_g, peak_g):
    psd_gate = (
        1
        / m_dot_slope**2
        * ((side_psd - off_psd) - side_g**2 / peak_g**2 * (peak_psd - off_psd))
    )
    return psd_gate


def _process_retune_data(
    dataset: spinqick_data.SpinqickData, m_dot: str, npts_fit: int
):
    peak = dataset.fit_param_dict["center"]
    sigma = dataset.fit_param_dict["sigma"]
    side = peak - sigma
    off = peak - 10 * sigma
    mbias_points = [peak, side, off]
    assert dataset.analyzed_data
    i_adc = dataset.analyzed_data[0][0]
    i_adc_filt = signal.savgol_filter(i_adc, 50, 2)
    vdcs = dataset.axes["x"][m_dot]["data"]

    side_pt = np.argmin(np.abs(vdcs - side))
    peak_i = i_adc_filt[np.argmin(np.abs(vdcs - peak))]
    side_i = i_adc_filt[side_pt]
    fit_pts = npts_fit // 2
    vfit = vdcs[side_pt - fit_pts : side_pt + fit_pts]
    ifit = i_adc_filt[side_pt - fit_pts : side_pt + fit_pts]
    linefit = analysis.fit_line(vfit, ifit)
    slope = linefit.params["slope"].value
    return mbias_points, side_i, peak_i, slope


def _average_charge_noise(
    times: np.ndarray, dataset: spinqick_data.SpinqickData, num_avgs: int
):
    assert dataset.analyzed_data
    for n in range(num_avgs):
        centered_data = (
            dataset.analyzed_data[0][0][n, :] - dataset.analyzed_data[0].mean()
        )
        samplerate = len(times) / (times[-1] - times[0])
        freq, power = signal.periodogram(centered_data, fs=samplerate)
        if n == 0:
            power_sum = power
        else:
            power_sum += power
    asd_tot = np.sqrt(power_sum / num_avgs)
    return freq, asd_tot


def _make_charge_noise_plots(
    freq, charge_noise, peak_asd, side_asd, off_asd, timestamp, integration_cutoff
):
    fig1 = plot_tools.plot1_simple(
        freq,
        np.sqrt(charge_noise),
        timestamp,
        marker=".",
        xlabel="freq (Hz)",
        ylabel="Vrms/RtHz",
        title="gate referred charge noise",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-9, 1e-3)

    fig2 = plot_tools.plot1_simple(
        freq,
        peak_asd,
        timestamp,
        marker=".",
        xlabel="freq (Hz)",
        ylabel="ADCrms/RtHz",
        title="gate referred charge noise",
        label="peak",
    )
    plt.plot(freq, side_asd, label="side")
    plt.plot(freq, off_asd, label="off")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("ADCrms/RtHz")
    plt.xlabel("freq (Hz)")
    plt.ylim(1e-4, 1)

    df = freq[-1] - freq[-2]

    fig3 = plot_tools.plot1_simple(
        freq[freq > integration_cutoff],
        np.sqrt(np.cumsum(charge_noise[freq > integration_cutoff] * df)),
        timestamp,
        marker=".",
        xlabel="freq (Hz)",
        ylabel="Vrms (V)",
        title="integrated noise",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-6, 1e-4)

    fignums = [fig1.number, fig2.number, fig3.number]
    figs = [fig1, fig2, fig3]
    return figs, fignums


class MeasureNoise(dot_experiment.DotExperiment):
    """This class holds functions that help the user characterize their system's noise.

    :param soccfg: QickConfig object
    :param soc: Qick object
    :param voltage_source: Initialized DC voltage source object. This is used here for saving the DC
        voltage state each time data is saved.
    """

    def __init__(
        self, soccfg, soc, voltage_source: hardware_manager.VoltageSource, **kwargs
    ):
        super().__init__(**kwargs)
        self.soccfg = soccfg
        self.soc = soc
        self.vdc = hardware_manager.DCSource(voltage_source=voltage_source)

    @dot_experiment.updater
    def readout_noise_at_bias(
        self,
        m_dot: spinqick_enums.GateNames,
        retune_data: spinqick_data.SpinqickData,
        measure_buffer: float,
        time_steps: int = 1000,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
        num_avgs: int = 1,
        bias_settle_time: float = 5,
        npts_fit: int = 20,
    ):
        """Measure noise spectrum at a specific bias configuration of the device."""

        current_m_bias = self.vdc.get_dc_voltage(m_dot)
        times = (
            1e-6
            * np.linspace(0, 2 * measure_buffer + self.dcs_config.length, time_steps)
            * time_steps
        )

        gvg_cfg = experiment_models.StaticConfig(
            measure_buffer=measure_buffer,
            points=time_steps,
            avgs=num_avgs,
            dcs_cfg=self.dcs_config,
            mode=mode,
        )
        meas = tune_electrostatics_programs_v2.Static(
            self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
        )
        m_bias_points, side_i, peak_i, slope = _process_retune_data(
            retune_data, m_dot, npts_fit
        )
        data_list = []
        asd_list = []
        for m_bias in m_bias_points:
            print(f"going to point {m_bias}")
            self.vdc.set_dc_voltage(m_bias, m_dot)
            time.sleep(bias_settle_time)
            data = meas.acquire(self.soc, progress=False)
            assert data
            qd = spinqick_data.SpinqickData(
                data,
                gvg_cfg,
                1,
                1,
                "_noise",
                voltage_state=self.vdc.all_voltages,
                prog=meas,
            )
            qd.add_axis([times], "x", [m_dot], time_steps, units=["us"], loop_no=1)
            qd.add_axis([np.arange(num_avgs)], "avgs", ["avgs"], num_avgs, loop_no=0)
            if mode == "sd_chop":
                analysis.calculate_conductance(
                    qd,
                    self.adc_unit_conversions,
                )
            else:
                analysis.calculate_transconductance(
                    qd,
                    self.adc_unit_conversions,
                )
            data_list.append(qd)
            freq, asd_avged = _average_charge_noise(times, qd, num_avgs)
            asd_list.append(asd_avged)
        self.vdc.set_dc_voltage(current_m_bias, m_dot)
        [peak, side, off] = m_bias_points
        [peak_asd, side_asd, off_asd] = asd_list
        charge_noise = _charge_noise_psd(
            slope, side_asd**2, peak_asd**2, off_asd**2, side_i, peak_i
        )
        dset_labels = ["peak", "side", "off"]

        full_dataset = spinqick_data.CompositeSpinqickData(
            data_list,
            dset_labels,
            "_charge_noise",
            dset_coordinates=freq,
            analyzed_data=charge_noise,
            dset_coordinate_units="Hz",
        )
        full_dataset.dset_coordinate_units = "Hz"

        if self.plot:
            _, fignums = _make_charge_noise_plots(
                freq,
                charge_noise,
                peak_asd,
                side_asd,
                off_asd,
                full_dataset.timestamp,
                20,
            )
        if self.save_data:
            plot_figs = fignums if self.plot else []
            self.finalize(full_dataset, fignums=plot_figs)
        return full_dataset

    @dot_experiment.updater
    def dcs_stability(
        self,
        m_dot: spinqick_enums.GateNames,
        m_range: Tuple[float, float, int],
        measure_buffer: float,
        time_steps: int = 1000,
        freq_cutoff: float = 0.1,
        wait_time: float | None = None,
        frequency_fit: bool = True,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
    ) -> spinqick_data.CompositeSpinqickData:
        """Track DCS peak for longer timescale (minutes) and look for drift and noise.  Right now
        this is coded to work with only one adc readout.

        :param m_dot: M dot to track
        :param m_range: List of start voltage, stop voltage and number of points. Voltages are
            relative to the current setpoint
        :param time_steps: number of times to run the sweep
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and
            the QICK starts a DCS measurement.
        :param wait_time: delay time in seconds between m-gate measurements, to measure longer
            timescale drift
        :param frequency_fit: if true, fits frequency spectrum
        :param freq_cutoff: fit frequencies below this value to a power law
        """

        ### M Sweep
        m_bias = self.vdc.get_dc_voltage(m_dot)
        n_vm = int(m_range[2])
        vm_start = m_range[0] + m_bias
        vm_stop = m_range[1] + m_bias
        vm_sweep = np.linspace(vm_start, vm_stop, n_vm)
        # setup the slow_dac step length
        slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer

        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=self.hardware_config.dac_settings.trig_pin,
            trig_length=self.hardware_config.dac_settings.trig_length,
            measure_buffer=measure_buffer,
            points=n_vm,
            dcs_cfg=self.dcs_config,
            mode=mode,
        )
        ### setup the slow_dac step length
        slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer

        data_list = []
        times = np.zeros((time_steps))
        data_array = np.zeros((len(times), n_vm))
        for step in range(time_steps):
            self.vdc.program_ramp(
                vm_start, vm_stop, slow_dac_step_len * 1e-6, n_vm, m_dot
            )
            self.vdc.arm_sweep(m_dot)
            meas = tune_electrostatics_programs_v2.GvG(
                self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
            )
            data = meas.acquire(self.soc, progress=False)
            assert data
            qd = spinqick_data.SpinqickData(
                data,
                gvg_cfg,
                1,
                1,
                "_m_gate_sweep",
                voltage_state=self.vdc.all_voltages,
                prog=meas,
            )
            qd.add_axis([vm_sweep], "x", [m_dot], n_vm, units=["V"])
            if mode == "sd_chop":
                analysis.calculate_conductance(
                    qd,
                    self.adc_unit_conversions,
                )
            else:
                analysis.calculate_transconductance(
                    qd,
                    self.adc_unit_conversions,
                )
            data_list.append(qd)
            times[step] = time.time_ns() / 1e9
            assert qd.analyzed_data is not None
            data_array[step, :] = qd.analyzed_data[0][0]
            # time.sleep(0.001)
            if wait_time:
                time.sleep(wait_time)

        # return to initial bias
        dset_labels = [str(times[i]) for i in range(time_steps)]
        self.vdc.set_dc_voltage(m_bias, m_dot)
        full_dataset = spinqick_data.CompositeSpinqickData(
            data_list,
            dset_labels,
            "_m_tracking",
            dset_coordinates=times,
            dset_coordinate_units="s",
        )

        ### now fit the data
        center_data = np.zeros((time_steps))
        for step in range(time_steps):
            step_data = full_dataset.qdata_array[step]
            assert step_data.analyzed_data is not None
            ydata = step_data.analyzed_data[0][0]
            xdata = vm_sweep
            try:
                _, out = analysis.fit_gaussian(xdata, ydata)
                if np.logical_and(
                    out.params["center"].value > vm_start,
                    out.params["center"].value < vm_stop,
                ):
                    center_data[step] = out.params["center"].value
                else:
                    center_data[step] = np.nan
            except Exception as exc:
                logger.error("fit failed: %s", exc, exc_info=True)
            if np.logical_and(
                out.params["center"].value > vm_start,
                out.params["center"].value < vm_stop,
            ):
                center_data[step] = out.params["center"].value
            else:
                center_data[step] = np.nan
            full_dataset.analyzed_data = center_data

        ### get the PSD
        if frequency_fit:
            samplerate = len(times) / (times[-1] - times[0])
            freq, power = signal.periodogram(
                center_data - np.mean(center_data), fs=samplerate
            )
            asd = np.sqrt(power)

            def linfit(f, pwr, a):
                return a * np.power(f, pwr)

            def linfit_log(logf, m, b):
                return m * logf + b

            freq_fit = None
            try:
                # pylint: disable-next=unbalanced-tuple-unpacking
                popt, _ = curve_fit(
                    linfit_log,
                    np.log10(freq[np.logical_and(freq > 0, freq < freq_cutoff)]),
                    np.log10(asd[np.logical_and(freq > 0, freq < freq_cutoff)]),
                )
                icept = np.power(10, popt[1])
                fit_params = {"power": popt[0], "intercept": popt[1]}
                full_dataset.fit_param_dict = fit_params
                print("pow equals %f" % popt[0])
                print("intercept equals %f" % icept)  # pylint: disable=possibly-used-before-assignment
                freq_fit = linfit(freq, popt[0], 10 ** popt[1])
            except ValueError:
                print("PSD fit failed")
            if self.plot:
                freq_fig = plt.figure()
                plt.loglog(freq, asd, "k.")
                if freq_fit is not None:
                    plt.plot(freq, freq_fit, "r-")
                plt.ylabel("Vrms/RtHz")
                plt.xlabel("freq (Hz)")
                plt.title("DCS peak location noise")
                plt.loglog(freq, asd)
                plt.ylim(1e-7, np.max(asd) * 1.1)
                freq_plot_num = freq_fig.number
        if self.plot:
            fig = plot_tools.plot2_simple(
                times,
                vm_sweep,
                np.transpose(data_array),
                full_dataset.timestamp,
                title=m_dot,
                ylabel=" %s voltage (V)" % m_dot,
                xlabel="time in seconds",
            )
            plt.plot(times, center_data)
            plt.title(m_dot)
            plt.ylabel(" %s voltage (V)" % m_dot)
            plt.xlabel("time in seconds")
            full_plot_num = fig.number

        if self.save_data:
            plot_figs_stability: list[int | str | None] = []
            if self.plot:
                plot_figs_stability.append(full_plot_num)
                if frequency_fit:
                    plot_figs_stability.append(freq_plot_num)
            self.finalize(
                full_dataset,
                fignums=plot_figs_stability if plot_figs_stability else None,
            )
        return full_dataset

    @dot_experiment.updater
    def readout_noise_raw(
        self,
        readout_time: float,
        n_averages: int = 2,
        add_tone: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grab raw time traces from one ADC and fft them.  Using DDR4 buffer we are able to get up
        to ~3 seconds of raw data.

        :param readout_time: time (in seconds) to collect data on ddr4 buffer
        :param n_averages: number of times to repeat data capture, averaging the psd every time
        :param add_tone: play the readout tone while capturing data
        :returns:
                -frequency of fft points
                -averaged amplitude spectral density
        """

        clock_tick = 1 / self.soc.get_cfg()["readouts"][0]["f_fabric"] * 1e-6
        n_transfers = (
            int(readout_time / clock_tick / 128) + self.soccfg["ddr4_buf"]["junk_len"]
        )

        qickprogram = measure_noise_programs_v2.grab_noise(
            self.soccfg,
            self.dcs_config,
            demodulate=False,
            readout_tone=add_tone,
            continuous_tone=add_tone,
        )

        for n in range(n_averages):
            self.soc.arm_ddr4(ch=self.dcs_config.ro_chs[0], nt=n_transfers)
            qickprogram.config_all(self.soc)
            self.soc.tproc.start()
            iq = self.soc.get_ddr4(n_transfers)
            self.soc.reset_gens()  # in case the readout tone was left on
            complex_iq = iq.dot([1, 1j])

            print("done, average number %d" % n)
            self.soc.reset_gens()
            if n == 0:
                freq, power = signal.periodogram(np.abs(complex_iq), fs=1 / clock_tick)
                power_fft = power
            else:
                freq, power_fft_single = signal.periodogram(
                    np.abs(complex_iq), fs=1 / clock_tick
                )
                power_fft += power_fft_single

        average_asd = np.sqrt(power_fft / n_averages)
        # print(qickprogram)

        if self.plot:
            plt.figure()
            plt.loglog(freq, average_asd)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("ASD (ADC units/rtHz)")
            plt.ylim(np.min(average_asd[1:]) / 2, np.max(average_asd) * 2)

        if self.save_data:
            # TODO add data saving
            pass
        return freq, average_asd

    @dot_experiment.updater
    def readout_noise_demodulate(
        self,
        readout_time: float,
        n_averages: int = 10,
        continuous_tone: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grab raw time traces from one ADC and fft them. Play a tone and turn on demodulation.
        Using DDR4 buffer we are able to capture up to ~3 seconds of raw data.

        :param readout_time: time in seconds to collect data on ddr4 buffer
        :param n_averages: number of times to repeat data capture, averaging the psd every time
        :param continuous_tone: play the readout tone constantly
        :returns:
                -frequency of fft points
                -averaged amplitude spectral density
        """

        clock_tick = 1 / self.soc.get_cfg()["readouts"][0]["f_fabric"] * 1e-6
        n_transfers = (
            int(readout_time / clock_tick / 128) + self.soccfg["ddr4_buf"]["junk_len"]
        )
        qickprogram = measure_noise_programs_v2.grab_noise(
            self.soccfg,
            self.dcs_config,
            demodulate=True,
            readout_tone=True,
            continuous_tone=continuous_tone,
        )

        for n in range(n_averages):
            self.soc.arm_ddr4(ch=self.dcs_config.ro_chs[0], nt=n_transfers)
            qickprogram.config_all(self.soc)
            self.soc.tproc.start()
            iq = self.soc.get_ddr4(n_transfers)
            complex_iq = iq.dot([1, 1j])

            print("done, average number %d" % n)
            if n == 0:
                freq, power = signal.periodogram(np.abs(complex_iq), fs=1 / clock_tick)
                power_fft = power
            else:
                freq, power_fft_single = signal.periodogram(
                    np.abs(complex_iq), fs=1 / clock_tick
                )
                power_fft += power_fft_single

        average_asd = np.sqrt(power_fft / n_averages)
        if self.plot:
            plt.figure()
            plt.loglog(freq, average_asd)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("ASD (ADC units/rtHz)")
            plt.ylim(np.min(average_asd[1:]) / 2, np.max(average_asd) * 2)

        if self.save_data:
            # TODO add data saving
            pass
        return freq, average_asd
