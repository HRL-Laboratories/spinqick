"""Defines the MeasureNoise class used to perform general noise measurements with QICK."""

import logging
import time
from typing import Literal, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import periodogram

from spinqick import settings
from spinqick.core import dot_experiment, spinqick_data
from spinqick.helper_functions import analysis, hardware_manager, plot_tools
from spinqick.models import experiment_models
from spinqick.qick_code_v2 import (
    measure_noise_programs_v2,
    tune_electrostatics_programs_v2,
)
from spinqick.settings import dac_settings

logger = logging.getLogger(__name__)

# TODO create FFT plotting function


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
        m_dot: settings.GateNames,
        m_bias: float,
        measure_buffer: float,
        time_steps: int = 1000,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
        num_avgs: int = 1,
    ) -> spinqick_data.SpinqickData:
        """Measure noise spectrum at a specific bias configuration of the device."""

        current_m_bias = self.vdc.get_dc_voltage(m_dot)
        times = (
            1e-6
            * np.linspace(0, 2 * measure_buffer + self.dcs_config.length, time_steps)
            * time_steps
        )

        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=dac_settings.trig_pin,
            trig_length=dac_settings.trig_length,
            measure_buffer=measure_buffer,
            points=time_steps,
            dcs_cfg=self.dcs_config,
            mode=mode,
        )
        self.vdc.set_dc_voltage(m_bias, m_dot)
        meas = tune_electrostatics_programs_v2.Static(
            self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
        )
        data_list = []
        for n in range(num_avgs):
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
            qd.add_axis([times], "x", [m_dot], time_steps, units=["us"])
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
            assert qd.analyzed_data
            centered_data = qd.analyzed_data[0][0] - qd.analyzed_data[0].mean()
            samplerate = len(times) / (times[-1] - times[0])
            freq, power = periodogram(centered_data, fs=samplerate)
            asd = np.sqrt(power)
            data_list.append(qd)
            if n == 0:
                asd_sum = asd
            else:
                asd_sum += asd
        dset_labels = [str(freq[i]) for i in range(len(freq))]
        asd_tot = asd_sum / num_avgs
        assert isinstance(asd_tot, np.ndarray)
        full_dataset = spinqick_data.CompositeSpinqickData(
            data_list,
            dset_labels,
            "_charge_noise",
            dset_coordinates=freq,
            analyzed_data=asd_tot,
            dset_coordinate_units="Hz",
        )
        full_dataset.dset_coordinate_units = "Hz"

        if self.plot:
            fig = plot_tools.plot1_simple(
                freq, asd_tot, full_dataset.timestamp, marker="."
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel("ADCrms/RtHz")
            plt.xlabel("freq (Hz)")
            plt.ylim(1e-4, 1)
            plt.title("mbias = %.4f" % m_bias)
            full_plot_num = fig.number
        # TODO implement data and plot saving

        self.vdc.set_dc_voltage(current_m_bias, m_dot)
        if self.save_data:
            nc_file = full_dataset.basic_composite_save()
            if self.plot:
                nc_file.save_last_plot(fignum=full_plot_num)
            nc_file.close()
            logger.info("data saved at %s", full_dataset.data_file)
        return qd

    @dot_experiment.updater
    def dcs_stability(
        self,
        m_dot: settings.GateNames,
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
            trig_pin=dac_settings.trig_pin,
            trig_length=dac_settings.trig_length,
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
            freq, power = periodogram(center_data - np.mean(center_data), fs=samplerate)
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
                times, vm_sweep, np.transpose(data_array), full_dataset.timestamp
            )
            plt.plot(times, center_data)
            plt.title(m_dot)
            plt.ylabel(" %s voltage (V)" % m_dot)
            plt.xlabel("time in seconds")
            full_plot_num = fig.number

        if self.save_data:
            nc_file = full_dataset.basic_composite_save()
            if self.plot:
                nc_file.save_last_plot(fignum=full_plot_num)
                if frequency_fit:
                    nc_file.save_last_plot(fignum=freq_plot_num)
            nc_file.close()
            logger.info("data saved at %s", full_dataset.data_file)
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
                freq, power = periodogram(np.abs(complex_iq), fs=1 / clock_tick)
                power_fft = power
            else:
                freq, power_fft_single = periodogram(
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
                freq, power = periodogram(np.abs(complex_iq), fs=1 / clock_tick)
                power_fft = power
            else:
                freq, power_fft_single = periodogram(
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
