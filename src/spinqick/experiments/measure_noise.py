"""
Perform noise measurements with QICK

"""

import logging
import os
import time
from typing import Tuple

import netCDF4
import numpy as np
from lmfit.models import GaussianModel
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import periodogram

from spinqick.experiments import dot_experiment
from spinqick.helper_functions import file_manager, hardware_manager
from spinqick.qick_code import measure_noise_programs, tune_electrostatics_programs
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)


class MeasureNoise(dot_experiment.DotExperiment):
    """This class holds functions that help the user characterize their system's noise."""

    def __init__(
        self,
        soccfg,
        soc,
        voltage_source: hardware_manager.VoltageSource,
        datadir: str = file_settings.data_directory,
    ):
        """
        :param soccfg: QickConfig object
        :param soc: Qick object
        :param datadir: data directory where all data is being stored. Experiment will make a folder here with today's date.
        """
        super().__init__(datadir=datadir)
        self.soccfg = soccfg
        self.soc = soc
        self.datadir = datadir
        self.vdc = hardware_manager.DCSource(voltage_source=voltage_source)

    @dot_experiment.updater
    def dcs_stability(
        self,
        m_dot: str,
        m_range: Tuple[float, float, int] = (-0.008, 0.008, 50),
        time_steps: int = 1000,
        freq_cutoff: float = 0.1,
        measure_buffer: float = 30,
        wait_time: float | None = None,
        frequency_fit: bool = True,
        save_data: bool = True,
        plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track DCS peak for longer timescale (minutes) and look for drift and noise.

        :param m_dot: M dot to track
        :param m_range: List of start voltage, stop voltage and number of points.  Voltages are relative
            to the current setpoint
        :param time_steps: number of times to run the sweep
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param wait_time: additional time in seconds between loops, measure longer timescale drift
        :param frequency_fit: fit frequencies below this value to a power law
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: data, times, center_data
        """

        ### M Sweep
        m_bias = self.vdc.get_dc_voltage(m_dot)
        n_vm = int(m_range[2])
        vm_start = m_range[0] + m_bias
        vm_stop = m_range[1] + m_bias
        vm_sweep = np.linspace(vm_start, vm_stop, n_vm)

        ### setup the slow_dac step length
        slow_dac_step_len = (
            self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
        )
        
        # parameters for GvG
        self.config.gvg_expt.measure_delay = measure_buffer
        self.config.expts = n_vm
        self.config.start = vm_start
        self.config.step = (vm_stop - vm_start) / n_vm
        self.config.gvg_expt.measure_delay = measure_buffer
        self.config.reps = 1

        data = np.zeros((n_vm, time_steps))
        times = np.zeros((time_steps))
        for step in range(time_steps):
            self.vdc.program_ramp(
                vm_start, vm_stop, slow_dac_step_len * 1e-6, n_vm, m_dot
            )
            self.vdc.arm_sweep(m_dot)

            ### Start a Vy sweep at a Vx increment and store the data
            meas = tune_electrostatics_programs.GvG(self.soccfg, self.config)
            expt_pts, avgi, avgq = meas.acquire(
                self.soc, load_pulses=True, progress=False
            )
            mag = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)
            data[:, step] = mag
            times[step] = time.time_ns() / 1e9
            time.sleep(0.2)
            if wait_time:
                time.sleep(wait_time)

        # return to initial bias
        self.vdc.set_dc_voltage(m_bias, m_dot)

        ### now fit the data
        center_data = np.zeros((time_steps))
        for step in range(time_steps):
            mag = data[:, step]
            gaussian = GaussianModel()
            pars = gaussian.guess(mag, x=vm_sweep)
            try:
                out = gaussian.fit(mag, pars, x=vm_sweep)
                if np.logical_and(
                    out.params["center"].value > vm_start,
                    out.params["center"].value < vm_stop,
                ):
                    center_data[step] = out.params["center"].value
                else:
                    center_data[step] = np.nan
            except Exception as exc:
                logger.error("fit failed: %s", exc, exc_info=True)

        data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
        ### plot the data
        if plot:
            plt.figure()
            plt.clf()
            plt.pcolormesh(
                times - times[0], vm_sweep, data, shading="nearest", cmap="binary_r"
            )
            plt.plot(times - times[0], center_data, "m")
            plt.title("dcs stability")
            plt.title("t: %d" % stamp, loc="right", fontdict={"fontsize": 6})
            plt.xlabel(" time (s)")
            plt.ylabel("vm")

        ### get the PSD and plot/fit it
        if frequency_fit:
            samplerate = len(times) / (times[-1] - times[0])
            freq, power = periodogram(center_data - np.mean(center_data), fs=samplerate)
            asd = np.sqrt(power)

            def linfit(f, pow, A):
                return A * np.power(f, pow)

            def linfit_log(logf, m, b):
                return m * logf + b

            popt, pcov = curve_fit(
                linfit_log,
                np.log10(freq[np.logical_and(freq > 0, freq < freq_cutoff)]),
                np.log10(asd[np.logical_and(freq > 0, freq < freq_cutoff)]),
            )

            icept = np.power(10, popt[1])

        if save_data:
            # make a directory for today's date and create a unique timestamp
            # data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_dcs_stability.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_dcs_stability.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_dcs_stability.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            if plot:
                plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("t", time_steps)
            nc_file.createDimension(m_dot, n_vm)

            # create variables and fill in data
            t = nc_file.createVariable("t", np.float32, ("t"))
            t.units = "ns"
            t[:] = times
            M = nc_file.createVariable(m_dot, np.float32, (m_dot))
            M.units = "V"
            M[:] = vm_sweep
            raw = nc_file.createVariable("processed", np.float32, (m_dot, "t"))
            raw[:] = data
            processed = nc_file.createVariable("centerdata", np.float32, ("t"))
            processed[:] = center_data
            nc_file.close()
            logger.info("data saved at %s" % data_file)
        if np.logical_and(plot, frequency_fit):
            plt.figure()
            plt.loglog(freq, asd, "k.")
            plt.plot(freq, linfit(freq, popt[0], 10 ** popt[1]), "r-")
            print("pow equals %f" % popt[0])
            print("intercept equals %f" % icept)
            plt.ylabel("Vrms/RtHz")
            plt.xlabel("freq (Hz)")
            plt.title("DCS peak location noise")
            plt.title("t: %d" % stamp, loc="right", fontdict={"fontsize": 6})
            plt.loglog(freq, asd)
            plt.ylim(1e-7, np.max(asd) * 1.1)
            if save_data:
                fig_file = os.path.join(
                    data_path, str(stamp) + "_fft_dcs_stability.png"
                )
                plt.savefig(fig_file)

        return times, data, center_data

    @dot_experiment.updater
    def readout_noise_raw(
        self,
        readout_time: float,
        n_averages: int = 2,
        add_tone: bool = False,
        save_data: bool = True,
        plot: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grab raw time traces from the ADC and fft them.  Using DDR4 buffer
        we are able to get up to ~3 seconds of raw data.

        :param readout_time: time (in seconds) to collect data on ddr4 buffer
        :param n_averages: number of times to repeat data capture, averaging the psd every time
        :param add_tone: play the readout tone while capturing data
        :param save_data: automatically save data
        :param plot: plot the final FFT
        :returns:
                -frequency of fft points
                -averaged amplitude spectral density
        """

        pulse_time = (
            self.config.DCS_cfg.length
        )  # not sure if we want to blast the device with a tone for the full period
        clock_tick = 1 / self.soc.get_cfg()["readouts"][0]["f_fabric"] * 1e-6
        n_transfers = (
            int(readout_time / clock_tick / 128) + self.soccfg["ddr4_buf"]["junk_len"]
        )

        qickprogram = measure_noise_programs.grab_noise(
            self.soccfg,
            self.config,
            pulse_time,
            demodulate=False,
            readout_tone=add_tone,
            continuous_tone=add_tone,
        )

        for n in range(n_averages):
            self.soc.arm_ddr4(ch=self.config.DCS_cfg.ro_ch, nt=n_transfers)
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
        print(qickprogram)

        if plot:
            plt.figure()
            plt.loglog(freq, average_asd)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("ASD (ADC units/rtHz)")
            plt.ylim(np.min(average_asd[1:]) / 2, np.max(average_asd) * 2)

        if save_data:
            # TODO add data saving
            pass
        return freq, average_asd

    @dot_experiment.updater
    def readout_noise_demodulate(
        self,
        readout_time: float,
        n_averages: int = 10,
        continuous_tone: bool = False,
        save_data: bool = True,
        plot: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grab raw time traces from the ADC and fft them. Play a tone and turn on demodulation.
        Using DDR4 buffer we are able to capture up to ~3 seconds of raw data

        :param readout_time: time in seconds to collect data on ddr4 buffer
        :param n_averages: number of times to repeat data capture, averaging the psd every time
        :param continuous_tone: play the readout tone constantly
        :param save_data: automatically save data
        :param plot: plot the final FFT
        :returns:
                -frequency of fft points
                -averaged amplitude spectral density
        """

        pulse_time = self.config.DCS_cfg.length  # check this
        clock_tick = 1 / self.soc.get_cfg()["readouts"][0]["f_fabric"] * 1e-6
        n_transfers = (
            int(readout_time / clock_tick / 128) + self.soccfg["ddr4_buf"]["junk_len"]
        )
        qickprogram = measure_noise_programs.grab_noise(
            self.soccfg,
            self.config,
            pulse_time,
            demodulate=True,
            readout_tone=True,
            continuous_tone=continuous_tone,
        )

        for n in range(n_averages):
            self.soc.arm_ddr4(ch=self.config.DCS_cfg.ro_ch, nt=n_transfers)
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
        if plot:
            plt.figure()
            plt.loglog(freq, average_asd)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("ASD (ADC units/rtHz)")
            plt.ylim(np.min(average_asd[1:]) / 2, np.max(average_asd) * 2)

        if save_data:
            pass
        return freq, average_asd
