"""
Module to hold functions that run Loss Divincenzo single-qubit experiments

"""

import logging
import os
from typing import List, Tuple

import netCDF4
import numpy as np
from matplotlib import pyplot as plt

from spinqick.experiments import dot_experiment
from spinqick.helper_functions import file_manager, plot_tools
from spinqick.qick_code import ld_single_qubit_programs
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)


class LDSingleQubit(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK code classes for single LD qubit experiments.
    These involve a microwave drive. These scripts are set up to generate a trigger which goes high during the RF pulse,
    so if the user is mixing their signal with an LO they can trigger an RF switch.

    """

    def __init__(self, soccfg, soc, datadir: str = file_settings.data_directory):
        """initialize with information about your rfsoc and your experimental setup

        :param soccfg: qick config object
        :param soc: QickConfig
        :param datadir: data directory where all data is being stored. Experiment will make a folder here with today's date.
        """
        super().__init__(datadir=datadir)
        self.soccfg = soccfg
        self.soc = soc
        self.datadir = datadir

    @dot_experiment.updater
    def rf_freq_scan(
        self,
        RF_gain: int,
        start_freq: float,
        stop_freq: float,
        num_pts: int,
        RF_gen: int,
        expt_avgs: int,
        RF_length: float = 10,
        plot: bool = True,
        save_data=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Play an RF pulse and scan frequency, look for resonance

        :param RF_gain: Gain of RF tone in DAC units
        :param start_freq: Lowest RF frequency in MHz
        :param stop_freq: Max RF frequency in MHz
        :param num_pts: Number of points in the frequency sweep
        :param RF_gen: RF channel number
        :param expt_avgs: Number of times to run sweep and average full experiemtn
        :param RF_length: Pulse length of RF drive in microseconds
        :param plot: Plot result
        :param save_data: Save data to netcdf
        """
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.start = start_freq
        self.config.RF_expt.stop = stop_freq
        self.config.RF_expt.expts = num_pts
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_length = self.soccfg.us2cycles(RF_length)
        self.config.expts = num_pts
        self.config.step = np.round((stop_freq - start_freq) / num_pts, decimals=6)
        self.config.reps = 1

        meas = ld_single_qubit_programs.ScanRFFrequency(self.soccfg, self.config)
        for avg in range(expt_avgs):
            expt_pts, temp_i, temp_q = meas.acquire(
                self.soc, load_pulses=True, progress=True
            )
            if avg == 0:
                avgi, avgq = temp_i[0], temp_q[0]
            else:
                avgi += temp_i[0]
                avgq += temp_q[0]
        avgi /= expt_avgs
        avgq /= expt_avgs

        self.soc.reset_gens()
        mag = np.sqrt(avgi**2 + avgq**2)
        differential_signal = mag[1] - mag[0]
        if plot:
            plt.figure()
            plt.plot(expt_pts, differential_signal)
            plt.ylabel("difference in conducatance (ADC raw units)")
            plt.xlabel("frequency (MHz)")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp()
            data_file = os.path.join(data_path, str(stamp) + "_RF_freq_scan.nc")
            cfg_file = os.path.join(data_path, str(stamp) + "_RF_freq_scan.yaml")

            # save the config
            file_manager.save_config(self.config, cfg_file)

            # save our scan data
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            nc_file.add_axis(label="frequency", data=expt_pts, units="MHz")
            nc_file.add_dataset(
                label="change_in_conductance",
                axes=["frequency"],
                data=differential_signal,
                units="adc_magnitude",
            )
            nc_file.data_flavour = "scan_rf_frequency"
            if plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s" % data_file)
        return expt_pts, differential_signal

    @dot_experiment.updater
    def rabi_chevron(
        self,
        RF_gain: int,
        RF_gen: int,
        freq_range: Tuple[float, float, int] = (1000, 2000, 50),
        time_range: Tuple[float, float, int] = (0.01, 10, 50),
        point_avgs: int = 10,
        RF_cooldown: float = 10,
        trig_offset: float = 0.1,
        plot: bool = True,
        save_data: bool = True,
    ):
        """sweeps frequency on FPGA and time in an outer python loop.

        :param RF_gain: DAC units
        :param RF_gen: DAC channel being used for RF
        :param freq_range: (start frequency (MHz), stop frequency (MHz), number of steps)
        :param time_range: (start time (us), stop time (us), number of steps)
        :param point_avgs: number of averages per point
        :param RF_cooldown: time to pause after RF drive before the second measurement
        :param trig_offset: time in microseconds between trigger on pin 0 and beginning of RF pulse
        """
        start_freq, stop_freq, freq_pts = freq_range
        start_time, stop_time, time_pts = time_range
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.start = start_freq
        self.config.RF_expt.stop = stop_freq
        self.config.RF_expt.expts = freq_pts
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.RF_expt.trig_offset = self.soccfg.us2cycles(trig_offset)
        self.config.expts = point_avgs
        # need these dummy parameters for our fake sweep
        self.config.start = 1
        self.config.stop = 10
        self.config.reps = 1

        times = np.linspace(start_time, stop_time, time_pts)
        data = np.zeros((2, 2, point_avgs, freq_pts, time_pts))
        for i, t in enumerate(times):
            print("loop number %d" % i)
            self.config.RF_expt.RF_time = self.soccfg.us2cycles(t)
            meas = ld_single_qubit_programs.RabiChevron(self.soccfg, self.config)
            expt_pts, avgi, avgq = meas.acquire(
                self.soc, load_pulses=True, progress=False
            )
            data[:, :, :, :, i] = [avgi[0], avgq[0]]
        if plot:
            if self.config.PSB_config.threshold:
                thresh = self.config.PSB_config.thresh
            else:
                thresh = None
            plot_tools.plot2_PSBdata(
                [times, expt_pts[0]], [data[0][0]], [data[0][1]], thresh=thresh
            )
            plt.xlabel("time (us)")
            if self.config.PSB_config.threshold:
                plt.ylabel("probability")
            else:
                plt.ylabel("DCS conductance (arbs)")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp()
            data_file = os.path.join(data_path, str(stamp) + "_RabiChevron_scan.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_RabiChevron_scan.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_RabiChevron_scan.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("frequency", freq_pts)
            nc_file.createDimension("time", time_pts)
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("shots", point_avgs)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata", np.float32, ("IQ", "triggers", "shots", "frequency", "time")
            )
            psbraw.units = "raw_adc"
            freq = nc_file.createVariable("frequency", np.float32, ("frequency"))
            freq.units = "MHz"
            freq[:] = expt_pts[0]
            time = nc_file.createVariable("time", np.float32, ("time"))
            time.units = "tproc"
            time.conversion = self.soccfg.cycles2us(1)
            time[:] = time_pts
            psbraw[:, :, :, :, :] = data
            processed = nc_file.createVariable(
                "processed", np.float32, ("frequency", "time")
            )
            processed[:] = plot_tools.interpret_data_PSB([data[0][0]], [data[0][1]])
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return times, expt_pts, data

    @dot_experiment.updater
    def rabi_chevron_v2(
        self,
        RF_gain: int,
        RF_gen: int,
        freq_range: Tuple[float, float, int] = (1000, 2000, 50),
        time_range: Tuple[float, float, int] = (0.01, 10, 50),
        point_avgs: int = 10,
        full_avgs: int = 1,
        RF_cooldown: float = 10,
        trig_ignore: bool = True,
        plot: bool = True,
        save_data: bool = True,
    ):
        """sweeps both frequency and time on FPGA to obtain Rabi Chevron data.
        Utilizes the FlexyPSBAveragerPrograms.

        :param RF_gain: DAC units
        :param RF_gen: DAC channel being used for RF
        :param freq_range: (start frequency (MHz), stop frequency (MHz), number of steps)
        :param time_range: (start time (us), stop time (us), number of steps)
        :param point_avgs: number of averages per point (shots)
        :param full_avgs: averages of full experiment (reps)
        :param RF_cooldown: time to pause after RF drive before the second measurement
        :param trig_ignore: Set to true to perform your measurement directly after the RF pulse.  If set to false, it waits for the trigger to return to zero
        :param plot: plot result
        :param save_data: save data and plot
        """

        start = self.soccfg.us2cycles(time_range[0])
        stop = self.soccfg.us2cycles(time_range[1])
        step = int((stop - start) / time_range[2])

        start_f = self.soccfg.freq2reg(freq_range[0], RF_gen)
        step_f = self.soccfg.freq2reg((freq_range[1] - freq_range[0]) / freq_range[2])

        self.config.start_outer = start_f
        self.config.step_outer = step_f
        self.config.expts_outer = freq_range[2]
        self.config.start = start
        self.config.step = step
        self.config.expts = time_range[2]
        self.config.RF_expt.trig_ignore = trig_ignore
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.RF_expt.RF_gen = RF_gen
        self.config.inner_loop = "time"
        self.config.outer_loop = "freq"
        self.config.shots = point_avgs
        self.config.reps = full_avgs

        meas = ld_single_qubit_programs.RabiChevronV2(self.soccfg, self.config)
        expt_pts, mag = meas.acquire(self.soc, load_pulses=True, progress=False)

        times = self.soccfg.cycles2us(expt_pts[0])
        frequencies = self.soccfg.reg2freq(expt_pts[1])
        if self.config.PSB_cfg.thresholding:
            meas_units = "singlet_probability"
        else:
            meas_units = "conductance_arbs"
        if plot:
            plot_tools.plot2_simple(
                xarray=times, yarray=frequencies, data=mag, cbar_label=meas_units
            )
            plt.xlabel("time (us)")
            plt.ylabel("frequency")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp()
            data_file = os.path.join(data_path, str(stamp) + "_RabiChevron_scan.nc")
            cfg_file = os.path.join(data_path, str(stamp) + "_RabiChevron_scan.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.add_axis("time", times, units="microseconds")
            nc_file.add_axis("frequency", frequencies, units="MHz")

            # create variables and fill in data
            nc_file.add_dataset(
                "data",
                [
                    "frequency",
                    "time",
                ],
                mag,
                units=meas_units,
            )
            nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, mag

    @dot_experiment.updater
    def time_rabi(
        self,
        RF_gain: int,
        RF_gen: int,
        RF_freq: float,
        time_range: Tuple[float, float, int] = (0.01, 10, 50),
        point_avgs: int = 10,
        RF_cooldown: float = 10,
        plot: bool = True,
        save_data: bool = True,
    ):
        """perform a time rabi experiment.  This script loops over pulse time in software, but since this experiment is 1D that is plenty fast for us.

        :param RF_gain: RF gain in DAC units
        :param RF_gen: RFSoC generator output corresponding to the RF generator
        :RF_freq: Drive frequency in MHz
        :time_range: (time_start, time_stop, points) in microseconds
        :point_avgs: number of averages per time point
        :RF_cooldown: time to pause after RF drive before the second measurement
        :plot: plot result
        :save_data: save data and plot
        """
        start_time, stop_time, time_pts = time_range
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_freq = RF_freq
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.expts = point_avgs
        self.config.start = 1
        self.config.stop = 10

        times = np.linspace(start_time, stop_time, time_pts)
        data = np.zeros((2, 2, point_avgs, time_pts))
        for i, t in enumerate(times):
            print("loop number %d" % i)
            self.config.RF_expt.RF_time = self.soccfg.us2cycles(t)
            meas = ld_single_qubit_programs.TimeRabi(self.soccfg, self.config)
            expt_pts, avgi, avgq = meas.acquire(
                self.soc, load_pulses=True, progress=False
            )
            if point_avgs == 1:
                data[:, :, 0, i] = [avgi[0], avgq[0]]
            data[:, :, :, i] = [avgi[0], avgq[0]]
        if plot:
            if self.config.PSB_config["thresholding"]:
                plot_tools.plot1_PSBdata(
                    [times],
                    [data[0][0]],
                    [data[0][1]],
                    thresh=self.config.PSB_config["threshold"],
                )

                plt.ylabel("probability")
            else:
                plot_tools.plot1_PSBdata(
                    [times], [data[0][0]], [data[0][1]], thresh=None
                )
                plt.ylabel("DCS conductance (arbs)")
            plt.xlabel("time (us)")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp()
            data_file = os.path.join(data_path, str(stamp) + "_TimeRabi_scan.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_TimeRabi_scan.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_TimeRabi_scan.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            if plot:
                plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("time", time_pts)
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("shots", point_avgs)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata", np.float32, ("IQ", "triggers", "shots", "time")
            )
            psbraw.units = "raw_adc"
            psbraw.frequency = RF_freq
            time = nc_file.createVariable("time", np.float32, ("time"))
            time.units = "tproc"
            time.conversion = self.soccfg.cycles2us(1)
            time[:] = time_pts
            psbraw[:, :, :, :] = data
            processed = nc_file.createVariable("processed", np.float32, ("time"))
            processed[:] = plot_tools.interpret_data_PSB([data[0][0]], [data[0][1]])
            nc_file.data_flavour = "time_rabi"
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return data, times, expt_pts

    @dot_experiment.updater
    def time_rabi_v2(
        self,
        RF_gain: int,
        RF_gen: int,
        RF_freq: float,
        time_range: Tuple[float, float, int] = (0.01, 10, 50),
        point_avgs: int = 10,
        RF_cooldown: float = 10,
        trig_offset: float = 0.1,
        trig_ignore: bool = True,
        plot: bool = True,
        save_data: bool = True,
    ):
        """perform a time rabi by sweeping pulse length on the FPGA
        Sadly we can't sweep the trigger width on the FPGA, so the RF trigger is left on for longer than the pulse time

        :param RF_gain: RF gain in DAC units
        :param RF_gen: RFSoC generator output corresponding to the RF generator
        :param RF_freq: Drive frequency in MHz
        :param time_range: (time_start, time_stop, points) in microseconds
        :param point_avgs: number of averages per time point
        :param RF_cooldown: time to pause after RF drive before the second measurement
        :param trig_ignore: Set to true to perform your measurement directly after the RF pulse.  If set to false, it waits for the trigger to return to zero
        :param plot: plot result
        :param save_data: save data
        """

        start_time, stop_time, time_pts = time_range

        start = self.soccfg.us2cycles(start_time)
        stop = self.soccfg.us2cycles(stop_time)
        step = int((stop - start) / time_pts)
        self.config.RF_expt.trig_ignore = trig_ignore
        self.config.start = start
        self.config.stop = stop
        self.config.step = step
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_freq = RF_freq
        self.config.RF_expt.trig_offset = self.soccfg.us2cycles(trig_offset)
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.expts = time_pts
        self.config.shots = point_avgs
        self.config.expts_outer = 1
        self.config.start_outer = 1
        self.config.step_outer = 1
        self.config.reps = 1

        meas = ld_single_qubit_programs.TimeRabiV2(self.soccfg, self.config)
        expt_pts, mag = meas.acquire(self.soc, load_pulses=True, progress=False)

        x_axis = self.soccfg.cycles2us(expt_pts[0])
        if self.config.PSB_cfg.thresholding:
            meas_units = "singlet_probability"
        else:
            meas_units = "conductance_arbs"
        if plot:
            plt.figure()
            plt.plot(x_axis, mag[0])
            plt.ylabel(meas_units)
            plt.xlabel("time (us)")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_TimeRabi_scan.nc")
            cfg_file = os.path.join(data_path, str(stamp) + "_TimeRabi_scan.yaml")

            # save the config, data and plot
            file_manager.save_config(self.config, cfg_file)
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            nc_file.add_axis("time", x_axis, units="microseconds")
            nc_file.add_dataset(meas_units, axes=["time"], data=mag, units=meas_units)
            nc_file.data_flavour = "time_rabi"
            if plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, mag

    @dot_experiment.updater
    def amplitude_rabi(
        self,
        RF_time: float,
        RF_gen: int,
        RF_freq: float,
        gain_range: Tuple[float, float, int] = (0, 5000, 50),
        RF_cooldown: float = 10,
        trigger_offset: float = 0.1,
        plot: bool = True,
        save_data: bool = True,
    ):
        """Rabi experiment which sweeps RF amplitude, keeping RF time constant. Currently this has no averaging built in, because I'm using Raverager.

        :param RF_time: pulse length, microseconds
        :param RF_gen: RF DAC channel
        :param RF_freq: MHz
        :param gain_range: gain sweep parameters in dac units; (start_gain, stop_gain, number of points)
        :param RF_cooldown: time between RF pulse and readout

        """

        self.config.RF_expt.start_gain = gain_range[0]
        self.config.RF_expt.step_gain = int(
            (gain_range[1] - gain_range[0]) / gain_range[2]
        )
        self.config.RF_expt.gain_pts = gain_range[2]
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_freq = RF_freq
        self.config.RF_expt.RF_time = self.soccfg.us2cycles(RF_time)
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.RF_expt.trig_offset = self.soccfg.us2cycles(trigger_offset)
        self.config.expts = gain_range[2]
        self.config.start = gain_range[0]

        meas = ld_single_qubit_programs.AmplitudeRabi(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=False)
        if self.config.PSB_cfg.thresholding:
            thresh = self.config.PSB_cfg.thresh
        else:
            thresh = None
        if plot:
            plot_tools.plot1_PSBdata([expt_pts], avgi, avgq, thresh=thresh)
            plt.xlabel("RF amplitude (DAC units)")
            plt.title("amplitude rabi")
            if thresh is None:
                plt.ylabel("DCS conductance (arbs)")
            else:
                plt.ylabel("probability")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_TimeRabi_scan.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_TimeRabi_scan.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_TimeRabi_scan.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("gain", gain_range[2])
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata", np.float32, ("IQ", "triggers", "gain")
            )
            psbraw.units = "raw_adc"
            psbraw.frequency = RF_freq
            psbraw.time = RF_time
            gain = nc_file.createVariable("gain", np.float32, ("gain"))
            gain[:] = expt_pts[0]
            psbraw[0, :, :] = avgi[0]
            psbraw[1, :, :] = avgq[0]
            processed = nc_file.createVariable("processed", np.float32, ("gain"))
            processed[:] = plot_tools.interpret_data_PSB(avgi, avgq, data_dim="1D")
            if thresh is not None:
                processed.thresh = thresh
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, avgi, avgq

    @dot_experiment.updater
    def allxy(
        self,
        RF_gain: int,
        RF_freq: float,
        RF_gen: int,
        RF_time: float,
        point_avgs: int = 10,
        expt_avgs: int = 10,
        RF_cooldown: float = 10,
        trig_offset: float = 0.1,
        plot: bool = True,
        save_data: bool = True,
    ):
        """Perform an all x-y experiment.  This experiment consists of a series of specific manipulations to demonstrate x-y control.

        :param RF_gain: amplitude of RF drive in dac units (pi/2 pulse)
        :param RF_freq: frequency of RF drive in MHz
        :param RF_gen: RFSoC DAC channel for RF drive
        :param RF_time: time of RF pulse in microseconds (pi/2 pulse)
        :param point_avgs: averages per point
        :param expt_avgs: averages of full experiment
        :param RF_cooldown: time between RF pulse and readout in microseconds
        :param trig_offset: time delay between RF trigger and RF pulse in microseconds
        """

        self.config.RF_expt.RF_freq = RF_freq
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.RF_time = RF_time
        self.config.RF_expt.RF_cooldown = RF_cooldown
        self.config.RF_expt.trig_offset = self.soccfg.us2cycles(trig_offset)
        self.config.RF_expt.pulse_delay = self.soccfg.us2cycles(0)
        # parameters for dummy sweep
        self.config.expts = point_avgs
        self.config.start = 1
        self.config.stop = 10

        AllXY_seq = [
            "I,I",
            "X180,X180",
            "Y180,Y180",
            "X180,Y180",
            "Y180,X180",
            "X90,I",
            "Y90,I",
            "X90,Y90",
            "Y90,X90",
            "X90,Y180",
            "Y90,X180",
            "X180,Y90",
            "Y180,X90",
            "X90,X180",
            "X180,X90",
            "Y90,Y180",
            "Y180,Y90",
            "X180,I",
            "Y180,I",
            "X90,X90",
            "Y90,Y90",
        ]

        data = np.zeros((2, 2, point_avgs, len(AllXY_seq), expt_avgs))
        # we found that lower frequency noise mattered a lot here, needed to introduce averages over the full experiment instead of just per point
        for n in range(expt_avgs):
            print("experiment average %d" % n)
            for i, gateset in enumerate(AllXY_seq):
                self.config.RF_expt.gate_set = gateset
                meas = ld_single_qubit_programs.AllXY(self.soccfg, self.config)
                expt_pts, avgi, avgq = meas.acquire(
                    self.soc, load_pulses=True, progress=False
                )
                data[:, :, :, i, n] = [avgi[0], avgq[0]]
        if self.config.PSB_cfg.thresholding:
            thresh = self.config.PSB_cfg.thresh
        else:
            thresh = None
        avged_data_1 = plot_tools.interpret_data_PSB(
            [data[0][0]], [data[0][1]], thresh=thresh
        )
        avged_data_2 = np.mean(avged_data_1, axis=1)
        if plot:
            plt.plot(
                AllXY_seq,
                avged_data_2,
                "ko-",
            )

            plt.title("AllXY")
            plt.tight_layout()
            plt.xticks(rotation=45)
            if thresh is not None:
                plt.ylabel("probability")
            else:
                plt.ylabel("DCS conductance")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_allxy.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_allxy.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_allxy.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("gateset", len(AllXY_seq))
            nc_file.createDimension("experiment_avgs", expt_avgs)
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("shots", point_avgs)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata",
                np.float32,
                ("IQ", "triggers", "shots", "gateset", "experiment_avgs"),
            )
            psbraw.units = "raw_adc"

            gates = nc_file.createVariable("gateset", str, ("gateset"))
            gates[:] = np.array(AllXY_seq)
            psbraw[:, :, :, :, :] = data
            processed = nc_file.createVariable("processed", np.float32, ("gateset"))
            processed[:] = avged_data_2
            nc_file.close()
            logger.info("data saved at %s" % data_file)
        return expt_pts, data

    @dot_experiment.updater
    def phase_control(
        self,
        RF_gain: int,
        RF_freq: float,
        RF_gen: int,
        RF_time: float,
        phase_steps: int = 10,
        point_avgs: int = 10,
        full_avgs: int = 10,
        RF_cooldown: float = 10,
        trigger_offset: float = 0.1,
        plot: bool = True,
        save_data: bool = True,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """this experiment performs two pi/2 pulses, incrementing the phase offset of the second pulse.
        If you are driving x-y rotations you will see a periodic output.  This is a simple way to demonstrate that you have
        x-y control of your qubit using RF drive phase.

        :param RF_gain: amplitude of RF drive in DAC units (corresponding to pi/2 pulse)
        :param RF_freq: frequency of RF drive in MHz
        :param RF_gen: RFSoC generator producing RF drive
        :param RF_time: time of RF pulse (corresponding to pi/2 pulse) in microseconds
        :param phase_steps: number of points in the phase sweep
        :param point_avgs: averages at each point in the sweep
        :param full_avgs: averages of full experiment
        :param RF_cooldown: time to let system settle before going to measurement point
        :param trigger_offset: time in microseconds to wait after triggering the RF switch before triggering a pulse
        """
        self.config.RF_expt.RF_freq = RF_freq
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.RF_time = self.soccfg.us2cycles(RF_time)
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.RF_expt.phase_pulse_1 = 0
        self.config.RF_expt.pulse_delay = self.soccfg.us2cycles(0)
        self.config.RF_expt.trig_offset = self.soccfg.us2cycles(trigger_offset)
        # dummy parameters for inner qicksweep loop
        self.config.expts = point_avgs
        self.config.start = 1
        self.config.stop = 10
        self.config.reps = 1
        thresh = self.config.PSB_cfg.thresh
        thresholding = self.config.PSB_cfg.thresholding
        phase_sweep = np.linspace(-180, 180, phase_steps)
        data = np.zeros((2, 2, point_avgs, phase_steps, full_avgs))
        # we found that lower frequency noise mattered a lot here, needed to introduce averages over the full experiment instead of just per point
        for n in range(full_avgs):
            print("experiment average %d" % n)
            for i, phase in enumerate(phase_sweep):
                self.config.RF_expt.phase_sweep = phase
                meas = ld_single_qubit_programs.SweepPhase(self.soccfg, self.config)
                expt_pts, avgi, avgq = meas.acquire(
                    self.soc, load_pulses=True, progress=False
                )
                data[:, :, :, i, n] = [avgi[0], avgq[0]]
        if thresholding:
            avged_data_1 = plot_tools.interpret_data_PSB(
                [data[0][0]], [data[0][1]], thresh=thresh
            )
        else:
            avged_data_1 = plot_tools.interpret_data_PSB([data[0][0]], [data[0][1]])
        avged_data_2 = np.mean(avged_data_1, axis=1)
        if plot:
            plt.plot(
                phase_sweep,
                avged_data_2,
                "ko-",
            )

            plt.title("phase control")
            plt.xlabel("phase offset (degrees)")
            if thresh:
                plt.ylabel("probability")
            else:
                plt.ylabel("conductance")
            plt.tight_layout()

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_phase_control.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_phase_control.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_phase_control.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("phase", phase_steps)
            nc_file.createDimension("experiment_avgs", full_avgs)
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("shots", point_avgs)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata",
                np.float32,
                ("IQ", "triggers", "shots", "phase", "experiment_avgs"),
            )
            psbraw.units = "raw_adc"
            phase = nc_file.createVariable("phase", np.float32, ("phase"))
            phase[:] = phase_sweep
            psbraw[:, :, :, :, :] = data
            processed = nc_file.createVariable("processed", np.float32, ("phase"))
            processed[:] = avged_data_2
            nc_file.close()
            logger.info("data saved at %s" % data_file)
        return expt_pts, data

    @dot_experiment.updater
    def ramsey_experiment(
        self,
        RF_gain: int,
        RF_freq: float,
        RF_gen: int,
        RF_pi_2: float,
        time_range: Tuple[float, float, int] = (1, 10, 20),
        full_avgs: int = 10,
        point_avgs: int = 10,
        trigger_offset: float = 0.1,
        ramsey_freq: float = 0,
        RF_cooldown: float = 10,
        plot: bool = True,
        save_data: bool = True,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """perform a ramsey experiment.  Sweep time delay between two pi/2 pulses.  If ramsey_freq
        is set to a nonzero value, it advances the phase of the second pi/2 pulse by 2pi x ramsey_freq.
        TODO figure out why this won't compile for certain time step values

        :param RF_gain: RF out of the RFSoC in DAC units
        :param RF_freq: Frequency of RF drive, in MHz
        :param RF_gen: RFSoC generator channel to power RF out
        :param RF_pi_2: Pi/2 pulse time, in microseconds. Obtain from an amplitude Rabi
        :time_range: (time_start, time_stop, points) in microseconds
        :param full_avgs: averages over the full experiment
        :param point_avgs: repeat a single data point measurement and average this many times
        :param RF_cooldown: time in us after the second pulse to wait before readout
        :param thresh: optionally turn thresholding on or off, regardless of the default in readout_cfg
        :param save_data: save data
        :param plot: display plot in ipython
        :param ramsey_freq: provide ramsey frequency in MHz if known.  This will be used to calculate the amount to advance the RF drive phase by.

        """
        self.config.RF_expt.RF_freq = RF_freq
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.RF_time = self.soccfg.us2cycles(RF_pi_2)
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.RF_expt.ramsey_freq = ramsey_freq
        self.config.RF_expt.trig_offset = self.soccfg.us2cycles(trigger_offset)
        self.config.expts = time_range[2]
        self.config.start = self.soccfg.us2cycles(time_range[0])
        self.config.step = int(
            self.soccfg.us2cycles(time_range[1] - time_range[0]) / time_range[2]
        )
        self.config.expts_outer = 1
        self.config.step_outer = 1
        self.config.start_outer = 1
        self.config.reps = full_avgs
        self.config.shots = point_avgs

        meas = ld_single_qubit_programs.RamseyFringes(self.soccfg, self.config)
        expt_pts, data = meas.acquire(self.soc, load_pulses=True, progress=False)
        expt_pts = self.soccfg.cycles2us(expt_pts[0])

        if plot:
            plt.figure()
            plt.plot(expt_pts, data[0])
            # TODO add fit to get T2 Ramsey

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_ramsey.nc")
            cfg_file = os.path.join(data_path, str(stamp) + "_ramsey.yaml")
            # save config, dataset and plot
            file_manager.save_config(self.config, cfg_file)
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            nc_file.add_axis(label="time", data=expt_pts, units="us")
            if self.config.PSB_cfg.thresholding:
                readout_label = "thresholded_readout"
                readout_units = "probability"
            else:
                readout_label = "readout"
                readout_units = "raw_adc"
            nc_file.add_dataset(
                label=readout_label,
                axes=["time"],
                data=data[0],
                units=readout_units,
            )
            nc_file.data_flavour = "ramsey"
            if plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, data

    @dot_experiment.updater
    def ramsey_2d(
        self,
        RF_gain: int,
        RF_gen: int,
        RF_pi_2: float,
        time_range: Tuple[float, float, int] = (1, 10, 20),
        freq_range: Tuple[float, float, int] = (1, 10, 20),
        full_avgs: int = 10,
        point_avgs: int = 10,
        RF_cooldown: float = 10,
        plot: bool = True,
        save_data: bool = True,
    ):
        """perform a series of ramsey experiments.  Sweep time delay between two pi/2 pulses, and sweep drive frequency

        :param RF_gain: RF out of the RFSoC in DAC units
        :param RF_freq: Frequency of RF drive, in MHz
        :param RF_gen: RFSoC generator channel to power RF out
        :param RF_pi_2: Pi/2 pulse time, in microseconds. Obtain from an amplitude Rabi
        :param time_range: (time_start, time_stop, points) in microseconds
        :param freq_range: (freq_start, freq_stop, points) in MHz
        :param full_avgs: averages over the full experiment
        :param point_avgs: repeat a single data point measurement and average this many times
        :param RF_cooldown: time in us after the second pulse to wait before readout
        :param save_data: save data
        :param plot: display plot in ipython
        :param ramsey_freq: provide ramsey frequency in MHz if known.  This will be used to calculate the amount to advance the RF drive phase by.

        """
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.RF_time = self.soccfg.us2cycles(RF_pi_2)
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.expts = time_range[2]
        self.config.start = self.soccfg.us2cycles(time_range[0])
        self.config.step = int(
            self.soccfg.us2cycles(time_range[1] - time_range[0]) / time_range[2]
        )
        self.config.expts_outer = freq_range[2]
        self.config.step_outer = int(
            self.soccfg.freq2reg(freq_range[1] - freq_range[0]) / freq_range[2]
        )
        self.config.start_outer = self.soccfg.freq2reg(freq_range[0])
        self.config.reps = full_avgs
        self.config.shots = point_avgs

        meas = ld_single_qubit_programs.Ramsey2D(self.soccfg, self.config)
        expt_pts, data = meas.acquire(self.soc, load_pulses=True, progress=False)

        times = self.soccfg.cycles2us(expt_pts[0])
        frequencies = self.soccfg.reg2freq(expt_pts[1])
        if self.config.PSB_cfg.thresholding:
            meas_units = "singlet_probability"
        else:
            meas_units = "conductance_arbs"
        if plot:
            plot_tools.plot2_simple(
                xarray=times, yarray=frequencies, data=data, cbar_label=meas_units
            )
            plt.xlabel("time (us)")
            plt.ylabel("frequency (MHz)")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_ramsey2d.nc")
            cfg_file = os.path.join(data_path, str(stamp) + "_ramsey2d.yaml")
            # save config, dataset and plot
            file_manager.save_config(self.config, cfg_file)
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")

            nc_file.add_axis(label="time", data=expt_pts, units="us")
            if self.config.PSB_cfg.thresholding:
                readout_label = "thresholded_readout"
                readout_units = "probability"
            else:
                readout_label = "readout"
                readout_units = "raw_adc"
            nc_file.add_dataset(
                label=readout_label,
                axes=["time"],
                data=data[0],
                units=readout_units,
            )
            nc_file.data_flavour = "ramsey2d"
            if plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, data

    @dot_experiment.updater
    def spin_echo(
        self,
        RF_gain: int,
        RF_freq: float,
        RF_gen: int,
        RF_pi_2: float,
        n_echoes: int = 0,
        time_range: Tuple[float, float, int] = (1, 10, 20),
        full_avgs: int = 10,
        point_avgs: int = 10,
        RF_cooldown: float = 10,
        thresh: bool | None = None,
        save_data: bool = True,
        plot: bool = True,
    ):
        """perform a hahn echo or CPMG experiment.

        :param RF_gain: RF out of the RFSoC in DAC units
        :param RF_freq: Frequency of RF drive, in MHz
        :param RF_gen: RFSoC generator channel to power RF out
        :param RF_pi_2: Pi/2 pulse time, in microseconds. Obtain from an amplitude Rabi
        :param n_echoes: Number of pi pulses to insert. n=1 for Hahn echo
        :time_range: (time_start, time_stop, points) in microseconds
        :param full_avgs: averages over the full experiment
        :param point_avgs: repeat a single data point measurement and average this many times
        :param RF_cooldown: time in us after the second pulse to wait before readout
        :param thresh: optionally turn thresholding on or off, regardless of the default in readout_cfg
        :param save_data: save data
        :param plot: display plot in ipython

        """
        self.config.RF_expt.RF_freq = RF_freq
        self.config.RF_expt.RF_gen = RF_gen
        self.config.RF_expt.RF_gain = RF_gain
        self.config.RF_expt.RF_time = self.soccfg.us2cycles(RF_pi_2)
        self.config.RF_expt.n_echoes = n_echoes
        self.config.RF_expt.RF_cooldown = self.soccfg.us2cycles(RF_cooldown)
        self.config.expts = time_range[2]
        self.config.start = self.soccfg.us2cycles(time_range[0])
        self.config.step = int(
            self.soccfg.us2cycles(time_range[1] - time_range[0]) / time_range[2]
        )
        self.config.expts_outer = 1
        self.config.step_outer = 1
        self.config.start_outer = 1
        self.config.reps = full_avgs
        self.config.shots = point_avgs
        if thresh is not None:
            self.config.PSB_config.thresholding = thresh
        meas = ld_single_qubit_programs.SpinEcho(self.soccfg, self.config)
        expt_pts, data = meas.acquire(self.soc, load_pulses=True, progress=False)
        expt_pts = self.soccfg.cycles2us(expt_pts[0])

        if plot:
            plt.figure()
            plt.plot(expt_pts, data[0])
            # TODO add a fit

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_echo.nc")
            cfg_file = os.path.join(data_path, str(stamp) + "_echo.yaml")
            # save config, dataset and plot
            file_manager.save_config(self.config, cfg_file)
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")

            nc_file.add_axis(label="time", data=expt_pts, units="us")
            if self.config.PSB_cfg.thresholding:
                readout_label = "thresholded_readout"
                readout_units = "probability"
            else:
                readout_label = "readout"
                readout_units = "raw_adc"
            nc_file.add_dataset(
                label=readout_label,
                axes=["time"],
                data=data[0],
                units=readout_units,
            )
            nc_file.data_flavour = "spin_echo"
            nc_file.n_echoes = n_echoes
            if plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, data
