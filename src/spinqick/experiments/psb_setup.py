"""
Module to hold functions that help you set up Pauli spin blockade.
Looking for SPAM? find it here!

"""

import os
from typing import Literal, Tuple
import logging

import numpy as np
from addict import Dict
from matplotlib import pyplot as plt
from scipy import optimize
from qick import QickConfig

from spinqick.helper_functions import file_manager, plot_tools
from spinqick.qick_code import psb_setup_programs
from spinqick.experiments import dot_experiment
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)


class PsbSetup(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK classes for setting up PSB readout."""

    def __init__(
        self, soccfg: QickConfig, soc, datadir: str = file_settings.data_directory
    ):
        """initialize with information about your rfsoc and your experimental setup

        :param soccfg: qick config object (QickConfig)
        :param soc: QickSoc object
        :param datadir: data directory where all data is being stored. Experiment will make a folder here with today's date.
        """

        super().__init__(datadir=datadir)
        self.soccfg = soccfg
        self.soc = soc
        self.datadir = datadir

    @dot_experiment.updater
    def meashist(
        self,
        num_measurements: int,
        flush_2: bool = False,
        plot: bool = True,
        fit: bool = True,
        save_data: bool = True,
    ):
        """Produce a measurement histogram to show relative prevalance of measured singlets and triplets.  Fits the data to two Gaussians if plot==True.

        :param num_measurements: Total number of measurements
        :param flush_2: Add a second flush, to obtain a random mixture of singlets and triplets for LD qubits.  If false, this program runs a typical EO psb sequence
        :param plot: plot results
        :param fit: fit results to a pair of gaussians
        :param save_data: save data and plot

        """
        self.config = Dict(self.config)
        self.config.expts = num_measurements
        self.config.flush_2 = flush_2
        self.config.reps = 1

        # dummy parameters to appease qicksweep
        self.config.stop = 10
        self.config.start = 1
        meas = psb_setup_programs.PSBExperiment(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=True)

        avgvals = plot_tools.interpret_data_psb(avgi, avgq, data_dim="1D")
        data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
        if plot:
            hist, bins = np.histogram(avgvals, bins=100, range=None, weights=None)
            x = bins + (bins[1] - bins[0]) / 2
            plt.figure()
            plt.plot(x[:-1], hist, ".")
        if fit:
            fit_exception = False

            def gauss(x, a1, sigma1, mu1, a2, sigma2, mu2):
                return a1 / (np.sqrt(2 * np.pi) * sigma1) * np.exp(
                    -0.5 * (x - mu1) ** 2 / sigma1**2
                ) + a2 / (np.sqrt(2 * np.pi) * sigma2) * np.exp(
                    -0.5 * (x - mu2) ** 2 / sigma2**2
                )

            guess = [
                num_measurements,
                5,
                0,
                num_measurements,
                5,
                np.max(x) * 0.9,
            ]  # make this more robust, maybe use lmfit
            try:
                # pylint: disable-next=unbalanced-tuple-unpacking
                popt, _ = optimize.curve_fit(gauss, x[:-1], hist, p0=guess)
                if plot:
                    plt.plot(x[:-1], gauss(x[:-1], *popt))
                    plt.title("meas hist, %d shots" % num_measurements)
                    plt.title("t: %d" % stamp, loc="right", fontdict={"fontsize": 6})
                    plt.xlabel("DCS conductance (arbs)")
                print("sigma1: %f, sigma2: %f" % (popt[1], popt[4]))
                print("fwhm1: %f, fwhm2: %f" % (popt[1] * 2.355, popt[4] * 2.355))
                print("mu1: %f, mu2: %f" % (popt[2], popt[5]))
                print("A1: %f, A2: %f" % (popt[0], popt[3]))
                snr = 2 * np.abs(popt[2] - popt[5]) / (popt[1] + popt[4])
                print("SNR: %f" % snr)
            except RuntimeError:
                fit_exception = True
                logger.error("fit failed")

        if save_data:
            data_file = os.path.join(data_path, str(stamp) + "_meashist.nc")
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("shots", num_measurements)
            nc_file.createDimension("IQ", 2)
            nc_file.createDimension("popt", 6)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata", np.float32, ("IQ", "triggers", "shots")
            )
            psbraw.units = "raw_adc"
            psbraw[0, :, :] = avgi[0]
            psbraw[1, :, :] = avgq[0]
            processed = nc_file.createVariable("processed", np.float32, ("shots"))
            processed[:] = avgvals
            # pylint: disable-next=possibly-used-before-assignment
            if fit and not fit_exception:
                popt_var = nc_file.createVariable("popt", np.float32, ("popt"))
                nc_file.snr = snr
                popt_var[:] = popt
            nc_file.save_config(self.config)
            nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s", data_file)

        return avgvals, expt_pts, popt, snr, stamp

    @dot_experiment.updater
    def idle_cell_scan(
        self,
        p_gates: Tuple[str, str],
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]] = (
            (-0.01, 0.01, 100),
            (-0.01, 0.01, 100),
        ),
        add_rf: bool = False,
        rf_gain: int | None = None,
        rf_freq: float | None = None,
        point_avgs: int = 10,
        plot: bool = True,
        save_data: bool = True,
    ):
        """
        2D sweep of idle coordinate.

        :param p_gates: specify the two plunger gates being used
        :param p_range: specify the range of each axis sweep.  ((px_start, px_stop, px_points), (py_start, py_stop, py_points))
        :param add_rf: add an RF pulse at the idle point
        :param rf_freq: frequency of RF pulse
        :param rf_gain: gain of RF pulse
        :param point_avgs: number of averages per point
        :param plot: whether to plot data
        :param save_data: saves data to netCDF and saves any figure generated as a png
        """

        px_gate, py_gate = p_gates
        px_start_voltage, px_stop_voltage, px_num_points = p_range[0]
        py_start_voltage, py_stop_voltage, py_num_points = p_range[1]

        self.config.psb_sweep_cfg.gates.py.gate = py_gate
        self.config.psb_sweep_cfg.gates.py.start = self.volts2dac(
            py_start_voltage, py_gate
        )
        self.config.psb_sweep_cfg.gates.py.stop = self.volts2dac(
            py_stop_voltage, py_gate
        )
        self.config.psb_sweep_cfg.gates.py.expts = py_num_points
        self.config.psb_sweep_cfg.gates.px.gate = px_gate
        self.config.psb_sweep_cfg.gates.px.start = self.volts2dac(
            px_start_voltage, px_gate
        )
        self.config.psb_sweep_cfg.gates.px.stop = self.volts2dac(
            px_stop_voltage, px_gate
        )
        self.config.psb_sweep_cfg.gates.px.expts = px_num_points
        self.config.psb_sweep_cfg.rf_gain = rf_gain
        self.config.psb_sweep_cfg.rf_freq = rf_freq
        self.config.psb_sweep_cfg.rf_gen = self.config.rf_expt.rf_gen
        self.config.psb_sweep_cfg.add_rf = add_rf
        if add_rf:
            # TODO this check should be functionalized here and elsewhere
            if rf_freq is not None:
                if rf_freq > 3000:
                    self.config.psb_sweep_cfg.nqz = 2
                else:
                    self.config.psb_sweep_cfg.nqz = 1
        # requirements for the averager function
        self.config.expts = point_avgs
        self.config.reps = 1
        self.config.start = 1
        self.config.stop = 10

        # run the scan
        meas = psb_setup_programs.IdleScan(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=True)
        x_pts = expt_pts[1]
        y_pts = expt_pts[0]
        self.soc.reset_gens()
        # make a directory for today's date and create a unique timestamp
        data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)

        # plot the data
        if plot:
            if self.config.psb_cfg.thresholding:
                mag = plot_tools.interpret_data_psb(
                    avgi, avgq, thresh=self.config.psb_cfg.thresh
                )
            else:
                mag = plot_tools.interpret_data_psb(avgi, avgq)
            avged_mag = np.transpose(mag)
            x_volts = self.dac2volts(expt_pts[1], px_gate) * 1000
            y_volts = self.dac2volts(expt_pts[0], py_gate) * 1000

            plt.figure()

            plt.pcolormesh(
                x_volts, y_volts, avged_mag, shading="nearest", cmap="binary_r"
            )
            if self.config.psb_cfg.thresholding:
                plt.colorbar(label="singlet probability")
            else:
                plt.colorbar(label="DCS conductance - reference measurement, arbs")
            plt.title("idle cell scan")
            plt.title("t: %d" % stamp, loc="right", fontdict={"fontsize": 6})
            plt.xlabel("%s (dac units)" % px_gate)
            plt.ylabel("%s (dac units)" % py_gate)
            plt.tight_layout()

        if save_data:
            data_file = os.path.join(data_path, str(stamp) + "_" + "_idlescan.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_" + "_idlescan.png")
            plt.savefig(fig_file)
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("Px", p_range[0][2])
            nc_file.createDimension("Py", p_range[1][2])
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("shots", point_avgs)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata", np.float32, ("IQ", "triggers", "shots", "Px", "Py")
            )
            psbraw.units = "raw_adc"
            px = nc_file.createVariable("Px", np.float32, ("Px"))
            px.units = "dac_units"
            px.dac2volts = self.dac2volts(1, px_gate)
            px[:] = x_pts
            py = nc_file.createVariable("Py", np.float32, ("Py"))
            py.units = "dac_units"
            py.dac2volts = self.dac2volts(1, py_gate)
            py[:] = y_pts
            psbraw[0, :, :, :, :] = avgi[0]
            psbraw[1, :, :, :, :] = avgq[0]
            processed = nc_file.createVariable("processed", np.float32, ("Px", "Py"))
            processed[:] = plot_tools.interpret_data_psb(avgi, avgq)
            nc_file.save_config(self.config)
            nc_file.close()
            logger.info("data saved at %s", data_file)

        return expt_pts, avgi, avgq, stamp

    @dot_experiment.updater
    def meas_window_scan(
        self,
        scan_type: Literal["flush", "meas"],
        p_gates: Tuple[str, str],
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]] = (
            (-0.01, 0.01, 100),
            (-0.01, 0.01, 100),
        ),
        x_init: bool = True,
        x_gate: str | None = None,
        point_avgs: int = 10,
        plot: bool = True,
        save_data: bool = True,
    ):
        """
        2D sweep of measurement, flush or idle window coordinates.  This is set up for use with parity readout, and
        it initializes a random spin state by going to a point denoted init_2 instead of dephasing at idle

        :param scan_type: Choose the coordinate that you want to scan.
        :param p_gates: specify the two plunger gates being used
        :param p_range: specify the range of each axis sweep.  ((px_start, px_stop, px_points), (py_start, py_stop, py_points))
        :param x_init: turn on x gate at idle point
        :param x_gate: if using x_init, specify the x-gate
        :param point_avgs: number of averages per point
        :param plot: whether to plot data
        :param save_data: saves data to netCDF and saves any figure generated as a png
        """

        # I'm trying to write everything generally in terms of px and py to help make it more readable
        self.config.psb_sweep_cfg.scan_type = scan_type
        self.config.psb_sweep_cfg.x_init = x_init
        px_gate, py_gate = p_gates
        px_start_voltage, px_stop_voltage, px_num_points = p_range[0]
        py_start_voltage, py_stop_voltage, py_num_points = p_range[1]

        self.config.psb_sweep_cfg.gates.py.gate = py_gate
        self.config.psb_sweep_cfg.gates.py.start = self.volts2dac(
            py_start_voltage, py_gate
        )
        self.config.psb_sweep_cfg.gates.py.stop = self.volts2dac(
            py_stop_voltage, py_gate
        )
        self.config.psb_sweep_cfg.gates.py.expts = py_num_points
        self.config.psb_sweep_cfg.gates.px.gate = px_gate
        self.config.psb_sweep_cfg.gates.px.start = self.volts2dac(
            px_start_voltage, px_gate
        )
        self.config.psb_sweep_cfg.gates.px.stop = self.volts2dac(
            px_stop_voltage, px_gate
        )
        self.config.psb_sweep_cfg.gates.px.expts = px_num_points
        self.config.psb_sweep_cfg.gates.x.gate = x_gate
        # requirements for the averager function
        self.config.expts = point_avgs
        self.config.reps = 1
        self.config.start = 1
        self.config.stop = 10

        # run the scan
        meas = psb_setup_programs.PSBScanGeneral(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=True)

        x_volts = self.dac2volts(expt_pts[1], px_gate) * 1000
        y_volts = self.dac2volts(expt_pts[0], py_gate) * 1000
        self.soc.reset_gens()
        # make a directory for today's date and create a unique timestamp
        data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)

        x_pts = expt_pts[1]
        y_pts = expt_pts[0]

        # plot the data
        if plot:
            mag = plot_tools.interpret_data_psb(avgi, avgq)
            avged_mag = np.transpose(mag)
            plt.figure()
            plt.pcolormesh(
                x_volts, y_volts, avged_mag, shading="nearest", cmap="binary_r"
            )
            plt.colorbar(label="DCS conductance - reference measurement, arbs")

            plt.title("psb scantype = %s" % scan_type)
            plt.title("t: %d" % stamp, loc="right", fontdict={"fontsize": 6})
            plt.xlabel("%s (mV)" % px_gate)
            plt.ylabel("%s (mV)" % py_gate)

        if save_data:
            data_file = os.path.join(
                data_path, str(stamp) + "_" + scan_type + "_psbscan.nc"
            )
            fig_file = os.path.join(
                data_path, str(stamp) + "_" + scan_type + "_psbscan.png"
            )

            plt.savefig(fig_file)
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("Px", p_range[0][2])
            nc_file.createDimension("Py", p_range[1][2])
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("shots", point_avgs)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata", np.float32, ("IQ", "triggers", "shots", "Px", "Py")
            )
            psbraw.units = "raw_adc"
            px = nc_file.createVariable("Px", np.float32, ("Px"))
            px.units = "dac_units"
            px.dac2volts = self.dac2volts(1, px_gate)
            px[:] = x_pts
            py = nc_file.createVariable("Py", np.float32, ("Py"))
            py.units = "dac_units"
            py.dac2volts = self.dac2volts(1, py_gate)
            py[:] = y_pts
            psbraw[0, :, :, :, :] = avgi[0]
            psbraw[1, :, :, :, :] = avgq[0]
            processed = nc_file.createVariable("processed", np.float32, ("Px", "Py"))
            processed[:] = plot_tools.interpret_data_psb(avgi, avgq)
            nc_file.save_config(self.config)
            nc_file.close()
            logger.info("data saved at %s", data_file)

        return expt_pts, avgi, avgq, stamp
