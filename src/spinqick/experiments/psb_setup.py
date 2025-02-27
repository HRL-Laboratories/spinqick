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
import netCDF4
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
        save_data: bool = True,
    ):
        """Produce a measurement histogram to show relative prevalance of measured singlets and triplets.  Fits the data to two Gaussians if plot==True.
        TODO utilize PSBAverager and update data saving to use helperfunctions.

        :param num_measurements: Total number of measurements
        :param flush_2: Add a second flush, to obtain a random mixture of singlets and triplets for LD qubits.  If false, this program runs a typical psb sequence
        :param plot: plot results
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

        avgvals = plot_tools.interpret_data_PSB(avgi, avgq, data_dim="1D")

        if plot:
            hist, bins = np.histogram(avgvals, bins=100, range=None, weights=None)
            x = bins + (bins[1] - bins[0]) / 2
            plt.figure()
            plt.plot(x[:-1], hist, ".")

            def gauss(x, A1, sigma1, mu1, A2, sigma2, mu2):
                return A1 / (np.sqrt(2 * np.pi) * sigma1) * np.exp(
                    -0.5 * (x - mu1) ** 2 / sigma1**2
                ) + A2 / (np.sqrt(2 * np.pi) * sigma2) * np.exp(
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
                popt, pcov = optimize.curve_fit(gauss, x[:-1], hist, p0=guess)
                plt.plot(x[:-1], gauss(x[:-1], *popt))
                print("sigma1: %f, sigma2: %f" % (popt[1], popt[4]))
                print("fwhm1: %f, fwhm2: %f" % (popt[1] * 2.355, popt[4] * 2.355))
                print("mu1: %f, mu2: %f" % (popt[2], popt[5]))
                print("A1: %f, A2: %f" % (popt[0], popt[3]))
                snr = 2 * np.abs(popt[2] - popt[5]) / (popt[1] + popt[4])
                print("SNR: %f" % snr)
            except Exception as exc:
                logger.error("fit failed: %s", exc, exc_info=True)

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_meashist.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_meashist.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_meashist_cfg.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("triggers", 2)
            nc_file.createDimension("shots", num_measurements)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata", np.float32, ("IQ", "triggers", "shots")
            )
            psbraw.units = "raw_adc"
            psbraw[0, :, :] = avgi[0]
            psbraw[1, :, :] = avgq[0]
            processed = nc_file.createVariable("processed", np.float32, ("shots"))
            processed[:] = avgvals
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return avgvals, expt_pts, avgi, avgq

    @dot_experiment.updater
    def meas_window_scan(
        self,
        scan_type: Literal["flush", "flush_2", "idle", "meas"],
        p_gates: Tuple[str, str],
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]] = (
            (-1, 1, 100),
            (-1, 1, 100),
        ),
        x_init: bool = True,
        x_gate: str | None = None,
        point_avgs: int = 100,
        plot: bool = True,
        save_data: bool = True,
    ):
        """
        2D sweep of measurement, flush or idle window coordinates.

        :param scan_type: Choose the coordinate that you want to scan.
        :param p_gates: specify the two plunger gates being used
        :param p_range: specify the range of each axis sweep.  ((px_start, px_stop, px_points), (py_start, py_stop, py_points))
        :param x_init: turn on x gate at idle point
        :param x_gate: if using x_init, specify the x-gate
        :param point_avgs: number of averages per point
        :param plot: whether to plot data
        :param save_data: saves data to netCDF and saves any figure generated as a png
        """

        # I'm trying to write everything generally in terms of Px and Py to help make it more readable
        self.config.PSB_sweep_cfg.scan_type = scan_type
        self.config.PSB_sweep_cfg.x_init = x_init
        Px_gate, Py_gate = p_gates

        self.config.PSB_sweep_cfg.gates.Py.gate = Py_gate
        self.config.PSB_sweep_cfg.gates.Py.start = self.volts2dac(
            p_range[1][0], Py_gate
        )
        self.config.PSB_sweep_cfg.gates.Py.stop = self.volts2dac(p_range[1][1], Py_gate)
        self.config.PSB_sweep_cfg.gates.Py.expts = p_range[1][2]
        self.config.PSB_sweep_cfg.gates.Px.gate = Px_gate
        self.config.PSB_sweep_cfg.gates.Px.start = self.volts2dac(
            p_range[0][0], Px_gate
        )
        self.config.PSB_sweep_cfg.gates.Px.stop = self.volts2dac(p_range[0][1], Px_gate)
        self.config.PSB_sweep_cfg.gates.Px.expts = p_range[0][2]
        self.config.PSB_sweep_cfg.gates.X.gate = x_gate
        self.config.expts = point_avgs

        # run the scan
        meas = psb_setup_programs.PSBScanGeneral(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=True)

        expt_pts[1] = self.dac2volts(expt_pts[1], Px_gate) * 1000
        expt_pts[0] = self.dac2volts(expt_pts[0], Py_gate) * 1000
        # plot the data
        if plot:
            mag = plot_tools.interpret_data_PSB(avgi, avgq)
            avged_mag = np.transpose(mag)
            x_pts = expt_pts[1]
            y_pts = expt_pts[0]

            plt.figure()

            plt.pcolormesh(x_pts, y_pts, avged_mag, shading="nearest", cmap="binary_r")

            plt.colorbar(label="DCS conductance - reference measurement, arbs")
            plt.title("scantype = %s" % scan_type)
            plt.xlabel("%s (mV)" % Px_gate)
            plt.ylabel("%s (mV)" % Py_gate)

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(
                data_path, str(stamp) + "_" + scan_type + "_psbscan.nc"
            )
            fig_file = os.path.join(
                data_path, str(stamp) + "_" + scan_type + "_psbscan.png"
            )
            cfg_file = os.path.join(
                data_path, str(stamp) + "_" + scan_type + "_psbscan_cfg.yaml"
            )

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

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
            Px = nc_file.createVariable("Px", np.float32, ("Px"))
            Px.units = "dac_units"
            Px.dac2volts = self.dac2volts(1, Px_gate)
            Px[:] = expt_pts[1]
            Py = nc_file.createVariable("Py", np.float32, ("Py"))
            Py.units = "dac_units"
            Py.dac2volts = self.dac2volts(1, Py_gate)
            Py[:] = expt_pts[0]
            psbraw[0, :, :, :, :] = avgi[0]
            psbraw[1, :, :, :, :] = avgq[0]
            processed = nc_file.createVariable("processed", np.float32, ("Px", "Py"))
            processed[:] = plot_tools.interpret_data_PSB(avgi, avgq)
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, avgi, avgq
