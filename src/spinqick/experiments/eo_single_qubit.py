"""Module to hold exchange only experiments"""

import logging
import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from spinqick.experiments import dot_experiment
from spinqick.helper_functions import file_manager, plot_tools
from spinqick.qick_code import eo_single_qubit_programs
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)


class EOSingleQubit(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK classes for setting up EO 1Qubit experiments."""

    def __init__(self, soccfg, soc, datadir=file_settings.data_directory):
        super().__init__(datadir=datadir)
        self.soccfg = soccfg
        self.soc = soc
        self.datadir = datadir

        self.detuning_axis = np.zeros((2, 2))  # [[p21,p31],[p22,p32]]
        self.symmetric_axis = np.zeros((2, 2, 2))  # [[p21,p31,X1],[p22,p32,X2]]

    @dot_experiment.updater
    def do_nonequilibrium_cell23(
        self,
        p_gates: Tuple[str, str],
        x_gate: str,
        x_amplitude: int,
        x_time: float,
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]] = (
            (-1, 1, 100),
            (-1, 1, 100),
        ),
        point_avgs: int = 10,
        full_avgs: int = 1,
        plot: bool = True,
        save_data: bool = True,
    ):
        """do a scan of the non-equilibrium cell.  Right now this implies PSB readout on dots 1 and 2, and scanning non-equilibrium cell for dots 2 and 3.
        This helps us to set the detuning axis for a fingerprint plot. TODO add capability to do dots 1 and 2

        :param p_gates: p gates corresponding to dots 2 and 3 (if dots 1 and 2 are for readout)
        :param x_gate: exchange gate between p_gates
        :param x_amplitude: x gate pulse amplitude in volts
        :param x_time: length of time of exchange pulse in microseconds
        :param p_range: range of p gate sweeps: (p2 start voltage, p2 stop voltage, p2 points), (p3 start voltage, p3 stop voltage, p3 points)
        :param point_avgs: number of times to measure/average each point of the scan
        :param plot: Plot result
        :param save_data: Save data to netcdf


        """
        p2_gate, p3_gate = p_gates
        p2_start, p2_stop, p2_pts = p_range[0]
        p3_start, p3_stop, p3_pts = p_range[1]

        self.config.eo_cfg.gates.p3.gate = p3_gate
        self.config.eo_cfg.gates.p3.gen = self.hardware_config.channels[
            p3_gate
        ].qick_gen
        self.config.eo_cfg.gates.p3.start = self.volts2dac(p3_start, p3_gate)
        self.config.eo_cfg.gates.p3.stop = self.volts2dac(p3_stop, p3_gate)
        self.config.eo_cfg.gates.p3.expts = p3_pts
        self.config.eo_cfg.gates.p2.gate = p2_gate
        self.config.eo_cfg.gates.p2.gen = self.hardware_config.channels[
            p2_gate
        ].qick_gen
        self.config.eo_cfg.gates.p2.start = self.volts2dac(p2_start, p2_gate)
        self.config.eo_cfg.gates.p2.stop = self.volts2dac(p2_stop, p2_gate)
        self.config.eo_cfg.gates.p2.expts = p2_pts
        self.config.eo_cfg.gates.x.gate = x_gate
        self.config.eo_cfg.gates.x.gen = self.hardware_config.channels[x_gate].qick_gen
        self.config.eo_cfg.gates.x.gain = self.volts2dac(x_amplitude, x_gate)
        self.config.eo_cfg.gates.x.pulse_time = self.soccfg.us2cycles(x_time)
        self.config.shots = point_avgs
        self.config.reps = full_avgs

        meas = eo_single_qubit_programs.DoNonEquilibriumCell(self.soccfg, self.config)
        expt_pts, mag = meas.acquire(self.soc, load_pulses=True, progress=True)

        expt_pts[0] = self.dac2volts(expt_pts[0], p2_gate) * 1000
        expt_pts[1] = self.dac2volts(expt_pts[1], p3_gate) * 1000
        p3_array = np.linspace(p3_start, p3_stop, p3_pts)
        p2_array = np.linspace(p2_start, p2_stop, p2_pts)

        if plot:
            plot_tools.plot2_simple(p3_pts, p2_pts, mag, cbar_label="probability")
            plt.title("non-equilibrium cell")
            plt.xlabel("%s (mV)" % p3_gate)
            plt.ylabel("%s (mV)" % p2_gate)

        if save_data:
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_nonequilibrium_cell.nc")
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            nc_file.add_axis("p2", data=p2_array, units="Volts")
            nc_file.add_axis("p3", data=p3_array, units="Volts")
            nc_file.add_dataset(
                "readout", axes=["p2", "p3"], data=mag, units="probability"
            )
            nc_file.save_config(self.config)
            nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s", data_file)

        return [p3_array, p2_array], mag

    def find_p3_detuning_axis(self, vp2):
        """Calculate P3 voltage along detuning axis given P2 voltage and detuning axis points"""
        m = (self.detuning_axis[1, 1] - self.detuning_axis[0, 1]) / (
            self.detuning_axis[1, 0] - self.detuning_axis[0, 0]
        )
        b = self.detuning_axis[0, 1] - self.detuning_axis[0, 0] * m
        return m * vp2 + b

    def find_x_symmetric_axis(self, vp2):
        """Calculate X voltage along detuning axis given P2 voltage and detuning axis points"""
        m = (self.symmetric_axis[1, 2] - self.symmetric_axis[0, 2]) / (
            self.symmetric_axis[1, 0] - self.symmetric_axis[0, 0]
        )
        b = self.symmetric_axis[0, 2] - self.symmetric_axis[0, 0] * m
        return m * vp2 + b

    @dot_experiment.updater
    def do_fingerprint23(
        self,
        p_gates: Tuple[str, str],
        x_gate: str,
        detune_range: Tuple[float, float, int] = (-1, 1, 100),
        x_range: Tuple[float, float, int] = (-1, 1, 100),
        x_time: float = 0.010,
        point_avgs: int = 10,
        save_data: bool = True,
        plot: bool = True,
    ):
        """do a fingerprint plot on dots 2 and 3.  PSB readout on dots 1 and 2 is implied.
        This helps us find the symmetric exchange axis. TODO add capability to do dots 1 and 2.

        :param p_gates: p gates corresponding to dots 2 and 3 (if dots 1 and 2 are for readout)
        :param x_gate: exchange gate between p_gates
        :param x_amplitude: x gate pulse amplitude in volts
        :param x_time: length of time of exchange pulse in microseconds
        :param p_range: range of p gate sweeps: (p2 start voltage, p2 stop voltage, p2 points), (p3 start voltage, p3 stop voltage, p3 points)
        :param point_avgs: number of times to measure/average each point of the scan
        :param plot: Plot result
        :param save_data: Save data to netcdf


        """
        p2_gate, p3_gate = p_gates
        p2_start, p2_stop, p2_pts = detune_range
        p3_start = self.find_p3_detuning_axis(p2_start)
        p3_stop = self.find_p3_detuning_axis(p2_stop)
        x_start, x_stop, x_pts = x_range

        self.config.eo_cfg.gates.p2.gate = p2_gate
        self.config.eo_cfg.gates.p2.start = self.volts2dac(p2_start, p2_gate)
        self.config.eo_cfg.gates.p2.stop = self.volts2dac(p2_stop, p2_gate)
        self.config.eo_cfg.gates.p2.expts = p2_pts
        self.config.eo_cfg.gates.p2.gen = self.hardware_config.channels[
            p2_gate
        ].qick_gen
        self.config.eo_cfg.gates.p3.gate = p3_gate
        self.config.eo_cfg.gates.p3.start = self.volts2dac(p3_start, p3_gate)
        self.config.eo_cfg.gates.p3.stop = self.volts2dac(p3_stop, p3_gate)
        self.config.eo_cfg.gates.p3.expts = p2_pts
        self.config.eo_cfg.gates.p3.gen = self.hardware_config.channels[
            p3_gate
        ].qick_gen
        self.config.eo_cfg.gates.x.gate = x_gate
        self.config.eo_cfg.gates.x.start = self.volts2dac(x_start, x_gate)
        self.config.eo_cfg.gates.x.stop = self.volts2dac(x_stop, x_gate)
        self.config.eo_cfg.gates.x.pulse_time = self.soccfg.us2cycles(
            x_time
        )  # TODO upgrade this for more precise pulse length
        self.config.eo_cfg.gates.x.expts = x_pts
        self.config.eo_cfg.gates.x.gen = self.hardware_config.channels[x_gate].qick_gen
        self.config.expts = point_avgs
        self.config.shots = point_avgs
        self.config.reps = point_avgs

        # run the scan
        meas = eo_single_qubit_programs.DoFingerprint(self.soccfg, self.config)
        _, mag = meas.acquire(self.soc, load_pulses=True, progress=True)

        p2_axis = np.linspace(p2_start, p2_stop, p2_pts)
        p3_axis = np.linspace(p3_start, p3_stop, p2_pts)
        detune_axis = np.sqrt(p2_axis**2 + p3_axis**2)
        x_axis = np.linspace(x_start, x_stop, x_pts)

        # plot the data
        if plot:
            plot_tools.plot2_simple(
                x_axis, detune_axis, mag.T, cbar_label="probability"
            )
            plt.title("fingerprint")
            plt.xlabel("%s (mV)" % x_gate)
            plt.ylabel("%s (mV)" % p3_gate)

        if save_data:
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_fingerprint.nc")
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            nc_file.add_axis("p2", data=p2_axis, units="Volts")
            nc_file.add_axis("p3", data=p3_axis, units="Volts")
            nc_file.add_axis("x", data=x_axis, units="Volts")
            nc_file.add_dataset(
                "readout", axes=["x", "p2"], data=mag, units="probability"
            )
            nc_file.save_config(self.config)
            nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s", data_file)

        return [x_axis, p2_axis, p3_axis], mag


# TODO add do_rotation_calibration
