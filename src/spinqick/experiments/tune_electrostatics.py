"""
Perform charge stability measurements

"""

import logging
import os
from typing import Literal
import time

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import GaussianModel

from spinqick.experiments import dot_experiment
from spinqick.helper_functions import file_manager, hardware_manager, plot_tools
from spinqick.qick_code import tune_electrostatics_programs
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)


class TuneElectrostatics(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK experiments for charge stability measurements.
    In general each function sets the necessary config parameters for you and then runs the qick program.
    They then optionally plot and save the data.
    """

    def __init__(
        self,
        soccfg,
        soc,
        voltage_source: hardware_manager.VoltageSource,
        datadir: str = file_settings.data_directory,
    ):
        """initialize with information about your rfsoc and your experimental setup

        :param soccfg: QickConfig object
        :param soc: Qick object
        :param datadir: data directory where all data is being stored. Experiment will make a folder here with today's date.
        :param init_slow_dac: initialize your DC source object before starting any experiments.
        """
        super().__init__(datadir=datadir)
        self.soccfg = soccfg
        self.soc = soc
        self.datadir = datadir
        self.vdc = hardware_manager.DCSource(voltage_source=voltage_source)

    @dot_experiment.updater
    def gvg_baseband(
        self,
        g_gates: tuple[str, str],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]] = (
            (-0.1, 0.1, 100),
            (-0.1, 0.1, 100),
        ),
        point_avgs: int = 10,
        measure_delay: float = 10,
        add_pat: bool = False,
        pat_gain: int = 500,
        pat_freq: float = 4200,
        pat_gen: int = 3,
        save_data: bool = True,
        plot: bool = True,
    ):
        """perform a basic PvP or PvT by baseband pulsing with the RFSoC.

        :param g_gates: gates to sweep on y and x axes. i.e. ('P1','P2')
        :param g_range: range to sweep gate x and gate y, and number of points for each
        :param point_avgs: repeat each data point this many times and average the result
        :param measure_delay: pause time between measurements
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: expt pts, avgi, avgq -- raw data from 'acquire'
        """
        gx_gate, gy_gate = g_gates
        gx_start, gx_stop, gx_pts = g_range[0]
        gy_start, gy_stop, gy_pts = g_range[1]

        # fill in the config dictionary
        self.config.gvg_expt.gates.gy.gate = gy_gate
        self.config.gvg_expt.gates.gy.gen = self.hardware_config.channels[
            gy_gate
        ].qick_gen
        self.config.gvg_expt.gates.gy.start = self.volts2dac(gy_start, gy_gate)
        self.config.gvg_expt.gates.gy.stop = self.volts2dac(gy_stop, gy_gate)
        self.config.gvg_expt.gates.gy.expts = gy_pts
        self.config.gvg_expt.gates.gx.gate = gx_gate
        self.config.gvg_expt.gates.gx.gen = self.hardware_config.channels[
            gx_gate
        ].qick_gen
        self.config.gvg_expt.gates.gx.start = self.volts2dac(gx_start, gx_gate)
        self.config.gvg_expt.gates.gx.stop = self.volts2dac(gx_stop, gx_gate)
        self.config.gvg_expt.gates.gx.expts = gx_pts
        self.config.gvg_expt.trig_pin = self.hardware_config.slow_dac_trig_pin
        self.config.gvg_expt.measure_delay = measure_delay
        self.config.gvg_expt.add_pat = add_pat
        if add_pat:
            self.config.gvg_expt.pat_gain = pat_gain
            self.config.gvg_expt.pat_freq = pat_freq
            self.config.gvg_expt.pat_gen = pat_gen
        self.config.expts = point_avgs
        self.config.reps = 1

        # run QICK code
        meas = tune_electrostatics_programs.BasebandPulseGvG(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=True)
        self.soc.reset_gens()

        expt_pts[0] = self.dac2volts(expt_pts[0], gx_gate) * 1000
        expt_pts[1] = self.dac2volts(expt_pts[1], gy_gate) * 1000
        mag = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)
        mag = np.transpose(mag)
        data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
        # plot the data
        if plot:
            plot_tools.plot2_simple(xarray=expt_pts[1], yarray=expt_pts[0], data=mag)
            plt.title("%s v %s" % (gx_gate, gy_gate))
            plt.title("t: %d" % stamp, loc="right", fontdict={"fontsize": 6})
            plt.xlabel("%s (mV)" % gx_gate)
            plt.ylabel("%s (mV)" % gy_gate)

        if save_data:
            # make a directory for today's date and create a unique timestamp
            # data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(
                data_path, str(stamp) + "_" + gx_gate + "v" + gy_gate + "_baseband.nc"
            )
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            vx_sweep = np.linspace(gx_start, gx_stop, gx_pts)
            vy_sweep = np.linspace(gy_start, gy_stop, gy_pts)
            nc_file.add_axis(label=gx_gate + "_sweep", data=vx_sweep, units="V")
            nc_file.add_axis(label=gy_gate + "_sweep", data=vy_sweep, units="V")
            nc_file.add_dataset(
                label="readout",
                axes=[gy_gate + "_sweep", gx_gate + "_sweep"],
                data=mag,
                units="adc_magnitude",
            )
            nc_file.data_flavour = "gvg_baseband"
            if plot:
                nc_file.save_last_plot()
            nc_file.save_config(self.config)
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, avgi, avgq, mag

    @dot_experiment.updater
    def gvg_dc(
        self,
        g_gates: tuple[list[str], list[str]],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]] = (
            (-1.0, 1.0, 100),
            (-1.0, 1.0, 100),
        ),
        compensate: str | None = None,
        sweep_direction: tuple[list[int], list[int]] | None = None,
        measure_buffer: float = 10,
        mode: Literal["sdchop", "transdc"] = "sdchop",
        twiddle_gate=None,
        plot=True,
        save_data=True,
    ):
        """GvG script which sweeps an external DC voltage source and reads out a DCS

        :param g_gates: gates to sweep on y and x axes. i.e. (['P1'],['P2']).  Option to provide a list of gates
            to sweep on each axis.
        :param g_range: voltage range to sweep gate x and gate y, and number of points for each.
        :param compensate: gate to compensate while changing other voltages. i.e. 'M1'
        :param sweep_direction: Allows user to sweep a gate backwards if desired. Provide a
            list of positive or negative ones corresponding to each gate in g_gates i.e. ([1,-1], [1]).
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param mode: "sdchop" selects typical source drain chop readout, "transdc" is for transcoductance mode
        :param twiddle_gate: if using transconductance mode, select a gate to apply the ac signal on
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: expt pts, avgi, avgq -- raw data from 'acquire'
        """

        if mode == "transdc":
            ### transconductance mode right now is in the 1.6K style assuming a DC offset on SD
            res_ch = self.config.DCS_cfg.res_ch
            self.config.DCS_cfg.res_ch = self.hardware_config.channels[
                str(twiddle_gate)
            ].qick_gen

        gx_gates, gy_gates = g_gates
        gx_start, gx_stop, n_vx = g_range[0]
        gy_start, gy_stop, n_vy = g_range[1]

        # Vx Sweep
        vx_0 = []
        vx_sweep = np.zeros((len(gx_gates), n_vx))
        for i, gx in enumerate(gx_gates):
            vx_0.append(self.vdc.get_dc_voltage(gx))
            # sweep backwards if sweep direction is set to -1
            if sweep_direction is not None:
                if sweep_direction[0][i] == -1:
                    vx_start = gx_stop + vx_0[i]
                    vx_stop = gx_start + vx_0[i]
                else:
                    vx_start = gx_start + vx_0[i]
                    vx_stop = gx_stop + vx_0[i]
            else:
                vx_start = gx_start + vx_0[i]
                vx_stop = gx_stop + vx_0[i]
            vx_sweep[i, :] = np.linspace(vx_start, vx_stop, n_vx)

        # Vy Sweep
        vy_0 = []
        vy_sweep = np.zeros((len(gy_gates), n_vy))
        for i, gy in enumerate(gy_gates):
            vy_0.append(self.vdc.get_dc_voltage(gy))
            if sweep_direction is not None:
                if sweep_direction[1][i] == -1:
                    vy_start = gy_stop + vy_0[i]
                    vy_stop = gy_start + vy_0[i]
                else:
                    vy_start = gy_start + vy_0[i]
                    vy_stop = gy_stop + vy_0[i]
            else:
                vy_start = gy_start + vy_0[i]
                vy_stop = gy_stop + vy_0[i]
            vy_sweep[i, :] = np.linspace(vy_start, vy_stop, n_vy)

        if isinstance(compensate, str):
            vm_0 = self.vdc.get_dc_voltage(compensate)

        # setup the slow_dac step length
        slow_dac_step_len = (
            self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
        )

        # self.config.measure_delay = measure_buffer
        self.config.expts = n_vy
        self.config.start = gy_start
        self.config.step = (gy_stop - gy_start) / n_vy
        self.config.gvg_expt.measure_delay = measure_buffer
        self.config.gvg_expt.trig_pin = self.hardware_config.slow_dac_trig_pin
        self.config.reps = 1

        # setup data structure
        readout = np.zeros((n_vy, n_vx))
        for vx_index in range(n_vx):
            # Set up SDAC voltages and sweeps
            if isinstance(compensate, str):
                self.vdc.set_dc_voltage_compensate(
                    [vx_sweep[i, vx_index] for i in range(len(gx_gates))]
                    + [vy_sweep[i, 0] for i in range(len(gy_gates))],
                    gx_gates + gy_gates,
                    compensate,
                )
                for m, gy_gate in enumerate(gy_gates):
                    self.vdc.program_ramp_compensate(
                        vy_sweep[m, 0],
                        vy_sweep[m, -1],
                        slow_dac_step_len * 1e-6,
                        n_vy,
                        gy_gate,
                        compensate,
                    )
                # right now compensation will only work for one y gate
                self.vdc.arm_sweep(compensate)
            else:
                for i, gx_gate in enumerate(gx_gates):
                    self.vdc.set_dc_voltage(vx_sweep[i, vx_index], gx_gate)
                for m, gy_gate in enumerate(gy_gates):
                    self.vdc.program_ramp(
                        vy_sweep[m, 0],
                        vy_sweep[m, -1],
                        slow_dac_step_len * 1e-6,
                        n_vy,
                        gy_gate,
                    )
            for gy_gate in gy_gates:
                self.vdc.arm_sweep(gy_gate)

            # Start a Vy sweep at a Vx increment and store the data
            meas = tune_electrostatics_programs.GvG(self.soccfg, self.config)
            expt_pts, avgi, avgq = meas.acquire(
                self.soc, load_pulses=True, progress=False
            )
            mag = np.sqrt(avgi[0] ** 2 + avgq[0] ** 2)
            readout[:, vx_index] = mag
            time.sleep(0.010)

        ### set config file res_ch back to SD channel
        if mode == "transdc":
            self.config.DCS_cfg.res_ch = res_ch

        ### return to starting point
        for i, gx_gate in enumerate(gx_gates):
            self.vdc.set_dc_voltage(vx_0[i], gx_gate)
        for k, gy_gate in enumerate(gy_gates):
            self.vdc.set_dc_voltage(vy_0[k], gy_gate)
        if isinstance(compensate, str):
            self.vdc.set_dc_voltage(vm_0, compensate)

        data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
        # plot the data
        if plot:
            x_label = ""
            for gate in gx_gates:
                x_label = x_label + " " + gate + ","
            y_label = ""
            for gate in gy_gates:
                y_label = y_label + " " + gate + ","
            plot_tools.plot2_simple(vx_sweep[0, :], vy_sweep[0, :], readout)
            plt.title("%s v %s" % (gx_gate, gy_gate))
            plt.title("t: %d" % stamp, loc="right", fontdict={"fontsize": 6})
            plt.xlabel("%s (V)" % x_label)
            plt.ylabel("%s (V)" % y_label)

        if save_data:
            # make a directory for today's date and create a unique timestamp
            # data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(
                data_path, str(stamp) + "_" + gx_gates[0] + "v" + gy_gates[0] + "dc.nc"
            )

            # save data in netcdf4 format
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            for i, gx_gate in enumerate(gx_gates):
                nc_file.add_axis(
                    label=gx_gate + "_sweep", data=vx_sweep[i, :], units="V"
                )
            for i, gy_gate in enumerate(gy_gates):
                nc_file.add_axis(
                    label=gy_gate + "_sweep", data=vy_sweep[i, :], units="V"
                )
            nc_file.add_dataset(
                label=mode + "_readout",
                axes=[gy_gates[0] + "_sweep", gx_gates[0] + "_sweep"],
                data=readout,
                units="adc",
            )
            nc_file.data_flavour = "gvg_dc"
            if plot:
                nc_file.save_last_plot()
            nc_file.save_config(self.config)
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return expt_pts, avgi, avgq

    def tune_mz(
        self,
        m_dot: str,
        m_range: list = [-0.1, 0.1, 50],
        z_range: list = [-0.1, 0.1, 50],
        tune_type: Literal["common", "diff"] = "common",
        measure_buffer: float = 5,
        save_data: bool = True,
        plot: bool = True,
    ):
        """Tune MZ using sleeperdac or other external DC voltage bias. This wraps the function gvg_dc.

        :param m_dot: M dot you're tuning, i.e. 'M1'
        :param m_range: voltage range to sweep m gate, and number of points.
        :param z_range: voltage range to sweep z gate, and number of points.
        :param tune_type: sweep z gates in same direction or opposite directions
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: expt pts, avgi, avgq -- raw data from 'acquire'
        """

        if m_dot == "M1":
            z_dots = ["Z1", "Z2"]
        elif m_dot == "M2":
            z_dots = ["Z3", "Z4"]

        if tune_type == "common":
            z_sweep_vector = [1, 1]
        if tune_type == "diff":
            z_sweep_vector = [1, -1]

        a = self.gvg_dc(
            g_gates=([m_dot], z_dots),
            g_range=(m_range, z_range),
            compensate=None,
            measure_buffer=measure_buffer,
            save_data=save_data,
            mode="sdchop",
            sweep_direction=([1], z_sweep_vector),
            plot=plot,
        )
        return a

    @dot_experiment.updater
    def retune_dcs(
        self,
        m_dot: str,
        m_range: list[float] = [-0.008, 0.008, 50],
        measure_buffer: float = 100,
        set_v=False,
        save_data: bool = True,
        plot: bool = True,
    ):
        """automated retune DCS script. Scan m-gate voltage, fit to conductance peak, extract an
        optimal m-gate voltage from peak

        :param m_dot: Dot to retune
        :param m_range: List of start voltage, stop voltage and number of points.  Voltages are relative
            to the current setpoint
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: vm_sweep, mag, out

        """

        # M Sweep
        m_bias = self.vdc.get_dc_voltage(m_dot)
        n_vm = int(m_range[2])
        vm_start = m_range[0] + m_bias
        vm_stop = m_range[1] + m_bias
        vm_sweep = np.linspace(vm_start, vm_stop, n_vm)

        slow_dac_step_len = (
            self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
        )


        self.config.gvg_expt.measure_delay = measure_buffer
        self.config.expts = n_vm
        self.config.reps = 1
        self.config.start = vm_start
        self.config.step = (vm_stop - vm_start) / n_vm
        self.config.gvg_expt.trig_pin = self.hardware_config.slow_dac_trig_pin

        self.vdc.program_ramp(vm_start, vm_stop, slow_dac_step_len * 1e-6, n_vm, m_dot)
        self.vdc.arm_sweep(m_dot)

        # Start a Vy sweep at a Vx increment and store the data
        meas = tune_electrostatics_programs.GvG(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=False)
        mag = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)

        # Ramp Vy voltage back down to the starting value
        t_min_slow_dac = 2.65
        self.vdc.program_ramp(vm_stop, vm_start, t_min_slow_dac * 1e-6, n_vm, m_dot)
        self.vdc.digital_trigger(m_dot)

        gaussian = GaussianModel()
        pars = gaussian.guess(mag, x=vm_sweep)
        try:
            # try the fit to a gaussian
            out = gaussian.fit(mag, pars, x=vm_sweep)
            center = out.params["center"].value
            fwhm = out.params["fwhm"].value
            halfmax = center - fwhm / 2
            print("best voltage = %f" % halfmax)
            # set to the half maximum value
            if set_v:
                # first make sure the voltage it picks is within limits of sweep for safety
                if np.logical_and(halfmax < vm_stop, halfmax > vm_start):
                    self.vdc.set_dc_voltage(halfmax, m_dot)
            else:
                self.vdc.set_dc_voltage(m_bias, m_dot)
        except Exception as exc:
            if set_v:
                self.vdc.set_dc_voltage(m_bias, m_dot)
            fwhm = 0
            center = 0
            halfmax = 0
            logger.error(
                "fit failed, setting m to initial value, error %s", exc, exc_info=True
            )

        # plot the data
        plt.figure()
        plt.plot(vm_sweep, mag)
        plt.plot(vm_sweep, out.best_fit)
        try:
            plt.plot(
                [halfmax], [gaussian.eval(out.params, x=halfmax)], "o"
            )  # not sure this will work!
        except Exception as exc:
            logger.error("fix halfmax code: %s", exc, exc_info=True)
        plt.title("retune dcs")
        plt.xlabel(" %s (mV)" % (m_dot))
        plt.ylabel("conductance (arbs)")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_retune_dcs.nc")

            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            nc_file.add_axis(label=m_dot, data=vm_sweep, units="V")
            nc_file.add_dataset(label="readout", axes=[m_dot], data=mag, units="adc")
            nc_file.data_flavour = "retune_dcs"
            nc_file.halfmax = halfmax
            nc_file.center = center
            nc_file.fwhm = fwhm
            nc_file.save_config(self.config)
            if plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return vm_sweep, mag, out

    @dot_experiment.updater
    def gate_action(
        self,
        gates: list[str],
        max_v: float = 1.2,
        num_points: int = 100,
        measure_buffer: float = 10,
        save_data: bool = True,
        plot: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """gate action sweep script.  Sweep gates individually up and back down to look for
        turn on voltages

        :param gates: list of gates to sweep
        :param max_v: voltage to sweep up to
        :param num_points: points in each sweep direction
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: vm_sweep, mag
        """

        n_v = num_points
        # setup the slow_dac step length
        slow_dac_step_len = (
            self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
        )
        if slow_dac_step_len < 2.65:
            measure_buffer = 2.65 - self.soccfg.cycles2us(self.config.DCS_cfg.length)
            slow_dac_step_len = (
                self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
            )
        self.config.gvg_expt.measure_delay = measure_buffer
        self.config.expts = n_v
        self.config.start = 0
        self.config.step = max_v / n_v
        self.config.gvg_expt.trig_pin = self.hardware_config.slow_dac_trig_pin
        self.config.reps = 1

        if save_data:
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_gate_action.nc")

        if plot:
            plt.figure()
        for gate_label in gates:
            # gate sweeps
            v_0 = self.vdc.get_dc_voltage(gate_label)
            v_start: float = 0.0
            v_stop = max_v
            v_sweep_up = np.linspace(v_start, v_stop, n_v)
            self.vdc.program_ramp(
                v_start, v_stop, slow_dac_step_len * 1e-6, n_v, gate_label
            )
            self.vdc.arm_sweep(gate_label)

            # Start a Vy sweep at a Vx increment and store the data
            meas = tune_electrostatics_programs.GvG(self.soccfg, self.config)
            expt_pts, avgi, avgq = meas.acquire(
                self.soc, load_pulses=True, progress=False
            )
            mag_up = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)

            time.sleep(0.5)

            # sweep back down
            v_start = max_v
            v_stop = 0
            v_sweep_down = np.linspace(v_start, v_stop, n_v)
            self.vdc.program_ramp(
                v_start, v_stop, slow_dac_step_len * 1e-6, n_v, gate_label
            )
            self.vdc.arm_sweep(gate_label)
            expt_pts, avgi, avgq = meas.acquire(
                self.soc, load_pulses=True, progress=False
            )
            mag_down = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)

            mag = np.append(mag_up, mag_down)
            v_sweep = np.append(v_sweep_up, v_sweep_down)

            # set voltage back to initial value!
            self.vdc.set_dc_voltage(v_0, gate_label)
            if plot:
                # plot the data
                plt.plot(v_sweep, mag, label=gate_label)
            if save_data:
                nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
                nc_file.add_axis(gate_label + "_V", data=v_sweep, units="V")
                nc_file.add_dataset(
                    gate_label, axes=[gate_label + "_V"], data=mag, units="adc_raw"
                )
                nc_file.close()
        if plot:
            plt.title("gate action")
            x_label = "gates:"
            for gate in gates:
                x_label = x_label + " " + gate + ","
            plt.xlabel(" %s (mV)" % (x_label))
            plt.ylabel("conductance (arbs)")
            plt.legend()

        if save_data:
            # save the config and the plot
            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            if plot:
                nc_file.save_last_plot()
            nc_file.save_config(self.config)
            nc_file.data_flavour = "gate_action_sweep"
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return v_sweep, mag

    @dot_experiment.updater
    def gate_turn_on(
        self,
        gates: list[str],
        max_v: float = 1.2,
        num_points: int = 100,
        measure_buffer: float = 10,
        save_data: bool = True,
        plot: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Global gate turn on script.  Sweep several gates simultaneously to look for global
        turn on of the device.

        :param gates: list of gates to sweep
        :param max_v: voltage to sweep up to
        :param num_points: points in each sweep direction
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: vm_sweep, mag"""

        n_v = num_points
        # setup the slow_dac step length
        slow_dac_step_len = (
            self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
        )
        if slow_dac_step_len < 2.65:
            measure_buffer = 2.65 - self.soccfg.cycles2us(self.config.DCS_cfg.length)
            slow_dac_step_len = (
                self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
            )
        self.config.gvg_expt.measure_delay = measure_buffer
        self.config.expts = n_v
        self.config.start = 0
        self.config.step = max_v / n_v
        self.config.gvg_expt.trig_pin = self.hardware_config.slow_dac_trig_pin
        self.config.reps = 1
        if plot:
            plt.figure()
        v_0 = np.zeros((len(gates)))
        for i, gate_label in enumerate(gates):
            # gate sweeps
            v_0[i] = self.vdc.get_dc_voltage(gate_label)
            v_start = 0
            v_stop = max_v
            v_sweep = np.linspace(v_start, v_stop, n_v)
            self.vdc.program_ramp(
                v_start, v_stop, slow_dac_step_len * 1e-6, n_v, gate_label
            )
        for gate_label in gates:
            self.vdc.arm_sweep(gate_label)

        # Start a Vy sweep at a Vx increment and store the data
        meas = tune_electrostatics_programs.GvG(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=False)
        mag = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)

        for i, gate_label in enumerate(gates):
            self.vdc.set_dc_voltage(v_0[i], gate_label)
        if plot:
            # plot the data
            x_label = "gates:"
            for gate in gates:
                x_label = x_label + " " + gate + ","
            plt.plot(v_sweep, mag, label=x_label)
            plt.title("global gate turn on")
            plt.xlabel(" %s (mV)" % (x_label))
            plt.ylabel("conductance (arbs)")

        if save_data:
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_gate_turn_on.nc")

            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            for gate_label in gates:
                nc_file.add_axis(gate_label + "_V", data=v_sweep, units="V")
            nc_file.add_dataset(
                gate_label, axes=[gates[0] + "_V"], data=mag, units="adc_raw"
            )
            if plot:
                nc_file.save_last_plot()
            nc_file.save_config(self.config)
            nc_file.data_flavour = "global_gate_turn_on"
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return v_sweep, mag

    @dot_experiment.updater
    def sweep_1D(
        self,
        gates: list[str],
        g_range: tuple[float, float, int],
        measure_buffer: float = 50,
        mc_temp: float | None = None,
        compensate: str | None = None,
        save_data: bool = True,
        filename_tag: str | None = None,
        plot: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sweep gates over desired range. No averaging built in currently.  Can be used for electron temperature measurements. Work in progress.

        :param gates: list of gates to sweep
        :param g_range: (start voltage, stop voltage, number of steps)
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param mc_temp: optionally note the temperature of the mixing chamber
        :param filename_tag: add a string to the filename ie 'electron temperature'
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: vm_sweep, mag"""

        g_start, g_stop, n_v = g_range
        # setup the slow_dac step length
        slow_dac_step_len = (
            self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
        )
        if slow_dac_step_len < 2.65:
            measure_buffer = 2.65 - self.soccfg.cycles2us(self.config.DCS_cfg.length)
            slow_dac_step_len = (
                self.soccfg.cycles2us(self.config.DCS_cfg.length) + 2 * measure_buffer
            )
        self.config.gvg_expt.measure_delay = measure_buffer
        self.config.expts = n_v
        self.config.start = g_start
        self.config.step = (g_stop - g_start) / n_v
        self.config.gvg_expt.trig_pin = self.hardware_config.slow_dac_trig_pin
        self.config.reps = 1
        if plot:
            plt.figure()
        v_0 = np.zeros((len(gates)))
        if compensate is not None:
            v_m = self.vdc.get_dc_voltage(compensate)
        for i, gate_label in enumerate(gates):
            # gate sweeps
            v_bias_0 = self.vdc.get_dc_voltage(gate_label)
            v_0[i] = v_bias_0
            v_start = v_bias_0 + g_start
            v_stop = v_bias_0 + g_stop
            v_sweep = np.linspace(v_start, v_stop, n_v)
            if compensate is None:
                self.vdc.program_ramp(
                    v_start, v_stop, slow_dac_step_len * 1e-6, n_v, gate_label
                )
            else:
                # currently this will only compensate for one gate
                self.vdc.program_ramp_compensate(
                    v_start,
                    v_stop,
                    slow_dac_step_len * 1e-6,
                    n_v,
                    gate_label,
                    compensate,
                )
                self.vdc.arm_sweep(compensate)
        for gate_label in gates:
            self.vdc.arm_sweep(gate_label)

        # Start a Vy sweep
        meas = tune_electrostatics_programs.GvG(self.soccfg, self.config)
        expt_pts, avgi, avgq = meas.acquire(self.soc, load_pulses=True, progress=False)
        mag = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)

        for i, gate_label in enumerate(gates):
            self.vdc.set_dc_voltage(v_0[i], gate_label)
        if compensate is not None:
            self.vdc.set_dc_voltage(v_m, compensate)
        if plot:
            # plot the data
            x_label = "gates:"
            for gate in gates:
                x_label = x_label + " " + gate + ","
            plt.plot(v_sweep, mag, label=x_label)
            if mc_temp is None:
                plt.title("1D voltage sweep")
            else:
                plt.title("1D voltage sweep, %.3f" % mc_temp)
            plt.xlabel(" %s (V)" % (x_label))
            plt.ylabel("conductance (arbs)")

        if save_data:
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            if filename_tag is None:
                data_file = os.path.join(data_path, str(stamp) + "_1D_sweep.nc")
            else:
                data_file = os.path.join(
                    data_path, str(stamp) + "_" + filename_tag + "_1D_sweep.nc"
                )

            nc_file = file_manager.SaveData(data_file, "a", format="NETCDF4")
            for gate_label in gates:
                nc_file.add_axis(gate_label + "_V", data=v_sweep, units="V")
            nc_file.add_dataset(
                gate_label, axes=[gates[0] + "_V"], data=mag, units="adc_raw"
            )
            if plot:
                nc_file.save_last_plot()
            nc_file.save_config(self.config)
            nc_file.data_flavour = "1Dsweep"
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return v_sweep, mag
