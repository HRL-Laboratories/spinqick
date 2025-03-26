"""
This module holds scripts which can be useful when setting up a new device or system.

TODO: adc_trig_offset measurement

"""

import logging
import os
from typing import Sequence, Tuple
import time

import netCDF4
import numpy as np
from lmfit.models import GaussianModel, LinearModel
from matplotlib import pyplot as plt

from spinqick.experiments import dot_experiment
from spinqick.helper_functions import file_manager, hardware_manager, plot_tools
from spinqick.qick_code import (
    system_calibrations_programs,
    tune_electrostatics_programs,
)
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)


class SystemCalibrations(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK classes for hardware calibrations"""

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
    def calibrate_baseband_voltage(
        self,
        p_gate: str,
        x_gate: str,
        p_dc_range: Tuple[float, float, int] = (-0.1, 0.1, 100),
        p_pulse_range: Tuple[float, float, int] = (0, 10000, 100),
        x_pulse_gain: int = 100,
        point_avgs: int = 10,
        chop_time: float = 0.05,
        measure_buffer: float = 10,
        plot: bool = True,
        save_data: bool = True,
    ):
        """Calibrate baseband voltage based off of low speed dacs.
        Scans a loading line with low speed dacs while sweeping pulse gain of high speed dacs"""

        p_dc_start, p_dc_stop, p_dc_npts = p_dc_range
        p_pulse_start, p_pulse_stop, p_pulse_npts = p_pulse_range
        self.config.calibrate_cfg.p_gate.gate = p_gate
        self.config.calibrate_cfg.p_gate.gen = self.hardware_config.channels[
            str(p_gate)
        ].qick_gen
        self.config.calibrate_cfg.x_gate.gate = x_gate
        self.config.calibrate_cfg.x_gate.gen = self.hardware_config.channels[
            str(x_gate)
        ].qick_gen
        self.config.calibrate_cfg.p_gate.gain = p_pulse_start
        self.config.calibrate_cfg.p_gate.chop_time = self.soccfg.us2cycles(chop_time)
        self.config.calibrate_cfg.x_gate.gain = x_pulse_gain
        # these phases could be used to modify phase offset of adc, p and x gates
        self.config.calibrate_cfg.res_phase = 0
        self.config.calibrate_cfg.res_phase_diff = 0
        self.config.calibrate_cfg.trig_pin = self.hardware_config.slow_dac_trig_pin
        self.config.expts = p_dc_npts
        self.config.start = p_pulse_start
        self.config.step = (p_pulse_stop - p_pulse_start) / p_pulse_npts
        self.config.reps = 1
        self.config.calibrate_cfg.measure_delay = measure_buffer

        t_min_slow_dac = 2.65  # microseconds

        # DC P Sweep
        p_bias = self.vdc.get_dc_voltage(p_gate)
        vp_start = p_dc_start + p_bias
        vp_stop = p_dc_stop + p_bias
        vp_sweep = np.linspace(vp_start, vp_stop, p_dc_npts)

        # Baseband P sweep
        baseband_sweep = np.linspace(p_pulse_start, p_pulse_stop, p_pulse_npts)

        # setup the slow_dac step length
        slow_dac_step_len = (
            self.soccfg.cycles2us(self.config.dcs_cfg.length) + 2 * measure_buffer
        )

        data = np.zeros((point_avgs, p_dc_npts, p_pulse_npts), dtype=complex)
        for avg in range(point_avgs):
            for i, gain in enumerate(baseband_sweep):
                self.config.calibrate_cfg.p_gate.gain = int(gain)
                self.vdc.program_ramp(
                    vp_start, vp_stop, slow_dac_step_len * 1e-6, p_dc_npts, p_gate
                )
                self.vdc.arm_sweep(p_gate)

                # Start QICK code and run the slow_dac ramp
                meas = system_calibrations_programs.BasebandVoltageCalibration(
                    self.soccfg, self.config
                )
                expt_pts, avgi, avgq = meas.acquire(
                    self.soc, load_pulses=True, progress=False
                )
                trans_data = avgi[0][:] + 1j * avgq[0][:]

                data[avg, :, i] = trans_data
                time.sleep(0.010)

        # Ramp voltage back down to the starting value
        self.vdc.set_dc_voltage(p_bias, p_gate)
        time.sleep(0.010)
        self.soc.reset_gens()

        # try to rotate iq data into one axis, plot
        phi_rot = np.mean(np.arctan2(data.imag, data.real))
        data_rot = data * np.exp(-1j * phi_rot)
        data_mean = np.mean(data_rot.real, axis=0)
        data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
        if plot:
            plot_tools.plot2_simple(
                baseband_sweep, vp_sweep, data_mean, cbar_label="transconductance"
            )
            plt.title("baseband voltage calibration")
            plt.title("t: %d" % stamp, loc="right", fontdict={"fontsize": 6})

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_file = os.path.join(data_path, str(stamp) + "_baseband_cal.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_baseband_cal.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_baseband_cal.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            if plot:
                plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("V_baseband", p_pulse_range[2])
            nc_file.createDimension("Vdc", p_dc_npts)
            nc_file.createDimension("shots", point_avgs)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata",
                np.float32,
                (
                    "IQ",
                    "shots",
                    "Vdc",
                    "V_baseband",
                ),
            )
            psbraw.units = "raw_adc"
            gx = nc_file.createVariable("baseband pulse", np.float32, ("V_baseband"))
            gx.units = "dac_units"
            gx[:] = baseband_sweep
            gy = nc_file.createVariable("slow_dac sweep", np.float32, ("Vdc"))
            gy.units = "gate_voltage"
            gy[:] = vp_sweep
            psbraw[0, :, :, :] = data.real
            psbraw[1, :, :, :] = data.imag
            processed = nc_file.createVariable(
                "processed", np.float32, ("Vdc", "V_baseband")
            )
            processed[:] = data_mean
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return vp_sweep, baseband_sweep, data, data_rot

    @dot_experiment.updater
    def tune_hsa(
        self,
        tune_gate: str,
        gate_pulse_gain: int,
        gate_dc_range: Tuple[float, float, int] = (-0.1, 0.1, 100),
        pulse_times: Sequence[float] = [0.1, 100],
        loop_avgs: int = 10,
        measure_buffer: float = 10,
        plot: bool = True,
        save_data: bool = True,
    ):
        """Linesplitting measurement for tuning a HSA [1]. Pulse down over charge transition to check if high speed adder is tuned correctly.

        1. HSA = High Speed Adder as presented in "Fast and High-Fidelity State Preparation and Measurement in Triple-Quantum-Dot Spin Qubits"
        DOI: https://doi.org/10.1103/PRXQuantum.3.010352
        """

        self.config.calibrate_cfg.tune_gate.gate = tune_gate
        self.config.calibrate_cfg.tune_gate.gen = self.hardware_config.channels[
            str(tune_gate)
        ].qick_gen
        self.config.calibrate_cfg.tune_gate.gain = (
            gate_pulse_gain  # this number should be negative
        )
        self.config.calibrate_cfg.tune_gate.pulse_time = self.soccfg.us2cycles(
            pulse_times[0]
        )
        self.config.expts = (
            1  # in our real life experiment we looped over a whole sweep each time
        )
        self.config.measure_delay = measure_buffer

        # DC Sweep
        gate_bias = self.vdc.get_dc_voltage(tune_gate)
        n_vg = gate_dc_range[2]
        vg_start = gate_dc_range[0] + gate_bias
        vg_stop = gate_dc_range[1] + gate_bias
        vg_sweep = np.linspace(vg_start, vg_stop, n_vg)

        data = np.zeros((n_vg, len(pulse_times), loop_avgs))
        data_raw = np.zeros((2, n_vg, len(pulse_times), loop_avgs))
        for i, pulse_time in enumerate(pulse_times):
            self.config.calibrate_cfg.pulse_time = self.soccfg.us2cycles(pulse_time)
            for k in range(loop_avgs):
                for n, volt in enumerate(vg_sweep):
                    self.vdc.set_dc_voltage(volt, tune_gate)
                    meas = system_calibrations_programs.HSATune(
                        self.soccfg, self.config
                    )
                    expt_pts, avgi, avgq = meas.acquire(
                        self.soc, load_pulses=True, progress=False
                    )
                    data[n, i, k] = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)
                    data_raw[0, n, i, k] = avgi[0][0]
                    data_raw[1, n, i, k] = avgq[0][0]

        self.vdc.set_dc_voltage(gate_bias, tune_gate)
        data_avg = np.mean(data, axis=2)
        if plot:
            plt.figure()
            for m, t in enumerate(pulse_times):
                plt.plot(vg_sweep, data_avg[:, m], label="time=%.3f us" % t)
            plt.title("HSA tune")
            plt.xlabel(" %s voltage (V)" % (tune_gate))
            plt.ylabel("conductance (arbs)")

        if save_data:
            # make a directory for today's date and create a unique timestamp
            data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
            data_file = os.path.join(data_path, str(stamp) + "_HSA_cal.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_HSA_cal.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_HSA_cal.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("time", len(pulse_times))
            nc_file.createDimension("Vdc", n_vg)
            nc_file.createDimension("avgs", loop_avgs)
            nc_file.createDimension("IQ", 2)

            # create variables and fill in data
            psbraw = nc_file.createVariable(
                "fulldata",
                np.float32,
                (
                    "IQ",
                    "Vdc",
                    "time",
                    "avgs",
                ),
            )
            psbraw.units = "raw_adc"
            gx = nc_file.createVariable("V", np.float32, ("Vdc"))
            gx.units = "Volts"
            gx[:] = vg_sweep
            times = nc_file.createVariable("times", np.float32, ("time"))
            times.units = "us"
            times[:] = pulse_times
            psbraw[:, :, :, :] = data_raw

            processed = nc_file.createVariable("processed", np.float32, ("Vdc", "time"))
            processed[:, :] = data_avg
            nc_file.close()
            logger.info("data saved at %s" % data_file)

        return vg_sweep, data_raw, data_avg

    @dot_experiment.updater
    def get_cross_caps(
        self,
        m_dot: str,
        gate: str,
        sdac_range: Tuple[float, float, int] = (-0.01, 0.01, 100),
        gate_range: Tuple[float, float, int] = (-0.01, 0.01, 100),
        measure_buffer: float = 50,
        plot: bool = True,
        save_data: bool = True,
    ):
        """get cross coupling between m dot and your gate of choice
        TODO implement ability to get cross coupling between other gates, add data saving
        """

        t_min_slow_dac = 2.65
        gate_start, gate_stop, n_vg = gate_range
        m_start, m_stop, n_vm = sdac_range
        m_bias = self.vdc.get_dc_voltage(m_dot)

        # M Sweep
        m_bias = self.vdc.get_dc_voltage(m_dot)
        vm_start = m_start + m_bias
        vm_stop = m_stop + m_bias
        vm_sweep = np.linspace(vm_start, vm_stop, n_vm)

        # gate sweep
        gate_bias = self.vdc.get_dc_voltage(gate)
        vg_start = gate_start + gate_bias
        vg_stop = gate_stop + gate_bias
        vg_sweep = np.linspace(vg_start, vg_stop, n_vg)

        # setup the slow_dac step length
        slow_dac_step_len = (
            self.soccfg.cycles2us(self.config.dcs_cfg.length) + 2 * measure_buffer
        )

        self.config.gvg_expt.measure_delay = measure_buffer
        self.config.expts = n_vm
        self.config.start = vg_start
        self.config.step = (vg_stop - vg_start) / n_vg
        self.config.reps = 1
        self.config.gvg_expt.trig_pin = self.hardware_config.slow_dac_trig_pin

        data = np.zeros_like(vg_sweep)
        mdata = np.zeros((len(vg_sweep), n_vm))
        for i, vgate in enumerate(vg_sweep):
            self.vdc.set_dc_voltage(vgate, gate)
            self.vdc.program_ramp(
                vm_start, vm_stop, slow_dac_step_len * 1e-6, n_vm, m_dot
            )
            self.vdc.arm_sweep(m_dot)

            # Start a Vy sweep at a Vx increment and store the data
            meas = tune_electrostatics_programs.GvG(self.soccfg, self.config)
            expt_pts, avgi, avgq = meas.acquire(
                self.soc, load_pulses=True, progress=False
            )
            mag = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2)
            self.vdc.program_ramp(vm_stop, vm_start, t_min_slow_dac * 1e-6, n_vm, m_dot)
            self.vdc.digital_trigger(m_dot)

            gaussian = GaussianModel()
            pars = gaussian.guess(mag, x=vm_sweep)
            try:
                out = gaussian.fit(mag, pars, x=vm_sweep)
            except Exception as exc:
                logger.error("fit failed, %s", exc, exc_info=True)
            if np.logical_and(
                out.params["center"].value > vm_start,
                out.params["center"].value < vm_stop,
            ):
                data[i] = out.params["center"].value
                mdata[i, :] = mag
            else:
                data[i] = np.nan
                mdata[i, :] = mag

        self.vdc.set_dc_voltage(m_bias, m_dot)
        self.vdc.set_dc_voltage(gate_bias, gate)

        line = LinearModel()
        pars = line.guess(data, x=vg_sweep)
        try:
            out = line.fit(data, pars, x=vg_sweep)
            slope = out.params["slope"].value
            logger.info("slope is %f" % slope)
        except Exception as exc:
            slope = np.nan
            logger.error(" line fit failed, %s", exc, exc_info=True)

        data_path, stamp = file_manager.get_new_timestamp(datadir=self.datadir)
        if plot:
            # plot the data
            plt.figure()
            plt.plot(vg_sweep, data, ".")
            plt.plot(vg_sweep, out.best_fit)

            plt.title("cross coupling")
            plt.xlabel(" %s (V)" % (gate))
            plt.ylabel("fitted center voltage (V)")
        if save_data:
            # make a directory for today's date and create a unique timestamp

            data_file = os.path.join(data_path, str(stamp) + "_caps_cal.nc")
            fig_file = os.path.join(data_path, str(stamp) + "_caps_cal.png")
            cfg_file = os.path.join(data_path, str(stamp) + "_caps_cal.yaml")

            # save the config and the plot
            file_manager.save_config(self.config, cfg_file)
            plt.savefig(fig_file)
            nc_file = netCDF4.Dataset(data_file, "a", format="NETCDF4")

            # create dimensions for all data
            nc_file.createDimension("vm", n_vm)
            nc_file.createDimension("vg", n_vg)

            # create variables and fill in data
            crosscaps = nc_file.createVariable(
                "fulldata",
                np.float32,
                (
                    "vm",
                    "vg",
                ),
            )
            crosscaps.units = "V"
            gx = nc_file.createVariable("Vg", np.float32, ("vg"))
            gx.units = "Volts"
            gx[:] = vg_sweep
            gy = nc_file.createVariable("Vm", np.float32, ("vm"))
            gy.units = "Volts"
            gy[:] = vm_sweep
            crosscaps[:, :] = data
            nc_file.close()
            logger.info("data saved at %s" % data_file)
        return data, mdata, slope

    @dot_experiment.updater
    def sweep_adc_trig_offset(
        self, times: Tuple[float, float, int] = (0, 1, 10), avgs: int = 10, plot=True
    ):
        """sweep the adc trigger offset parameter, which sets the offset between when a pulse is fired and when readout is turned on.

        :param times:
        :param avgs: Averages per time point
        """
        self.config.reps = avgs
        t_array = np.linspace(*times)
        amplitudes = np.zeros((len(t_array)))
        for i, t in enumerate(t_array):
            self.config.dcs_cfg.adc_trig_offset = self.soccfg.us2cycles(t)
            prog = system_calibrations_programs.PulseAndMeasure(
                self.soccfg, self.config
            )
            data = prog.acquire(self.soc, load_pulses=True)
            mag = np.sqrt(data[0][0] ** 2 + data[1][0] ** 2)
            amplitudes[i] = mag
        if plot:
            plt.figure()
            plt.plot(t_array, amplitudes, ".-")
            plt.ylabel("ADC units")
            plt.xlabel("times (us)")
        return t_array, amplitudes
