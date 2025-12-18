"""Perform charge stability measurements.

Converted to tprocv2 compatibility.
"""

import logging
import time
from typing import List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pydantic
from lmfit import models
from qick.asm_v2 import QickProgramV2

from spinqick.core import dot_experiment, spinqick_data
from spinqick.helper_functions import analysis, hardware_manager, plot_tools, spinqick_enums
from spinqick.models import experiment_models, hardware_config_models
from spinqick.qick_code_v2 import system_calibrations_programs_v2, tune_electrostatics_programs_v2

logger = logging.getLogger(__name__)


def plot_gvg_2d(
    data: spinqick_data.SpinqickData,
    x_gate: str,
    y_gate: str,
    x_label: str,
    y_label: str,
    adc_units: str | List[str] = "arbs",
):
    """Plotting function for 2D electrostatics type sweeps."""
    if data.analyzed_data is not None:
        fignums = []
        if isinstance(adc_units, str):
            units = [adc_units for i in data.analyzed_data]
        else:
            units = adc_units
        for i, adc_data in enumerate(data.analyzed_data):
            if data.analysis_type == "sd_chop":
                mag = adc_data
            else:
                mag = np.real(adc_data)

            if mag.shape[0] == 1:
                plot_data = mag[0]
            else:
                plot_data = np.mean(mag, axis=0)
            plot_mag = np.transpose(plot_data)
            x_data = data.axes["x"]["sweeps"][x_gate]["data"]
            y_data = data.axes["y"]["sweeps"][y_gate]["data"]
            fig = plot_tools.plot2_simple(
                xarray=x_data * 1000,
                yarray=y_data * 1000,
                data=plot_mag,
                timestamp=data.timestamp,
                cbar_label=data.analysis_type + "_" + units[i],
            )
            plt.title("%s v %s, adc %d" % (x_gate, y_gate, i))
            plt.xlabel("%s (mV)" % x_label)
            plt.ylabel("%s (mV)" % y_label)
            fignums.append(fig.number)
    else:
        raise Exception("no analyzed data exists on spinqickdata object")
    return fignums


def plot_g_1d(
    data: spinqick_data.SpinqickData,
    gate: str,
    x_label: str,
    y_label: str,
    dset_label: str | None = None,
):
    """Generate a plot of a 1D sweep of a gate voltage."""
    if data.analyzed_data is not None:
        fignums = []
        for i, adc_data in enumerate(data.analyzed_data):
            x_data = data.axes["x"]["sweeps"][gate]["data"]
            y_data = adc_data
            fig = plot_tools.plot1_simple(x_data, y_data[0], data.timestamp, dset_label=dset_label)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(" adc %d" % i)
            fignums.append(fig.number)
    return fignums


def add_g_1d(
    data: spinqick_data.SpinqickData,
    gate: str,
    x_label: str,
    y_label: str,
    fignums: list,
    dset_label: str | None = None,
):
    """Add a 1d trace to a plot generated with plot_g_1d."""
    if data.analyzed_data is not None:
        for i, adc_data in enumerate(data.analyzed_data):
            plt.figure(fignums[i])
            x_data = data.axes["x"]["sweeps"][gate]["data"]
            y_data = adc_data
            plot_tools.plot1_simple(
                x_data,
                y_data[0],
                data.timestamp,
                dset_label=dset_label,
                new_figure=False,
            )
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(" adc %d" % i)


def analyze_cross_caps(
    data_obj: spinqick_data.SpinqickData,
    adc: int = 0,
    fit_type: Literal["gaussian", "abs_max", "abs_min"] = "gaussian",
):
    """Analysis routine for the cross capacitance experiment."""
    slow_gate_dict = list(data_obj.axes["x"]["sweeps"])[0]
    fast_gate_dict = list(data_obj.axes["y"]["sweeps"])[0]
    n_vx = data_obj.axes["x"]["size"]
    center_data = np.zeros(n_vx)
    # fit each slice (fast sweep) to a gaussian
    assert data_obj.analyzed_data is not None
    for x_pt in range(n_vx):
        xdata = data_obj.axes["y"]["sweeps"][fast_gate_dict]["data"]
        ydata = data_obj.analyzed_data[adc][0][x_pt, :]
        if fit_type == "gaussian":
            try:
                _, out = analysis.fit_gaussian(xdata, ydata)
                if np.logical_or(
                    out.params["center"].value > xdata[0],
                    out.params["center"].value < xdata[-1],
                ):
                    center_data[x_pt] = out.params["center"].value
                else:
                    center_data[x_pt] = np.nan
            except Exception as exc:
                logger.error("fit failed: %s", exc, exc_info=True)
        elif fit_type == "abs_max":
            center_data[x_pt] = xdata[np.argmax(ydata)]
        elif fit_type == "abs_min":
            center_data[x_pt] = xdata[np.argmin(ydata)]
    line = models.LinearModel()
    pars = line.guess(center_data, x=data_obj.axes["x"]["sweeps"][slow_gate_dict]["data"])
    try:
        out = line.fit(center_data, pars, x=data_obj.axes["x"]["sweeps"][slow_gate_dict]["data"])
        slope = out.params["slope"].value
        logger.info("slope is %f", slope)
    except Exception as exc:
        slope = np.nan
        logger.error("line fit failed, %s", exc, exc_info=True)
    fit_params = {
        "slope": slope,
        "intercept": out.params["intercept"].value,
    }
    data_obj.add_fit_params(fit_params, center_data, "x")


class TuneElectrostatics(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK experiments for charge stability measurements.
    In general each function sets the necessary config parameters for you and then runs the qick
    program. They then optionally plot and save the data.Initialize with information about your
    rfsoc and your experimental setup.

    :param soccfg: QickConfig object
    :param soc: QickSoc object
    :param voltage_source: Initialized DC voltage source object. This is used here for saving the DC
        voltage state each time data is saved.
    """

    def __init__(self, soccfg, soc, voltage_source: hardware_manager.VoltageSource, **kwargs):
        super().__init__(**kwargs)
        self.soccfg = soccfg
        self.soc = soc
        self.vdc = hardware_manager.DCSource(voltage_source=voltage_source)

    @dot_experiment.updater
    def gvg_baseband(
        self,
        g_gates: tuple[spinqick_enums.GateNames, spinqick_enums.GateNames],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]],
        measure_buffer: float,
    ) -> spinqick_data.SpinqickData:
        """Perform a basic PvP or PvT by baseband pulsing with the RFSoC.

        :param g_gates: gates to sweep on y and x axes. i.e. ('P1','P2')
        :param g_range: range to sweep gate x and gate y, and number of points for each
        :param measure_buffer: sets the delay time between measurements. The actual measurement is
            sandwiched between two measure_buffer delays
        :return: SpinqickData object
        """

        gx_gate, gy_gate = g_gates
        gx_start, gx_stop, gx_expts = g_range[0]
        gy_start, gy_stop, gy_expts = g_range[1]

        gx_cfg = self.hardware_config.channels[gx_gate]
        gy_cfg = self.hardware_config.channels[gy_gate]
        assert isinstance(gx_cfg, hardware_config_models.FastGate)
        assert isinstance(gy_cfg, hardware_config_models.FastGate)
        gx_gen = gx_cfg.qick_gen
        gy_gen = gy_cfg.qick_gen

        # create a pydantic model and fill it in
        gvg_cfg = experiment_models.GvgBasebandConfig(
            measure_buffer=measure_buffer,
            gx_gen=gx_gen,
            gy_gen=gy_gen,
            gx_gate=gx_gate,
            gy_gate=gy_gate,
            gx_start=self.volts2dac(gx_start, gx_gate),
            gx_stop=self.volts2dac(gx_stop, gx_gate),
            gx_expts=gx_expts,
            gy_start=self.volts2dac(gy_start, gy_gate),
            gy_stop=self.volts2dac(gy_stop, gy_gate),
            gy_expts=gy_expts,
            dcs_cfg=self.dcs_config,
        )

        meas = tune_electrostatics_programs_v2.BasebandPulseGvG(
            self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()

        x_pts = np.linspace(gx_start, gx_stop, gx_expts)
        y_pts = np.linspace(gy_start, gy_stop, gy_expts)
        assert data is not None
        data_obj = spinqick_data.SpinqickData(
            data,
            gvg_cfg,
            1,
            1,
            "_gvg_baseband",
            voltage_state=self.vdc.all_voltages,
            prog=meas,
        )
        data_obj.add_axis([x_pts], "x", [gx_gate], gx_expts, loop_no=0, units=["V"])
        data_obj.add_axis([y_pts], "y", [gy_gate], gy_expts, loop_no=1, units=["V"])
        analysis.calculate_conductance(data_obj, self.adc_unit_conversions)

        # plot the data
        if self.plot:
            plot_gvg_2d(
                data_obj,
                gx_gate,
                gy_gate,
                gx_gate,
                gy_gate,
                adc_units=self.adc_units[0],
            )

        if self.save_data:
            save_obj = data_obj.save_data()
            if self.plot:
                save_obj.save_last_plot()
            save_obj.close()

        return data_obj

    @dot_experiment.updater
    def gvg_quack(
        self,
        g_gates: tuple[list[spinqick_enums.GateNames], list[spinqick_enums.GateNames]],
        g_vals: tuple[tuple[list[float], list[float], int], tuple[list[float], list[float], int]],
        measure_buffer: float,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
    ) -> spinqick_data.SpinqickData:
        """Perform a basic PvP or PvT by baseband pulsing with the RFSoC.

        :param g_gates: list of gates to sweep on x and y axes. i.e. (['P1'], ['P2', 'P3'])
        :param g_vals: voltage values to sweep x and y, and step size for each. THIS IS DIFFERENT
            THAN TYPICAL SPINQICK RANGE FORMAT. Its not relative to current setpoint.
        :param measure_buffer: sets the delay time between measurements. The actual measurement is
            sandwiched between two measure_buffer delays
        :return: SpinqickData object
        """

        gx_gates, gy_gates = g_gates
        gx_starts, gx_steps, gx_expts = g_vals[0]
        gy_starts, gy_steps, gy_expts = g_vals[1]

        gx_cfgs = [self.hardware_config.channels[gx_gate] for gx_gate in gx_gates]
        gy_cfgs = [self.hardware_config.channels[gy_gate] for gy_gate in gy_gates]
        x_conversions = []
        x_gens = []
        for gx_cfg in gx_cfgs:
            assert isinstance(
                gx_cfg,
                Union[hardware_config_models.FastGate, hardware_config_models.SlowGate],
            )
            gx_convert = gx_cfg.dc_conversion_factor
            gx_gen = gx_cfg.slow_dac_channel
            x_conversions.append(gx_convert)
            x_gens.append(gx_gen)

        y_conversions = []
        y_gens = []
        for gy_cfg in gy_cfgs:
            assert isinstance(
                gy_cfg,
                Union[hardware_config_models.FastGate, hardware_config_models.SlowGate],
            )
            gy_convert = gy_cfg.dc_conversion_factor
            gy_gen = gy_cfg.slow_dac_channel
            y_conversions.append(gy_convert)
            y_gens.append(gy_gen)

        # create a pydantic model and fill it in
        gvg_cfg = experiment_models.Quack2DConfig(
            measure_buffer=measure_buffer,
            x_gens=x_gens,
            y_gens=y_gens,
            x_starts=[gx_starts[i] / x_conversions[i] for i in range(len(x_gens))],
            x_steps=[gx_steps[i] / x_conversions[i] for i in range(len(x_gens))],
            x_points=gx_expts,
            y_starts=[gy_starts[i] / y_conversions[i] for i in range(len(y_gens))],
            y_steps=[gy_steps[i] / y_conversions[i] for i in range(len(y_gens))],
            y_points=gy_expts,
            dcs_cfg=self.dcs_config,
            mode=mode,
        )
        pin_list = self.soccfg["tprocs"][0]["output_pins"]
        if len(pin_list) < 8:
            pin_list.append(("trig", 7, 0, "trigger_pmod"))
        self.soc.clear_dac_regs()
        self.soc.setup_quack_sweep(
            loop_0_cfg={
                "start": gvg_cfg.y_starts,
                "step": gvg_cfg.y_steps,
                "channels": gvg_cfg.y_gens,
                "loop_size": gvg_cfg.y_points,
            },
            loop_1_cfg={
                "start": gvg_cfg.x_starts,
                "step": gvg_cfg.x_steps,
                "channels": gvg_cfg.x_gens,
                "loop_size": gvg_cfg.x_points,
            },
            dwell_cycles=100
            * (
                gvg_cfg.dcs_cfg.readout_length + gvg_cfg.measure_buffer
            ),  # 100MHz clock, this defines time after trigger plays
        )

        self.soc.set_trigger_source("qick")
        meas = tune_electrostatics_programs_v2.Quack_2D(
            self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        self.soc.quit_pvp()  # make sure it actually quits

        x_pts = [
            np.arange(gx_starts[i], gx_steps[i] * gx_expts, gx_steps[i])
            for i in range(len(gx_gates))
        ]
        y_pts = [
            np.arange(gy_starts[i], gy_steps[i] * gy_expts, gy_steps[i])
            for i in range(len(gy_gates))
        ]
        assert data is not None
        data_obj = spinqick_data.SpinqickData(
            data,
            gvg_cfg,
            1,
            1,
            "_gvg_quack",
            voltage_state=self.vdc.all_voltages,
            prog=meas,
        )
        data_obj.add_axis(
            [x_pts[i] for i in range(len(gx_gates))],
            "x",
            gx_gates,
            gx_expts,
            loop_no=0,
            units=["V" for i in range(len(gx_gates))],
        )
        data_obj.add_axis(
            [y_pts[i] for i in range(len(gy_gates))],
            "y",
            gy_gates,
            gy_expts,
            loop_no=1,
            units=["V" for i in range(len(gy_gates))],
        )

        analysis.calculate_conductance(data_obj, self.adc_unit_conversions)

        # plot the data
        if self.plot:
            x_label = ""
            for gate in gx_gates:
                x_label = x_label + " " + gate + ","
            y_label = ""
            for gate in gy_gates:
                y_label = y_label + " " + gate + ","
            plot_gvg_2d(
                data_obj,
                gx_gates[
                    0
                ],  # TODO figure out a better way to plot axes so its accurate for all gates
                gy_gates[0],
                x_label,
                y_label,
                adc_units=self.adc_units,
            )

        if self.save_data:
            save_obj = data_obj.save_data()
            if self.plot:
                save_obj.save_last_plot()
            save_obj.close()

        return data_obj

    def gvg_arb_prog(
        self,
        prog: QickProgramV2,
        expt_name: str,
        cfg: pydantic.BaseModel,
        g_gates: tuple[list[spinqick_enums.GateNames], list[spinqick_enums.GateNames]],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]],
        measure_buffer: float,
        compensate: spinqick_enums.GateNames | None = None,
        sweep_direction: tuple[list[int], list[int]] | None = None,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
        save_data: bool = False,
    ) -> spinqick_data.SpinqickData:
        """Python outer loop for 2 dimensional gate sweep-type experiments which sweep DC voltages.
        This method runs the 2D sweep and returns a spinqick data object.

        :param prog: QickProgram to run at each sweep
        :param expt_name: experiment name string to pass to SpinqickData object
        :param cfg: pydantic config model to pass in to the QickProgram when it is run
        :param g_gates: specify the gates to sweep on the x and y axes. The fast sweep runs along
            the y axis, which is the second list in the tuple. An outer python loop steps through
            and sets the x-axis values one by one.
        :param g_range: voltage range to sweep gate x and gate y, and number of points for each in
            this format ((x start, x end, number of points), (x start, x end, number of points) )
        :param measure_buffer: time in microseconds at each voltage point before and after the
            readout pulse.
        :param compensate: whether to apply compensation on another gate.  Typically an M gate.
        :param sweep_direction: specify direction of sweep. This arguement takes a tuple of lists
            the same size as g_gates. The values of the inputs must be +/-1, with +1 corresponding
            to sweeping from start to end voltage values, and -1 sweeping from end voltage to start
            voltage.
        :param mode: toggles the readout mode between source-drain chop and transconductance. In
            transconductance mode, an AC signal is applied to a gate and a DC bias is applied to the
            device, while current is read out the normal way.
        """

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
                if np.sign(sweep_direction[0][i]) == -1:
                    vx_start = np.abs(sweep_direction[0][i]) * gx_stop + vx_0[i]
                    vx_stop = np.abs(sweep_direction[0][i]) * gx_start + vx_0[i]
                else:
                    vx_start = np.abs(sweep_direction[0][i]) * gx_start + vx_0[i]
                    vx_stop = np.abs(sweep_direction[0][i]) * gx_stop + vx_0[i]
            else:
                vx_start = gx_start + vx_0[i]
                vx_stop = gx_stop + vx_0[i]
            # self.vdc.set_dc_voltage(vx_start, gx)
            vx_sweep[i, :] = np.linspace(vx_start, vx_stop, n_vx)
        if compensate:
            vm_0 = self.vdc.get_dc_voltage(compensate)
            _, delta_vm, _ = self.vdc.calculate_compensated_voltage(
                [gx_stop - gx_start for i in range(len(gx_gates))],
                gx_gates,
                [compensate],
            )
            _, delta_vm_start, _ = self.vdc.calculate_compensated_voltage(
                [gx_start for i in range(len(gx_gates))] + [gy_start for i in range(len(gy_gates))],
                gx_gates + gy_gates,
                [compensate],
            )
            vm_start = vm_0 + delta_vm_start[-1]
            vm_sweep = np.linspace(vm_start, vm_start + delta_vm[-1], n_vx)
            # self.vdc.set_dc_voltage(vm_start, compensate)
        # Vy Sweep
        vy_0 = []
        vy_sweep = np.zeros((len(gy_gates), n_vy))
        for i, gy in enumerate(gy_gates):
            vy_0.append(self.vdc.get_dc_voltage(gy))
            if sweep_direction is not None:
                if np.sign(sweep_direction[1][i]) == -1:
                    vy_start = gy_stop + vy_0[i]
                    vy_stop = gy_start + vy_0[i]
                else:
                    vy_start = gy_start + vy_0[i]
                    vy_stop = gy_stop + vy_0[i]
            else:
                vy_start = gy_start + vy_0[i]
                vy_stop = gy_stop + vy_0[i]
            vy_sweep[i, :] = np.linspace(vy_start, vy_stop, n_vy)
            # self.vdc.set_dc_voltage(vy_start, gy)

        # setup the slow_dac step length
        step_length = self.dcs_config.length + 2 * measure_buffer
        if step_length < self.hardware_config.dac_settings.t_min_slow_dac:
            slow_dac_step_len = self.hardware_config.dac_settings.t_min_slow_dac
        else:
            slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer

        raw_list = []
        for i in range(len(self.dcs_config.ro_chs)):
            raw = np.zeros((1, n_vx, n_vy, 2))
            raw_list.append(raw)
        for vx_index in range(n_vx):
            for i, gx_gate in enumerate(gx_gates):
                self.vdc.set_dc_voltage(vx_sweep[i, vx_index], gx_gate)
            for i, gy_gate in enumerate(gy_gates):
                if gy_gate in gx_gates:
                    v_simultaenous = vx_sweep[gx_gates.index(gy_gate), vx_index]
                    v_simultaenous -= vx_0[gx_gates.index(gy_gate)]
                else:
                    v_simultaenous = 0
                self.vdc.set_dc_voltage(vy_sweep[i, 0] + v_simultaenous, gy_gate)
            if compensate:
                self.vdc.set_dc_voltage(vm_sweep[vx_index], compensate)
                self.vdc.program_ramp_compensate(
                    vy_sweep[0, 0],
                    vy_sweep[0, -1],
                    slow_dac_step_len * 1e-6,
                    n_vy,
                    gy_gates,
                    compensate,
                )
                self.vdc.arm_sweep(compensate)
            else:
                for m, gy_gate in enumerate(gy_gates):
                    if gy_gate in gx_gates:
                        v_simultaenous = vx_sweep[gx_gates.index(gy_gate), vx_index]
                        v_simultaenous -= vx_0[gx_gates.index(gy_gate)]
                    else:
                        v_simultaenous = 0
                    self.vdc.program_ramp(
                        vy_sweep[m, 0] + v_simultaenous,
                        vy_sweep[m, -1] + v_simultaenous,
                        slow_dac_step_len * 1e-6,
                        n_vy,
                        gy_gate,
                    )
            for gy_gate in gy_gates:
                self.vdc.arm_sweep(gy_gate)
            ### Start a Vy sweep and store the data
            data = prog.acquire(self.soc, progress=False)
            for i, adc_data in enumerate(data):
                raw_list[i][0, vx_index, :, :] = adc_data
            time.sleep(slow_dac_step_len * 1e-6 * n_vy)
            ### ramp Vy back to start
            if compensate:
                # TODO fix the program_ramp_compensate so it takes an array of start and stop
                self.vdc.program_ramp_compensate(
                    vy_sweep[0, -1],
                    vy_sweep[0, 0],
                    slow_dac_step_len * 1e-6,
                    n_vy,
                    gy_gates,
                    compensate,
                )
                self.vdc.arm_sweep(compensate)
                self.vdc.digital_trigger(compensate)
            else:
                for m, gy_gate in enumerate(gy_gates):
                    self.vdc.program_ramp(
                        vy_sweep[m, -1],
                        vy_sweep[m, 0],
                        slow_dac_step_len * 1e-6,
                        n_vy,
                        gy_gate,
                    )
            for gy_gate in gy_gates:
                self.vdc.arm_sweep(gy_gate)
                self.vdc.digital_trigger(gy_gate)
                time.sleep(slow_dac_step_len * 1e-6 * n_vy)  # leave some time to ramp down

        for k, gx_gate in enumerate(gx_gates):
            v_final = self.vdc.get_dc_voltage(gx_gate)
            self.vdc.program_ramp(
                v_final,
                vx_0[k],
                self.hardware_config.dac_settings.t_min_slow_dac * 1e-6,
                n_vx,
                gx_gate,
            )
            self.vdc.arm_sweep(gx_gate)
            self.vdc.digital_trigger(gx_gate)
            time.sleep(
                self.hardware_config.dac_settings.t_min_slow_dac * n_vx * 1e-6
            )  # leave some time to ramp down
        for k, gy_gate in enumerate(gy_gates):
            v_final = self.vdc.get_dc_voltage(gy_gate)
            self.vdc.program_ramp(
                v_final,
                vy_0[k],
                self.hardware_config.dac_settings.t_min_slow_dac * 1e-6,
                n_vy,
                gy_gate,
            )
            self.vdc.arm_sweep(gy_gate)
            self.vdc.digital_trigger(gy_gate)
            time.sleep(slow_dac_step_len * 1e-6 * n_vy)  # leave some time to ramp down
        if isinstance(compensate, str):
            vm = self.vdc.get_dc_voltage(compensate)
            self.vdc.program_ramp(
                vm, vm_0, self.hardware_config.dac_settings.t_min_slow_dac * 1e-6, n_vy, compensate
            )
            self.vdc.arm_sweep(compensate)
            self.vdc.digital_trigger(compensate)
            time.sleep(slow_dac_step_len * 1e-6 * n_vy)  # leave some time to ramp down
        data_obj = spinqick_data.SpinqickData(
            raw_list,
            cfg,
            1,
            1,
            expt_name,
            voltage_state=self.vdc.all_voltages,
            prog=prog,
        )
        data_obj.add_axis(
            [vx_sweep[i, :] for i in range(len(gx_gates))],
            "x",
            gx_gates,
            n_vx,
            loop_no=0,
            units=["V" for i in range(len(gx_gates))],
        )
        data_obj.add_axis(
            [vy_sweep[i, :] for i in range(len(gy_gates))],
            "y",
            gy_gates,
            n_vy,
            loop_no=1,
            units=["V" for i in range(len(gy_gates))],
        )

        if mode == "sd_chop":
            analysis.calculate_conductance(data_obj, self.adc_unit_conversions)
        else:
            analysis.calculate_transconductance(
                data_obj,
                self.adc_unit_conversions,
            )
        # plot the data
        if self.plot:
            x_label = ""
            for gate in gx_gates:
                x_label = x_label + " " + gate + ","
            y_label = ""
            for gate in gy_gates:
                y_label = y_label + " " + gate + ","
            plot_gvg_2d(
                data_obj,
                gx_gates[0],
                gy_gates[0],
                x_label,
                y_label,
                adc_units=self.adc_units,
            )

        if save_data:
            ncdf = data_obj.save_data()
            if self.plot:
                ncdf.save_last_plot()
            ncdf.close()
            logger.info("data saved at %s", data_obj.data_file)

        return data_obj

    @dot_experiment.updater
    def gvg_dc(
        self,
        g_gates: tuple[list[spinqick_enums.GateNames], list[spinqick_enums.GateNames]],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]],
        measure_buffer: float,
        compensate: spinqick_enums.GateNames | None = None,
        sweep_direction: tuple[list[int], list[int]] | None = None,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
    ) -> spinqick_data.SpinqickData:
        """GvG script which sweeps an external DC voltage source and reads out a DCS.

        :param g_gates: gates to sweep on y and x axes. i.e. (['P1'],['P2']). Option to provide a
            list of gates to sweep on each axis.
        :param g_range: voltage range to sweep gate x and gate y, and number of points for each.
        :param compensate: gate to compensate while changing other voltages. i.e. 'M1'
        :param sweep_direction: Allows user to sweep a gate backwards if desired. Provide a list of
            positive or negative ones corresponding to each gate in g_gates i.e. ([1,-1], [1]).
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and
            the QICK starts a DCS measurement.
        :param mode: "sdchop" selects typical source drain chop readout, "transdc" is for
            transcoductance mode
        """
        _, g_range_y = g_range
        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=self.hardware_config.dac_settings.trig_pin,
            measure_buffer=measure_buffer,
            points=g_range_y[2],
            dcs_cfg=self.dcs_config,
            trig_length=self.hardware_config.dac_settings.trig_length,
            mode=mode,
        )

        meas = tune_electrostatics_programs_v2.GvG(self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg)

        expt_name = "_gvg_dc"
        data_obj = self.gvg_arb_prog(
            meas,
            expt_name,
            gvg_cfg,
            g_gates,
            g_range,
            measure_buffer,
            compensate=compensate,
            sweep_direction=sweep_direction,
            mode=mode,
            save_data=self.save_data,
        )

        return data_obj

    @dot_experiment.updater
    def gvg_pat(
        self,
        g_gates: tuple[list[spinqick_enums.GateNames], list[spinqick_enums.GateNames]],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]],
        measure_buffer: float,
        pat_freq: float,
        pat_gain: float,
        pat_gen: int,
        compensate: spinqick_enums.GateNames | None = None,
        sweep_direction: tuple[list[int], list[int]] | None = None,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
    ) -> spinqick_data.SpinqickData:
        """GvG script which applies an RF tone to probe photon assisted tunneling.

        :param g_gates: gates to sweep on y and x axes. i.e. (['P1'],['P2']). Option to provide a
            list of gates to sweep on each axis.
        :param g_range: voltage range to sweep gate x and gate y, and number of points for each.
        :param compensate: gate to compensate while changing other voltages. i.e. 'M1'
        :param sweep_direction: Allows user to sweep a gate backwards if desired. Provide a list of
            positive or negative ones corresponding to each gate in g_gates i.e. ([1,-1], [1]).
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and
            the QICK starts a DCS measurement.
        :param pat_freq: frequency of microwave tone
        :param pat_gain: gain of pat signal in QICK units (-1, 1)
        :param pat_gen: generator used for pat signal
        :param mode: "sdchop" selects typical source drain chop readout, "transdc" is for
            transcoductance mode
        """
        _, gy_range = g_range

        pat_cfg = experiment_models.PatConfig(
            pat_freq=pat_freq,
            pat_gain=pat_gain,
            pat_gen=pat_gen,
        )

        gvg_cfg = experiment_models.GvgPatConfig(
            trig_pin=self.hardware_config.dac_settings.trig_pin,
            measure_buffer=measure_buffer,
            points=gy_range[2],
            dcs_cfg=self.dcs_config,
            trig_length=self.hardware_config.dac_settings.trig_length,
            pat_cfg=pat_cfg,
        )

        meas = tune_electrostatics_programs_v2.GvGPat(
            self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
        )

        expt_name = "_gvg_pat"
        data_obj = self.gvg_arb_prog(
            meas,
            expt_name,
            gvg_cfg,
            g_gates,
            g_range,
            measure_buffer,
            compensate=compensate,
            sweep_direction=sweep_direction,
            mode=mode,
            save_data=self.save_data,
        )

        return data_obj

    @dot_experiment.updater
    def get_cross_caps(
        self,
        x_gate: spinqick_enums.GateNames,
        y_gate: spinqick_enums.GateNames,
        x_range: Tuple[float, float, int],
        y_range: Tuple[float, float, int],
        measure_buffer: float,
        compensate: spinqick_enums.GateNames | None = None,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
        fit_type: Literal["gaussian", "abs_max", "abs_min"] = "gaussian",
    ) -> spinqick_data.SpinqickData:
        """Measure cross-capacitance between gates.  This is set to fit a gaussian to a feature as
        it is scanned on the y-axis.

        :param x_gate: gate to step in python outer loop
        :param y_gate: either an M gate, or the gate that compensation will be applied to
        :param x_range: voltage range to sweep x gate over
        :param y_range: voltage range to sweep y (fast sweep) gate over
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and
            the QICK starts a DCS measurement.
        :param compensate: the m-gate to compensate with for the y-gate. If y-gate is an m-gate,
            leave as None
        :param mode: "sdchop" selects typical source drain chop readout, "transdc" is for
            transcoductance mode
        :param fit_type:
        """
        _, _, n_vy = y_range

        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=self.hardware_config.dac_settings.trig_pin,
            measure_buffer=measure_buffer,
            points=n_vy,
            dcs_cfg=self.dcs_config,
            trig_length=self.hardware_config.dac_settings.trig_length,
            mode=mode,
        )

        meas = tune_electrostatics_programs_v2.GvG(self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg)

        expt_name = "_cross_caps"
        data_obj = self.gvg_arb_prog(
            meas,
            expt_name,
            gvg_cfg,
            ([x_gate], [y_gate]),
            (x_range, y_range),
            measure_buffer,
            compensate=compensate,
            save_data=False,
            mode=mode,
        )
        analyze_cross_caps(data_obj, fit_type=fit_type)
        if self.plot:
            plt.plot(
                data_obj.axes["x"]["sweeps"][x_gate]["data"] * 1000,
                data_obj.best_fit * 1000,
                "--",
            )
            text_xcoord = data_obj.axes["x"]["sweeps"][x_gate]["data"][0] * 1000
            y_max = data_obj.axes["y"]["sweeps"][y_gate]["data"][-1] * 1000
            text_ycoord = y_max - (y_max - data_obj.best_fit[0] * 1000) / 2
            plt.text(
                text_xcoord,
                text_ycoord,
                "slope = %.6f" % data_obj.fit_param_dict["slope"],
                fontdict={"color": "red"},
            )
            line_fit = (
                data_obj.fit_param_dict["slope"]
                * data_obj.axes["x"]["sweeps"][x_gate]["data"]
                * 1000
                + data_obj.fit_param_dict["intercept"] * 1000
            )
            plt.plot(data_obj.axes["x"]["sweeps"][x_gate]["data"] * 1000, line_fit)
        if self.save_data:
            ncdf = data_obj.save_data()
            if self.plot:
                ncdf.save_last_plot()
            ncdf.close()
            logger.info("data saved at %s", data_obj.data_file)
        return data_obj

    def tune_mz(
        self,
        m_dot: spinqick_enums.GateNames,
        m_range: tuple[float, float, int],
        z_range: tuple[float, float, int],
        measure_buffer: float,
        tune_type: Literal["common", "diff"] = "common",
    ) -> spinqick_data.SpinqickData:
        """Tune MZ using sleeperdac or other external DC voltage bias. This wraps the function
        gvg_dc.

        :param m_dot: M dot you're tuning, i.e. 'M1'
        :param m_range: voltage range to sweep m gate, and number of points.
        :param z_range: voltage range to sweep z gate, and number of points.
        :param tune_type: sweep z gates in same direction or opposite directions
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and
            the QICK starts a DCS measurement.
        """

        if m_dot == "M1":
            z_dots = [spinqick_enums.GateNames.Z1, spinqick_enums.GateNames.Z2]
        else:
            z_dots = [spinqick_enums.GateNames.Z3, spinqick_enums.GateNames.Z4]

        if tune_type == "common":
            z_sweep_vector = [1, 1]
        else:
            z_sweep_vector = [1, -1]

        result = self.gvg_dc(
            g_gates=([m_dot], z_dots),
            g_range=(m_range, z_range),
            compensate=None,
            measure_buffer=measure_buffer,
            mode="sd_chop",
            sweep_direction=([1], z_sweep_vector),
        )
        return result

    @dot_experiment.updater
    def retune_dcs(
        self,
        m_dot: spinqick_enums.GateNames,
        m_range: tuple[float, float, int],
        measure_buffer: float,
        set_v=False,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
    ) -> spinqick_data.SpinqickData:
        """Automated retune DCS script. Scan m-gate voltage, fit to conductance peak, extract an
        optimal m-gate voltage from peak.

        :param m_dot: Dot to retune
        :param m_range: List of start voltage, stop voltage and number of points. Voltages are
            relative to the current setpoint
        :param measure_buffer: time in microseconds between when the slow dac steps in voltage and
            the QICK starts a DCS measurement.
        :param set_v: after retuning, automatically set the voltage to the optimal point
        :param mode: "sdchop" selects typical source drain chop readout, "transdc" is for
            transcoductance mode
        """

        # M Sweep
        m_bias = self.vdc.get_dc_voltage(m_dot)
        n_vm = int(m_range[2])
        vm_start = m_range[0] + m_bias
        vm_stop = m_range[1] + m_bias
        vm_sweep = np.linspace(vm_start, vm_stop, n_vm)

        slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer

        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=self.hardware_config.dac_settings.trig_pin,
            trig_length=self.hardware_config.dac_settings.trig_length,
            measure_buffer=measure_buffer,
            points=n_vm,
            dcs_cfg=self.dcs_config,
            mode=mode,
        )

        self.vdc.program_ramp(vm_start, vm_stop, slow_dac_step_len * 1e-6, n_vm, m_dot)
        self.vdc.arm_sweep(m_dot)

        # Start a Vy sweep at a Vx increment and store the data
        meas = tune_electrostatics_programs_v2.GvG(self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg)
        data = meas.acquire(self.soc, progress=False)
        assert data
        data_obj = spinqick_data.SpinqickData(
            data,
            gvg_cfg,
            1,
            1,
            "_retune_m",
            voltage_state=self.vdc.all_voltages,
            prog=meas,
        )
        data_obj.add_axis([vm_sweep], "x", [m_dot], n_vm, units=["V"])
        if mode == "sd_chop":
            analysis.calculate_conductance(data_obj, self.adc_unit_conversions)
        else:
            analysis.calculate_transconductance(data_obj, self.adc_unit_conversions)

        # Ramp Vy voltage back down to the starting value
        return_step_time = self.hardware_config.dac_settings.t_min_slow_dac
        self.vdc.program_ramp(vm_stop, m_bias, return_step_time * 1e-6, n_vm, m_dot)
        self.vdc.digital_trigger(m_dot)
        time.sleep(return_step_time * 1e-6 * n_vm)

        try:
            x_data = data_obj.axes["x"]["sweeps"][m_dot]["data"]
            y_data = data_obj.analyzed_data
            assert y_data is not None
            gaussian, out = analysis.fit_gaussian(x_data, y_data[0][0])
            center = out.params["center"].value
            fwhm = out.params["fwhm"].value
            halfmax = center - fwhm / 2
            if mode == "sd_chop":
                best_v = halfmax
            else:
                best_v = center
            print("best voltage = %f" % best_v)
            # first make sure the voltage it picks is within limits of sweep for safety
            if set_v:
                if np.logical_and(best_v < vm_stop, best_v > vm_start):
                    self.vdc.set_dc_voltage(best_v, m_dot)
                else:
                    self.vdc.set_dc_voltage(m_bias, m_dot)
            fit_dict = {
                "center": center,
                "fwhm": fwhm,
                "best_voltage": halfmax,
                "amplitude": out.params["amplitude"].value,
                "c": out.params["c"].value,
                "sigma": out.params["sigma"].value,
            }
            data_obj.add_fit_params(fit_dict, out.best_fit, "x")
        except Exception as exc:
            if set_v:
                self.vdc.set_dc_voltage(m_bias, m_dot)
            fwhm = 0
            center = 0
            halfmax = 0
            logger.error("fit failed, setting m to initial value, error %s", exc, exc_info=True)

        if self.plot:
            # plot the data
            adc_units = self.adc_units[0]
            plot_g_1d(data_obj, m_dot, "M voltage (V)", "%s (%s)" % (mode, adc_units))
            plt.plot(vm_sweep, out.best_fit)
            plt.plot([best_v], [gaussian.eval(out.params, x=best_v)], "o")  # type: ignore
            plt.title("retune dcs")

        if self.save_data:
            nc_file = data_obj.save_data()
            if self.plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s", data_obj.data_file)

        return data_obj

    @dot_experiment.updater
    def gate_action(
        self,
        gates: list[spinqick_enums.GateNames],
        max_v: float,
        num_points: int,
        measure_buffer: float,
    ) -> spinqick_data.CompositeSpinqickData:
        """Performs a voltage sweep of a list of gates down to zero and back to a max voltage.  Each
        gate is swept individually.  This function holds other gates at the voltages they were set
        to when it was called.

        :param gates: list of gates to sweep
        :param max_v: voltage to sweep up to
        :param num_points: points in each sweep direction
        :param measure_buffer: time in microseconds between when the precision DAC steps in voltage
            and QICK starts a DCS measurement.
        """

        # setup the slow_dac step length
        slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer
        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=self.hardware_config.dac_settings.trig_pin,
            trig_length=self.hardware_config.dac_settings.trig_length,
            measure_buffer=measure_buffer,
            points=num_points,
            dcs_cfg=self.dcs_config,
        )

        qickdata_list = []
        data_labels = []

        for gate_label in gates:
            # gate sweeps
            v_0 = self.vdc.get_dc_voltage(gate_label)
            v_start: float = 0.0
            v_stop = max_v
            v_sweep_up = np.linspace(v_start, v_stop, num_points)
            self.vdc.program_ramp(v_start, v_stop, slow_dac_step_len * 1e-6, num_points, gate_label)
            self.vdc.arm_sweep(gate_label)

            # Start a Vy sweep at a Vx increment and store the data
            meas = tune_electrostatics_programs_v2.GvG(
                self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
            )
            data = meas.acquire(self.soc, progress=False)
            expt_label = gate_label + "_up"
            data_labels.append(expt_label)
            assert data
            qd_up = spinqick_data.SpinqickData(
                data,
                gvg_cfg,
                1,
                1,
                expt_label,
                voltage_state=self.vdc.all_voltages,
                prog=meas,
            )
            time.sleep(slow_dac_step_len * 1e-6 * num_points)
            qd_up.add_axis([v_sweep_up], "x", [gate_label], num_points, units=["V"])
            analysis.calculate_conductance(qd_up, self.adc_unit_conversions)
            qickdata_list.append(qd_up)

            # sweep back down
            v_start = max_v
            v_stop = 0
            v_sweep_down = np.linspace(v_start, v_stop, num_points)
            self.vdc.program_ramp(v_start, v_stop, slow_dac_step_len * 1e-6, num_points, gate_label)
            self.vdc.arm_sweep(gate_label)
            meas = tune_electrostatics_programs_v2.GvG(
                self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
            )
            data = meas.acquire(self.soc, progress=False)
            expt_label = gate_label + "_down"
            data_labels.append(expt_label)
            assert data
            qd_down = spinqick_data.SpinqickData(
                data,
                gvg_cfg,
                1,
                1,
                expt_label,
                voltage_state=self.vdc.all_voltages,
                prog=meas,
            )
            time.sleep(slow_dac_step_len * 1e-6 * num_points)
            qd_down.add_axis([v_sweep_down], "x", [gate_label], num_points, units=["V"])
            analysis.calculate_conductance(qd_down, self.adc_unit_conversions)
            qickdata_list.append(qd_down)

            self.vdc.set_dc_voltage(v_0, gate_label)

        qd_composite = spinqick_data.CompositeSpinqickData(
            qickdata_list, data_labels, "_gate_action"
        )

        if self.plot:
            x_label = "gates:"
            for gate in gates:
                x_label = x_label + " " + gate + ","

            for i, dset in enumerate(qickdata_list):
                adc_units = self.adc_unit_conversions[0]
                if i == 0:
                    fignums = plot_g_1d(
                        dset,
                        gates[i],
                        x_label,
                        "conductance (%s)" % adc_units,
                        dset_label=dset.experiment_name,
                    )
                else:
                    gatename = list(dset.axes["x"]["sweeps"].keys())
                    add_g_1d(
                        dset,
                        gatename[0],
                        x_label,
                        "conductance (%s)" % adc_units,
                        dset_label=dset.experiment_name,
                        fignums=fignums,
                    )
                for num in fignums:
                    plt.figure(num)
                    plt.legend()

        if self.save_data:
            nc_file = qd_composite.basic_composite_save()
            if self.plot:
                for adc in fignums:
                    nc_file.save_last_plot(fignum=adc)
            nc_file.close()
        return qd_composite

    @dot_experiment.updater
    def sweep_1d(
        self,
        gates: list[spinqick_enums.GateNames],
        g_range: tuple[float, float, int],
        measure_buffer: float = 50,
        compensate: spinqick_enums.GateNames | None = None,
        sweep_direction: list[int] | None = None,
        filename_tag: str | None = None,
        mode: Literal["sd_chop", "trans"] = "sd_chop",
    ) -> spinqick_data.SpinqickData:
        """Sweep gates over desired range. No averaging built in currently.

        :param gates: list of gates to sweep
        :param g_range: (start voltage, stop voltage, number of steps)
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and
            the QICK starts a DCS measurement.
        :param compensate: specify a gate to compensate on while the sweep is running
        :param sweep_direction: Allows user to sweep a gate backwards if desired. Provide a list of
            values equal to +/-1 to specify backwards sweeps.
        :param filename_tag: add a string to the filename, for example 'electron_temperature'
        :param mode: "sdchop" selects typical source drain chop readout, "transdc" is for
            transcoductance mode
        """

        g_start, g_stop, num_points = g_range
        # setup the slow_dac step length
        slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer
        if filename_tag is not None:
            experiment_str = filename_tag
        else:
            experiment_str = "_1d_sweep"
        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=self.hardware_config.dac_settings.trig_pin,
            trig_length=self.hardware_config.dac_settings.trig_length,
            measure_buffer=measure_buffer,
            points=num_points,
            dcs_cfg=self.dcs_config,
        )

        v_0 = np.zeros((len(gates)))
        vsweeps = []
        if compensate is not None:
            v_m = self.vdc.get_dc_voltage(compensate)
        for i, gate_label in enumerate(gates):
            v_bias_0 = self.vdc.get_dc_voltage(gate_label)
            v_0[i] = v_bias_0

            v_start = v_bias_0 + g_start
            v_stop = v_bias_0 + g_stop
            if sweep_direction is not None:
                if sweep_direction[i] == -1:
                    v_start = v_bias_0 + g_stop
                    v_stop = v_bias_0 + g_start
            v_sweep = np.linspace(v_start, v_stop, num_points)
            vsweeps.append(v_sweep)
            if compensate is None:
                self.vdc.program_ramp(
                    v_start, v_stop, slow_dac_step_len * 1e-6, num_points, gate_label
                )
            else:
                self.vdc.program_ramp_compensate(
                    v_start,
                    v_stop,
                    slow_dac_step_len * 1e-6,
                    num_points,
                    gate_label,
                    compensate,
                )
                self.vdc.arm_sweep(compensate)
        for gate_label in gates:
            self.vdc.arm_sweep(gate_label)

        meas = tune_electrostatics_programs_v2.GvG(self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg)
        data = meas.acquire(self.soc, progress=False)
        assert data
        qd = spinqick_data.SpinqickData(
            data,
            gvg_cfg,
            1,
            1,
            experiment_str,
            voltage_state=self.vdc.all_voltages,
            prog=meas,
        )
        qd.add_axis(vsweeps, "x", gates, num_points, units=["V" for i in range(len(gates))])
        if mode == "sd_chop":
            analysis.calculate_conductance(qd, self.adc_unit_conversions)
        else:
            analysis.calculate_transconductance(
                qd,
                self.adc_unit_conversions,
            )

        # TODO convert these to ramps
        for i, gate_label in enumerate(gates):
            self.vdc.set_dc_voltage(v_0[i], gate_label)
        if compensate is not None:
            self.vdc.set_dc_voltage(v_m, compensate)

        if self.plot:
            # plot the data
            x_label = "gates:"
            for gate in gates:
                x_label = x_label + " " + gate + ","
            adc_units = self.adc_units[0]
            plot_g_1d(qd, gates[0], x_label, "%s (%s)" % (mode, adc_units))

        if self.save_data:
            nc_file = qd.save_data()
            if self.plot:
                nc_file.save_last_plot()
            nc_file.close()
        return qd

    def gate_turn_on(
        self,
        gates: list[spinqick_enums.GateNames],
        max_v: float,
        num_points: int,
        measure_buffer: float,
    ) -> spinqick_data.SpinqickData:
        """Global gate turn on script.  Sweep several gates simultaneously to look for current
        through the device.

        :param gates: list of gates to sweep
        :param max_v: voltage to sweep up to
        :param num_points: points in each sweep direction
        :param measure_buffer: time in microseconds between when the precision DC DAC steps in
            voltage and QICK starts a DCS measurement.
        """

        sqd_obj = self.sweep_1d(
            gates,
            (0, max_v, num_points),
            measure_buffer=measure_buffer,
            filename_tag="_global_turn_on",
        )

        return sqd_obj

    @dot_experiment.updater
    def calibrate_baseband_voltage(
        self,
        gate: spinqick_enums.GateNames,
        gate_dc_range: Tuple[float, float, int],
        gate_pulse_range: Tuple[float, float, int],
        gate_step: float,
        gate_freq: float,
        measure_buffer: float,
    ) -> spinqick_data.CompositeSpinqickData:
        """Calibrate baseband voltage based off of low speed dacs. Scans a loading line with low
        speed dacs while sweeping pulse gain of high speed dacs.  The pulse is a high frequency AC
        signal.  We perform two measurements at each point, each with a slightly different DC offset
        on the sine wave output by the rfsoc.

        :param gate: name of gate to calibrate
        :param gate_dc_range: range of DC source voltages (start voltage, end voltage, number of
            points)
        :param gate_pulse_range: range of qick pulse gains
        :param gate_step: pulse gain value used to get a differential measurement. The Rfsoc pulses
            to a value, takes a measurement, then pulses to the value minus the gate_step parameter.
            Then we take a difference between the two conductance values. This gives us a
            differential measurement so that we can see the calibration lines more clearly.
        :param gate_freq: pulse frequency applied by rfsoc.
        :param measure_buffer: time in microseconds between when the slow dac steps in voltage and
            the QICK starts a DCS measurement.
        """

        p_dc_start, p_dc_stop, p_dc_npts = gate_dc_range
        v_p0 = self.vdc.get_dc_voltage(gate)
        dc_sweep = np.linspace(p_dc_start + v_p0, p_dc_stop + v_p0, p_dc_npts)
        p_pulse_start, p_pulse_stop, p_pulse_npts = gate_pulse_range
        gate_cfg = self.hardware_config.channels[gate]
        if isinstance(gate_cfg, hardware_config_models.FastGate):
            gate_gen = gate_cfg.qick_gen
        else:
            raise KeyError(" specified gate does not have an associated qick channel ")

        pulse_gain_sweep = np.linspace(p_pulse_start, p_pulse_stop, p_pulse_npts)
        raw_list = []
        data_list = []
        data_array = np.zeros((p_pulse_npts, p_dc_npts))
        for k in range(len(self.dcs_config.ro_chs)):
            raw = np.zeros((1, 2, p_pulse_npts, p_dc_npts, 2))
            raw_list.append(raw)
        for i, pulse_gain in enumerate(pulse_gain_sweep):
            bb_cal_config = experiment_models.LineSplitting(
                trig_pin=self.hardware_config.dac_settings.trig_pin,
                measure_buffer=measure_buffer,
                points=p_dc_npts,
                dcs_cfg=self.dcs_config,
                trig_length=self.hardware_config.dac_settings.trig_length,
                mode="sd_chop",
                differential_channel=gate_gen,
                differential_ac_freq=gate_freq,
                differential_ac_gain=pulse_gain,
                differential_step_gain=gate_step,
            )

            step_time = 2 * self.dcs_config.length + 4 * measure_buffer

            self.vdc.program_ramp(dc_sweep[0], dc_sweep[-1], step_time * 1e-6, p_dc_npts, gate)
            self.vdc.arm_sweep(gate)

            meas = system_calibrations_programs_v2.BasebandVoltageCalibration(
                self.soccfg, reps=1, final_delay=0, cfg=bb_cal_config
            )
            data = meas.acquire(self.soc, progress=False)
            assert data
            for k, ro in enumerate(data):
                raw_list[k][:, :, i, :, :] = ro

            self.vdc.program_ramp(
                p_dc_stop + v_p0, p_dc_start + v_p0, step_time * 1e-6, p_dc_npts, gate
            )
            self.vdc.arm_sweep(gate)
            self.vdc.digital_trigger(gate)
            time.sleep(step_time / 1e6 * p_dc_npts)
            self.soc.reset_gens()

            qd = spinqick_data.PsbData(
                data,
                bb_cal_config,
                2,
                1,
                "_bb_cal",
                voltage_state=self.vdc.all_voltages,
                prog=meas,
            )
            qd.add_axis([dc_sweep], "x", [gate], p_dc_npts, units=["V"])
            analysis.calculate_conductance(
                qd,
                self.adc_unit_conversions,
            )
            analysis.calculate_difference(qd)
            data_list.append(qd)
            diff = qd.difference_data
            assert diff is not None
            data_array[i, :] = diff[0]
        self.vdc.set_dc_voltage(v_p0, gate)
        dset_labels = [str(pulse_gain_sweep[i]) for i in range(p_pulse_npts)]
        full_dataset = spinqick_data.CompositeSpinqickData(
            data_list, dset_labels, "_bb_cal", dset_coordinates=pulse_gain_sweep
        )
        if self.plot:
            plot_tools.plot2_simple(
                pulse_gain_sweep,
                dc_sweep,
                np.transpose(data_array),
                full_dataset.timestamp,
                cbar_label="differential conductance",
            )
            plt.title(gate)
            plt.ylabel(" %s voltage (V)" % gate)
            plt.xlabel("rfsoc dac gain")

        if self.save_data:
            nc_file = full_dataset.basic_composite_save()
            if self.plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s", full_dataset.data_file)
        return full_dataset

    @dot_experiment.updater
    def measure_bandwidth(
        self,
        gate: spinqick_enums.GateNames,
        gate_dc_range: Tuple[float, float, int],
        gate_freq_range: Tuple[float, float, int],
        gate_step: float,
        pulse_gain: float,
        measure_buffer: float,
    ) -> spinqick_data.CompositeSpinqickData:
        """Performs a frequency sweep of an AC signal generated by qick in order to look for a roll-
        off. Scans a loading line with low speed dacs while sweeping pulse gain of high speed dacs.
        The pulse is a high frequency AC signal.  We perform two measurements at each point, each
        with a slightly different DC offset on the sine wave output by the rfsoc.

        :param gate: name of gate to calibrate
        :param gate_dc_range: range of DC source voltages (start voltage, end voltage, number of
            points)
        :param gate_freq_range: range of qick pulse frequency values (start frequency, end
            frequency, number of points)
        :param gate_step: pulse gain value used to get a differential measurement. The Rfsoc pulses
            to a value, takes a measurement, then pulses to the value minus the gate_step parameter.
            Then we take a difference between the two conductance values. This gives us a
            differential measurement so that we can see the calibration lines more clearly.
        :param pulse_gain: RFSoC pulse gain.
        :param measure_buffer: time in microseconds between when the slow dac steps in voltage and
            the QICK starts a DCS measurement.
        """

        p_dc_start, p_dc_stop, p_dc_npts = gate_dc_range
        v_p0 = self.vdc.get_dc_voltage(gate)
        dc_sweep = np.linspace(p_dc_start + v_p0, p_dc_stop + v_p0, p_dc_npts)
        p_pulse_start, p_pulse_stop, p_pulse_npts = gate_freq_range
        gate_cfg = self.hardware_config.channels[gate]
        if isinstance(gate_cfg, hardware_config_models.FastGate):
            gate_gen = gate_cfg.qick_gen
        else:
            raise KeyError(" specified gate does not have an associated qick channel ")

        pulse_freq_sweep = np.logspace(
            np.log10(p_pulse_start), np.log10(p_pulse_stop), p_pulse_npts
        )
        raw_list = []
        data_list = []
        data_array = np.zeros((p_pulse_npts, p_dc_npts))
        for k in range(len(self.dcs_config.ro_chs)):
            raw = np.zeros((1, 2, p_pulse_npts, p_dc_npts, 2))
            raw_list.append(raw)
        for i, pulse_freq in enumerate(pulse_freq_sweep):
            bb_cal_config = experiment_models.LineSplitting(
                trig_pin=self.hardware_config.dac_settings.trig_pin,
                measure_buffer=measure_buffer,
                points=p_dc_npts,
                dcs_cfg=self.dcs_config,
                trig_length=self.hardware_config.dac_settings.trig_length,
                mode="sd_chop",
                differential_channel=gate_gen,
                differential_ac_freq=pulse_freq,
                differential_ac_gain=pulse_gain,
                differential_step_gain=gate_step,
            )

            step_time = 2 * self.dcs_config.length + 4 * measure_buffer

            self.vdc.program_ramp(dc_sweep[0], dc_sweep[-1], step_time * 1e-6, p_dc_npts, gate)
            self.vdc.arm_sweep(gate)

            meas = system_calibrations_programs_v2.BasebandVoltageCalibration(
                self.soccfg, reps=1, final_delay=0, cfg=bb_cal_config
            )
            data = meas.acquire(self.soc, progress=False)
            assert data
            for k, ro in enumerate(data):
                raw_list[k][:, :, i, :, :] = ro

            self.vdc.program_ramp(
                p_dc_stop + v_p0, p_dc_start + v_p0, step_time * 1e-6, p_dc_npts, gate
            )
            self.vdc.arm_sweep(gate)
            self.vdc.digital_trigger(gate)
            time.sleep(step_time / 1e6 * p_dc_npts)
            self.soc.reset_gens()

            qd = spinqick_data.PsbData(
                data,
                bb_cal_config,
                2,
                1,
                "_freq_cal",
                voltage_state=self.vdc.all_voltages,
                prog=meas,
            )
            qd.add_axis([dc_sweep], "x", [gate], p_dc_npts, units=["V"])
            analysis.calculate_conductance(
                qd,
                self.adc_unit_conversions,
            )
            analysis.calculate_difference(qd)
            data_list.append(qd)
            assert qd.difference_data
            data_array[i, :] = qd.difference_data[0]
        self.vdc.set_dc_voltage(v_p0, gate)
        dset_labels = [str(pulse_freq_sweep[i]) for i in range(p_pulse_npts)]
        full_dataset = spinqick_data.CompositeSpinqickData(
            data_list, dset_labels, "_freq_cal", dset_coordinates=pulse_freq_sweep
        )
        if self.plot:
            plot_tools.plot2_simple(
                pulse_freq_sweep,
                dc_sweep,
                np.transpose(data_array),
                full_dataset.timestamp,
                cbar_label="differential conductance",
            )
            plt.title(gate)
            plt.ylabel(" %s voltage (V)" % gate)
            plt.xlabel("rfsoc dac freq")
            plt.xscale("log")

        if self.save_data:
            nc_file = full_dataset.basic_composite_save()
            if self.plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s", full_dataset.data_file)
        return full_dataset

    @dot_experiment.updater
    def tune_hsa(
        self,
        gate: spinqick_enums.GateNames,
        gate_dc_range: Tuple[float, float, int],
        pulse_time: float,
        pulse_gain: float,
        measure_buffer: float,
    ) -> spinqick_data.SpinqickData:
        """Performs a scan of voltage and plays a baseband pulse followed immediately with a measurement.  This is intended for tuning the high speed adder circuit described in https://doi.org/10.1103/PRXQuantum.3.010352 .

        :param gate: name of gate to calibrate
        :param gate_dc_range: range of DC source voltages (start voltage, end voltage, number of points)
        :param pulse_time: duration of pulse in microseconds
        :param pulse_gain: pulse gain in RFSoC units (between -1, and 1)
        :param measure_buffer: time in microseconds between when the slow dac steps in voltage and
            the QICK starts a DCS measurement.
        """

        p_dc_start, p_dc_stop, p_dc_npts = gate_dc_range
        v_p0 = self.vdc.get_dc_voltage(gate)
        dc_sweep = np.linspace(p_dc_start + v_p0, p_dc_stop + v_p0, p_dc_npts)
        gate_cfg = self.hardware_config.channels[gate]
        if isinstance(gate_cfg, hardware_config_models.FastGate):
            gate_gen = gate_cfg.qick_gen
        else:
            raise KeyError(" specified gate does not have an associated qick channel ")
        pgain_dac = pulse_gain * gate_cfg.dac_conversion_factor
        cal_config = experiment_models.HsaTune(
            dcs_cfg=self.dcs_config,
            point_avgs=p_dc_npts,
            tune_gate=gate,
            tune_gate_gen=gate_gen,
            pulse_time=pulse_time,
            pulse_gain=pgain_dac,
            measure_buffer=measure_buffer,
        )

        step_time = self.dcs_config.length + measure_buffer + pulse_time

        self.vdc.program_ramp(dc_sweep[0], dc_sweep[-1], step_time * 1e-6, p_dc_npts, gate)
        self.vdc.arm_sweep(gate)

        meas = system_calibrations_programs_v2.HSATune(
            self.soccfg, reps=1, final_delay=0, cfg=cal_config
        )
        data = meas.acquire(self.soc, progress=False)
        assert data

        self.soc.reset_gens()

        qd = spinqick_data.PsbData(
            data,
            cal_config,
            2,
            1,
            "_bb_cal",
            voltage_state=self.vdc.all_voltages,
            prog=meas,
        )
        qd.add_axis([dc_sweep], "x", [gate], p_dc_npts, units=["V"])
        analysis.calculate_conductance(
            qd,
            self.adc_unit_conversions,
        )
        self.vdc.set_dc_voltage(v_p0, gate)
        if self.plot:
            plot_g_1d(qd, gate, " %s voltage (V)" % gate, "conductance")

        if self.save_data:
            nc_file = qd.save_data()
            if self.plot:
                nc_file.save_last_plot()
            nc_file.close()
            logger.info("data saved at %s", qd.data_file)
        return qd
