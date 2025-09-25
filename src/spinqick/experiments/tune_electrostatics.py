"""
Perform charge stability measurements.  Converted to tprocv2 compatibility.

"""

import time
import logging
from typing import Literal, Tuple, List
from lmfit import models
import matplotlib.pyplot as plt
import numpy as np
from spinqick.core import dot_experiment
from spinqick.core import spinqick_data
from spinqick.helper_functions import (
    hardware_manager,
    plot_tools,
    analysis,
)
from spinqick.qick_code_v2 import (
    tune_electrostatics_programs_v2,
    system_calibrations_programs_v2,
)
from spinqick.settings import file_settings, dac_settings
from spinqick import settings
from spinqick.models import experiment_models, hardware_config_models


logger = logging.getLogger(__name__)


def plot_gvg_2d(
    data: spinqick_data.SpinqickData,
    x_gate: str,
    y_gate: str,
    x_label: str,
    y_label: str,
    adc_units: str | List[str] = "arbs",
):
    """plotting function for 2D electrostatics type sweeps"""
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
    """generate a plot of a 1D sweep of a gate voltage"""
    if data.analyzed_data is not None:
        fignums = []
        for i, adc_data in enumerate(data.analyzed_data):
            x_data = data.axes["x"]["sweeps"][gate]["data"]
            y_data = adc_data
            fig = plot_tools.plot1_simple(
                x_data, y_data[0], data.timestamp, dset_label=dset_label
            )
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
    """add a 1d trace to a plot generated with plot_g_1d"""
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
    """analysis routine for the cross capacitance experiment"""
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
    pars = line.guess(
        center_data, x=data_obj.axes["x"]["sweeps"][slow_gate_dict]["data"]
    )
    try:
        out = line.fit(
            center_data, pars, x=data_obj.axes["x"]["sweeps"][slow_gate_dict]["data"]
        )
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
    In general each function sets the necessary config parameters for you and then runs the qick program.
    They then optionally plot and save the data.
    """

    def __init__(
        self,
        soccfg,
        soc,
        voltage_source: hardware_manager.VoltageSource,
        datadir: str = file_settings.data_directory,
        save_data: bool = True,
        plot: bool = True,
    ):
        """initialize with information about your rfsoc and your experimental setup

        :param soccfg: QickConfig object
        :param soc: Qick object
        :param voltage_source: instantiated dc VoltageSource object
        :param datadir: data directory where all data is being stored. Experiment will make a folder here with today's date.
        """
        super().__init__(datadir=datadir)
        self.soccfg = soccfg
        self.soc = soc
        self.save_data = save_data
        self.plot = plot
        self.vdc = hardware_manager.DCSource(voltage_source=voltage_source)

    @dot_experiment.updater
    def gvg_baseband(
        self,
        g_gates: tuple[settings.GateNames, settings.GateNames],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]],
        measure_buffer: float,
    ) -> spinqick_data.SpinqickData:
        """perform a basic PvP or PvT by baseband pulsing with the RFSoC.

        :param g_gates: gates to sweep on y and x axes. i.e. ('P1','P2')
        :param g_range: range to sweep gate x and gate y, and number of points for each
        :param measure_buffer: pause time between measurements
        :param add_pat: apply RF during measurement
        :param pat_gain: gain of RF signal in range [-1, 1]
        :param pat_gen: qick generator for RF signal
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

    def gvg_arb_prog(
        self,
        prog,
        expt_name,
        cfg,
        g_gates: tuple[list[settings.GateNames], list[settings.GateNames]],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]],
        measure_buffer: float,
        compensate: settings.GateNames | None = None,
        sweep_direction: tuple[list[int], list[int]] | None = None,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
        save_data: bool = False,
    ) -> spinqick_data.SpinqickData:
        """python outer loop for 2 dimensional gate sweep-type experiments.  Currently user can only compensate on one gate"""

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
                [gx_start for i in range(len(gx_gates))]
                + [gy_start for i in range(len(gy_gates))],
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
        if step_length < dac_settings.t_min_slow_dac:
            slow_dac_step_len = dac_settings.t_min_slow_dac
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
                time.sleep(
                    slow_dac_step_len * 1e-6 * n_vy
                )  # leave some time to ramp down

        for k, gx_gate in enumerate(gx_gates):
            v_final = self.vdc.get_dc_voltage(gx_gate)
            self.vdc.program_ramp(
                v_final, vx_0[k], dac_settings.t_min_slow_dac * 1e-6, n_vx, gx_gate
            )
            self.vdc.arm_sweep(gx_gate)
            self.vdc.digital_trigger(gx_gate)
            time.sleep(
                dac_settings.t_min_slow_dac * n_vx * 1e-6
            )  # leave some time to ramp down
        for k, gy_gate in enumerate(gy_gates):
            v_final = self.vdc.get_dc_voltage(gy_gate)
            self.vdc.program_ramp(
                v_final, vy_0[k], dac_settings.t_min_slow_dac * 1e-6, n_vy, gy_gate
            )
            self.vdc.arm_sweep(gy_gate)
            self.vdc.digital_trigger(gy_gate)
            time.sleep(slow_dac_step_len * 1e-6 * n_vy)  # leave some time to ramp down
        if isinstance(compensate, str):
            vm = self.vdc.get_dc_voltage(compensate)
            self.vdc.program_ramp(
                vm, vm_0, dac_settings.t_min_slow_dac * 1e-6, n_vy, compensate
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
        g_gates: tuple[list[settings.GateNames], list[settings.GateNames]],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]],
        measure_buffer: float,
        compensate: settings.GateNames | None = None,
        sweep_direction: tuple[list[int], list[int]] | None = None,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
    ) -> spinqick_data.SpinqickData:
        """GvG script which sweeps an external DC voltage source and reads out a DCS

        :param g_gates: gates to sweep on y and x axes. i.e. (['P1'],['P2']).  Option to provide a list of gates
            to sweep on each axis.
        :param g_range: voltage range to sweep gate x and gate y, and number of points for each.
        :param compensate: gate to compensate while changing other voltages. i.e. 'M1'
        :param sweep_direction: Allows user to sweep a gate backwards if desired. Provide a
            list of positive or negative ones corresponding to each gate in g_gates i.e. ([1,-1], [1]).
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param mode: "sdchop" selects typical source drain chop readout, "transdc" is for transcoductance mode
        :return: expt pts, avgi, avgq -- raw data from 'acquire'
        """
        _, g_range_y = g_range
        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=dac_settings.trig_pin,
            measure_buffer=measure_buffer,
            points=g_range_y[2],
            dcs_cfg=self.dcs_config,
            trig_length=dac_settings.trig_length,
            mode=mode,
        )

        meas = tune_electrostatics_programs_v2.GvG(
            self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
        )

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
        g_gates: tuple[list[settings.GateNames], list[settings.GateNames]],
        g_range: tuple[tuple[float, float, int], tuple[float, float, int]],
        measure_buffer: float,
        pat_freq: float,
        pat_gain: float,
        pat_gen: int,
        compensate: settings.GateNames | None = None,
        sweep_direction: tuple[list[int], list[int]] | None = None,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
    ) -> spinqick_data.SpinqickData:
        """GvG script which applies an RF tone to probe photon assisted tunneling

        :param g_gates: gates to sweep on y and x axes. i.e. (['P1'],['P2']).  Option to provide a list of gates
            to sweep on each axis.
        :param g_range: voltage range to sweep gate x and gate y, and number of points for each.
        :param compensate: gate to compensate while changing other voltages. i.e. 'M1'
        :param sweep_direction: Allows user to sweep a gate backwards if desired. Provide a
            list of positive or negative ones corresponding to each gate in g_gates i.e. ([1,-1], [1]).
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param pat_freq: frequency of microwave tone
        :param pat_gain: gain of pat signal in QICK units (-1, 1)
        :param pat_gen: generator used for pat signal
        :param mode: "sdchop" selects typical source drain chop readout, "transdc" is for transcoductance mode
        :return: expt pts, avgi, avgq -- raw data from 'acquire'
        """
        _, gy_range = g_range

        pat_cfg = experiment_models.PatConfig(
            pat_freq=pat_freq,
            pat_gain=pat_gain,
            pat_gen=pat_gen,
        )

        gvg_cfg = experiment_models.GvgPatConfig(
            trig_pin=dac_settings.trig_pin,
            measure_buffer=measure_buffer,
            points=gy_range[2],
            dcs_cfg=self.dcs_config,
            trig_length=dac_settings.trig_length,
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
        x_gate: settings.GateNames,
        y_gate: settings.GateNames,
        x_range: Tuple[float, float, int],
        y_range: Tuple[float, float, int],
        measure_buffer: float,
        compensate: settings.GateNames | None = None,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
        fit_type: Literal["gaussian", "abs_max", "abs_min"] = "gaussian",
    ) -> spinqick_data.SpinqickData:
        """Measure cross-capacitance between gates
        :param x_gate: gate to step in python outer loop
        :param y_gate: either an M gate, or the gate that compensation will be applied to
        :param x_range: voltage range to sweep x gate over
        :param y_range: voltage range to sweep y (fast sweep) gate over
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param compensate: the m-gate to compensate with for the y-gate.  If y-gate is an m-gate, leave as None
        """
        _, _, n_vy = y_range

        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=dac_settings.trig_pin,
            measure_buffer=measure_buffer,
            points=n_vy,
            dcs_cfg=self.dcs_config,
            trig_length=dac_settings.trig_length,
            mode=mode,
        )

        meas = tune_electrostatics_programs_v2.GvG(
            self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
        )

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
        m_dot: settings.GateNames,
        m_range: tuple[float, float, int],
        z_range: tuple[float, float, int],
        measure_buffer: float,
        tune_type: Literal["common", "diff"] = "common",
    ) -> spinqick_data.SpinqickData:
        """Tune MZ using sleeperdac or other external DC voltage bias. This wraps the function gvg_dc.

        :param m_dot: M dot you're tuning, i.e. 'M1'
        :param m_range: voltage range to sweep m gate, and number of points.
        :param z_range: voltage range to sweep z gate, and number of points.
        :param tune_type: sweep z gates in same direction or opposite directions
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param save_data: whether or not to autosave the results
        :param plot: whether or not to plot in ipython
        :return: Electrostatics data object
        """

        if m_dot == "M1":
            z_dots = [settings.GateNames.Z1, settings.GateNames.Z2]
        else:
            z_dots = [settings.GateNames.Z3, settings.GateNames.Z4]

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
        m_dot: settings.GateNames,
        m_range: tuple[float, float, int],
        measure_buffer: float,
        set_v=False,
        mode: Literal["sd_chop", "transdc"] = "sd_chop",
    ) -> spinqick_data.SpinqickData:
        """automated retune DCS script. Scan m-gate voltage, fit to conductance peak, extract an
        optimal m-gate voltage from peak

        :param m_dot: Dot to retune
        :param m_range: List of start voltage, stop voltage and number of points.  Voltages are relative
            to the current setpoint
        :param measure_buffer: time in microseconds between when the slow dac steps in voltage and the QICK starts a DCS measurement.
        :param set_v: after retuning, set the voltage to the optimal point
        """

        # M Sweep
        m_bias = self.vdc.get_dc_voltage(m_dot)
        n_vm = int(m_range[2])
        vm_start = m_range[0] + m_bias
        vm_stop = m_range[1] + m_bias
        vm_sweep = np.linspace(vm_start, vm_stop, n_vm)

        slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer

        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=dac_settings.trig_pin,
            trig_length=dac_settings.trig_length,
            measure_buffer=measure_buffer,
            points=n_vm,
            dcs_cfg=self.dcs_config,
            mode=mode,
        )

        self.vdc.program_ramp(vm_start, vm_stop, slow_dac_step_len * 1e-6, n_vm, m_dot)
        self.vdc.arm_sweep(m_dot)

        # Start a Vy sweep at a Vx increment and store the data
        meas = tune_electrostatics_programs_v2.GvG(
            self.soccfg, reps=1, final_delay=0, cfg=gvg_cfg
        )
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
        return_step_time = dac_settings.t_min_slow_dac
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
            fit_dict = {"center": center, "fwhm": fwhm, "best_voltage": halfmax}
            data_obj.add_fit_params(fit_dict, out.best_fit, "x")
        except Exception as exc:
            if set_v:
                self.vdc.set_dc_voltage(m_bias, m_dot)
            fwhm = 0
            center = 0
            halfmax = 0
            logger.error(
                "fit failed, setting m to initial value, error %s", exc, exc_info=True
            )

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
        gates: list[settings.GateNames],
        max_v: float,
        num_points: int,
        measure_buffer: float,
    ) -> spinqick_data.CompositeSpinqickData:
        """gate action sweep script.  Sweep gates individually up and back down to look for
        turn on voltages.

        :param gates: list of gates to sweep
        :param max_v: voltage to sweep up to
        :param num_points: points in each sweep direction
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        """

        # setup the slow_dac step length
        slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer
        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=dac_settings.trig_pin,
            trig_length=dac_settings.trig_length,
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
            self.vdc.program_ramp(
                v_start, v_stop, slow_dac_step_len * 1e-6, num_points, gate_label
            )
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
            self.vdc.program_ramp(
                v_start, v_stop, slow_dac_step_len * 1e-6, num_points, gate_label
            )
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
        gates: list[settings.GateNames],
        g_range: tuple[float, float, int],
        measure_buffer: float = 50,
        compensate: settings.GateNames | None = None,
        sweep_direction: list[int] | None = None,
        filename_tag: str | None = None,
        mode: Literal["sd_chop", "trans"] = "sd_chop",
    ):
        """
        Sweep gates over desired range. No averaging built in currently.  Can be used for electron temperature measurements. Work in progress.

        :param gates: list of gates to sweep
        :param g_range: (start voltage, stop voltage, number of steps)
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :param sweep_direction: Allows user to sweep a gate backwards if desired. Provide a
        :param filename_tag: add a string to the filename, for example 'electron_temperature'
        """

        g_start, g_stop, num_points = g_range
        # setup the slow_dac step length
        slow_dac_step_len = self.dcs_config.length + 2 * measure_buffer
        if filename_tag is not None:
            experiment_str = filename_tag
        else:
            experiment_str = "_1d_sweep"
        gvg_cfg = experiment_models.GvgDcConfig(
            trig_pin=dac_settings.trig_pin,
            trig_length=dac_settings.trig_length,
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
            experiment_str,
            voltage_state=self.vdc.all_voltages,
            prog=meas,
        )
        qd.add_axis(
            vsweeps, "x", gates, num_points, units=["V" for i in range(len(gates))]
        )
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
        gates: list[settings.GateNames],
        max_v: float,
        num_points: int,
        measure_buffer: float,
    ) -> spinqick_data.SpinqickData:
        """
        Global gate turn on script.  Sweep several gates simultaneously to look for global
        turn on of the device.

        :param gates: list of gates to sweep
        :param max_v: voltage to sweep up to
        :param num_points: points in each sweep direction
        :param measure_buffer: time in microseconds between when the sleeperdac steps in voltage and the QICK starts a DCS measurement.
        :return: vm_sweep, mag"""

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
        gate: settings.GateNames,
        gate_dc_range: Tuple[float, float, int],
        gate_pulse_range: Tuple[float, float, int],
        gate_step: float,
        gate_freq: float,
        measure_buffer: float,
    ):
        """Calibrate baseband voltage based off of low speed dacs.
        Scans a loading line with low speed dacs while sweeping pulse gain of high speed dacs
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
                trig_pin=dac_settings.trig_pin,
                measure_buffer=measure_buffer,
                points=p_dc_npts,
                dcs_cfg=self.dcs_config,
                trig_length=dac_settings.trig_length,
                mode="sd_chop",
                differential_channel=gate_gen,
                differential_ac_freq=gate_freq,
                differential_ac_gain=pulse_gain,
                differential_step_gain=gate_step,
            )

            step_time = 2 * self.dcs_config.length + 4 * measure_buffer

            self.vdc.program_ramp(
                dc_sweep[0], dc_sweep[-1], step_time * 1e-6, p_dc_npts, gate
            )
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
            data_array[i, :] = qd.difference_data[0]
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
        gate: settings.GateNames,
        gate_dc_range: Tuple[float, float, int],
        gate_freq_range: Tuple[float, float, int],
        gate_step: float,
        pulse_gain: float,
        measure_buffer: float,
    ):
        """Scan frequency of qick generator to see rolloff of circuit.
        Scans a loading line with low speed dacs while sweeping pulse freq of high speed dacs
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
                trig_pin=dac_settings.trig_pin,
                measure_buffer=measure_buffer,
                points=p_dc_npts,
                dcs_cfg=self.dcs_config,
                trig_length=dac_settings.trig_length,
                mode="sd_chop",
                differential_channel=gate_gen,
                differential_ac_freq=pulse_freq,
                differential_ac_gain=pulse_gain,
                differential_step_gain=gate_step,
            )

            step_time = 2 * self.dcs_config.length + 4 * measure_buffer

            self.vdc.program_ramp(
                dc_sweep[0], dc_sweep[-1], step_time * 1e-6, p_dc_npts, gate
            )
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
