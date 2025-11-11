"""Defines the EOSingleQubit class used to perform coherent control of a single exchange-only
qubit."""

import logging
from typing import Literal, Tuple

import numpy as np
import pylab as plt

from spinqick.core import dot_experiment, spinqick_data
from spinqick.experiments import eo_analysis
from spinqick.helper_functions import analysis, hardware_manager, plot_tools, spinqick_enums
from spinqick.models import experiment_models, qubit_models
from spinqick.qick_code_v2 import eo_single_qubit_programs_v2

logger = logging.getLogger(__name__)


class EOSingleQubit(dot_experiment.DotExperiment):
    """Contains methods that wrap the QICK classes for setting up EO single qubit experiments.
    Initialize with information about your rfsoc and your experimental setup.

    :param soccfg: qick config object (QickConfig)
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
    def calc_fingerprint_vectors(
        self,
        qubit: str,
        axis: str,
        house_point_1: tuple[float, float],
        house_point_2: tuple[float, float],
        x_volts: float,
    ):
        """Calculates detuning and exchange axes from a pair of points traversing the non-
        equilibrium cell.This is mainly useful for using spinqick without crosstalk compensation.

        :param house_point_1: ('Px', 'Py') pair of voltage coordinates defining a point on one edge
            of the cell. The detuning axis is the vector connecting the two pairs of house_points.
        :param house_point_2: ('Px', 'Py') pair of voltage coordinates defining a point on other
            edge of the cell.
        :param x_volts: voltage applied to x-gate during the measurement
        """

        # first convert volts to dac units
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        px_points = [
            self.volts2dac(point[0], axis_cfg.gates.px.name)
            for point in (house_point_1, house_point_2)
        ]
        py_points = [
            self.volts2dac(point[1], axis_cfg.gates.py.name)
            for point in (house_point_1, house_point_2)
        ]
        gates = axis_cfg.gates
        assert isinstance(gates, qubit_models.ExchangeGateMap)
        idle_points = [
            gates.px.gains.idle_gain,
            gates.py.gains.idle_gain,
            gates.x.gains.idle_gain,
        ]
        x_gain = self.volts2dac(x_volts, axis_cfg.gates.x.name)
        detuning, exchange = eo_analysis.define_fingerprint_vectors(
            np.array(px_points), np.array(py_points), np.array(idle_points), x_gain
        )
        return detuning, exchange

    @dot_experiment.updater
    def calculate_exchange_gate_vals(self, qubit: str, axis: str, detuning: float, x_throw: float):
        """Calculate exchange gain at gates in rfsoc units, given detuning and x_throw in volts."""

        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        detuning_convert = self.volts2dac(detuning, axis_cfg.gates.px.name)
        x_convert = self.volts2dac(x_throw, axis_cfg.gates.x.name)
        gates = axis_cfg.gates
        assert isinstance(gates, qubit_models.ExchangeGateMap)
        idle_points = [
            gates.px.gains.idle_gain,
            gates.py.gains.idle_gain,
            gates.x.gains.idle_gain,
        ]
        gate_vals = eo_analysis.calculate_fingerprint_gate_vals(
            detuning_convert,
            x_convert,
            np.array(axis_cfg.detuning_vector),
            np.array(axis_cfg.exchange_vector),
            idle_points,
        )
        return gate_vals

    @dot_experiment.updater
    def calc_symmetric_vector(self, qubit: str, axis: str, detuning_volts: float, x_volts: float):
        """Calculates the symmetric vector, assuming the vector starts at the idle point and ends at
        the point specified here. Returns the vectors after dac unit conversion.

        :param detuning_volts: detuning value of symmetric vector endpoint
        :param x_volts: exchange gate voltage at endpoint of symmetric vector
        """

        # first convert volts to dac units
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        detuning_convert = self.volts2dac(detuning_volts, axis_cfg.gates.px.name)
        gates = axis_cfg.gates
        assert isinstance(gates, qubit_models.ExchangeGateMap)
        idle_points = [
            gates.px.gains.idle_gain,
            gates.py.gains.idle_gain,
            gates.x.gains.idle_gain,
        ]
        x_gain = self.volts2dac((x_volts - gates.x.gains.idle_gain), axis_cfg.gates.x.name)
        gate_vals = eo_analysis.calculate_fingerprint_gate_vals(
            detuning_convert,
            x_gain,
            np.array(axis_cfg.detuning_vector),
            np.array(axis_cfg.symmetric_vector),
            idle_points,
        )
        symmetric_raw = gate_vals - idle_points
        symmetric = symmetric_raw / (x_gain)
        return symmetric

    @dot_experiment.updater
    def calculate_exchange_volts(self, qubit: str, axis: str, detuning: float, x_throw: float):
        """Calculates the exchange gain at gates in voltage units."""

        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        gate_vals = self.calculate_exchange_gate_vals(qubit, axis, detuning, x_throw)
        gate_volts = [
            self.dac2volts(gate_vals[0], axis_cfg.gates.px.name),
            self.dac2volts(gate_vals[1], axis_cfg.gates.py.name),
            self.dac2volts(gate_vals[2], axis_cfg.gates.x.name),
        ]
        return gate_volts

    @dot_experiment.updater
    def theta_to_voltage(self, qubit: str, axis: str, theta: float):
        """Use finecal curve to convert angle to gate voltages."""
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        cal = axis_cfg.exchange_cal
        assert cal is not None
        tlist = cal.cal_parameters.theta_list
        vlist = cal.cal_parameters.volt_list
        assert tlist is not None
        assert vlist is not None
        v_x = eo_analysis.fine_cal_voltage(
            theta,
            tlist,
            vlist,
            cal.cal_parameters.A,
            cal.cal_parameters.B,
            cal.cal_parameters.theta_max,
        )
        exchange_vals = self.calculate_exchange_volts(qubit, axis, 0, v_x)
        return exchange_vals

    @dot_experiment.updater
    def do_nonequilibrium(
        self,
        qubit: str,
        axis: spinqick_enums.ExchangeAxis,
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        point_avgs: int = 1,
        full_avgs: int = 1,
        t1j: bool = False,
    ):
        """Perform a scan of the non-equilibrium cell.  This helps us to set the detuning axis for a
        fingerprint plot.

        :param qubit: Qubit label, describing the qubit of interest in the experiment config
        :param axis: Qubit axis (n, z or m)
        :param p_range: range of p gate sweeps: (p2 start voltage, p2 stop voltage, p2 points), (p3
            start voltage, p3 stop voltage, p3 points)
        :param t1j: if true, does not apply a prerotation on 'n' axis for z axis measurement. This
            way the user can easily see if there are parts of the charge cell where singlets are
            decaying due to x-gate throw.
        """

        px_start, px_stop, px_pts = p_range[0]
        py_start, py_stop, py_pts = p_range[1]
        assert self.experiment_config.qubit_configs is not None
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        px_cfg = axis_cfg.gates.px
        py_cfg = axis_cfg.gates.py
        assert isinstance(px_cfg, qubit_models.ExchangeGate)
        assert isinstance(py_cfg, qubit_models.ExchangeGate)

        ne_config = experiment_models.NonEquilibriumConfig(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            gx_gen=px_cfg.gen,
            gx_gate=px_cfg.name,
            gy_gen=py_cfg.gen,
            gy_gate=py_cfg.name,
            gx_start=self.volts2dac(px_start, px_cfg.name),
            gx_stop=self.volts2dac(px_stop, px_cfg.name),
            gy_start=self.volts2dac(py_start, py_cfg.name),
            gy_stop=self.volts2dac(py_stop, py_cfg.name),
            gx_expts=px_pts,
            gy_expts=py_pts,
            qubit=eo1qubit,
            axis=axis,
            t1j=t1j,
        )
        meas = eo_single_qubit_programs_v2.NonEquilibriumCell(
            self.soccfg, reps=1, initial_delay=1, final_delay=0, cfg=ne_config
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if eo1qubit.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            ne_config,
            trigs,
            1,
            "_noneq_cell",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        sq_data.add_axis(
            [np.linspace(px_start, px_stop, px_pts)],
            "x",
            [ne_config.gx_gate],
            px_pts,
            loop_no=1,
            units=["volts"],
        )
        sq_data.add_axis(
            [np.linspace(py_start, py_stop, py_pts)],
            "y",
            [ne_config.gy_gate],
            py_pts,
            loop_no=2,
            units=["volts"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 3)

        # create logic for averaging sequence
        analysis.calculate_conductance(
            sq_data,
            self.adc_unit_conversions,
            average_level=None,
        )
        if eo1qubit.ro_cfg.reference:
            if eo1qubit.ro_cfg.thresh:
                avg_lvl = None
            else:
                avg_lvl = spinqick_enums.AverageLevel.BOTH
            analysis.calculate_difference(sq_data, average_level=avg_lvl)
        if eo1qubit.ro_cfg.thresh:
            assert eo1qubit.ro_cfg.threshold
            analysis.calculate_thresholded(
                sq_data,
                [eo1qubit.ro_cfg.threshold],
                average_level=spinqick_enums.AverageLevel.BOTH,
            )
        if self.plot:
            plot_tools.plot2_psb(sq_data, ne_config.gx_gate, ne_config.gy_gate)
            plt.ylabel(ne_config.gy_gate + "(V)")
            plt.xlabel(ne_config.gx_gate + "(V)")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def do_fingerprint(
        self,
        qubit: str,
        axis: spinqick_enums.ExchangeAxis,
        detuning_range: Tuple[float, float, int],
        exchange_range: Tuple[float, float, int],
        n_pulses: int = 1,
        point_avgs: int = 1,
        full_avgs: int = 1,
    ):
        """Perform a 2D sweep of axis detuning vs exchange gate voltage.

        :param qubit: Qubit label, describing the qubit of interest in the experiment config
        :param axis: Qubit axis (n, z or m)
        :param detuning_range: range of detuning axis sweep (start, stop, num_points)
        :param detuning_range: range of exchange axis sweep (start, stop, num_points)
        :param n_pulses: optionally play more than one pulse
        """
        d_start, d_stop, d_pts = detuning_range
        x_start, x_stop, x_pts = exchange_range

        assert self.experiment_config.qubit_configs
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        gates = axis_cfg.gates
        assert isinstance(gates, qubit_models.ExchangeGateMap)
        px_cfg = gates.px
        py_cfg = gates.py
        x_cfg = gates.x
        assert isinstance(px_cfg, qubit_models.ExchangeGate)
        assert isinstance(py_cfg, qubit_models.ExchangeGate)
        ne_config = experiment_models.FingerprintConfig(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            gx_gen=x_cfg.gen,
            gx_gate=x_cfg.name,
            gy_gen=py_cfg.gen,
            gy_gate=py_cfg.name,
            gx_start=self.volts2dac(x_start, x_cfg.name),
            gx_stop=self.volts2dac(x_stop, x_cfg.name),
            gy_start=self.volts2dac(d_start, px_cfg.name),
            gy_stop=self.volts2dac(d_stop, px_cfg.name),
            gx_expts=x_pts,
            gy_expts=d_pts,
            qubit=eo1qubit,
            axis=axis,
            n_pulses=n_pulses,
        )
        meas = eo_single_qubit_programs_v2.Fingerprint(
            self.soccfg, reps=1, initial_delay=1, final_delay=0, cfg=ne_config
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if eo1qubit.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            ne_config,
            trigs,
            1,
            "_fingerprint",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        sq_data.add_axis(
            [np.linspace(x_start, x_stop, ne_config.gx_expts)],
            "x",
            [ne_config.gx_gate],
            ne_config.gx_expts,
            loop_no=1,
            units=["volts"],
        )
        sq_data.add_axis(
            [np.linspace(d_start, d_stop, ne_config.gy_expts)],
            "y",
            [ne_config.gy_gate],
            ne_config.gy_expts,
            loop_no=2,
            units=["volts"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 3)

        analysis.calculate_conductance(
            sq_data,
            self.adc_unit_conversions,
            average_level=None,
        )
        if eo1qubit.ro_cfg.reference:
            if eo1qubit.ro_cfg.thresh:
                avg_lvl = None
            else:
                avg_lvl = spinqick_enums.AverageLevel.BOTH
            analysis.calculate_difference(sq_data, average_level=avg_lvl)
        if eo1qubit.ro_cfg.thresh:
            assert eo1qubit.ro_cfg.threshold
            analysis.calculate_thresholded(
                sq_data,
                [eo1qubit.ro_cfg.threshold],
                average_level=spinqick_enums.AverageLevel.BOTH,
            )

        if self.plot:
            plot_tools.plot2_psb(sq_data, ne_config.gx_gate, ne_config.gy_gate)
            plt.ylabel("detuning (V)")
            plt.xlabel(ne_config.gx_gate + "(V)")
            plt.title("fingerprint")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def time_sweep(
        self,
        qubit: str,
        axis: spinqick_enums.ExchangeAxis,
        max_time: float,
        time_pts: int,
        point_avgs: int,
        full_avgs: int,
    ):
        """Perform a time-rabi style measurement, sweeping pulse time.  Note that the minimum
        allowed time step is the length of one tproc clock cycle.

        :param max_time: pulse duration at the end of the sweep in microseconds
        :param time_pts: number of steps in time sweep
        """
        start, stop, points = 0.005, max_time, time_pts
        assert self.experiment_config.qubit_configs
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        x_cfg = axis_cfg.gates.x
        assert isinstance(x_cfg, qubit_models.ExchangeGate)
        nosc_cfg = experiment_models.NoscConfig(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            qubit=eo1qubit,
            axis=axis,
            start=start,
            stop=stop,
            expts=points,
        )
        meas = eo_single_qubit_programs_v2.Nosc(
            self.soccfg, reps=1, initial_delay=1, final_delay=0, cfg=nosc_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if eo1qubit.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            nosc_cfg,
            trigs,
            1,
            "_time_rabi",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        t_min_actual = self.soccfg.cycles2us(3, gen_ch=x_cfg.gen)
        times_actual = t_min_actual + np.linspace(start, stop, time_pts)
        sq_data.add_axis(
            [times_actual],
            "x",
            ["time"],
            time_pts,
            loop_no=1,
            units=["microseconds"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 2)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            eo1qubit.ro_cfg.reference,
            eo1qubit.ro_cfg.thresh,
            eo1qubit.ro_cfg.threshold,
        )

        if self.plot:
            plot_tools.plot1_psb(sq_data, "time")
            plt.xlabel("pulse time (us)")
            plt.title("time rabi")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def fid_sweep(
        self,
        qubit: str,
        axis: spinqick_enums.ExchangeAxis,
        max_time: float,
        time_pts: int,
        point_avgs: int,
        full_avgs: int,
    ):
        """Perform a free induction decay measurement. If n or m axis is selected, adds a pi pulse
        on this axis before and after the decay time. Note that the minimum allowed time step is the
        length of one tproc clock cycle.

        :param max_time: maximum delay duration in microseconds
        :param time_pts: number of steps in time sweep
        """
        start, stop, points = 0, max_time, time_pts
        assert self.experiment_config.qubit_configs
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        t_cfg = experiment_models.T2StarConfig(
            psb_cfg=eo1qubit.ro_cfg.psb_cfg,
            qubit=eo1qubit,
            dcs_cfg=eo1qubit.ro_cfg.dcs_cfg,
            reference=eo1qubit.ro_cfg.reference,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            start=start,
            stop=stop,
            expts=points,
            axis=axis,
        )
        meas = eo_single_qubit_programs_v2.T2Star(
            self.soccfg, reps=1, initial_delay=1, final_delay=0, cfg=t_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if eo1qubit.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            t_cfg,
            trigs,
            1,
            "_fid",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        t_min_actual = 3 * self.soccfg.cycles2us(3, gen_ch=3)
        times_actual = t_min_actual + np.linspace(start, stop, time_pts)
        sq_data.add_axis(
            [times_actual],
            "x",
            ["time"],
            time_pts,
            loop_no=1,
            units=["microseconds"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 2)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            eo1qubit.ro_cfg.reference,
            eo1qubit.ro_cfg.thresh,
            eo1qubit.ro_cfg.threshold,
        )
        if self.plot:
            plot_tools.plot1_psb(sq_data, "time")
            plt.xlabel("time at idle (us)")
            plt.ylabel("singlet probability")
            plt.title("fid")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def course_cal(
        self,
        qubit: str,
        axis: spinqick_enums.ExchangeAxis,
        exchange_range: Tuple[float, float, int],
        n_pulses: int = 3,
        point_avgs: int = 10,
        full_avgs: int = 10,
        fit: bool = True,
    ):
        """Perform a course calibration of exchange angle.

        :param exchange_range: range of x-gate voltages to perform the calibration over (start
            voltage, end voltage, number of points)
        :param n_pulses: number of pulses to apply for the calibration
        """

        start, stop, points = exchange_range
        assert self.experiment_config.qubit_configs
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        x_cfg = axis_cfg.gates.x
        cc_cfg = experiment_models.CourseCalConfig(
            qubit=eo1qubit,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            start=self.volts2dac(start, x_cfg.name),
            stop=self.volts2dac(stop, x_cfg.name),
            expts=points,
            n_pulses=n_pulses,
            axis=axis,
        )
        meas = eo_single_qubit_programs_v2.CourseCalComp(
            self.soccfg, reps=1, initial_delay=1, final_delay=0, cfg=cc_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if eo1qubit.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            cc_cfg,
            trigs,
            1,
            "_coursecal",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )

        sq_data.add_axis(
            [np.linspace(start, stop, points)],
            "x",
            [x_cfg.name],
            points,
            loop_no=1,
            units=["volts"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 2)

        analysis.calculate_conductance(
            sq_data,
            self.adc_unit_conversions,
            average_level=None,
        )
        if eo1qubit.ro_cfg.reference:
            if eo1qubit.ro_cfg.thresh:
                analysis.calculate_difference(sq_data, average_level=None)
            else:
                analysis.calculate_difference(
                    sq_data, average_level=spinqick_enums.AverageLevel.BOTH
                )
        if eo1qubit.ro_cfg.thresh:
            assert eo1qubit.ro_cfg.threshold
            analysis.calculate_thresholded(
                sq_data,
                [eo1qubit.ro_cfg.threshold],
                average_level=spinqick_enums.AverageLevel.BOTH,
            )
        if fit:
            angle_array, d_filt, best_fit, _, v_array, p0_array = eo_analysis.course_cal_fit(
                sq_data, n_pulses, x_cfg.name
            )
        if self.plot:
            fig = plot_tools.plot1_psb(sq_data, x_cfg.name)
            if fit:
                plt.plot(np.linspace(start, stop, points), d_filt, label="filtered data")
                plt.plot(v_array, p0_array, "*", label="extrema")
                plt.legend()
            plt.xlabel(x_cfg.name + "(V)")
            plt.title("coursecal")
            fignum = fig.number
            if fit:
                fig = plt.figure()
                plt.plot(angle_array, v_array, "o", label="estimated extrema")
                plt.plot(angle_array, best_fit, label="angle fit")
                plt.legend()
                plt.ylabel("x gate voltage")
                plt.xlabel("angle (radians)")
                fig2 = fig.number
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot(fignum)
                if fit:
                    nc.save_last_plot(fig2)
            nc.close()
        return sq_data

    @dot_experiment.updater
    def fine_cal(
        self,
        qubit: str,
        axis: spinqick_enums.ExchangeAxis,
        theta_range: Tuple[float, float, int],
        t_res: Literal["fs", "fabric"] = "fs",
        n_pulses: int = 10,
        point_avgs: int = 5,
        full_avgs: int = 2,
    ):
        """Perform a fine calibration of exchange angle.

        Use analyzed data as calibration curve
        """

        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        px_cfg = axis_cfg.gates.px
        py_cfg = axis_cfg.gates.py
        assert isinstance(px_cfg, qubit_models.ExchangeGate)
        assert isinstance(py_cfg, qubit_models.ExchangeGate)
        assert axis_cfg.exchange_cal is not None
        theta_array = np.linspace(*theta_range)
        v_array = eo_analysis.course_cal_function(
            theta_array,
            axis_cfg.exchange_cal.cal_parameters.A,
            axis_cfg.exchange_cal.cal_parameters.B,
            axis_cfg.exchange_cal.cal_parameters.theta_max,
        )
        sqd_list = []
        combined_data = np.zeros((full_avgs, len(v_array)))
        labels = []
        for avg in range(full_avgs):
            for i, x_volts in enumerate(v_array):
                fc_cfg = experiment_models.FineCalConfig(
                    qubit=eo1qubit,
                    point_avgs=point_avgs,
                    n_pulses=n_pulses,
                    axis=axis,
                    exchange_gain=self.volts2dac(x_volts, axis_cfg.gates.x.name),
                    t_res=t_res,
                )
                meas = eo_single_qubit_programs_v2.FineCalComp(
                    self.soccfg, reps=1, initial_delay=1, final_delay=0, cfg=fc_cfg
                )
                data = meas.acquire(self.soc, progress=False)
                self.soc.reset_gens()
                trigs = 2 if eo1qubit.ro_cfg.reference else 1
                sqd = spinqick_data.PsbData(
                    data,
                    fc_cfg,
                    trigs,
                    1,
                    "_finecal",
                    prog=meas,
                    voltage_state=self.vdc.all_voltages,
                )
                sqd.add_point_average(point_avgs, 0)
                analysis.analyze_psb_standard(
                    sqd,
                    self.adc_unit_conversions,
                    eo1qubit.ro_cfg.reference,
                    eo1qubit.ro_cfg.thresh,
                    eo1qubit.ro_cfg.threshold,
                    final_avg_lvl=spinqick_enums.AverageLevel.INNER,
                )
                sqd_list.append(sqd)
                if sqd.threshed_data is not None:
                    combined_data[avg, i] = sqd.threshed_data[0]
                else:
                    if sqd.difference_data is not None:
                        combined_data[avg, i] = sqd.difference_data[0]
                    else:
                        assert sqd.analyzed_data
                        combined_data[avg, i] = sqd.analyzed_data[0]
            dset_labels = [str(round(theta, ndigits=2)) + "_" + str(avg) for theta in theta_array]
            labels += dset_labels
        finecal_composite = spinqick_data.CompositeSpinqickData(
            sqd_list,
            labels,
            "_finecal",
            dset_coordinates=v_array,
            dset_coordinate_units="x_gate_volts",
            analyzed_data=combined_data,
        )
        avged_data = np.mean(combined_data, axis=0)
        fignum, theta_fit = eo_analysis.process_fine_cal(
            theta_array, v_array, avged_data, n_pulses, finecal_composite.timestamp
        )
        finecal_composite.analyzed_data = theta_fit
        if self.save_data:
            nc = finecal_composite.basic_composite_save()
            nc.save_last_plot(fignum=fignum[0])
            nc.save_last_plot(fignum=fignum[1])
        return finecal_composite
