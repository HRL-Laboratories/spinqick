"""Module to hold exchange only experiments"""

import logging
from typing import Tuple, Literal
import numpy as np
import pylab as plt
from scipy import signal, optimize
import memspectrum
from spinqick.core import dot_experiment, eo_pulses
from spinqick.core import spinqick_data, spinqick_utils
from spinqick.helper_functions import analysis, plot_tools, hardware_manager
from spinqick.settings import file_settings
from spinqick.models import experiment_models, qubit_models
from spinqick.qick_code_v2 import eo_single_qubit_programs_v2

logger = logging.getLogger(__name__)


def course_cal_function(theta, afit, bfit, theta_max):
    """fit function used for course cal, from https://doi.org/10.1038/s41565-019-0500-4"""
    return -2 * afit * np.log((theta_max - theta) / theta_max / np.sqrt(theta)) + bfit


def course_cal(threshed_data, volts_data, n_pulses):
    """perform the course calibration procedure"""
    d_filt = signal.savgol_filter(threshed_data, 10, 2)
    mval = np.max(d_filt) - (np.max(d_filt) - np.min(d_filt)) / 2
    find_pk_data = np.abs(d_filt - mval)
    maxpts, _ = signal.find_peaks(find_pk_data, width=None, prominence=0.1)
    angle_array = np.pi * (1 + np.arange(len(maxpts))) / n_pulses
    v_array = volts_data[maxpts]
    p0_array = threshed_data[maxpts]
    # pylint: disable-next=unbalanced-tuple-unpacking
    popt, _ = optimize.curve_fit(
        course_cal_function, angle_array, v_array, p0=[0.1, 1, angle_array[-1] * 2]
    )
    best_fit = course_cal_function(angle_array, *popt)
    fit_dict = {"A": popt[0], "B": popt[1], "theta_max": popt[2]}
    return angle_array, d_filt, best_fit, popt, v_array, p0_array, fit_dict, maxpts


def course_cal_fit(sqd: spinqick_data.PsbData, n_pulses: int, x_gate: str):
    """Course calibration fit, appends data to the spinqick data object"""
    threshed_data = sqd.threshed_data
    assert threshed_data is not None
    data = threshed_data[0]
    xvolts = sqd.axes["x"]["sweeps"][x_gate]["data"]
    angle_array, d_filt, best_fit, popt, v_array, p0_array, fit_dict, _ = course_cal(
        data, xvolts, n_pulses
    )
    sqd.add_fit_params(fit_dict, best_fit=d_filt, fit_axis="x")
    return angle_array, d_filt, best_fit, popt, v_array, p0_array


def process_fine_cal(
    theta_array: np.ndarray,
    voltage_array: np.ndarray,
    data_array: np.ndarray,
    n_pulses: int,
    timestamp: int,
    plot: bool = True,
):
    """fitting procedure for fine calibration from https://doi.org/10.1038/s41565-019-0500-4"""
    mesa = memspectrum.MESA()
    mesa.solve(data_array, method="standard", optimisation_method="FPE")
    extension_length = 5000
    n_sim = 200
    m1 = mesa.forecast(
        data_array[::-1], length=extension_length, number_of_simulations=n_sim
    )
    m2 = mesa.forecast(data_array, length=extension_length, number_of_simulations=n_sim)
    extended_data = np.concatenate(
        [sum([m[::-1] for m in m1]) / n_sim, data_array, sum(m2) / n_sim]
    )
    signal_freq = theta_array[-1] / len(data_array) / np.pi / 2 * n_pulses
    sos_filt = signal.butter(
        1, [signal_freq / 2, signal_freq * 4], "bandpass", output="sos"
    )
    filtered_extended = signal.sosfiltfilt(sos_filt, extended_data)
    fignums = []
    if plot:
        fig1 = plot_tools.plot1_simple(
            theta_array,
            filtered_extended[extension_length : extension_length + len(data_array)]
            + np.mean(data_array),
            timestamp,
            dset_label="extended",
        )
        plot_tools.plot1_simple(
            theta_array, data_array, timestamp, dset_label="fit", new_figure=False
        )
        plt.xlabel("estimated theta (rad)")
        plt.ylabel("singlet probability")
        plt.legend()
        first_fignum = fig1.number
        fignums.append(first_fignum)
    filtered_transformed = signal.hilbert(filtered_extended)
    theta_fit = (
        np.unwrap(
            np.angle(filtered_transformed)[
                extension_length : extension_length + len(data_array)
            ]
        )
        / n_pulses
    )
    if plot:
        fig2 = plot_tools.plot1_simple(
            voltage_array, theta_fit, timestamp, dset_label="new theta vals"
        )
        plt.plot(voltage_array, theta_array, "o", label="initial theta vals")
        plt.yscale("log")
        plt.xlabel("x gate voltage (V)")
        plt.ylabel("theta (rad)")
        plt.legend()
        fignum = fig2.number
        fignums.append(fignum)
    return fignums, theta_fit


def fine_cal_voltage(
    theta: float,
    theta_list: list,
    voltage_list: list,
    afit: float,
    bfit: float,
    theta_max: float,
):
    """finecal interpolation function from https://doi.org/10.1038/s41565-019-0500-4"""
    theta_sampling = theta_list[-1] - theta_list[-2]
    mask = [np.abs(theta - t) < theta_sampling for t in theta_list]
    theta_array = np.array(theta_list)
    voltage_array = np.array(voltage_list)
    t_range = theta_array[mask]
    v_range = voltage_array[mask]
    alpha = (theta - t_range[0]) / (t_range[1] - t_range[0])
    exp_i = np.exp((bfit - v_range[0]) / (2 * afit))
    exp_j = np.exp((bfit - v_range[1]) / (2 * afit))
    f_inv_i = (theta_max * (-1 * exp_i + np.sqrt(exp_i**2 + 4 / theta_max)) / 2) ** 2
    f_inv_j = (theta_max * (-1 * exp_j + np.sqrt(exp_j**2 + 4 / theta_max)) / 2) ** 2
    theta_adj = (1 - alpha) * f_inv_i + alpha * f_inv_j
    vfinal = course_cal_function(theta_adj, afit, bfit, theta_max)
    return vfinal


class EOSingleQubit(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK classes for setting up EO single qubit experiments."""

    def __init__(
        self,
        soccfg,
        soc,
        voltage_source: hardware_manager.VoltageSource,
        datadir=file_settings.data_directory,
    ):
        super().__init__(datadir=datadir)
        self.soccfg = soccfg
        self.soc = soc
        self.datadir = datadir
        self.save_data = True
        self.plot = True
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
        """calculate detuning and exchange axes.  Returns the vectors after dac unit conversion!"""
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
        detuning, exchange = eo_pulses.define_fingerprint_vectors(
            np.array(px_points), np.array(py_points), np.array(idle_points), x_gain
        )
        return detuning, exchange

    @dot_experiment.updater
    def calculate_exchange_gate_vals(
        self, qubit: str, axis: str, detuning: float, x_throw: float
    ):
        """calculate exchange gain at gates in rfsoc units, given detuning and x_throw in volts"""
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
        gate_vals = eo_pulses.calculate_fingerprint_gate_vals(
            detuning_convert,
            x_convert,
            np.array(axis_cfg.detuning_vector),
            np.array(axis_cfg.exchange_vector),
            idle_points,
        )
        return gate_vals

    @dot_experiment.updater
    def calc_symmetric_vector(
        self, qubit: str, axis: str, detuning_volts: float, x_volts: float
    ):
        """calculate detuning and exchange axes.  Returns the vectors after dac unit conversion!"""
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
        x_gain = self.volts2dac(
            (x_volts - gates.x.gains.idle_gain), axis_cfg.gates.x.name
        )
        gate_vals = eo_pulses.calculate_fingerprint_gate_vals(
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
    def calculate_exchange_volts(
        self, qubit: str, axis: str, detuning: float, x_throw: float
    ):
        """calculate exchange gain at gates in voltage units"""
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
        """use finecal data to convert theta to gate voltages"""
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        cal = axis_cfg.exchange_cal
        assert cal is not None
        tlist = cal.cal_parameters.theta_list
        vlist = cal.cal_parameters.volt_list
        assert tlist is not None
        assert vlist is not None
        v_x = fine_cal_voltage(
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
        axis: Literal["n", "z", "m"],
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        point_avgs: int = 1,
        full_avgs: int = 1,
        t1j: bool = False,
    ):
        """Perform a scan of the non-equilibrium cell.  This helps us to set the detuning axis for a fingerprint plot.

        :param qubit: Qubit label, describing the qubit of interest in the experiment config
        :param axis: Qubit axis (n, z or m)
        :param p_range: range of p gate sweeps: (p2 start voltage, p2 stop voltage, p2 points), (p3 start voltage, p3 stop voltage, p3 points)

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
                avg_lvl = spinqick_utils.AverageLevel.BOTH
            analysis.calculate_difference(sq_data, average_level=avg_lvl)
        if eo1qubit.ro_cfg.thresh:
            assert eo1qubit.ro_cfg.threshold
            analysis.calculate_thresholded(
                sq_data,
                [eo1qubit.ro_cfg.threshold],
                average_level=spinqick_utils.AverageLevel.BOTH,
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
        axis: Literal["n", "z", "m"],
        detuning_range: Tuple[float, float, int],
        exchange_range: Tuple[float, float, int],
        n_pulses: int = 1,
        point_avgs: int = 1,
        full_avgs: int = 1,
    ):
        """
        :param qubit: Qubit label, describing the qubit of interest in the experiment config
        :param axis: Qubit axis (n, z or m)
        :param detuning_range: range of detuning axis sweep (start, stop, num_points)
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
                avg_lvl = spinqick_utils.AverageLevel.BOTH
            analysis.calculate_difference(sq_data, average_level=avg_lvl)
        if eo1qubit.ro_cfg.thresh:
            assert eo1qubit.ro_cfg.threshold
            analysis.calculate_thresholded(
                sq_data,
                [eo1qubit.ro_cfg.threshold],
                average_level=spinqick_utils.AverageLevel.BOTH,
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
        axis: Literal["n", "z", "m"],
        max_time: float,
        time_pts: int,
        point_avgs: int,
        full_avgs: int,
    ):
        """Perform a time-rabi style measurement"""
        start, stop, points = 0, max_time, time_pts
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
        t_min_actual = 3 * self.soccfg.cycles2us(3, gen_ch=x_cfg.gen)
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
        max_time: float,
        time_pts: int,
        point_avgs: int,
        full_avgs: int,
    ):
        """Perform a free induction decay measurement"""
        start, stop, points = 0, max_time, time_pts
        assert self.experiment_config.qubit_configs
        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        t_cfg = experiment_models.T2StarConfig(
            psb_cfg=eo1qubit.ro_cfg.psb_cfg,
            dcs_cfg=eo1qubit.ro_cfg.dcs_cfg,
            reference=eo1qubit.ro_cfg.reference,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            start=start,
            stop=stop,
            expts=points,
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
        axis: Literal["n", "z", "m"],
        exchange_range: Tuple[float, float, int],
        n_pulses: int = 3,
        point_avgs: int = 10,
        full_avgs: int = 10,
    ):
        """perform a course calibration of exchange angle"""

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
        meas = eo_single_qubit_programs_v2.CourseCal(
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
                    sq_data, average_level=spinqick_utils.AverageLevel.BOTH
                )
        if eo1qubit.ro_cfg.thresh:
            assert eo1qubit.ro_cfg.threshold
            analysis.calculate_thresholded(
                sq_data,
                [eo1qubit.ro_cfg.threshold],
                average_level=spinqick_utils.AverageLevel.BOTH,
            )

        angle_array, d_filt, best_fit, _, v_array, p0_array = course_cal_fit(
            sq_data, n_pulses, x_cfg.name
        )
        if self.plot:
            fig = plot_tools.plot1_psb(sq_data, x_cfg.name)
            plt.plot(np.linspace(start, stop, points), d_filt, label="filtered data")
            plt.plot(v_array, p0_array, "*", label="extrema")
            plt.legend()
            plt.xlabel(x_cfg.name + "(V)")
            plt.title("coursecal")
            fignum = fig.number
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
                nc.save_last_plot(fig2)
            nc.close()
        return sq_data

    @dot_experiment.updater
    def fine_cal(
        self,
        qubit: str,
        axis: Literal["n", "z", "m"],
        theta_range: Tuple[float, float, int],
        t_res: Literal["fs", "fabric"] = "fabric",
        n_pulses: int = 10,
        point_avgs: int = 10,
    ):
        """perform a fine calibration of exchange angle.  Use analyzed data as calibration curve"""

        eo1qubit = self.experiment_config.qubit_configs[qubit]
        assert isinstance(eo1qubit, qubit_models.Eo1Qubit)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(eo1qubit, axis)
        px_cfg = axis_cfg.gates.px
        py_cfg = axis_cfg.gates.py
        assert isinstance(px_cfg, qubit_models.ExchangeGate)
        assert isinstance(py_cfg, qubit_models.ExchangeGate)
        assert axis_cfg.exchange_cal is not None
        theta_array = np.linspace(*theta_range)
        v_array = course_cal_function(
            theta_array,
            axis_cfg.exchange_cal.cal_parameters.A,
            axis_cfg.exchange_cal.cal_parameters.B,
            axis_cfg.exchange_cal.cal_parameters.theta_max,
        )
        sqd_list = []
        combined_data = np.zeros(len(v_array))
        for i, x_volts in enumerate(v_array):
            fc_cfg = experiment_models.FineCalConfig(
                qubit=eo1qubit,
                point_avgs=point_avgs,
                n_pulses=n_pulses,
                axis=axis,
                exchange_gain=self.volts2dac(x_volts, axis_cfg.gates.x.name),
                t_res=t_res,
            )
            meas = eo_single_qubit_programs_v2.FineCal(
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
                final_avg_lvl=spinqick_utils.AverageLevel.INNER,
            )
            sqd_list.append(sqd)
            if sqd.threshed_data is not None:
                combined_data[i] = sqd.threshed_data[0]
            else:
                if sqd.difference_data is not None:
                    combined_data[i] = sqd.difference_data[0]
                else:
                    assert sqd.analyzed_data
                    combined_data[i] = sqd.analyzed_data[0]
        dset_labels = [str(round(theta, ndigits=2)) for theta in theta_array]
        finecal_composite = spinqick_data.CompositeSpinqickData(
            sqd_list,
            dset_labels,
            "_finecal",
            dset_coordinates=v_array,
            dset_coordinate_units="x_gate_volts",
            analyzed_data=combined_data,
        )
        fignum, theta_fit = process_fine_cal(
            theta_array, v_array, combined_data, n_pulses, finecal_composite.timestamp
        )
        finecal_composite.analyzed_data = theta_fit
        if self.save_data:
            nc = finecal_composite.basic_composite_save()
            nc.save_last_plot(fignum=fignum[0])
            nc.save_last_plot(fignum=fignum[1])
        return finecal_composite
