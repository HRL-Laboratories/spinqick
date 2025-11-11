"""Defines the PsbSetup class which is used for setting up Pauli Spin Blockade spin-to-charge
conversion."""

import logging
from typing import Tuple

import matplotlib.patches as mpl_patches
import numpy as np
from matplotlib import pyplot as plt
from qick import QickConfig
from scipy import optimize
from scipy.stats import norm

from spinqick import settings
from spinqick.core import dot_experiment, spinqick_data
from spinqick.helper_functions import (
    analysis,
    hardware_manager,
    plot_tools,
    spinqick_enums,
)
from spinqick.models import experiment_models, hardware_config_models
from spinqick.qick_code_v2 import psb_setup_programs_v2

logger = logging.getLogger(__name__)


class PsbSetup(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK classes for setting up PSB readout. Initialize
    with information about your rfsoc and your experimental setup.

    :param soccfg: qick config object (QickConfig)
    :param soc: QickSoc object
    :param voltage_source: Initialized DC voltage source object. This is used here for saving the DC
        voltage state each time data is saved.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        soc,
        voltage_source: hardware_manager.VoltageSource,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.soccfg = soccfg
        self.soc = soc
        self.reference = True
        self.thresh = True
        self.threshold = 0
        self.vdc = hardware_manager.DCSource(voltage_source=voltage_source)

    @property
    def psb_config(self):
        """Get psb config for selected qubit from experiment config."""
        try:
            qubits = self.experiment_config.qubit_configs
            assert isinstance(qubits, dict)
            qubit = qubits[self.qubit]
            assert qubit is not None
            ro = qubit.ro_cfg
        except AssertionError:
            logger.error("specified qubit parameters not found in experiment config")
        ### maybe separate these into their own properties
        self.thresh = ro.thresh
        self.threshold = ro.threshold
        self.reference = ro.reference
        return ro.psb_cfg

    @dot_experiment.updater
    def meashist(
        self,
        num_measurements: int,
        reference: bool = True,
        fit: bool = True,
        n_gaussians: int = 3,
        use_gmm: bool = False,
        log_scale: bool = False,
    ):
        """Perform some number of spam sequences and produce a measurement histogram to show
        relative prevalance of measured 0 and 1 states.  This simply plays the spam sequence as
        defined in the config.

        :param num_measurements: Total number of measurements
        :param reference: perform a reference measurement and report the difference between the two
            measurements
        :param fit: if True, fit results to a pair of gaussians
        :param use_gmm: use gaussian mixtures model to fit gaussians :log_scale: if true, plot y
            axis in log scale
        """

        mh_cfg = experiment_models.MeashistConfig(
            dcs_cfg=self.dcs_config,
            psb_cfg=self.psb_config,
            reference=reference,
            thresh=False,
            threshold=None,
            num_measurements=num_measurements,
        )

        meas = psb_setup_programs_v2.MeasHist(
            self.soccfg, reps=1, initial_delay=1, final_delay=0, cfg=mh_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        assert data is not None
        trigs = 2 if reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            mh_cfg,
            trigs,
            1,
            "_meas_hist",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        analysis.calculate_conductance(sq_data, self.adc_unit_conversions)
        if reference:
            analysis.calculate_difference(sq_data)
        sq_data.add_axis(
            [np.arange(num_measurements) for i in data],
            "shots",
            ["shots"],
            num_measurements,
        )
        plot_data = sq_data.difference_data if reference else sq_data.analyzed_data
        if plot_data is not None:
            if self.plot and not use_gmm:
                hist, bins = np.histogram(
                    plot_data[0], bins=100, range=None, weights=None
                )
                x = bins + (bins[1] - bins[0]) / 2
                plot_tools.plot1_simple(x[:-1], hist, sq_data.timestamp)
            if fit:
                if use_gmm:
                    plot_data_gmm = np.squeeze(plot_data[0])
                    mean, covs, weights = analysis.fit_blobs(plot_data_gmm, n_gaussians)
                    # mus = mean[0, :2]
                    # mus_locs = np.argsort(mus)
                    # x_begin = mus[mus_locs][0] - 4 * np.sqrt(covs[0, mus_locs[0]])
                    # x_end = mus[mus_locs][1] + 4 * np.sqrt(covs[0, mus_locs[1]])
                    x_begin = np.min(plot_data_gmm)
                    x_end = np.max(plot_data_gmm)
                    x_axis = np.linspace(x_begin, x_end, 50)
                    hist, bins = np.histogram(plot_data_gmm, density=True, bins=x_axis)
                    bins = bins + (bins[1] - bins[0]) / 2

                    yaxis0 = (
                        norm.pdf(x_axis, mean[0, 0], np.sqrt(covs[0, 0]))
                        * weights[0, 0]
                    )
                    yaxis1 = (
                        norm.pdf(x_axis, mean[0, 1], np.sqrt(covs[0, 1]))
                        * weights[0, 1]
                    )

                    if self.plot:
                        _, ax = plt.subplots()
                        ax.bar(
                            bins[:-1],
                            hist,
                            color=np.array([0.4, 0.5, 0.95, 0.5]),
                            width=0.6,
                            edgecolor="k",
                        )
                        ax.plot(x_axis, yaxis0, color="k", ls="--", lw=2)
                        ax.plot(x_axis, yaxis1, color="k", ls="--", lw=2)
                        ax.set_title(
                            "t: %d" % sq_data.timestamp,
                            loc="right",
                            fontdict={"fontsize": 6},
                        )
                        ax.set_xlabel("DCS conductance (arbs)")
                        ax.set_ylabel("Probability")
                        if log_scale:
                            ax.set_yscale("log")
                            ax.set_ylim(1e-3, 1e-1)

                        snr = (
                            2
                            * np.abs(mean[0, 1] - mean[0, 0])
                            / (np.sqrt(covs[0, 1]) + np.sqrt(covs[0, 0]))
                        )
                        pops = weights[0][[0, 1]][np.argsort(mean[0][[0, 1]])]
                        pops = 1e2 * pops[[0, 1]] / (pops[0] + pops[1])
                        # leakage = 1e2 * np.sum(weights[0, 2:]) / np.sum(weights)
                        thresh_sign = np.sign(np.sum(weights))
                        threshold_val = (
                            np.abs((mean[0, 1] - mean[0, 0]) / 2) * thresh_sign
                        )

                        labels = [
                            f"SNR = {snr:.1f}",
                            f"Pop. = {int(pops[0])} %/ {int(pops[1])} %",
                            f"threshold = {threshold_val:.2f}",
                        ]
                        handles = [
                            mpl_patches.Rectangle(
                                (0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0
                            )
                        ] * len(labels)
                        ax.legend(handles, labels, loc="best", frameon=False)
                else:

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
                    ]  # TODO make this more robust
                    try:
                        # pylint: disable-next=unbalanced-tuple-unpacking
                        popt, _ = optimize.curve_fit(gauss, x[:-1], hist, p0=guess)
                        if self.plot:
                            plt.plot(x[:-1], gauss(x[:-1], *popt))
                            plt.xlabel("DCS conductance (arbs)")
                        print("sigma1: %f, sigma2: %f" % (popt[1], popt[4]))
                        print(
                            "fwhm1: %f, fwhm2: %f" % (popt[1] * 2.355, popt[4] * 2.355)
                        )
                        print("mu1: %f, mu2: %f" % (popt[2], popt[5]))
                        print("A1: %f, A2: %f" % (popt[0], popt[3]))
                        snr = 2 * np.abs(popt[2] - popt[5]) / (popt[1] + popt[4])
                        print("SNR: %f" % snr)
                    except RuntimeError:
                        logger.error("fit failed")

        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()

        return sq_data

        """TODO 2d sweep of readout parameters for optimizing SNR."""

    @dot_experiment.updater
    def idle_cell_scan(
        self,
        p_gates: Tuple[settings.GateNames, settings.GateNames],
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        point_avgs: int = 10,
        full_avgs: int = 10,
    ):
        """Performs a 2D sweep of idle coordinate.

        :param p_gates: specify the two plunger gates being used
        :param p_range: specify the range of each axis sweep. ((px_start, px_stop, px_points),
            (py_start, py_stop, py_points))
        """

        px_gate, py_gate = p_gates
        px_start_voltage, px_stop_voltage, px_num_points = p_range[0]
        py_start_voltage, py_stop_voltage, py_num_points = p_range[1]

        px_start_dacval = self.volts2dac(px_start_voltage, px_gate)
        px_stop_dacval = self.volts2dac(px_stop_voltage, px_gate)
        py_start_dacval = self.volts2dac(py_start_voltage, py_gate)
        py_stop_dacval = self.volts2dac(py_stop_voltage, py_gate)

        x_cfg = self.hardware_config.channels[px_gate]
        y_cfg = self.hardware_config.channels[py_gate]
        assert isinstance(x_cfg, hardware_config_models.FastGate)
        assert isinstance(y_cfg, hardware_config_models.FastGate)

        idle_cfg = experiment_models.PsbScanConfig(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            dcs_cfg=self.dcs_config,
            psb_cfg=self.psb_config,
            reference=self.reference,
            thresh=self.thresh,
            threshold=self.threshold,
            gx_gate=px_gate,
            gy_gate=py_gate,
            gx_gen=x_cfg.qick_gen,
            gy_gen=y_cfg.qick_gen,
            gx_start=px_start_dacval,
            gx_stop=px_stop_dacval,
            gy_start=py_start_dacval,
            gy_stop=py_stop_dacval,
            gy_expts=py_num_points,
            gx_expts=px_num_points,
        )

        meas = psb_setup_programs_v2.IdleScan(
            self.soccfg, reps=1, final_delay=1, initial_delay=1, cfg=idle_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if self.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            idle_cfg,
            trigs,
            1,
            "_idle_scan",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        sq_data.add_axis(
            [np.linspace(px_start_voltage, px_stop_voltage, px_num_points)],
            "x",
            [px_gate],
            px_num_points,
            loop_no=1,
            units=["volts"],
        )
        sq_data.add_axis(
            [np.linspace(py_start_voltage, py_stop_voltage, py_num_points)],
            "y",
            [py_gate],
            py_num_points,
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
        if self.reference:
            if self.thresh:
                avg_lvl = None
            else:
                avg_lvl = spinqick_enums.AverageLevel.BOTH
            analysis.calculate_difference(sq_data, average_level=avg_lvl)
        if self.thresh:
            assert self.threshold
            analysis.calculate_thresholded(
                sq_data,
                [self.threshold],
                average_level=spinqick_enums.AverageLevel.BOTH,
            )
        if self.plot:
            plot_tools.plot2_psb(sq_data, px_gate, py_gate)
            plt.ylabel(py_gate + "(V)")
            plt.xlabel(px_gate + "(V)")
            plt.title("idle cell scan")
            qubits = self.experiment_config_params.qubit_configs
            assert qubits is not None
            idle_x = qubits[self.qubit].ro_cfg.psb_cfg.idle.gate_list[px_gate].voltage
            idle_y = qubits[self.qubit].ro_cfg.psb_cfg.idle.gate_list[py_gate].voltage
            plt.plot([idle_x], [idle_y], "o")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def flush_window_scan(
        self,
        p_gates: Tuple[settings.GateNames, settings.GateNames],
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        point_avgs: int = 10,
        full_avgs: int = 10,
    ):
        """Performs a 2D sweep of flush coordinate.

        :param p_gates: specify the two plunger gates being used
        :param p_range: specify the range of each axis sweep. ((px_start, px_stop, px_points),
            (py_start, py_stop, py_points))
        """

        px_gate, py_gate = p_gates
        px_start_voltage, px_stop_voltage, px_num_points = p_range[0]
        py_start_voltage, py_stop_voltage, py_num_points = p_range[1]

        px_start_dacval = self.volts2dac(px_start_voltage, px_gate)
        px_stop_dacval = self.volts2dac(px_stop_voltage, px_gate)
        py_start_dacval = self.volts2dac(py_start_voltage, py_gate)
        py_stop_dacval = self.volts2dac(py_stop_voltage, py_gate)

        x_cfg = self.hardware_config.channels[px_gate]
        y_cfg = self.hardware_config.channels[py_gate]
        assert isinstance(x_cfg, hardware_config_models.FastGate)
        assert isinstance(y_cfg, hardware_config_models.FastGate)

        flush_cfg = experiment_models.PsbScanConfig(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            dcs_cfg=self.dcs_config,
            psb_cfg=self.psb_config,
            reference=self.reference,
            thresh=self.thresh,
            threshold=self.threshold,
            gx_gate=px_gate,
            gy_gate=py_gate,
            gx_gen=x_cfg.qick_gen,
            gy_gen=y_cfg.qick_gen,
            gx_start=px_start_dacval,
            gx_stop=px_stop_dacval,
            gy_start=py_start_dacval,
            gy_stop=py_stop_dacval,
            gy_expts=py_num_points,
            gx_expts=px_num_points,
        )

        meas = psb_setup_programs_v2.FlushScan(
            self.soccfg,
            reps=1,
            final_delay=1,
            initial_delay=1,
            cfg=flush_cfg,
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if self.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            flush_cfg,
            trigs,
            1,
            "_flush_scan",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        sq_data.add_axis(
            [np.linspace(px_start_voltage, px_stop_voltage, px_num_points)],
            "x",
            [px_gate],
            px_num_points,
            loop_no=1,
            units=["dac_units"],
        )
        sq_data.add_axis(
            [np.linspace(py_start_voltage, py_stop_voltage, py_num_points)],
            "y",
            [py_gate],
            py_num_points,
            loop_no=2,
            units=["dac_units"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 3)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            self.reference,
            self.thresh,
            self.threshold,
            final_avg_lvl=spinqick_enums.AverageLevel.BOTH,
        )
        if self.plot:
            plot_tools.plot2_psb(sq_data, px_gate, py_gate)
            plt.ylabel(py_gate + "(V)")
            plt.xlabel(px_gate + "(V)")
            plt.title("flush window scan")
            qubits = self.experiment_config_params.qubit_configs
            assert qubits is not None
            f_x = qubits[self.qubit].ro_cfg.psb_cfg.flush.gate_list[px_gate].voltage
            f_y = qubits[self.qubit].ro_cfg.psb_cfg.flush.gate_list[py_gate].voltage
            plt.plot([f_x], [f_y], "o")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def meas_window_scan(
        self,
        p_gates: Tuple[settings.GateNames, settings.GateNames],
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        step_time: float,
        point_avgs: int = 10,
        full_avgs: int = 10,
    ):
        """Performs a 2D sweep of the measurement coordinate.  Here we are using a series of fast
        steps to approximate a ramp.

        :param p_gates: specify the two plunger gates being used
        :param p_range: specify the range of each axis sweep. ((px_start, px_stop, px_points),
            (py_start, py_stop, py_points))
        :param step_time: time per step in each ramp, in microseconds. This uses a series of steps
            to approximate a ramp because it is challenging to code a sweep of the start and
            endpoints of a ramp in qick api.
        """

        px_gate, py_gate = p_gates
        px_start_voltage, px_stop_voltage, px_num_points = p_range[0]
        py_start_voltage, py_stop_voltage, py_num_points = p_range[1]

        px_start_dacval = self.volts2dac(px_start_voltage, px_gate)
        px_stop_dacval = self.volts2dac(px_stop_voltage, px_gate)
        py_start_dacval = self.volts2dac(py_start_voltage, py_gate)
        py_stop_dacval = self.volts2dac(py_stop_voltage, py_gate)

        x_cfg = self.hardware_config.channels[px_gate]
        y_cfg = self.hardware_config.channels[py_gate]
        assert isinstance(x_cfg, hardware_config_models.FastGate)
        assert isinstance(y_cfg, hardware_config_models.FastGate)

        meas_cfg = experiment_models.MeasScanConfig(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            dcs_cfg=self.dcs_config,
            psb_cfg=self.psb_config,
            reference=self.reference,
            thresh=self.thresh,
            threshold=self.threshold,
            gx_gate=px_gate,
            gy_gate=py_gate,
            gx_gen=x_cfg.qick_gen,
            gy_gen=y_cfg.qick_gen,
            gx_start=px_start_dacval,
            gx_stop=px_stop_dacval,
            gy_start=py_start_dacval,
            gy_stop=py_stop_dacval,
            gy_expts=py_num_points,
            gx_expts=px_num_points,
            step_time=step_time,
        )

        meas = psb_setup_programs_v2.MeasureScanStepRamp(
            self.soccfg, reps=1, final_delay=1, initial_delay=1, cfg=meas_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if self.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            meas_cfg,
            trigs,
            1,
            "_meas_scan",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        sq_data.add_axis(
            [np.linspace(px_start_voltage, px_stop_voltage, px_num_points)],
            "x",
            [px_gate],
            px_num_points,
            loop_no=1,
            units=["V"],
        )
        sq_data.add_axis(
            [np.linspace(py_start_voltage, py_stop_voltage, py_num_points)],
            "y",
            [py_gate],
            py_num_points,
            loop_no=2,
            units=["V"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 3)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            self.reference,
            self.thresh,
            self.threshold,
            final_avg_lvl=spinqick_enums.AverageLevel.BOTH,
        )
        if self.plot:
            plot_tools.plot2_psb(sq_data, px_gate, py_gate)
            plt.ylabel(py_gate + "(V)")
            plt.xlabel(px_gate + "(V)")
            plt.title("meas window scan")
            qubits = self.experiment_config_params.qubit_configs
            assert qubits is not None
            f_x = qubits[self.qubit].ro_cfg.psb_cfg.meas.gate_list[px_gate].voltage
            f_y = qubits[self.qubit].ro_cfg.psb_cfg.meas.gate_list[py_gate].voltage
            plt.plot([f_x], [f_y], "o")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data
