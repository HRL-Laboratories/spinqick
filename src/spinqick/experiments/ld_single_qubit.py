"""Defines the LDSingleQubit class, which contains methods for basic single- qubit."""

import logging
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from spinqick.core import dot_experiment, spinqick_data
from spinqick.helper_functions import analysis, hardware_manager, plot_tools, spinqick_enums
from spinqick.models import experiment_models, hardware_config_models, ld_qubit_models
from spinqick.qick_code_v2 import ld_single_qubit_programs_v2, psb_setup_programs_v2

logger = logging.getLogger(__name__)


class LDSingleQubit(dot_experiment.DotExperiment):
    """Contains methods that wrap the QICK code classes for single LD qubit experiments. These
    involve a microwave drive. These scripts are set up to generate a trigger which goes high during
    the RF pulse, so if the user is mixing their signal with an LO they can trigger an RF switch.
    Initialize with information about your rfsoc and your experimental setup.

    :param soccfg : QickConfig object
    :param soc : initialized QickSoc object
    :param voltage_source: Initialized DC voltage source object. This is used here for saving the DC
        voltage state each time data is saved.
    """

    def __init__(self, soccfg, soc, voltage_source: hardware_manager.VoltageSource, **kwargs):
        super().__init__(**kwargs)
        self.soccfg = soccfg
        self.soc = soc
        self.vdc = hardware_manager.DCSource(voltage_source=voltage_source)

    @dot_experiment.updater
    def idle_cell_scan_with_rf(
        self,
        p_gates: Tuple[spinqick_enums.GateNames, spinqick_enums.GateNames],
        p_range: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        rf_gain: int,
        rf_freq: float,
        rf_length: float,
        point_avgs: int = 10,
        full_avgs: int = 1,
    ) -> spinqick_data.PsbData:
        """Performs a 2D sweep of idle coordinate with RF on.  RF is turned on a the idle point in
        the spam sequence.

        :param p_gates: specify the two plunger gates being used
        :param p_range: specify the range of each axis sweep. ((px_start, px_stop, px_points),
            (py_start, py_stop, py_points))
        :param add_rf: add an RF pulse at the idle point
        :param rf_freq: frequency of RF pulse
        :param rf_gain: gain of RF pulse
        :param rf_length: length (in microseconds) of RF pulse
        """

        px_gate, py_gate = p_gates
        px_start_voltage, px_stop_voltage, px_num_points = p_range[0]
        py_start_voltage, py_stop_voltage, py_num_points = p_range[1]

        px_start_dacval = self.volts2dac(px_start_voltage, px_gate)
        px_stop_dacval = self.volts2dac(px_stop_voltage, px_gate)
        py_start_dacval = self.volts2dac(py_start_voltage, py_gate)
        py_stop_dacval = self.volts2dac(py_stop_voltage, py_gate)
        ro_cfg = self.experiment_config.qubit_configs[self.qubit].ro_cfg
        rf_gen = self.hardware_config.rf_gen
        assert rf_gen
        px = self.hardware_config.channels[px_gate]
        py = self.hardware_config.channels[py_gate]
        assert isinstance(px, hardware_config_models.FastGate)
        assert isinstance(py, hardware_config_models.FastGate)

        idle_cfg = experiment_models.IdleScanConfig(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            dcs_cfg=self.dcs_config,
            psb_cfg=ro_cfg.psb_cfg,
            reference=ro_cfg.reference,
            thresh=ro_cfg.thresh,
            threshold=ro_cfg.threshold,
            gx_gen=px.qick_gen,
            gy_gen=py.qick_gen,
            gx_start=px_start_dacval,
            gx_stop=px_stop_dacval,
            gy_start=py_start_dacval,
            gy_stop=py_stop_dacval,
            gx_gate=px_gate,
            gy_gate=py_gate,
            gx_expts=px_num_points,
            gy_expts=py_num_points,
            add_rf=True,
            rf_gen=rf_gen,
            rf_freq=rf_freq,
            rf_gain=rf_gain,
            rf_length=rf_length,
        )

        meas = psb_setup_programs_v2.IdleScan(
            self.soccfg, reps=1, final_delay=1, initial_delay=1, cfg=idle_cfg
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(data, idle_cfg, trigs, 1, "_idle_scan_rf", prog=meas)
        sq_data.add_axis(
            [np.linspace(px_start_dacval, px_stop_dacval, px_num_points)],
            "x",
            [px_gate],
            px_num_points,
            loop_no=1,
            units=["dac units"],
        )
        sq_data.add_axis(
            [np.linspace(py_start_dacval, py_stop_dacval, py_num_points)],
            "y",
            [py_gate],
            py_num_points,
            loop_no=2,
            units=["dac units"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 3)
        analysis.calculate_conductance(
            sq_data,
            self.adc_unit_conversions,
            average_level=spinqick_enums.AverageLevel.BOTH,
        )
        if ro_cfg.reference:
            analysis.calculate_difference(sq_data)
        if ro_cfg.thresh:
            assert ro_cfg.threshold
            analysis.calculate_thresholded(sq_data, [ro_cfg.threshold])
        if self.plot:
            plot_tools.plot2_psb(sq_data, px_gate, py_gate)
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return data

    @dot_experiment.updater
    def rf_freq_scan(
        self,
        rf_gain: float,
        start_freq: float,
        stop_freq: float,
        num_pts: int,
        rf_length: float,
        point_avgs: int = 10,
        full_avgs: int = 10,
    ) -> spinqick_data.PsbData:
        """Play a frequency sweep.

        :param rf_gain: Gain of RF tone in DAC units
        :param start_freq: Lowest RF frequency in MHz
        :param stop_freq: Max RF frequency in MHz
        :param num_pts: Number of points in the frequency sweep
        :param rf_length: Pulse length of RF drive in microseconds
        :param point_avgs: Number of times to repeat each measurement and average
        :param full_avgs: Number of times to run sweep and average full experiment
        """
        assert self.experiment_config.qubit_configs is not None
        qubit_cfg = self.experiment_config.qubit_configs[self.qubit]
        rf_gen = self.hardware_config.rf_gen
        assert isinstance(rf_gen, int)
        rf_cfg = experiment_models.RfSweep(
            ro_cfg=qubit_cfg.ro_cfg,
            gen=rf_gen,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            start=start_freq,
            stop=stop_freq,
            expts=num_pts,
            pulse_gain=rf_gain,
            pulse_length=rf_length,
        )

        meas = ld_single_qubit_programs_v2.ScanRfFrequency(
            self.soccfg, reps=1, final_delay=0, cfg=rf_cfg, initial_delay=1
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            rf_cfg,
            trigs,
            1,
            "_frequency_scan",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        freq_axis = np.linspace(start_freq, stop_freq, num_pts)
        sq_data.add_axis(
            [freq_axis],
            "x",
            ["frequency"],
            num_pts,
            loop_no=1,
            units=["MHz"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 2)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
        )
        if self.plot:
            plot_tools.plot1_psb(sq_data, "frequency")
            plt.xlabel("applied frequency (MHz)")
            plt.ylabel("singlet probability")
            plt.title("frequency scan")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def rabi_chevron(
        self,
        rf_gain: int,
        freq_range: Tuple[float, float, int],
        time_range: Tuple[float, float, int],
        point_avgs: int = 10,
        full_avgs: int = 10,
    ) -> spinqick_data.PsbData:
        """Performs a 2D sweep of RF pulse frequency and pulse length.

        :param rf_gain: RF pulse amplitude in DAC units
        :param freq_range: (start frequency (MHz), stop frequency (MHz), number of steps)
        :param time_range: (start time (us), stop time (us), number of steps)
        """
        start_freq, stop_freq, freq_pts = freq_range
        start_time, stop_time, time_pts = time_range
        qubit_cfg = self.experiment_config.qubit_configs[self.qubit]
        assert isinstance(qubit_cfg, ld_qubit_models.Ld1Qubit)
        rf_gen = self.hardware_config.rf_gen
        assert isinstance(rf_gen, int)
        rc_cfg = experiment_models.RfSweepTwo(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            qubit=qubit_cfg,
            gain=rf_gain,
            gx_start=start_freq,
            gx_stop=stop_freq,
            gx_expts=freq_pts,
            gy_start=start_time,
            gy_stop=stop_time,
            gy_expts=time_pts,
        )
        meas = ld_single_qubit_programs_v2.RabiChevron(
            self.soccfg,
            reps=1,
            final_delay=1,
            initial_delay=1,
            cfg=rc_cfg,
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            rc_cfg,
            trigs,
            1,
            "_rabi_chevron",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        sq_data.add_axis(
            [np.linspace(*freq_range)],
            "x",
            ["frequency"],
            freq_pts,
            loop_no=1,
            units=["MHz"],
        )
        sq_data.add_axis(
            [np.linspace(*time_range)],
            "y",
            ["time"],
            time_pts,
            loop_no=2,
            units=["microseconds"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 3)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
            final_avg_lvl=spinqick_enums.AverageLevel.BOTH,
        )
        if self.plot:
            plot_tools.plot2_psb(sq_data, "frequency", "time")
            plt.ylabel("time (us)")
            plt.xlabel("frequency (MHz)")
            plt.title("rabi chevron")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def time_rabi(
        self,
        rf_gain: int,
        time_range: Tuple[float, float, int],
        point_avgs: int = 10,
        full_avgs: int = 10,
    ) -> spinqick_data.PsbData:
        """Performs a time rabi experiment, sweeping the length of the RF pulse time.

        :param rf_gain: RF gain in DAC units :time_range: (time_start, time_stop, num_points) in
            microseconds
        """
        start_time, stop_time, time_pts = time_range
        qubit_cfg = self.experiment_config.qubit_configs[self.qubit]
        assert isinstance(qubit_cfg, ld_qubit_models.Ld1Qubit)
        tr_cfg = experiment_models.TimeRabi(
            qubit=qubit_cfg,
            gain=rf_gain,
            start=start_time,
            stop=stop_time,
            expts=time_pts,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
        )

        meas = ld_single_qubit_programs_v2.TimeRabi(
            self.soccfg, reps=1, final_delay=0, cfg=tr_cfg, initial_delay=1
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            tr_cfg,
            trigs,
            1,
            "_time_rabi",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        times = np.linspace(start_time, stop_time, time_pts)
        sq_data.add_axis(
            [times],
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
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
        )
        if self.plot:
            plot_tools.plot1_psb(sq_data, "time")
            plt.xlabel("evolution time (microseconds)")
            plt.ylabel("singlet probability")
            plt.title("time rabi")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def amplitude_rabi(
        self,
        rf_time: float,
        gain_range: Tuple[float, float, int],
        point_avgs: int = 10,
        full_avgs: int = 1,
    ) -> spinqick_data.PsbData:
        """Performs an experiment which sweeps RF pulse gain, keeping RF pulse length constant.

        :param rf_time: pulse length in microseconds
        :param gain_range: gain sweep parameters in dac units; (start_gain, stop_gain, number of
            points)
        """
        start, stop, pts = gain_range
        qubit_cfg = self.experiment_config.qubit_configs[self.qubit]
        assert isinstance(qubit_cfg, ld_qubit_models.Ld1Qubit)
        tr_cfg = experiment_models.AmplitudeRabi(
            qubit=qubit_cfg,
            time=rf_time,
            start=start,
            stop=stop,
            expts=pts,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
        )

        meas = ld_single_qubit_programs_v2.AmplitudeRabi(
            self.soccfg, reps=1, final_delay=0, cfg=tr_cfg, initial_delay=1
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            tr_cfg,
            trigs,
            1,
            "_amplitude_rabi",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        rf_gain = np.linspace(*gain_range)
        sq_data.add_axis(
            [rf_gain],
            "x",
            ["gain"],
            pts,
            loop_no=1,
            units=["dac units"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 2)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
        )
        if self.plot:
            plot_tools.plot1_psb(sq_data, "gain")
            plt.xlabel("rf gain (dac units)")
            plt.ylabel("singlet probability")
            plt.title("amplitude rabi")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def allxy(
        self,
        qubit: str,
        point_avgs: int = 10,
        full_avgs: int = 10,
    ) -> spinqick_data.PsbData:
        """Perform an all x-y experiment.  This experiment consists of a series of manipulations to
        demonstrate x-y control.

        :param qubit: specify qubit config label
        :param point_avgs: averages per measurement point
        :param full_averages: averages of full experiment
        """

        allxy_seq = [
            ["I", "I"],
            ["X180", "X180"],
            ["Y180", "Y180"],
            ["X180", "Y180"],
            ["Y180", "X180"],
            ["X90", "I"],
            ["Y90", "I"],
            ["X90", "Y90"],
            ["Y90", "X90"],
            ["X90", "Y180"],
            ["Y90", "X180"],
            ["X180", "Y90"],
            ["Y180", "X90"],
            ["X90", "X180"],
            ["X180", "X90"],
            ["Y90", "Y180"],
            ["Y180", "Y90"],
            ["X180", "I"],
            ["Y180", "I"],
            ["X90", "X90"],
            ["Y90", "Y90"],
        ]

        qubit_cfg = self.experiment_config.qubit_configs[qubit]
        assert isinstance(qubit_cfg, ld_qubit_models.Ld1Qubit)
        xy_cfg = experiment_models.PlayXY(
            qubit=qubit_cfg,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            gate_set=allxy_seq,
        )

        meas = ld_single_qubit_programs_v2.play_xy(self.soccfg, cfg=xy_cfg)
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            xy_cfg,
            trigs,
            1,
            "_allxy",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        xarray = np.arange(len(allxy_seq))
        sq_data.add_axis(
            [xarray],
            "x",
            ["gates"],
            len(allxy_seq),
            loop_no=1,
            units=["gates"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 2)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
        )
        if self.plot:
            plot_tools.plot1_psb(sq_data, "gates")
            plt.xlabel("gates")
            plt.ylabel("singlet probability")
            plt.title("allxy")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def phase_control(
        self,
        qubit: str,
        phase_range: Tuple[float, float, int],
        point_avgs: int = 10,
        full_avgs: int = 10,
    ) -> spinqick_data.PsbData:
        """Performs two pi/2 pulses, incrementing the phase offset of the second pulse. If you are
        driving x-y rotations you will see a periodic output.  This is a simple way to demonstrate
        that you have x-y control of your qubit using RF drive phase.

        :param qubit: specify qubit config label
        :param phase_range: (start_phase, end_phase, number of points)
        """

        phase_start, phase_stop, phase_steps = phase_range
        qubit_cfg = self.experiment_config.qubit_configs[qubit]
        assert isinstance(qubit_cfg, ld_qubit_models.Ld1Qubit)
        phase_cfg = experiment_models.LdSweepOne(
            qubit=qubit_cfg,
            start=phase_start,
            stop=phase_stop,
            expts=phase_steps,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
        )

        meas = ld_single_qubit_programs_v2.SweepPhase(
            self.soccfg, reps=1, final_delay=0, cfg=phase_cfg, initial_delay=1
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            phase_cfg,
            trigs,
            1,
            "_phase_sweep",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        phase = np.linspace(phase_start, phase_stop, phase_steps)
        sq_data.add_axis(
            [phase],
            "x",
            ["phase"],
            phase_steps,
            loop_no=1,
            units=["rad"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 2)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
        )
        if self.plot:
            plot_tools.plot1_psb(sq_data, "phase")
            plt.xlabel("applied rf phase offset (radians)")
            plt.ylabel("singlet probability")
            plt.title("phase control")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def ramsey_experiment(
        self,
        qubit: str,
        time_range: Tuple[float, float, int],
        point_avgs: int = 10,
        full_avgs: int = 10,
    ) -> spinqick_data.PsbData:
        """Perform a 1D Ramsey experiment, by sweeping time delay between two pi/2 pulses.

        :param qubit: specify qubit config label
        :param time_range: (start, end, number of points) in microseconds
        """
        start_time, stop_time, time_pts = time_range
        qubit_cfg = self.experiment_config.qubit_configs[qubit]
        assert isinstance(qubit_cfg, ld_qubit_models.Ld1Qubit)
        ramsey_cfg = experiment_models.LdSweepOne(
            qubit=qubit_cfg,
            start=start_time,
            stop=stop_time,
            expts=time_pts,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
        )

        meas = ld_single_qubit_programs_v2.RamseyFringe(
            self.soccfg, reps=1, final_delay=0, cfg=ramsey_cfg, initial_delay=1
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            ramsey_cfg,
            trigs,
            1,
            "_ramsey_fringe",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        t = np.linspace(*time_range)
        sq_data.add_axis(
            [t],
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
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
        )
        if self.plot:
            plot_tools.plot1_psb(sq_data, "time")
            plt.xlabel("pulse delay time (us)")
            plt.ylabel("singlet probability")
            plt.title("ramsey fringe")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def ramsey_2d(
        self,
        qubit: str,
        time_range: Tuple[float, float, int],
        freq_range: Tuple[float, float, int],
        full_avgs: int = 10,
        point_avgs: int = 10,
    ) -> spinqick_data.PsbData:
        """Performs a series of ramsey experiments, sweeping both delay time between two pulses and
        pulse frequency.

        :param qubit: specify qubit config label
        :param time_range: (time_start, time_stop, points) in microseconds
        :param freq_range: (freq_start, freq_stop, points) in MHz
        """
        start_freq, stop_freq, freq_pts = freq_range
        start_time, stop_time, time_pts = time_range
        qubit_cfg = self.experiment_config.qubit_configs[qubit]
        assert isinstance(qubit_cfg, ld_qubit_models.Ld1Qubit)
        rf_gen = self.hardware_config.rf_gen
        assert isinstance(rf_gen, int)
        rc_cfg = experiment_models.LdSweepTwo(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            qubit=qubit_cfg,
            gx_start=start_freq,
            gx_stop=stop_freq,
            gx_expts=freq_pts,
            gy_start=start_time,
            gy_stop=stop_time,
            gy_expts=time_pts,
        )
        meas = ld_single_qubit_programs_v2.Ramsey2D(
            self.soccfg,
            reps=1,
            final_delay=1,
            initial_delay=1,
            cfg=rc_cfg,
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            rc_cfg,
            trigs,
            1,
            "_ramsey_2d",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        sq_data.add_axis(
            [np.linspace(*freq_range)],
            "x",
            ["frequency"],
            freq_pts,
            loop_no=1,
            units=["MHz"],
        )
        sq_data.add_axis(
            [np.linspace(*time_range)],
            "y",
            ["time"],
            time_pts,
            loop_no=2,
            units=["microseconds"],
        )
        sq_data.add_full_average(full_avgs)
        sq_data.add_point_average(point_avgs, 3)
        analysis.analyze_psb_standard(
            sq_data,
            self.adc_unit_conversions,
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
            final_avg_lvl=spinqick_enums.AverageLevel.BOTH,
        )
        if self.plot:
            plot_tools.plot2_psb(sq_data, "frequency", "time")
            plt.ylabel("time between pulses (us)")
            plt.xlabel("frequency (MHz)")
            plt.title("ramsey")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data

    @dot_experiment.updater
    def spin_echo(
        self,
        qubit: str,
        n_echoes: int,
        time_range: Tuple[float, float, int],
        full_avgs: int = 10,
        point_avgs: int = 10,
    ) -> spinqick_data.PsbData:
        """Performs a hahn echo or CPMG experiment.

        :param n_echoes: Number of pi pulses to insert. n=1 for Hahn echo. :time_range: (time_start,
            time_stop, points) in microseconds
        """
        start_time, stop_time, time_pts = time_range
        qubit_cfg = self.experiment_config.qubit_configs[qubit]
        assert isinstance(qubit_cfg, ld_qubit_models.Ld1Qubit)
        tr_cfg = experiment_models.SpinEcho(
            qubit=qubit_cfg,
            start=start_time,
            stop=stop_time,
            expts=time_pts,
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            n_echoes=n_echoes,
        )

        meas = ld_single_qubit_programs_v2.TimeRabi(
            self.soccfg, reps=1, final_delay=0, cfg=tr_cfg, initial_delay=1
        )
        data = meas.acquire(self.soc, progress=True)
        self.soc.reset_gens()
        trigs = 2 if qubit_cfg.ro_cfg.reference else 1
        sq_data = spinqick_data.PsbData(
            data,
            tr_cfg,
            trigs,
            1,
            "_spin_echo",
            prog=meas,
            voltage_state=self.vdc.all_voltages,
        )
        times = np.linspace(start_time, stop_time, time_pts)
        sq_data.add_axis(
            [times],
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
            qubit_cfg.ro_cfg.reference,
            qubit_cfg.ro_cfg.thresh,
            qubit_cfg.ro_cfg.threshold,
        )
        if self.plot:
            plot_tools.plot1_psb(sq_data, "time")
            plt.xlabel("evolution time (microseconds)")
            plt.ylabel("singlet probability")
            plt.title("spin echo")
        if self.save_data:
            nc = sq_data.save_data()
            if self.plot:
                nc.save_last_plot()
            nc.close()
        return sq_data
