"""Single DFS qubit setup and control."""

import numpy as np
from qick import asm_v2

from spinqick.core import eo_pulses, readout_v2
from spinqick.experiments import eo_analysis
from spinqick.helper_functions import spinqick_enums
from spinqick.models import experiment_models, qubit_models


class T2Star(asm_v2.AveragerProgramV2):
    """Go to idle, sweep dephase time and measure singlet probability."""

    def _initialize(self, cfg: experiment_models.T2StarConfig):
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.psb_cfg)
        if cfg.axis in ["n", "m"]:
            eo_pulses.setup_pi_pulse(self, cfg.qubit, [cfg.axis], "fine")
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("dephase", cfg.expts)
        self.add_loop("shots", cfg.point_avgs)

    def _body(self, cfg: experiment_models.T2StarConfig):
        if cfg.reference:
            readout_v2.psb_fm(self, cfg.psb_cfg, cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.psb_cfg)
        if cfg.axis in ["n", "m"]:
            eo_pulses.play_pi(self, cfg.axis, cfg.qubit, t=0, t_res="fine")
        self.delay_auto(asm_v2.QickSweep1D("dephase", cfg.start, cfg.stop))  # type: ignore
        if cfg.axis in ["n", "m"]:
            eo_pulses.play_pi(self, cfg.axis, cfg.qubit, t=0, t_res="fine")
        self.delay_auto()
        readout_v2.psb_em(self, cfg.psb_cfg, cfg.dcs_cfg)


class NonEquilibriumCell(asm_v2.AveragerProgramV2):
    """Turn on exchange and scan the non-equilibrium cell."""

    def _initialize(self, cfg: experiment_models.NonEquilibriumConfig):
        x_gains = asm_v2.QickSweep1D("x_sweep", cfg.gx_start, cfg.gx_stop)
        y_gains = asm_v2.QickSweep1D("y_sweep", cfg.gy_start, cfg.gy_stop)
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        eo_pulses.setup_eo_gens(self, cfg.qubit)
        if cfg.axis == "z" and not cfg.t1j:
            eo_pulses.setup_pi_pulse(
                self, cfg.qubit, [spinqick_enums.ExchangeAxis.N], "fine"
            )
        eo_pulses.setup_evol_sweep(
            self, cfg.qubit, [cfg.axis], {cfg.axis: {"px": x_gains, "py": y_gains}}
        )
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("x_sweep", cfg.gx_expts)
        self.add_loop("y_sweep", cfg.gy_expts)
        self.add_loop("shots", cfg.point_avgs)

    def _body(self, cfg: experiment_models.NonEquilibriumConfig):
        x_axis_cfg: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        idle_time = eo_pulses.round_up_pulses(x_axis_cfg.times.idle_time, self.soccfg)
        exchange_time = eo_pulses.round_up_pulses(
            x_axis_cfg.times.exchange_time, self.soccfg
        )
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto(idle_time / 2)
        if cfg.axis == "z" and not cfg.t1j:
            eo_pulses.play_pi(
                self, spinqick_enums.ExchangeAxis.N, cfg.qubit, t=0, t_res="fine"
            )
            t_pulse = exchange_time + idle_time * 1.5
        else:
            t_pulse = 0
        eo_pulses.play_evol_course(self, cfg.axis, cfg.qubit, t=t_pulse)
        if cfg.axis == "z" and not cfg.t1j:
            t_second_swap = t_pulse + idle_time / 2 + exchange_time
            eo_pulses.play_pi(
                self,
                spinqick_enums.ExchangeAxis.N,
                cfg.qubit,
                t=t_second_swap,
                t_res="fine",
            )
        self.delay_auto(idle_time)
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)


class Fingerprint(asm_v2.AveragerProgramV2):
    """Scan the gain of the exchange pulse while sweeping along a detuning axis and symmetric
    axis."""

    def _initialize(self, cfg: experiment_models.FingerprintConfig):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        eo_pulses.setup_eo_gens(self, cfg.qubit)

        # calculate actual points from detuning
        exp_axis: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        gates = exp_axis.gates
        assert isinstance(gates, qubit_models.ExchangeGateMap)
        idle_points = [
            gates.px.gains.idle_gain,
            gates.py.gains.idle_gain,
            gates.x.gains.idle_gain,
        ]
        detuning_vector = np.array(exp_axis.detuning_vector)
        symmetric_vector = np.array(exp_axis.exchange_vector)

        first_point = eo_analysis.calculate_fingerprint_gate_vals(
            cfg.gy_start, cfg.gx_start, detuning_vector, symmetric_vector, idle_points
        )
        last_point_first_loop = eo_analysis.calculate_fingerprint_gate_vals(
            cfg.gy_stop, cfg.gx_start, detuning_vector, symmetric_vector, idle_points
        )
        first_point_last_loop = eo_analysis.calculate_fingerprint_gate_vals(
            cfg.gy_start, cfg.gx_stop, detuning_vector, symmetric_vector, idle_points
        )

        dy = last_point_first_loop - first_point
        dx = first_point_last_loop - first_point

        # setup the sweeps
        x_gains = asm_v2.QickSweep1D("x_sweep", first_point[2], first_point[2] + dx[2])
        px_gains = (
            first_point[0]
            + asm_v2.QickSpan("y_sweep", span=dy[0])
            + asm_v2.QickSpan("x_sweep", span=dx[0])
        )
        py_gains = (
            first_point[1]
            + asm_v2.QickSpan("y_sweep", span=dy[1])
            + asm_v2.QickSpan("x_sweep", span=dx[1])
        )

        if cfg.axis == "z":
            eo_pulses.setup_evol_sweep(
                self,
                cfg.qubit,
                [spinqick_enums.ExchangeAxis.N, spinqick_enums.ExchangeAxis.Z],
                {"z": {"px": px_gains, "py": py_gains, "x": x_gains}},
            )
            eo_pulses.setup_pi_pulse(
                self, cfg.qubit, [spinqick_enums.ExchangeAxis.N], "fine"
            )
        else:
            eo_pulses.setup_evol_sweep(
                self,
                cfg.qubit,
                [cfg.axis],
                {cfg.axis: {"px": px_gains, "py": py_gains, "x": x_gains}},
            )
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("x_sweep", cfg.gx_expts)
        self.add_loop("y_sweep", cfg.gy_expts)
        self.add_loop("shots", cfg.point_avgs)

    def _body(self, cfg: experiment_models.FingerprintConfig):
        x_axis_cfg: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        idle_time = eo_pulses.round_up_pulses(x_axis_cfg.times.idle_time, self.soccfg)
        exchange_time = eo_pulses.round_up_pulses(
            x_axis_cfg.times.exchange_time, self.soccfg
        )
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto(0)
        if cfg.axis == "z":
            eo_pulses.play_pi(
                self, spinqick_enums.ExchangeAxis.N, cfg.qubit, t=0, t_res="fine"
            )
            t_pulse = exchange_time + idle_time * 1.5
        else:
            t_pulse = 0
        for n in range(cfg.n_pulses):
            eo_pulses.play_evol_course(self, cfg.axis, cfg.qubit, t=t_pulse)
            t_pulse += exchange_time + idle_time
        if cfg.axis == "z":
            eo_pulses.play_pi(
                self,
                spinqick_enums.ExchangeAxis.N,
                cfg.qubit,
                t=t_pulse - idle_time / 2,
                t_res="fine",
            )
        self.delay_auto(idle_time)
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)


class Nosc(asm_v2.AveragerProgramV2):
    """Sweep exchange pulse time for a given exchange pulse voltage.

    This is set up to sweep pulse times in increments of tproc clock cycles.
    """

    def _initialize(self, cfg: experiment_models.NoscConfig):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        # set the exchange time to be the minimum pulse length
        ax_cfg: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        ax_cfg.times.exchange_time = 0.005
        eo_pulses.setup_eo_gens(self, cfg.qubit)
        eo_pulses.setup_evol(self, cfg.qubit, [cfg.axis])
        if cfg.axis == "z":
            eo_pulses.setup_pi_pulse(
                self, cfg.qubit, [spinqick_enums.ExchangeAxis.N], "fine"
            )
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("pulse_time", cfg.expts)
        self.add_loop("shots", cfg.point_avgs)

    def _body(self, cfg: experiment_models.NoscConfig):
        t_sweep = asm_v2.QickSweep1D("pulse_time", cfg.start, cfg.stop)
        axis_cfg: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        idle_time = eo_pulses.round_up_pulses(axis_cfg.times.idle_time, self.soccfg)
        exchange_time = eo_pulses.round_up_pulses(
            axis_cfg.times.exchange_time, self.soccfg
        )
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto(idle_time / 2)
        if cfg.axis == "z":
            eo_pulses.play_pi(self, spinqick_enums.ExchangeAxis.N, cfg.qubit, t=0)
            t_pulse = exchange_time + idle_time * 1.5
        else:
            t_pulse = 0
        gate = "x"
        gate_obj: qubit_models.ExchangeGate = getattr(axis_cfg.gates, gate)
        gen = gate_obj.gen
        gate_name = gate_obj.name
        pulse_name = gate_name + "_" + cfg.axis + "_evol"
        self.pulse(gen, pulse_name, t=t_pulse)
        return_name = gate_name + "_" + "idle_return"
        self.pulse(gen, return_name, t=t_sweep)  # type: ignore
        if cfg.axis == "z":
            eo_pulses.play_pi(
                self, spinqick_enums.ExchangeAxis.N, cfg.qubit, t=t_sweep + t_pulse
            )
        self.delay_auto(idle_time)
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)


class CourseCalComp(asm_v2.AveragerProgramV2):
    """Scan along the symmetric axis linearly in x gate voltage.

    This assumes compensation in firmware!
    """

    def _initialize(self, cfg: experiment_models.CourseCalConfig):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        eo_pulses.setup_eo_gens(self, cfg.qubit)

        # calculate actual points from detuning and exchange values
        exp_axis: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        gates = exp_axis.gates
        assert isinstance(gates, qubit_models.ExchangeGateMap)
        x_gains = asm_v2.QickSweep1D("x_sweep", cfg.start, cfg.stop)
        if cfg.axis == "z":
            eo_pulses.setup_evol_sweep_comp(
                self,
                cfg.qubit,
                [spinqick_enums.ExchangeAxis.Z],
                {"z": {"x": x_gains}},
                cfg.n_pulses,
            )
            eo_pulses.setup_pi_pulse(
                self, cfg.qubit, [spinqick_enums.ExchangeAxis.N], "fine"
            )
        else:
            eo_pulses.setup_evol_sweep_comp(
                self, cfg.qubit, [cfg.axis], {cfg.axis: {"x": x_gains}}, cfg.n_pulses
            )
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("x_sweep", cfg.expts)
        self.add_loop("shots", cfg.point_avgs)

    def _body(self, cfg: experiment_models.CourseCalConfig):
        x_axis_cfg: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        idle_time = eo_pulses.round_up_pulses(x_axis_cfg.times.idle_time, self.soccfg)
        exchange_time = eo_pulses.round_up_pulses(
            x_axis_cfg.times.exchange_time, self.soccfg
        )
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto(idle_time / 2)
        if cfg.axis == "z":
            eo_pulses.play_pi(
                self, spinqick_enums.ExchangeAxis.N, cfg.qubit, t=0, t_res="fine"
            )
            t_pulse = exchange_time + idle_time
        else:
            t_pulse = 0
        eo_pulses.play_evol_fine(self, cfg.axis, cfg.qubit, t=t_pulse, comp=True)
        if cfg.axis == "z":
            t_pulse += cfg.n_pulses * exchange_time + (cfg.n_pulses) * idle_time
            eo_pulses.play_pi(
                self, spinqick_enums.ExchangeAxis.N, cfg.qubit, t=t_pulse, t_res="fine"
            )
        self.delay_auto(idle_time / 2)
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)


class FineCalComp(asm_v2.AveragerProgramV2):
    """Scan along the symmetric axis linearly in phase.

    This sweep is hard to code in qick API, so that part can be done in a python outer loop. TODO
    make this a dmem sweep
    """

    def _initialize(self, cfg: experiment_models.FineCalConfig):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        eo_pulses.setup_eo_gens(self, cfg.qubit)

        exp_axis: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        gates = exp_axis.gates
        assert isinstance(gates, qubit_models.ExchangeGateMap)
        if cfg.axis == "z":
            eo_pulses.setup_evol_comp(
                self,
                cfg.qubit,
                [spinqick_enums.ExchangeAxis.Z],
                sweep_dict={
                    "z": {"x": cfg.exchange_gain},
                },
                n_pulses=cfg.n_pulses,
                ptime_res=cfg.t_res,
            )
            eo_pulses.setup_pi_pulse(
                self, cfg.qubit, [spinqick_enums.ExchangeAxis.N], "fine"
            )
        else:
            eo_pulses.setup_evol_comp(
                self,
                cfg.qubit,
                [cfg.axis],
                sweep_dict={cfg.axis: {"x": cfg.exchange_gain}},
                n_pulses=cfg.n_pulses,
                ptime_res=cfg.t_res,
            )
        self.add_loop("shots", cfg.point_avgs)

    def _body(self, cfg: experiment_models.FineCalConfig):
        x_axis_cfg: qubit_models.ExchangeAxisConfig = getattr(cfg.qubit, cfg.axis)
        idle_time = eo_pulses.round_up_pulses(x_axis_cfg.times.idle_time, self.soccfg)
        exchange_time = eo_pulses.round_up_pulses(
            x_axis_cfg.times.exchange_time, self.soccfg
        )
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto(idle_time / 2)
        if cfg.axis == "z":
            eo_pulses.play_pi(
                self, spinqick_enums.ExchangeAxis.N, cfg.qubit, t=0, t_res="fine"
            )
            t_pulse = exchange_time + idle_time
        else:
            t_pulse = 0
        if cfg.t_res == "fs":
            eo_pulses.play_evol_fine(self, cfg.axis, cfg.qubit, t=t_pulse, comp=True)
            t_pulse += cfg.n_pulses * exchange_time + cfg.n_pulses * idle_time
        else:
            for n in range(cfg.n_pulses):
                eo_pulses.play_evol_course(self, cfg.axis, cfg.qubit, t=t_pulse)
                t_pulse += exchange_time + idle_time
        if cfg.axis == "z":
            t_pulse -= idle_time
            eo_pulses.play_pi(
                self, spinqick_enums.ExchangeAxis.N, cfg.qubit, t=t_pulse, t_res="fine"
            )
        self.delay_auto(idle_time / 2)
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
