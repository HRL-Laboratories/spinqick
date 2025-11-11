"""Programs to help set up pauli spin blockade."""

import numpy as np
from qick import asm_v2

from spinqick.core import readout_v2
from spinqick.models import experiment_models, spam_models


class MeasHist(asm_v2.AveragerProgramV2):
    """This is a bare bones class to make a PSB measurement histogram."""

    def _initialize(self, cfg: experiment_models.MeashistConfig):
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.psb_cfg)
        self.add_loop("shots", cfg.num_measurements)

    def _body(self, cfg: experiment_models.MeashistConfig):
        if cfg.reference:
            readout_v2.psb_fm(self, cfg.psb_cfg, cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.psb_cfg)
        readout_v2.psb_em(self, cfg.psb_cfg, cfg.dcs_cfg)


class IdleScan(asm_v2.AveragerProgramV2):
    """Perform a 2D sweep of the idle point."""

    def _initialize(self, cfg: experiment_models.PsbScanConfig):
        idle_x_gains = asm_v2.QickSweep1D("x_sweep", cfg.gx_start, cfg.gx_stop)
        idle_y_gains = asm_v2.QickSweep1D("y_sweep", cfg.gy_start, cfg.gy_stop)
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        readout_v2.setup_spam_gens(self, cfg.psb_cfg)
        readout_v2.init_spam_point_sweep(
            self,
            cfg.psb_cfg,
            "idle",
            cfg.gx_gen,
            cfg.gy_gen,
            idle_x_gains,
            idle_y_gains,
        )
        # self.trigger(pins=[0])  # for testing only

        self.add_loop("full_avgs", cfg.full_avgs)  # add a loop
        self.add_loop("x_sweep", cfg.gx_expts)
        self.add_loop("y_sweep", cfg.gy_expts)
        self.add_loop("shots", cfg.point_avgs)  # add a loop

    def _body(self, cfg: experiment_models.PsbScanConfig):
        if cfg.reference:
            readout_v2.psb_fm(self, cfg.psb_cfg, cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.psb_cfg)
        readout_v2.psb_em(self, cfg.psb_cfg, cfg.dcs_cfg)


class FlushScan(asm_v2.AveragerProgramV2):
    """Perform a 2D sweep of the flush point."""

    def _initialize(self, cfg: experiment_models.PsbScanConfig):
        flush_x_gains = asm_v2.QickSweep1D("x_sweep", cfg.gx_start, cfg.gx_stop)
        flush_y_gains = asm_v2.QickSweep1D("y_sweep", cfg.gy_start, cfg.gy_stop)
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        readout_v2.setup_spam_gens(self, cfg.psb_cfg)
        readout_v2.init_spam_point_sweep(
            self,
            cfg.psb_cfg,
            "flush",
            cfg.gx_gen,
            cfg.gy_gen,
            flush_x_gains,
            flush_y_gains,
        )
        self.trigger(pins=[0])  # for testing only
        self.add_loop("full_avgs", cfg.full_avgs)  # add a loop
        self.add_loop("x_sweep", cfg.gx_expts)
        self.add_loop("y_sweep", cfg.gy_expts)
        self.add_loop("shots", cfg.point_avgs)  # add a loop

    def _body(self, cfg: experiment_models.PsbScanConfig):
        if cfg.reference:
            readout_v2.psb_fm(self, cfg.psb_cfg, cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.psb_cfg)
        readout_v2.psb_em(self, cfg.psb_cfg, cfg.dcs_cfg)


class MeasureScanStepRamp(asm_v2.AveragerProgramV2):
    """Scan measurement window point, this time use steps to emulate a ramp.

    Right now this only ramps two P gates.
    """

    def ramp_prog(self, cfg: experiment_models.MeasScanConfig, step_label):
        """asm_v2 chunk which steps."""
        sub = asm_v2.AsmV2()
        cfg_step: spam_models.SpamStepDac = getattr(cfg.psb_cfg, step_label)
        ramp_length = cfg_step.duration
        n_steps = np.ceil(ramp_length / cfg.step_time)
        if n_steps > 29:
            raise ValueError(
                "%d steps in ramp %s, number of steps needs to be less than 29. Raise step_time."
                % (n_steps, step_label)
            )

        for gate, gen, addr in [
            (cfg.gx_gate, cfg.gx_gen, 2),
            (cfg.gy_gate, cfg.gy_gen, 3),
        ]:
            wname = gate + "_" + step_label + "_w0"
            sub.read_wmem(name=wname)
            sub.write_dmem(addr=addr, src="w_gain")

        sub.open_loop(n_steps, "step_loop_" + step_label)
        for gate, gen in [(cfg.gx_gate, cfg.gx_gen), (cfg.gy_gate, cfg.gy_gen)]:
            wname = gate + "_" + step_label + "_w0"
            sub.read_wmem(name=wname)
            sub.pulse(ch=gen, name=gate + "_" + step_label, t=0)
            dac_step = int(self.sweep_dict[step_label][gate] * 32766 // n_steps)
            sub.inc_reg(dst="w_gain", src=dac_step)
            sub.write_wmem(name=wname)
        sub.close_loop()
        for gate, addr in [(cfg.gx_gate, 2), (cfg.gy_gate, 3)]:
            wname = gate + "_" + step_label + "_w0"
            sub.read_wmem(name=wname)
            sub.read_dmem(addr=addr, dst="w_gain")
            sub.write_wmem(name=wname)
        sub.delay(ramp_length)
        self.add_subroutine(step_label + "_stepramp", sub)

    def _initialize(self, cfg: experiment_models.MeasScanConfig):
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        readout_v2.setup_spam_gens(self, cfg.psb_cfg)
        for step in ["idle", "flush"]:
            readout_v2.program_spam_step_waveforms(self, cfg.psb_cfg, step)
        # set up the 1D sweep objects
        # measurement window
        meas_x_gains = asm_v2.QickSweep1D("x_sweep", cfg.gx_start, cfg.gx_stop)
        meas_y_gains = asm_v2.QickSweep1D("y_sweep", cfg.gy_start, cfg.gy_stop)
        readout_v2.init_point_multisweep(
            self,
            cfg.psb_cfg,
            "meas",
            [cfg.gx_gen],
            [cfg.gy_gen],
            [meas_x_gains],
            [meas_y_gains],
            pulse_length=None,
        )
        self.sweep_dict = {}
        # set up sweeps of the ramps
        for step in ["entry_11", "entry_20", "exit_11"]:
            step_cfg: spam_models.SpamStepDac = getattr(
                cfg.psb_cfg, step
            )  # check for whether the step is a ramp
            x_pulse = step_cfg.gate_list[cfg.gx_gate]
            y_pulse = step_cfg.gate_list[cfg.gy_gate]
            assert isinstance(x_pulse, spam_models.SpamRampDac)
            assert isinstance(y_pulse, spam_models.SpamRampDac)
            x_amp = x_pulse.coordinate_2 - x_pulse.coordinate
            y_amp = y_pulse.coordinate_2 - y_pulse.coordinate

            # set up the sweep so that it is sweeping the first coordinate of the ramp
            if step == "exit_11":
                x_gains = asm_v2.QickSweep1D("x_sweep", cfg.gx_start, cfg.gx_stop)
                y_gains = asm_v2.QickSweep1D("y_sweep", cfg.gy_start, cfg.gy_stop)
            else:
                x_gains = asm_v2.QickSweep1D("x_sweep", cfg.gx_start - x_amp, cfg.gx_stop - x_amp)
                y_gains = asm_v2.QickSweep1D("y_sweep", cfg.gy_start - y_amp, cfg.gy_stop - y_amp)
            readout_v2.init_point_multisweep(
                self,
                cfg.psb_cfg,
                step,
                [cfg.gx_gen],
                [cfg.gy_gen],
                [x_gains],
                [y_gains],
                pulse_length=cfg.step_time,
            )
            self.sweep_dict[step] = {cfg.gx_gate: x_amp, cfg.gy_gate: y_amp}
            self.ramp_prog(cfg, step)

        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("x_sweep", cfg.gx_expts)
        self.add_loop("y_sweep", cfg.gy_expts)
        self.add_loop("shots", cfg.point_avgs)
        self.delay(1)

    def _body(self, cfg: experiment_models.PsbScanConfig):
        if cfg.reference:
            readout_v2.play_spam_step(self, cfg.psb_cfg, "flush")
            self.call("entry_20_stepramp")
            readout_v2.play_spam_step(self, cfg.psb_cfg, "meas")
            readout_v2.readout_dcs(self, cfg.dcs_cfg)
            self.wait_auto(gens=True, ros=True)  # wait until readout is complete

        readout_v2.play_spam_step(self, cfg.psb_cfg, "flush")
        self.call("entry_20_stepramp")
        self.call("exit_11_stepramp")
        readout_v2.play_spam_step(self, cfg.psb_cfg, "idle")
        self.call("entry_11_stepramp")
        readout_v2.play_spam_step(self, cfg.psb_cfg, "meas")
        readout_v2.readout_dcs(self, cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)
