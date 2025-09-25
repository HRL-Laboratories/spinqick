"""
AveragerProgram -type qick code for the tune_electrostatics module
Now for tprocv2
"""

from qick import asm_v2
from spinqick.core import awg_pulse
from spinqick.core import readout_v2
from spinqick.core.qick_utils import (
    Mode,
    Stdysel,
    Waveform,
    check_nyquist,
)
from spinqick.models import experiment_models


class BasebandPulseGvG(asm_v2.AveragerProgramV2):
    """QICK class to sweep two gate voltages against each other, using QICK not slow DAC"""

    def _initialize(self, cfg: experiment_models.GvgBasebandConfig):
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        self.declare_gen(ch=cfg.gx_gen, nqz=1)
        self.declare_gen(ch=cfg.gy_gen, nqz=1)
        x_sweep = asm_v2.QickSweep1D("xloop", cfg.gx_start, cfg.gx_stop)
        y_sweep = asm_v2.QickSweep1D("yloop", cfg.gy_start, cfg.gy_stop)

        awg_pulse.add_long_baseband(self, cfg.gy_gen, "bb_y", y_sweep, self.soccfg)
        awg_pulse.add_long_baseband(self, cfg.gx_gen, "bb_x", x_sweep, self.soccfg)

        self.add_loop("xloop", cfg.gx_expts)
        self.add_loop("yloop", cfg.gy_expts)
        self.delay(cfg.measure_buffer)

    def _body(self, cfg: experiment_models.GvgBasebandConfig):
        self.pulse(ch=cfg.gx_gen, name="bb_x", t=0)
        self.pulse(ch=cfg.gy_gen, name="bb_y", t=0)
        self.delay_auto(t=cfg.measure_buffer, gens=True)  # type: ignore
        readout_v2.readout_dcs(self, cfg.dcs_cfg)
        self.wait_auto(t=0)


class Static(asm_v2.AveragerProgramV2):
    """Run a series of readout triggers without triggering an external instrument"""

    def _initialize(self, cfg: experiment_models.GvgDcConfig):
        readout_v2.init_dcs(self, cfg.dcs_cfg, cfg.mode)
        self.delay(1)
        # self.trigger(pins=[cfg.trig_pin], width=cfg.trig_length)
        self.add_loop("shots", cfg.points)

    def _body(self, cfg: experiment_models.GvgDcConfig):
        self.delay(cfg.measure_buffer)
        readout_v2.readout_dcs(self, cfg.dcs_cfg, cfg.mode)
        self.delay_auto(cfg.measure_buffer)  # type: ignore
        self.wait_auto(t=0)


class GvG(asm_v2.AveragerProgramV2):
    """Time this carefully with your DC voltage source.  This runs once per dac ramp, based on a trigger"""

    def _initialize(self, cfg: experiment_models.GvgDcConfig):
        readout_v2.init_dcs(self, cfg.dcs_cfg, cfg.mode)
        self.add_loop("repeat_body", cfg.points)
        self.delay(1)
        self.trigger(pins=[cfg.trig_pin], width=cfg.trig_length)

    def _body(self, cfg: experiment_models.GvgDcConfig):
        self.delay(cfg.measure_buffer)
        readout_v2.readout_dcs(self, cfg.dcs_cfg, cfg.mode)
        self.delay_auto(cfg.measure_buffer)  # type: ignore
        self.wait_auto(t=0)


class GvGPat(asm_v2.AveragerProgramV2):
    """Runs GvG with an RF pulse on each step"""

    def _initialize(self, cfg: experiment_models.GvgPatConfig):
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        self.add_loop("repeat_body", cfg.points)
        if cfg.pat_cfg is not None:
            nqz = check_nyquist(cfg.pat_cfg.pat_freq, cfg.pat_cfg.pat_gen, self.soccfg)
            self.declare_gen(cfg.pat_cfg.pat_gen, nqz=nqz)
            self.add_pulse(
                ch=cfg.pat_cfg.pat_gen,
                name="rf",
                style=Waveform.CONSTANT,
                freq=cfg.pat_cfg.pat_freq,
                phase=0,
                mode=Mode.ONESHOT,
                gain=cfg.pat_cfg.pat_gain,
                stdysel=Stdysel.ZERO,
                length=cfg.measure_buffer + cfg.dcs_cfg.length,
            )
        self.delay(1)
        self.trigger(pins=[cfg.trig_pin], width=cfg.trig_length)

    def _body(self, cfg: experiment_models.GvgPatConfig):
        if cfg.pat_cfg is not None:
            self.pulse(ch=cfg.pat_cfg.pat_gen, name="rf", t=0)
        self.delay(cfg.measure_buffer)
        readout_v2.readout_dcs(self, cfg.dcs_cfg)
        self.delay_auto(cfg.measure_buffer)  # type: ignore
        self.wait_auto(t=0)
