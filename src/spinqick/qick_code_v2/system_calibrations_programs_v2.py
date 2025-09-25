"""
Storing the actual qick class code for the experiments in this file
"""

from qick import asm_v2
from spinqick.core import readout_v2
from spinqick.core import qick_utils
from spinqick.helper_functions import dac_pulses
from spinqick.models import experiment_models


class BasebandVoltageCalibration(asm_v2.AveragerProgramV2):
    """Use a loading line to calibrate baseband pulse voltages at the device,
    given known DC bias voltage.
    """

    def _initialize(self, cfg: experiment_models.LineSplitting):
        readout_v2.init_dcs(self, cfg.dcs_cfg, "sd_chop")
        self.declare_gen(cfg.differential_channel, nqz=1)
        env_pos_plus = dac_pulses.offset_sine_wave(
            cfg.differential_channel,
            cfg.differential_ac_gain,
            cfg.differential_ac_freq,
            cfg.differential_step_gain,
            self.soccfg,
        )
        env_pos_minus = dac_pulses.offset_sine_wave(
            cfg.differential_channel,
            cfg.differential_ac_gain,
            cfg.differential_ac_freq,
            -1 * cfg.differential_step_gain,
            self.soccfg,
        )
        self.add_envelope(cfg.differential_channel, "plus", idata=env_pos_plus)
        self.add_envelope(cfg.differential_channel, "minus", idata=env_pos_minus)
        # self.add_envelope(cfg.differential_channel, "minus", idata=-1*env_pos_plus)
        self.add_pulse(
            cfg.differential_channel,
            "plus_wf",
            style="arb",
            freq=0,
            phase=0,
            gain=1.0,
            outsel="input",
            mode="periodic",
            stdysel="last",
            envelope="plus",
        )
        self.add_pulse(
            cfg.differential_channel,
            "minus_wf",
            style="arb",
            freq=0,
            phase=0,
            gain=1.0,
            outsel="input",
            mode="periodic",
            stdysel="last",
            envelope="minus",
        )
        ### copied from gvg script
        self.add_loop("repeat_body", cfg.points)
        self.delay(1)
        self.trigger(pins=[cfg.trig_pin], width=cfg.trig_length)

    def _body(self, cfg: experiment_models.LineSplitting):
        self.pulse(cfg.differential_channel, "plus_wf")
        self.delay(cfg.measure_buffer)
        readout_v2.readout_dcs(self, cfg.dcs_cfg, cfg.mode)
        self.delay_auto(cfg.measure_buffer)  # type: ignore
        self.wait_auto()
        self.pulse(cfg.differential_channel, "minus_wf")
        self.delay(cfg.measure_buffer)
        readout_v2.readout_dcs(self, cfg.dcs_cfg, cfg.mode)
        self.delay_auto(cfg.measure_buffer)  # type: ignore
        self.wait_auto()


class HSATune(asm_v2.AveragerProgramV2):
    """baseband pulse for a given amount of time, and measure after the pulse ends."""

    def _initialize(self, cfg: experiment_models.HsaTune):
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        self.declare_gen(ch=cfg.tune_gate_gen, nqz=1)
        self.add_envelope(
            ch=cfg.tune_gate_gen,
            name="baseband",
            idata=dac_pulses.baseband(maxv=qick_utils.Defaults.MAX_GAIN_BITS),
        )
        self.add_pulse(
            cfg.tune_gate_gen,
            "baseband_pulse",
            style="arb",
            freq=0,
            gain=cfg.pulse_gain,
            stdysel="last",
            envelope="baseband",
        )
        self.add_pulse(
            cfg.tune_gate_gen,
            "zero_pulse",
            style="arb",
            freq=0,
            gain=0,
            stdysel="last",
            envelope="baseband",
        )
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.HsaTune):
        self.pulse(cfg.tune_gate_gen, "baseband_pulse", t=0)
        self.delay_auto(t=cfg.pulse_time)  # type: ignore
        self.pulse(cfg.tune_gate_gen, "zero_pulse")
        self.delay_auto(cfg.meas_time)  # type: ignore
        readout_v2.readout_dcs(self, cfg.dcs_cfg)
        self.wait_auto()


class PulseAndMeasure(asm_v2.AveragerProgramV2):
    """simple loopback program"""

    def _initialize(self, cfg: experiment_models.AvgedReadout):
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("points", cfg.point_avgs)

    def _body(self, cfg: experiment_models.AvgedReadout):
        readout_v2.readout_dcs(self, cfg.dcs_cfg)
        self.wait_auto()


class SweepAdcDelay(asm_v2.AveragerProgramV2):
    """simple loopback program"""

    def _initialize(self, cfg: experiment_models.SweepDelay):
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        # self.trig_delay = asm_v2.QickSweep1D("delay", cfg.dcs_cfg.length + cfg.delay_start, cfg.dcs_cfg.length + cfg.delay_stop)
        self.trig_delay = asm_v2.QickSweep1D("delay", cfg.delay_start, cfg.delay_stop)
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("delay", cfg.delay_points)
        self.add_loop("points", cfg.point_avgs)

    def _body(self, cfg: experiment_models.SweepDelay):
        self.pulse(ch=cfg.dcs_cfg.sd_gen, name="sourcedrain", t=0)  # readout pulse
        self.trigger(
            pins=[0],
            ros=cfg.dcs_cfg.ro_chs,
            t=self.trig_delay,  # type: ignore
        )  # trigger ADC
        self.delay_auto(cfg.dcs_cfg.slack_delay, gens=True, ros=True)  # type: ignore
        self.delay(self.trig_delay)
        self.wait_auto(ros=True)
