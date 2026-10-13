"""Hardware-free tests for QICK readout and PSB sequencing helpers."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, call

from spinqick.core import readout_v2
from spinqick.models import spam_models


class SpamConfig:
    """Small stand-in for the Pydantic SPAM configuration model."""

    def __init__(self, **steps):
        self.model_fields_set = set(steps)
        for name, value in steps.items():
            setattr(self, name, value)


def make_dcs_config():
    return SimpleNamespace(
        sd_gen=0,
        ac_gate_gen=1,
        pulse_gain_readout=100,
        ac_gate_gain=50,
        ro_chs=[2, 3],
        readout_length=16,
        readout_freq=100.0,
        dds_freq=10.0,
        length=32,
        adc_trig_offset=4,
        slack_delay=8,
    )


def make_program():
    program = Mock()
    program.soccfg = {
        "gens": {
            0: {"has_mixer": True},
            1: {"has_mixer": False},
        }
    }
    return program


def test_init_dcs_declares_readout_channels_and_source_pulse():
    program = make_program()
    cfg = make_dcs_config()

    readout_v2.init_dcs(program, cfg, mode="sd_chop")

    program.declare_gen.assert_called_once_with(0, nqz=1, mixer_freq=0)
    assert program.declare_readout.call_args_list == [
        call(ch=2, length=16),
        call(ch=3, length=16),
    ]
    assert program.send_readoutconfig.call_args_list == [
        call(ch=2, name="ro_2", t=0),
        call(ch=3, name="ro_3", t=0),
    ]
    program.add_pulse.assert_called_once_with(
        ch=0,
        name="sourcedrain",
        ro_ch=2,
        style="const",
        freq=10.0,
        length=32,
        phase=0,
        phrst=1,
        gain=100,
    )


def test_init_dcs_transdc_uses_ac_gate_and_transdc_phase_reset():
    program = make_program()
    cfg = make_dcs_config()

    readout_v2.init_dcs(program, cfg, mode="transdc")

    program.declare_gen.assert_called_once_with(1, nqz=1)
    program.add_pulse.assert_called_once_with(
        ch=1,
        name="sourcedrain",
        ro_ch=2,
        style="const",
        freq=10.0,
        length=32,
        phase=0,
        phrst=0,
        gain=50,
    )


def test_readout_dcs_pulses_triggers_and_delays():
    program = make_program()
    cfg = make_dcs_config()

    readout_v2.readout_dcs(program, cfg)

    program.pulse.assert_called_once_with(ch=0, name="sourcedrain", t=0)
    program.trigger.assert_called_once_with(ros=[2, 3], t=4)
    program.delay_auto.assert_called_once_with(8, gens=True, ros=True)


def test_program_spam_step_waveforms_delegates_long_baseband(monkeypatch):
    program = make_program()
    cfg = SpamConfig(
        meas=SimpleNamespace(
            duration=10,
            gate_list={"P1": SimpleNamespace(coordinate=123, gen=0)},
        )
    )
    add_long_baseband = Mock()
    monkeypatch.setattr(readout_v2.awg_pulse, "add_long_baseband", add_long_baseband)

    readout_v2.program_spam_step_waveforms(
        program, cast(spam_models.DefaultSpamDac, cfg), "meas"
    )

    add_long_baseband.assert_called_once_with(
        program,
        0,
        "P1_meas",
        123,
        program.soccfg,
    )


def test_play_spam_step_pulses_each_gate_and_pauses():
    program = make_program()
    cfg = SpamConfig(
        meas=SimpleNamespace(
            duration=12,
            gate_list={
                "P1": SimpleNamespace(gen=0),
                "P2": SimpleNamespace(gen=1),
            },
        )
    )

    readout_v2.play_spam_step(program, cast(spam_models.DefaultSpamDac, cfg), "meas")

    assert program.pulse.call_args_list == [
        call(0, "P1_meas"),
        call(1, "P2_meas"),
    ]
    program.delay.assert_called_once_with(t=12)


def test_play_spam_step_can_skip_pause():
    program = make_program()
    cfg = SpamConfig(
        meas=SimpleNamespace(
            duration=12,
            gate_list={"P1": SimpleNamespace(gen=0)},
        )
    )

    readout_v2.play_spam_step(
        program, cast(spam_models.DefaultSpamDac, cfg), "meas", pause=False
    )

    program.pulse.assert_called_once_with(0, "P1_meas")
    program.delay.assert_not_called()


def test_setup_spam_gens_declares_each_generator_once():
    program = make_program()
    cfg = SpamConfig(
        flush=SimpleNamespace(
            gate_list={
                "P1": SimpleNamespace(gen=0),
                "P2": SimpleNamespace(gen=1),
            }
        ),
        meas=SimpleNamespace(
            gate_list={"P3": SimpleNamespace(gen=0)},
        ),
    )

    readout_v2.setup_spam_gens(program, cast(spam_models.DefaultSpamDac, cfg))

    assert program.declare_gen.call_args_list == [
        call(0, nqz=1, mixer_freq=0),
        call(1, nqz=1),
    ]


def test_psb_f11_runs_expected_spam_steps(monkeypatch):
    program = make_program()
    spam_cfg = object()
    play_step = Mock()
    monkeypatch.setattr(readout_v2, "play_spam_step", play_step)

    readout_v2.psb_f11(program, cast(spam_models.DefaultSpamDac, spam_cfg))

    assert play_step.call_args_list == [
        call(program, spam_cfg, "flush"),
        call(program, spam_cfg, "entry_20"),
        call(program, spam_cfg, "exit_11"),
    ]


def test_psb_fm_reads_flush_entry_and_measurement(monkeypatch):
    program = make_program()
    spam_cfg = object()
    dcs_cfg = make_dcs_config()
    play_step = Mock()
    readout = Mock()
    monkeypatch.setattr(readout_v2, "play_spam_step", play_step)
    monkeypatch.setattr(readout_v2, "readout_dcs", readout)

    readout_v2.psb_fm(program, cast(spam_models.DefaultSpamDac, spam_cfg), dcs_cfg)

    assert play_step.call_args_list == [
        call(program, spam_cfg, "flush"),
        call(program, spam_cfg, "entry_20"),
        call(program, spam_cfg, "meas"),
    ]
    readout.assert_called_once_with(program, dcs_cfg)
    program.delay.assert_called_once_with(dcs_cfg.length)
    program.wait_auto.assert_called_once_with(gens=True, ros=True)
