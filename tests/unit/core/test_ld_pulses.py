"""Hardware-free tests for linear-dot pulse helpers."""

from types import SimpleNamespace
from unittest.mock import Mock, call

from spinqick.core import ld_pulses


def make_ld_config():
    return SimpleNamespace(
        f0=100.0,
        rf_gen=2,
        pulses=SimpleNamespace(gain_90=0.25, time_90=0.05),
    )


def test_setup_1_ld_qubit_declares_generator_and_x_y_pulses(monkeypatch):
    program = Mock()
    program.soccfg = {"gens": {2: {"fs": 1000.0}}}
    monkeypatch.setattr(ld_pulses.utils, "check_nyquist", lambda *_: 2)
    cfg = make_ld_config()

    ld_pulses.setup_1_ld_qubit(program, cfg)

    program.declare_gen.assert_called_once_with(2, nqz=2)
    assert program.add_pulse.call_count == 2
    program.add_pulse.assert_any_call(
        2,
        "x90",
        style="const",
        freq=100.0,
        phase=0,
        gain=0.25,
        length=0.05,
    )
    program.add_pulse.assert_any_call(
        2,
        "y90",
        style="const",
        freq=100.0,
        phase=90,
        gain=0.25,
        length=0.05,
    )


def test_parse_and_play_1q_plays_180_degree_pulse_twice():
    program = Mock()
    cfg = make_ld_config()

    ld_pulses.parse_and_play_1q(program, cfg, ["X 180", "Y 90"])

    assert program.pulse.call_args_list == [
        call(2, "x90"),
        call(2, "x90"),
        call(2, "y90"),
    ]


def test_parse_and_play_1q_updates_phase_for_virtual_z():
    program = Mock()
    program.soccfg.deg2reg.return_value = 123
    cfg = make_ld_config()

    ld_pulses.parse_and_play_1q(program, cfg, ["Z 45"])

    assert program.read_wmem.call_args_list == [
        call(name="x90_w0"),
        call(name="y90_w0"),
    ]
    assert program.inc_reg.call_count == 2
    assert program.write_wmem.call_args_list == [
        call(name="x90_w0"),
        call(name="y90_w0"),
    ]
