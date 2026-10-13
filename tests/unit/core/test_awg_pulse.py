"""Hardware-free tests for AWG pulse delegation."""

from unittest.mock import Mock

import numpy as np

from spinqick.core import awg_pulse
from spinqick.helper_functions import qick_enums


def test_add_arb_wf_adds_envelope_and_arbitrary_pulse():
    program = Mock()
    pulse = np.array([0.0, 1.0, 0.0])

    awg_pulse.add_arb_wf(program, 2, "test", 0.5, pulse)

    envelope_args = program.add_envelope.call_args.args
    assert envelope_args[:2] == (2, "test_env")
    np.testing.assert_array_equal(envelope_args[2], pulse)
    program.add_pulse.assert_called_once_with(
        ch=2,
        name="test",
        envelope="test_env",
        freq=0,
        phase=0,
        style=qick_enums.Waveform.ARB,
        mode=qick_enums.Mode.ONESHOT,
        outsel=qick_enums.Outsel.INPUT,
        stdysel=qick_enums.Stdysel.ZERO,
        gain=0.5,
    )


def test_add_arb_wf_delegates_predistortion(monkeypatch):
    program = Mock()
    pulse = np.array([0.0, 1.0])
    predistort = Mock()
    monkeypatch.setattr(awg_pulse, "add_predistorted_envelope", predistort)

    awg_pulse.add_arb_wf(
        program,
        1,
        "filtered",
        1.0,
        pulse,
        predistort=True,
        filter_sandwich=True,
    )

    predistort.assert_called_once_with(
        program, 1, "filtered_env", pulse, filter_sandwich=True
    )
    program.add_pulse.assert_called_once()


def test_add_predistorted_envelope_applies_configured_filter(monkeypatch):
    program = Mock()
    program.soccfg = {"gens": {3: {"f_dds": 1000.0}}}
    pulse = np.array([1.0, 2.0])
    filter_path = Mock()
    filter_path.apply.return_value = np.array([3.0, 4.0])
    monkeypatch.setattr(awg_pulse, "get_filter_path", lambda _: filter_path)

    awg_pulse.add_predistorted_envelope(
        program,
        3,
        "filtered",
        pulse,
        filter_sandwich=True,
    )

    filter_path.apply.assert_called_once_with(
        pulse,
        pre=1,
        post=1,
        fs=1000.0,
    )
    envelope_args = program.add_envelope.call_args.args
    assert envelope_args[:2] == (3, "filtered")
    np.testing.assert_array_equal(envelope_args[2], np.array([3.0, 4.0]))


def test_add_short_baseband_generates_and_delegates_waveform(monkeypatch):
    program = Mock()
    soccfg = Mock()
    generated = np.array([1.0, 2.0])
    generate = Mock(return_value=generated)
    add_arb = Mock()
    monkeypatch.setattr(awg_pulse.dac_pulses, "generate_baseband", generate)
    monkeypatch.setattr(awg_pulse, "add_arb_wf", add_arb)

    awg_pulse.add_short_baseband(program, 2, "short", 0.25, 0.5, soccfg)

    generate.assert_called_once_with(1.0, 0.5, 2, soccfg)
    args, kwargs = add_arb.call_args
    assert args[:4] == (program, 2, "short", 0.25)
    np.testing.assert_array_equal(args[4], generated)
    assert kwargs == {
        "stdysel": qick_enums.Stdysel.LAST,
        "predistort": False,
        "filter_sandwich": False,
    }


def test_add_ramp_generates_ramp_and_delegates_waveform(monkeypatch):
    program = Mock()
    soccfg = Mock()
    generated = np.array([0.0, 1.0])
    generate_ramp = Mock(return_value=generated)
    monkeypatch.setattr(awg_pulse.dac_pulses, "generate_ramp", generate_ramp)
    add_arb = Mock()
    monkeypatch.setattr(awg_pulse, "add_arb_wf", add_arb)

    awg_pulse.add_ramp(program, 1, "ramp", 0.0, 1.0, 2.0, soccfg)

    generate_ramp.assert_called_once_with(0.0, 1.0, 2.0, 1, soccfg)
    args, kwargs = add_arb.call_args
    assert args[:4] == (program, 1, "ramp", 1)
    np.testing.assert_array_equal(args[4], generated)
    assert kwargs == {"stdysel": qick_enums.Stdysel.LAST}
