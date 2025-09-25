"""Module for adding commonly used arbitrary waveform pulses to a qickprogram"""

import numpy as np
from qick import asm_v2, QickConfig
from spinqick.helper_functions import dac_pulses
from spinqick.core import qick_utils


def add_arb_wf(
    prog: asm_v2.QickProgramV2,
    ch: int,
    name: str,
    gain: float | asm_v2.QickParam,
    pulse: np.ndarray,
    stdysel: qick_utils.Stdysel = qick_utils.Stdysel.ZERO,
):
    """Add an arbitrary waveform pulse to a qickprogram."""
    prog.add_envelope(ch=ch, name=name + "_env", idata=pulse)
    prog.add_pulse(
        ch=ch,
        name=name,
        envelope=name + "_env",
        freq=0,
        phase=0,
        style=qick_utils.Waveform.ARB,
        mode=qick_utils.Mode.ONESHOT,
        outsel=qick_utils.Outsel.INPUT,
        stdysel=stdysel,
        gain=gain,
    )


def add_long_baseband(
    prog: asm_v2.QickProgramV2,
    ch: int,
    name: str,
    gain: float | asm_v2.QickParam,
    soccfg: QickConfig,
):
    """Add a long baseband pulse to the program.  This adds a pulse which simply goes to the specified gain and stays there until the next command on that channel"""
    add_arb_wf(
        prog,
        ch,
        name,
        gain,
        dac_pulses.generate_baseband(1, 3, ch, soccfg, min_pulse=True),
        stdysel=qick_utils.Stdysel.LAST,
    )


def add_short_baseband(
    prog: asm_v2.QickProgramV2,
    ch: int,
    name: str,
    gain: float | asm_v2.QickParam,
    length: float,
    soccfg: QickConfig,
):
    """Add a short baseband pulse to the program's pulse library."""
    add_arb_wf(
        prog,
        ch,
        name,
        gain,
        dac_pulses.generate_baseband(1.0, length, ch, soccfg),
        stdysel=qick_utils.Stdysel.LAST,
    )


def add_ramp(
    prog: asm_v2.QickProgramV2,
    ch: int,
    name: str,
    gain_1: float,
    gain_2: float,
    ramp_time: float,
    soccfg: QickConfig,
    stdysel: qick_utils.Stdysel = qick_utils.Stdysel.LAST,
):
    """Add a ramp waveform to the program's pulse library"""
    ramp = dac_pulses.generate_ramp(gain_1, gain_2, ramp_time, ch, soccfg)
    add_arb_wf(prog, ch, name, 1, ramp, stdysel=stdysel)
