"""Module for adding commonly used arbitrary waveform pulses to a qickprogram."""

import numpy as np
from qick import QickConfig, asm_v2

from spinqick.helper_functions import dac_pulses, qick_enums


def add_predistorted_envelope(
    prog: asm_v2.QickProgramV2,
    ch: int,
    name: str,
    idata: np.ndarray,
    filter_sandwich: bool = False,
):
    """Apply predistortion to the pulses.

    :param prog: QickProgram to add pulse envelope to
    :param ch: channel to play pulse_envelope on
    :param name: pulse envelope name
    :param idata: array pulse envelope values. This should be defined point-by-point at the dac
        sampling rate
    :param filter_sandwich: if true, sandwich the envelope between two identical envelopes before
        applying predistortion, then remove the additional waveforms before adding to the pulse
        library.
    """
    if filter_sandwich:
        pre = 1
        post = 1
    else:
        pre = None
        post = None
    idata_filt = dac_pulses.add_wf_filter(idata, pre, post)
    prog.add_envelope(ch, name, idata_filt)


def add_arb_wf(
    prog: asm_v2.QickProgramV2,
    ch: int,
    name: str,
    gain: float | asm_v2.QickParam,
    pulse: np.ndarray,
    stdysel: qick_enums.Stdysel = qick_enums.Stdysel.ZERO,
    predistort: bool = False,
    filter_sandwich: bool = False,
):
    """Add an arbitrary waveform envelope and pulse to a qickprogram.

    :param prog: QickProgram to add pulse envelope to
    :param ch: channel to play pulse_envelope on
    :param name: pulse envelope name
    :param gain: pulse gain.  Can be a float or a QickParam if the user is defining a sweep.
    :param stdysel: whether to set the DAC to stay at the value of the last point defined in the
        pulse
    :param predistort: if true, applies predistortion filter to the pulse.
    :param filter_sandwich: if true, sandwich the envelope between two identical envelopes before
        applying predistortion, then remove the additional waveforms before adding to the pulse
        library.
    """
    if predistort:
        add_predistorted_envelope(prog, ch, name + "_env", pulse, filter_sandwich=filter_sandwich)
    else:
        prog.add_envelope(ch, name + "_env", pulse)

    prog.add_pulse(
        ch=ch,
        name=name,
        envelope=name + "_env",
        freq=0,
        phase=0,
        style=qick_enums.Waveform.ARB,
        mode=qick_enums.Mode.ONESHOT,
        outsel=qick_enums.Outsel.INPUT,
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
    """Add a long baseband pulse to the program.  This adds a pulse which simply goes to the
    specified gain and stays there until the next command on that channel.

    :param prog: QickProgram to add pulse envelope to
    :param ch: channel to play pulse_envelope on
    :param name: pulse envelope name
    :param gain: pulse gain.  Can be a float or a QickParam if the user is defining a sweep.
    :param soccfg: the Qick Config associated with the user's experiment.
    """
    add_arb_wf(
        prog,
        ch,
        name,
        gain,
        dac_pulses.generate_baseband(1, 3, ch, soccfg, min_pulse=True),
        stdysel=qick_enums.Stdysel.LAST,
    )


def add_short_baseband(
    prog: asm_v2.QickProgramV2,
    ch: int,
    name: str,
    gain: float | asm_v2.QickParam,
    length: float,
    soccfg: QickConfig,
    stdysel: qick_enums.Stdysel = qick_enums.Stdysel.LAST,
    predistort: bool = False,
    filter_sandwich: bool = False,
):
    """Add a short baseband pulse to the program's pulse library.  The duration of this pulse needs
    to be short enough to fit into the qick waveform memory.  This currently rounds the pulse length
    up to the next fabric clock cycle.

    :param prog: QickProgram to add pulse envelope to
    :param ch: channel to play pulse_envelope on
    :param name: pulse envelope name
    :param gain: pulse gain.  Can be a float or a QickParam if the user is defining a sweep.
    :param length: pulse length in microseconds
    :param soccfg: the Qick Config associated with the user's experiment.
    :param stdysel: if "last", the generator output will remain at "gain" until the next pulse is
        played on that channel.
    """
    add_arb_wf(
        prog,
        ch,
        name,
        gain,
        dac_pulses.generate_baseband(1.0, length, ch, soccfg),
        stdysel=stdysel,
        predistort=predistort,
        filter_sandwich=filter_sandwich,
    )


def add_ramp(
    prog: asm_v2.QickProgramV2,
    ch: int,
    name: str,
    gain_1: float,
    gain_2: float,
    ramp_time: float,
    soccfg: QickConfig,
    stdysel: qick_enums.Stdysel = qick_enums.Stdysel.LAST,
):
    """Add a ramp waveform to the program's pulse library.

    :param prog: QickProgram to add pulse envelope to
    :param ch: channel to play pulse_envelope on
    :param name: pulse envelope name
    :param gain_1: pulse gain at start of pulse in RFSoC gain units
    :param gain_2: pulse gain at end of pulse in RFSoC gain units
    :param ramp_time: duration of ramp in microseconds
    :param soccfg: the Qick Config associated with the user's experiment.
    :param stdysel: if "last", the generator output will remain at "gain_2" until the next pulse is
        played on that channel.
    """
    ramp = dac_pulses.generate_ramp(gain_1, gain_2, ramp_time, ch, soccfg)
    add_arb_wf(prog, ch, name, 1, ramp, stdysel=stdysel)
