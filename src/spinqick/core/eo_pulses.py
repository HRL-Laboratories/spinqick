"""Module for setting up exchange-only pulse sequences."""

from typing import List, Literal

import numpy as np
from qick import QickConfig, asm_v2

from spinqick.core import awg_pulse
from spinqick.experiments import eo_analysis
from spinqick.helper_functions import dac_pulses, qick_enums, spinqick_enums
from spinqick.models import qubit_models

# composite cliffords from Andrews et al 2019 supplement https://doi.org/10.1038/s41565-019-0500-4
THETA1 = np.arctan(np.sqrt(8))
THETA2 = np.pi - np.arctan(np.sqrt(5) / 2)
THETA3 = 74.755 / 180 * np.pi
THETA4 = 201.625 / 180 * np.pi
CLIFFORDS_1Q = {
    "I": [0, 0, 0, 0],
    "X": [0, np.pi - THETA1, THETA1, np.pi - THETA1],
    "Y": [np.pi, np.pi - THETA1, THETA1, np.pi - THETA1],
    "Z": [np.pi, 0, 0, 0],
    "S": [3 * np.pi / 2, 0, 0, 0],
    "Sd": [np.pi / 2, 0, 0, 0],
    "SX": [0, THETA3, THETA2, THETA4],
    "SdX": [0, THETA4, THETA2, THETA3],
    "H": [(np.pi - THETA1) / 2, np.pi + THETA1, (np.pi - THETA1) / 2, 0],
    "XH": [(np.pi + THETA1) / 2, np.pi - THETA1, (3 * np.pi + THETA1) / 2, 0],
    "YH": [(np.pi + THETA1) / 2, np.pi - THETA1, (np.pi + THETA1) / 2, 0],
    "ZH": [(3 * np.pi + THETA1) / 2, np.pi - THETA1, (np.pi + THETA1) / 2, 0],
    "SH": [(np.pi - THETA1) / 2, np.pi + THETA1, 2 * np.pi - THETA1 / 2, 0],
    "HS": [2 * np.pi - THETA1 / 2, np.pi + THETA1, (np.pi - THETA1) / 2, 0],
    "SdH": [(3 * np.pi + THETA1) / 2, np.pi - THETA1, THETA1 / 2, 0],
    "HSd": [THETA1 / 2, np.pi - THETA1, (3 * np.pi + THETA1) / 2, 0],
    "HSH": [THETA1 / 2, np.pi - THETA1, THETA1 / 2, 0],
    "HSdH": [np.pi + THETA1 / 2, np.pi - THETA1, np.pi + THETA1 / 2, 0],
    "SdHS": [np.pi + THETA1 / 2, np.pi - THETA1, THETA1 / 2, 0],
    "SHSd": [THETA1 / 2, np.pi - THETA1, np.pi + THETA1 / 2, 0],
    "HSX": [THETA1 / 2, np.pi - THETA1, (np.pi + THETA1) / 2, 0],
    "SdXH": [(np.pi + THETA1) / 2, np.pi - THETA1, THETA1 / 2, 0],
    "HSdX": [2 * np.pi - THETA1 / 2, np.pi + THETA1, (3 * np.pi - THETA1) / 2, 0],
    "SXH": [(3 * np.pi - THETA1) / 2, np.pi + THETA1, 2 * np.pi - THETA1 / 2, 0],
}


def round_up_pulses(pulsetime: float, soccfg: QickConfig):
    """Calculate the pulse length after rounding to fabric cycles."""

    cycles = round(np.ceil(pulsetime / soccfg.cycles2us(1)))
    return soccfg.cycles2us(cycles)


def setup_eo_gens(prog: asm_v2.QickProgramV2, qubit_cfg: qubit_models.Eo1Qubit):
    """Declare all generators associated with a qubit."""

    for exchange_axis in ["n", "z", "m"]:
        cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, exchange_axis)
        if cfg is not None:
            for gate in cfg.gates.model_fields_set:
                gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
                gen = gate_obj.gen
                if gen in prog.gen_chs.items():
                    continue
                else:
                    if prog.soccfg["gens"][gen]["has_mixer"]:
                        prog.declare_gen(gen, nqz=1, mixer_freq=0)
                    else:
                        prog.declare_gen(gen, nqz=1)


def setup_pi_pulse(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    exchange_axis: list[spinqick_enums.ExchangeAxis],
    pulse_time_cal: Literal["course", "fine"] = "fine",
):
    """Setup a pi pulse for one or more axes of a qubit.

    :param pulse_time_cal: if "course", function assumes that the calibration was performed with
        pulse time rounded to the nearest fabric clock cycle. If "fine", pulse times are defined and
        rounded to the nearest dac clock cycle.
    """

    for axis in exchange_axis:
        cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, axis)
        gate = "x"
        gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
        gen = gate_obj.gen
        gate_name = gate_obj.name
        gain = gate_obj.gains.exchange_gain
        idle_gain = gate_obj.gains.idle_gain
        time = cfg.times.exchange_time
        idle_time = cfg.times.idle_time
        pulse_name = gate_name + "_" + "pi_" + axis
        if pulse_time_cal == "fine":
            idata_array = dac_pulses.baseband_centered(
                gain, idle_gain, time, idle_time, gen, prog.soccfg
            )
            awg_pulse.add_arb_wf(
                prog,
                gen,
                pulse_name,
                1,
                idata_array,
                predistort=True,
                filter_sandwich=False,
            )
        else:
            idata_array = dac_pulses.generate_baseband(1, time, gen, prog.soccfg, pad_pulse=False)
            awg_pulse.add_arb_wf(
                prog,
                gen,
                pulse_name,
                1,
                idata_array,
                predistort=False,
                filter_sandwich=False,
            )
        return_name = gate_name + "_" + "idle_return"
        if return_name not in prog.pulses:
            idle = dac_pulses.generate_baseband(
                1, cfg.times.idle_time, gen, prog.soccfg, min_pulse=True
            )
            awg_pulse.add_arb_wf(
                prog,
                gen,
                return_name,
                idle_gain,
                idle,
                predistort=False,
                filter_sandwich=False,
            )


def play_pi(
    prog: asm_v2.AsmV2,
    exchange_axis: spinqick_enums.ExchangeAxis,
    qubit_cfg: qubit_models.Eo1Qubit,
    t: float = 0,
    t_res: Literal["course", "fine"] = "course",
):
    """Play pi pulse programmed by setup_pi_pulse.

    :param t: time in program to play pulse (in microseconds). This goes directly into the qick api
        "pulse" command.
    :param t_res: if "course", function assumes that the calibration was performed with pulse time
        rounded to the nearest fabric clock cycle. In this case, the script plays a pulse directly
        after the pi pulse which is at the idle voltage amplitude. If "fine", pulse times are
        defined and rounded to the nearest dac clock cycle.
    """

    cfg = getattr(qubit_cfg, exchange_axis)
    gate = "x"
    gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
    gen = gate_obj.gen
    gate_name = gate_obj.name
    pulse_name = gate_name + "_pi_" + exchange_axis
    prog.pulse(gen, pulse_name, t=t)  # type: ignore
    if t_res == "course":
        gate = "x"
        gate_obj = getattr(cfg.gates, gate)
        gen = gate_obj.gen
        gate_name = gate_obj.name
        return_name = gate_name + "_" + "idle_return"
        prog.pulse(gen, return_name, t="auto")  # type: ignore


def setup_evol_sweep(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    exchange_axis: List[spinqick_enums.ExchangeAxis],
    sweep_dict: dict,
):
    """Set up an evol pulse for one or both axes of a qubit.

    This automatically rounds the pulse times up to the nearest dac fabric cycle
    :param sweep_dict: {axis: {gate: {sweep: 1dqicksweep}} dictionary to specify gates to sweep and
        supply a 1dqicksweep object.
    """
    for axis in exchange_axis:
        cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, axis)
        for gate in cfg.gates.model_fields_set:
            gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
            gen = gate_obj.gen
            gate_name = gate_obj.name
            idle_gain = gate_obj.gains.idle_gain
            if sweep_dict is not None and axis in sweep_dict.keys():
                if gate in sweep_dict[axis].keys():
                    exchange_gain = sweep_dict[axis][gate]
                else:
                    exchange_gain = gate_obj.gains.exchange_gain
            else:
                exchange_gain = gate_obj.gains.exchange_gain
            pulse_name = gate_name + "_" + axis + "_evol"
            return_name = gate_name + "_" + "idle_return"
            env = dac_pulses.generate_baseband(1, cfg.times.exchange_time, gen, prog.soccfg)
            idle = dac_pulses.generate_baseband(1, cfg.times.idle_time, gen, prog.soccfg)
            prog.add_envelope(
                gen,
                "evol",
                idata=env,
            )
            prog.add_envelope(
                gen,
                "idle_return",
                idata=idle,
            )
            prog.add_pulse(
                gen,
                pulse_name,
                style="arb",
                freq=0,
                phase=0,
                gain=exchange_gain,
                envelope="evol",
                stdysel=qick_enums.Stdysel.LAST,
                outsel=qick_enums.Outsel.INPUT,
            )
            if return_name not in prog.pulses:
                prog.add_pulse(
                    gen,
                    return_name,
                    style="arb",
                    freq=0,
                    phase=0,
                    gain=idle_gain,
                    envelope="idle_return",
                    stdysel=qick_enums.Stdysel.LAST,
                    outsel=qick_enums.Outsel.INPUT,
                )


def setup_evol(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    exchange_axis: list[spinqick_enums.ExchangeAxis],
    n_pulses: int = 10,
    ptime_res: Literal["fabric", "fs"] = "fabric",
    sweep_dict: dict | None = None,
):
    """Setup an exchange pulse for one axis of a qubit .

    :param n_pulses: program n_pulses to play in a row, if ptime_res is set to "fs"
    :param ptime_res: time resolution of pulse time. If "fs", set to sampling rate of dac, and pad
        end of pulse with idle_gain values. If "fabric", round pulse length up to next fabric clock
        length.
    :param sweep_dict: {axis: {gate: gain} dictionary to specify gates to modulate when applying
        exchange and supply a gain for each gate.
    """
    for axis in exchange_axis:
        cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, axis)
        for gate in cfg.gates.model_fields_set:
            gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
            gen = gate_obj.gen
            gate_name = gate_obj.name
            if sweep_dict is not None and axis in sweep_dict.keys():
                exchange_gain = sweep_dict[axis][gate]
            else:
                exchange_gain = gate_obj.gains.exchange_gain
            idle_gain = gate_obj.gains.idle_gain
            pulse_name = gate_name + "_" + axis + "_evol"
            return_name = gate_name + "_" + "idle_return"
            if ptime_res == "fs":
                env = dac_pulses.baseband_pulse_train(
                    exchange_gain,
                    idle_gain,
                    cfg.times.exchange_time,
                    cfg.times.idle_time,
                    n_pulses,
                    gen,
                    prog.soccfg,
                )
                awg_pulse.add_predistorted_envelope(
                    prog,
                    gen,
                    "evol",
                    idata=env,
                )
            else:
                env = dac_pulses.generate_baseband(
                    exchange_gain,
                    cfg.times.exchange_time,
                    gen,
                    prog.soccfg,
                    pad_pulse=False,
                )
                idle = dac_pulses.generate_baseband(
                    idle_gain, cfg.times.idle_time, gen, prog.soccfg, min_pulse=True
                )
                prog.add_envelope(
                    gen,
                    "evol",
                    idata=env,
                )
                prog.add_envelope(
                    gen,
                    "idle_return",
                    idata=idle,
                )
                if return_name not in prog.pulses:
                    prog.add_pulse(
                        gen,
                        return_name,
                        style="arb",
                        freq=0,
                        phase=0,
                        gain=idle_gain,
                        envelope="idle_return",
                        stdysel=qick_enums.Stdysel.LAST,
                        outsel=qick_enums.Outsel.INPUT,
                    )
            prog.add_pulse(
                gen,
                pulse_name,
                style="arb",
                freq=0,
                phase=0,
                gain=1,
                envelope="evol",
                stdysel=qick_enums.Stdysel.LAST,
                outsel=qick_enums.Outsel.INPUT,
            )


def setup_evol_comp(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    exchange_axis: list[spinqick_enums.ExchangeAxis],
    n_pulses: int = 10,
    ptime_res: Literal["fabric", "fs"] = "fs",
    sweep_dict: dict | None = None,
):
    """Setup an evol pulse for one axis of a qubit.  This assumes that the user is using the
    compensation firmware to pulse along the exchange axis, so it only commands the x-gate to pulse
    during an exchange pulse.

    :param n_pulses: program n_pulses to play in a row, if ptime_res is set to "fs"
    :param ptime_res: time resolution of pulse time. If "fs", set to sampling rate of dac, and pad
        end of pulse with idle_gain values. If "fabric", round pulse length up to next fabric clock
        length.
    :param sweep_dict: {axis: {gate: gain} dictionary to specify gates to modulate when applying
        exchange and supply a gain for each gate.
    """
    for axis in exchange_axis:
        cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, axis)
        gate = "x"
        gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
        gen = gate_obj.gen
        gate_name = gate_obj.name
        if sweep_dict is not None and axis in sweep_dict.keys():
            exchange_gain = sweep_dict[axis][gate]
        else:
            exchange_gain = gate_obj.gains.exchange_gain
        idle_gain = gate_obj.gains.idle_gain
        pulse_name = gate_name + "_" + axis + "_evol"
        return_name = gate_name + "_" + "idle_return"
        if ptime_res == "fs":
            env = dac_pulses.baseband_pulse_train(
                0.9,
                0,
                cfg.times.exchange_time,
                cfg.times.idle_time,
                n_pulses,
                gen,
                prog.soccfg,
            )
            awg_pulse.add_predistorted_envelope(
                prog,
                gen,
                "evol",
                idata=env,
            )
        else:
            env = dac_pulses.generate_baseband(
                0.9, cfg.times.exchange_time, gen, prog.soccfg, pad_pulse=False
            )
            idle = dac_pulses.generate_baseband(
                0.9, cfg.times.idle_time, gen, prog.soccfg, min_pulse=True
            )
            prog.add_envelope(
                gen,
                "evol",
                idata=env,
            )
            prog.add_envelope(
                gen,
                "idle_return",
                idata=idle,
            )
            if return_name not in prog.pulses:
                prog.add_pulse(
                    gen,
                    return_name,
                    style="arb",
                    freq=0,
                    phase=0,
                    gain=idle_gain / 0.9,
                    envelope="idle_return",
                    stdysel=qick_enums.Stdysel.LAST,
                    outsel=qick_enums.Outsel.INPUT,
                )
        prog.add_pulse(
            gen,
            pulse_name,
            style="arb",
            freq=0,
            phase=0,
            gain=exchange_gain / 0.9,
            envelope="evol",
            stdysel=qick_enums.Stdysel.LAST,
            outsel=qick_enums.Outsel.INPUT,
        )


def setup_evol_sweep_comp(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    exchange_axis: List[spinqick_enums.ExchangeAxis],
    sweep_dict: dict,
    n_pulses: int,
):
    """Set up an evol pulse for one or more axes of a qubit.This automatically rounds the pulse
    times up to the nearest dac fabric cycle.

    :param n_pulses: program n_pulses to play in a row, if ptime_res is set to "fs"
    :param ptime_res: time resolution of pulse time. If "fs", set to sampling rate of dac, and pad
        end of pulse with idle_gain values. If "fabric", round pulse length up to next fabric clock
        length.
    :param sweep_dict: {axis: {gate: {sweep: 1dqicksweep}} specify gates to sweep and supply a
        1dqicksweep object describing the sweep.
    """
    for axis in exchange_axis:
        cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, axis)
        if axis in sweep_dict.keys():
            for gate in sweep_dict[axis].keys():
                exchange_gain = sweep_dict[axis][gate]
                gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
                gen = gate_obj.gen
                gate_name = gate_obj.name
                pulse_name = gate_name + "_" + axis + "_evol"
                env = dac_pulses.baseband_pulse_train(
                    0.9,
                    0,
                    cfg.times.exchange_time,
                    cfg.times.idle_time,
                    n_pulses,
                    gen,
                    prog.soccfg,
                )
                awg_pulse.add_predistorted_envelope(
                    prog,
                    gen,
                    "evol",
                    idata=env,
                )
                prog.add_pulse(
                    gen,
                    pulse_name,
                    style="arb",
                    freq=0,
                    phase=0,
                    gain=exchange_gain / 0.9,
                    envelope="evol",
                    stdysel=qick_enums.Stdysel.LAST,
                    outsel=qick_enums.Outsel.INPUT,
                )
        else:
            exchange_gain = gate_obj.gains.exchange_gain


def play_evol_fine(
    prog: asm_v2.AsmV2,
    exchange_axis: spinqick_enums.ExchangeAxis,
    qubit_cfg: qubit_models.Eo1Qubit,
    t: float = 0,
    comp: bool = False,
):
    """Play evol pulses programmed by setup_evol."""

    cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, exchange_axis)
    if comp:
        gate = "x"
        gate_obj = getattr(cfg.gates, gate)
        gen = gate_obj.gen
        gate_name = gate_obj.name
        pulse_name = gate_name + "_" + exchange_axis + "_evol"
        prog.pulse(gen, pulse_name, t=t)  # type: ignore
    else:
        for gate in cfg.gates.model_fields_set:
            gate_obj = getattr(cfg.gates, gate)
            gen = gate_obj.gen
            gate_name = gate_obj.name
            pulse_name = gate_name + "_" + exchange_axis + "_evol"
            prog.pulse(gen, pulse_name, t=t)  # type: ignore


def play_evol_course(
    prog: asm_v2.AsmV2,
    exchange_axis: spinqick_enums.ExchangeAxis,
    qubit_cfg: qubit_models.Eo1Qubit,
    t: float = 0,
    n_pulses: int = 1,
):
    """Play evol pulses programmed by setup_evol_sweep or setup_evol.

    :param comp: if using pulse compensation firmware, only play a pulse on x-gate
    """

    cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, exchange_axis)
    t_pulse_interval = cfg.times.idle_time + cfg.times.exchange_time
    t_play = t
    for n in range(n_pulses):
        for gate in cfg.gates.model_fields_set:
            gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
            gen = gate_obj.gen
            gate_name = gate_obj.name
            pulse_name = gate_name + "_" + exchange_axis + "_evol"
            prog.pulse(gen, pulse_name, t=t_play)  # type: ignore
        for gate in cfg.gates.model_fields_set:
            gate_obj = getattr(cfg.gates, gate)
            gen = gate_obj.gen
            gate_name = gate_obj.name
            return_name = gate_name + "_" + "idle_return"
            prog.pulse(gen, return_name, t="auto")  # type: ignore
        t_play += t_pulse_interval


def theta_to_x_gain(
    exchange_axis: spinqick_enums.ExchangeAxis,
    qubit_cfg: qubit_models.Eo1Qubit,
    theta: float,
    x_gate_conversion: float,
):
    """Convert desired angle to x-gate gain.  This assumes the user is using the crosstalk
    compensation firmware.

    :param theta: desired angle in radians.
    :param x_gate_conversion: specify the conversion from voltage to dac units
    """

    cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, exchange_axis)
    exchange_cal = cfg.exchange_cal
    assert exchange_cal is not None
    cal = exchange_cal.cal_parameters
    tlist = cal.theta_list
    glist = cal.volt_list  # volts are converted to gain in the config stored on dot experiment
    assert tlist
    assert glist
    if theta != 0:
        exchange_v = eo_analysis.fine_cal_voltage(theta, tlist, glist, cal.A, cal.B, cal.theta_max)
    else:
        exchange_v = 0
    exchange_gain = exchange_v * x_gate_conversion
    return exchange_gain


def dumb_1q_compiler(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    n_recipe: List,
    z_recipe: List,
    x_n_conversion: float,
    x_z_conversion: float,
):
    """Takes a list of angles on z and n axes, and converts them into waveforms.  It assumes that.

    :param x_n_conversion: specify the conversion from voltage to dac units for the n axis
    :param x_z_conversion: specify the conversion from voltage to dac units for the z axis
    """

    time = qubit_cfg.n.times.exchange_time
    idle_time = qubit_cfg.n.times.idle_time
    n_x = qubit_cfg.n.gates.x
    z_x = qubit_cfg.z.gates.x
    assert isinstance(n_x, qubit_models.ExchangeGate)
    assert isinstance(z_x, qubit_models.ExchangeGate)
    z_wf = np.array([])
    n_wf = np.array([])
    for theta in z_recipe:
        zgain = theta_to_x_gain(spinqick_enums.ExchangeAxis.Z, qubit_cfg, theta, x_z_conversion)
        wf = dac_pulses.baseband_pulse_train(zgain, 0, time, idle_time, 1, z_x.gen, prog.soccfg)
        z_wf = np.append(z_wf, wf)
    for theta in n_recipe:
        ngain = theta_to_x_gain(spinqick_enums.ExchangeAxis.N, qubit_cfg, theta, x_n_conversion)
        wf = dac_pulses.baseband_pulse_train(ngain, 0, time, idle_time, 1, n_x.gen, prog.soccfg)
        n_wf = np.append(n_wf, wf)
    return n_wf, z_wf


def clifford_to_recipe(clifford: str):
    """Return separate lists of angles for the n and z axes for a given clifford.

    This uses the one qubit clifford definitions listed above in CLIFFORD_1Q.
    """

    c_list = CLIFFORDS_1Q[clifford]
    z_recipe = [c_list[0], 0, c_list[2], 0]
    n_recipe = [0, c_list[1], 0, c_list[3]]
    return n_recipe, z_recipe


def setup_1q_cliffords(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    cliffords: List[str],
    x_n_conversion: float,
    x_z_conversion: float,
):
    """Setup pulses for a series of single qubit cliffords.

    :param cliffords: list of strings describing clifford gates from CLIFFORD_1Q.
    :param x_n_conversion: specify the conversion from voltage to dac units for the n axis
    :param x_z_conversion: specify the conversion from voltage to dac units for the z axis
    """

    n_x = qubit_cfg.n.gates.x
    z_x = qubit_cfg.z.gates.x
    assert isinstance(n_x, qubit_models.ExchangeGate)
    assert isinstance(z_x, qubit_models.ExchangeGate)
    label = ""
    n_wf_total = np.array([])
    z_wf_total = np.array([])
    for clifford in cliffords:
        nr, zr = clifford_to_recipe(clifford)
        n_wf, z_wf = dumb_1q_compiler(prog, qubit_cfg, nr, zr, x_n_conversion, x_z_conversion)
        n_wf_total = np.append(n_wf_total, n_wf)
        z_wf_total = np.append(z_wf_total, z_wf)
        label = label + clifford
    if label + "_n" not in prog.pulses:
        awg_pulse.add_arb_wf(prog, n_x.gen, label + "_n", 1, n_wf_total, predistort=True)
    if label + "_z" not in prog.pulses:
        awg_pulse.add_arb_wf(prog, z_x.gen, label + "_z", 1, z_wf_total, predistort=True)


def play_1q_clifford(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    cliffords: List[str],
    t: float = 0,
):
    """Play a single qubit clifford.

    :param cliffords: a list of strings specifying clifford gates from CLIFFORD_1Q.
    :param t: time in qickprogram to execute the cliffords in microseconds.
    """

    ncfg = getattr(qubit_cfg, "n")
    ngate_obj: qubit_models.ExchangeGate = getattr(ncfg.gates, "x")
    n_gen = ngate_obj.gen

    zcfg = getattr(qubit_cfg, "z")
    zgate_obj: qubit_models.ExchangeGate = getattr(zcfg.gates, "x")
    z_gen = zgate_obj.gen
    label = ""
    for clifford in cliffords:
        label = label + clifford
    prog.pulse(n_gen, label + "_n", t=t)  # type: ignore
    prog.pulse(z_gen, label + "_z", t=t)  # type: ignore
