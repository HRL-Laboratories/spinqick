"""Module for setting up exchange-only pulse sequences
TODO use AWG pulses module
"""

from typing import Literal, List
import numpy as np
from qick import asm_v2
from spinqick.models import qubit_models
from spinqick.helper_functions import dac_pulses
from spinqick.core import qick_utils

THETA1 = np.arctan(np.sqrt(8))
THETA2 = np.pi - np.arctan(np.sqrt(5) / 2)
THETA3 = 74.755 / 180 * np.pi
THETA4 = 201.625 / 180 * np.pi
CLIFFORDS_1Q = {}


def define_fingerprint_vectors(
    px_points: np.ndarray,
    py_points: np.ndarray,
    idle_point: np.ndarray,
    x_point: float,
):
    """define the detuning and exchange axes based on nonequilibrium cell parameters
    :param px_points: [Px1, Px2] format, defines the start and endpoints of a line being used to
    define detuning vector
    :param py_points: [Py1, Py2]
    :param x_point: x value used during nonequilibrium cell sweep
    """
    delta_px = px_points[1] - px_points[0]
    delta_py = py_points[1] - py_points[0]
    detuning_raw = np.array([delta_px, delta_py, 0])
    detuning = detuning_raw / np.linalg.norm(detuning_raw)
    midpoint = np.array([px_points[0] + delta_px / 2, py_points[0] + delta_py / 2])
    symmetric_raw = np.array([midpoint[0], midpoint[1], x_point]) - np.array(
        [idle_point[0], idle_point[1], idle_point[2]]
    )
    symmetric = symmetric_raw / (x_point - idle_point[2])
    return detuning, symmetric


def calculate_fingerprint_gate_vals(
    detuning, x, detuning_vector, symmetric_vector, idle_point
):
    """calculate individual gate voltages given detuning, x-gate gain, and detuning and symmetric vectors"""
    vector_out = np.zeros(3)
    vector_out = detuning_vector * detuning + symmetric_vector * x + idle_point
    return vector_out


def add_predistorted_envelope(prog: asm_v2.QickProgramV2, ch, name, idata, qdata=None):
    """Apply predistortion to the pulses"""
    idata_filt = dac_pulses.add_wf_filter(idata)
    prog.add_envelope(ch, name, idata_filt, qdata)


def setup_eo_gens(prog: asm_v2.QickProgramV2, qubit_cfg: qubit_models.Eo1Qubit):
    """declare all generators associated with a qubit"""
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
    exchange_axis: list[Literal["n", "z", "m"]],
    pulse_time_cal: Literal["course", "fine"] = "course",
):
    """setup a pi pulse for one or both axes of a qubit
    Right now this is assuming course pulse time definition
    """
    for axis in exchange_axis:
        cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, axis)
        for gate in cfg.gates.model_fields_set:
            gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
            gen = gate_obj.gen
            gate_name = gate_obj.name
            gain = gate_obj.gains.exchange_gain
            idle_gain = gate_obj.gains.idle_gain
            time = cfg.times.exchange_time
            # idle_time = cfg.times.idle_time
            pulse_name = gate_name + "_" + "pi"
            if pulse_time_cal == "fine":
                idle_norm = idle_gain / gain if gain != 0 else 0
                idata_array = dac_pulses.generate_baseband(
                    1, time, gen, prog.soccfg, gain_2=idle_norm, pad_pulse=True
                )
            else:
                idata_array = dac_pulses.generate_baseband(
                    1, time, gen, prog.soccfg, pad_pulse=False
                )
            add_predistorted_envelope(prog, gen, "pi", idata=idata_array)
            prog.add_pulse(
                gen,
                pulse_name,
                style="arb",
                freq=0,
                phase=0,
                gain=gain,
                envelope="pi",
                stdysel=qick_utils.Stdysel.LAST,
                outsel=qick_utils.Outsel.INPUT,
            )
            add_predistorted_envelope(
                prog,
                gen,
                "idle_return",
                idata=dac_pulses.generate_baseband(
                    1, cfg.times.idle_time, gen, prog.soccfg, min_pulse=True
                ),
            )
            return_name = gate_name + "_" + "idle_return"
            if return_name not in prog.pulses:
                prog.add_pulse(
                    gen,
                    return_name,
                    style="arb",
                    freq=0,
                    phase=0,
                    gain=idle_gain,
                    envelope="idle_return",
                    stdysel=qick_utils.Stdysel.LAST,
                    outsel=qick_utils.Outsel.INPUT,
                )


def play_pi(
    prog: asm_v2.QickProgramV2,
    exchange_axis: Literal["n", "z", "m"],
    qubit_cfg: qubit_models.Eo1Qubit,
    t: float = 0,
):
    """play pi pulse programmed by setup_pi_pulse"""
    cfg = getattr(qubit_cfg, exchange_axis)
    for gate in cfg.gates.model_fields_set:
        # gate = "x"
        gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
        gen = gate_obj.gen
        gate_name = gate_obj.name
        pulse_name = gate_name + "_" + "pi"
        prog.pulse(gen, pulse_name, t=t)
    for gate in cfg.gates.model_fields_set:
        gate_obj = getattr(cfg.gates, gate)
        gen = gate_obj.gen
        gate_name = gate_obj.name
        return_name = gate_name + "_" + "idle_return"
        prog.pulse(gen, return_name, t="auto")


def setup_evol_sweep(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    exchange_axis: List[Literal["n", "z", "m"]],
    sweep_dict: dict,
):
    """Set up an evol pulse for one or both axes of a qubit.
    This automatically rounds the pulse times up to the nearest dac fabric cycle
    :param sweep_dict:  {axis: {gate: {sweep: 1dqicksweep}} specify gates to sweep and supply a 1dqicksweep object
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
            env = dac_pulses.generate_baseband(
                1, cfg.times.exchange_time, gen, prog.soccfg
            )
            idle = dac_pulses.generate_baseband(
                1, cfg.times.idle_time, gen, prog.soccfg
            )
            add_predistorted_envelope(
                prog,
                gen,
                "evol",
                idata=env,
            )
            add_predistorted_envelope(
                prog,
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
                stdysel=qick_utils.Stdysel.LAST,
                outsel=qick_utils.Outsel.INPUT,
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
                    stdysel=qick_utils.Stdysel.LAST,
                    outsel=qick_utils.Outsel.INPUT,
                )


def setup_evol(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    exchange_axis: list[Literal["n", "z", "m"]],
    n_pulses: int = 10,
    ptime_res: Literal["fabric", "fs"] = "fabric",
    sweep_dict: dict | None = None,
):
    """setup an evol pulse for one axis of a qubit
    :param ptime_res: time resolution of pulse time. If fs, set to sampling rate of dac,
    and pad end of pulse with idle_gain values. If fabric, round pulse length up to next
    fabric clock length
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
            else:
                env = dac_pulses.generate_baseband(
                    1, cfg.times.exchange_time, gen, prog.soccfg, pad_pulse=False
                )
            idle = dac_pulses.generate_baseband(
                1, cfg.times.idle_time, gen, prog.soccfg, min_pulse=True
            )
            add_predistorted_envelope(
                prog,
                gen,
                "evol",
                idata=env,
            )
            add_predistorted_envelope(
                prog,
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
                gain=exchange_gain if ptime_res == "fabric" else 1,
                envelope="evol",
                stdysel=qick_utils.Stdysel.LAST,
                outsel=qick_utils.Outsel.INPUT,
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
                    stdysel=qick_utils.Stdysel.LAST,
                    outsel=qick_utils.Outsel.INPUT,
                )


def play_evol_fine(
    prog: asm_v2.QickProgramV2,
    exchange_axis: Literal["n", "z", "m"],
    qubit_cfg: qubit_models.Eo1Qubit,
    t: float = 0,
):
    """play evol pulses programmed by setup_evol"""
    cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, exchange_axis)
    for gate in cfg.gates.model_fields_set:
        gate_obj = getattr(cfg.gates, gate)
        gen = gate_obj.gen
        gate_name = gate_obj.name
        pulse_name = gate_name + "_" + exchange_axis + "_evol"
        prog.pulse(gen, pulse_name, t=t)


def play_evol_course(
    prog: asm_v2.QickProgramV2,
    exchange_axis: Literal["n", "z", "m"],
    qubit_cfg: qubit_models.Eo1Qubit,
    t: float = 0,
    n_pulses: int = 1,
):
    """play evol pulses programmed by setup_evol_sweep or setup_evol"""
    cfg: qubit_models.ExchangeAxisConfig = getattr(qubit_cfg, exchange_axis)
    t_pulse_interval = cfg.times.idle_time + cfg.times.exchange_time
    t_play = t
    for n in range(n_pulses):
        for gate in cfg.gates.model_fields_set:
            gate_obj: qubit_models.ExchangeGate = getattr(cfg.gates, gate)
            gen = gate_obj.gen
            gate_name = gate_obj.name
            pulse_name = gate_name + "_" + exchange_axis + "_evol"
            prog.pulse(gen, pulse_name, t=t_play)
        for gate in cfg.gates.model_fields_set:
            gate_obj = getattr(cfg.gates, gate)
            # ex_time = cfg.times.exchange_time
            gen = gate_obj.gen
            gate_name = gate_obj.name
            return_name = gate_name + "_" + "idle_return"
            prog.pulse(gen, return_name, t="auto")
        t_play += t_pulse_interval


def play_1q_clifford(
    prog: asm_v2.QickProgramV2,
    qubit_cfg: qubit_models.Eo1Qubit,
    gate: str,
    t: float = 0,
):
    """play a single qubit clifford
    :param gate: string describing clifford gate from CLIFFORD_1Q
    """
    # TODO play pulse sequence from the clifford dictionary above using finecal in qubit config
