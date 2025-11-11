"""Module for setting up and playing LD qubit pulse sequences."""

from typing import List

from qick import asm_v2

from spinqick.core import utils
from spinqick.models import ld_qubit_models


def setup_1_ld_qubit(prog: asm_v2.QickProgramV2, cfg: ld_qubit_models.Ld1Qubit):
    """Sets up a single qubit pi/2 pulse for x and y axes."""

    nqz = utils.check_nyquist(cfg.f0, cfg.rf_gen, prog.soccfg)
    prog.declare_gen(cfg.rf_gen, nqz=nqz)
    prog.add_pulse(
        cfg.rf_gen,
        "x90",
        style="const",
        freq=cfg.f0,
        phase=0,
        gain=cfg.pulses.gain_90,
        length=cfg.pulses.time_90,
    )
    prog.add_pulse(
        cfg.rf_gen,
        "y90",
        style="const",
        freq=cfg.f0,
        phase=90,
        gain=cfg.pulses.gain_90,
        length=cfg.pulses.time_90,
    )


def parse_and_play_1q(
    prog: asm_v2.QickProgramV2, cfg: ld_qubit_models.Ld1Qubit, gate_list: List[str]
):
    """Parses the gate and angle and play pulses.  This plays a 180 degree x- or y- pulse as two 90
    degree pulses.

    :param gate_list: takes a list of strings in the form of "axis" + "angle in degrees". For
        example, ["X90", "X180", "Y180"]
    """
    for gate in gate_list:
        gate_desc = gate.split()
        if gate_desc[0] in ["X", "Y"]:
            if gate_desc[0] == "X":
                pulse_name = "x90"
            else:
                pulse_name = "y90"
            prog.pulse(cfg.rf_gen, pulse_name)
            if gate_desc[1] == "180":
                # play a second pulse
                prog.pulse(cfg.rf_gen, pulse_name)
        elif gate_desc[0] == "Z":
            # virtual Z.  Increment phase on future x and y pulses.
            for wf in ["x90_w0", "y90_w0"]:
                prog.read_wmem(name=wf)
                ang_incr = prog.soccfg.deg2reg(int(gate_desc[1]), gen_ch=cfg.rf_gen)
                prog.inc_reg(dst="w_phase", src=ang_incr)
                prog.write_wmem(name=wf)
