"""Module to implement dot charge sensor readout with qick, and pauli spin blockade spin-to-charge
conversion."""

from typing import Literal

from qick import asm_v2

from spinqick.core import awg_pulse
from spinqick.helper_functions import qick_enums
from spinqick.models import dcs_model, spam_models

MAX_GAIN = qick_enums.Defaults.MAX_GAIN_BITS
DEFAULT_SPAM_NAMES = ["flush", "idle", "entry_20", "entry_11", "exit_11", "meas"]


def init_dcs(
    prog: asm_v2.QickProgramV2,
    cfg: dcs_model.DcsConfig,
    mode: Literal["sd_chop", "transdc"] = "sd_chop",
):
    """Initialize DCS generator and readout channel."""
    # TODO check if generator has a DDS
    ro_gen = cfg.sd_gen if mode == "sd_chop" else cfg.ac_gate_gen
    ro_gain = cfg.pulse_gain_readout if mode == "sd_chop" else cfg.ac_gate_gain
    if prog.soccfg["gens"][ro_gen]["has_mixer"]:
        prog.declare_gen(ro_gen, nqz=1, mixer_freq=0)
    else:
        prog.declare_gen(ro_gen, nqz=1)
    for ro_ch in cfg.ro_chs:
        prog.declare_readout(ch=ro_ch, length=cfg.readout_length)
        ro_pulse = "ro_" + str(ro_ch)
        prog.add_readoutconfig(
            ch=ro_ch,
            name=ro_pulse,
            freq=cfg.readout_freq,
            gen_ch=ro_gen,
            phrst=0 if mode == "transdc" else 1,
            outsel="product",
        )
        prog.send_readoutconfig(ch=ro_ch, name=ro_pulse, t=0)
    prog.add_pulse(
        ch=ro_gen,
        name="sourcedrain",
        ro_ch=cfg.ro_chs[0],  # frequency match to one of the readout channels
        style="const",
        freq=cfg.dds_freq,
        length=cfg.length,
        phase=0,
        phrst=0 if mode == "transdc" else 1,
        gain=ro_gain,
    )


def readout_dcs(
    prog: asm_v2.QickProgramV2,
    cfg: dcs_model.DcsConfig,
    mode: Literal["sd_chop", "transdc"] = "sd_chop",
):
    """Basic dcs readout.

    Plays an AC signal and triggers the ADC to read out the signal.
    """
    ro_gen = cfg.sd_gen if mode == "sd_chop" else cfg.ac_gate_gen
    prog.pulse(ch=ro_gen, name="sourcedrain", t=0)  # readout pulse
    prog.trigger(ros=cfg.ro_chs, t=cfg.adc_trig_offset)  # trigger ADC #type: ignore
    prog.delay_auto(cfg.slack_delay, gens=True, ros=True)  # type: ignore


def program_spam_step_waveforms(
    prog: asm_v2.QickProgramV2, cfg: spam_models.DefaultSpamDac, step: str
):
    """Program waveforms for a given spam step."""
    duration = getattr(cfg, step).duration
    gate_list = getattr(cfg, step).gate_list
    for gate in gate_list:
        pulse_gain = gate_list[gate].coordinate
        gen = gate_list[gate].gen
        pulse_name = gate + "_" + step
        if isinstance(gate_list[gate], spam_models.SpamRampDac):
            # if the pulse is a ramp-type, program a ramp
            pulse_gain_2 = gate_list[gate].coordinate_2
            awg_pulse.add_ramp(
                prog,
                gen,
                pulse_name,
                pulse_gain,
                pulse_gain_2,
                duration,
                prog.soccfg,
                stdysel=qick_enums.Stdysel.LAST,
            )
        else:
            awg_pulse.add_long_baseband(prog, gen, pulse_name, pulse_gain, prog.soccfg)


def program_spam_waveforms(prog: asm_v2.QickProgramV2, cfg: spam_models.DefaultSpamDac):
    """Iterate through each spam step and program waveforms."""

    for step in cfg.model_fields_set:
        program_spam_step_waveforms(prog, cfg, step)


def play_spam_step(
    prog: asm_v2.QickProgramV2,
    cfg: spam_models.DefaultSpamDac,
    step: str,
    pause: bool = True,
):
    """Iterate through spam gates and play pulse."""
    gate_list = getattr(cfg, step).gate_list
    gate_list = getattr(cfg, step).gate_list
    duration = getattr(cfg, step).duration
    for gate in gate_list:
        pulse_name = gate + "_" + step
        gen = gate_list[gate].gen
        prog.pulse(gen, pulse_name)
    if pause:
        prog.delay(t=duration)  # wait at this voltage for the duration of the step
        # prog.wait_auto(t=0, gens=True) # make sure the ramps are done before proceeding


def setup_spam_gens(prog: asm_v2.QickProgramV2, cfg: spam_models.DefaultSpamDac):
    """Declare all generators in the spam sequence."""
    gens = []
    for step in cfg.model_fields_set:
        step_obj = getattr(cfg, step)
        gate_list = step_obj.gate_list
        for gate in gate_list:
            gen = gate_list[gate].gen
            if gen not in gens:
                gens.append(gen)
    for gen in gens:
        if prog.soccfg["gens"][gen]["has_mixer"]:
            prog.declare_gen(gen, nqz=1, mixer_freq=0)
        else:
            prog.declare_gen(gen, nqz=1)


def init_spam_point_sweep(
    prog: asm_v2.QickProgramV2,
    cfg: spam_models.DefaultSpamDac,
    sweep_step: str,
    gx_gen: int,
    gy_gen: int,
    x_sweep: asm_v2.QickParam,
    y_sweep: asm_v2.QickParam,
):
    """Declares generators and program waveforms for a 2D sweep of a spam point.  The user can
    specify a sweep over any two gates which are defined in their spam sequence for that point.

    :param sweep_step: the name of the spam step which will be swept.
    :param gx_gen: the generator whose voltage is being swept on the x-axis
    :param gy_gen: the generator whose voltage is being swept on the y-axis
    :param x_sweep: QickParam object describing the x-axis sweep
    :param y_sweep: QickParam object describing the y-axis sweep
    """
    for step in DEFAULT_SPAM_NAMES:
        if step == sweep_step:
            continue
        else:
            program_spam_step_waveforms(prog, cfg, step)

    # now program the sweep step
    sweep_step_cfg = getattr(cfg, sweep_step)
    gate_list = sweep_step_cfg.gate_list
    for gate in gate_list:
        pulse_gain = gate_list[gate].coordinate
        gen = gate_list[gate].gen
        if gen == gx_gen:
            gain_val = x_sweep
        elif gen == gy_gen:
            gain_val = y_sweep
        else:
            gain_val = pulse_gain
        pulse_name = gate + "_" + sweep_step
        awg_pulse.add_long_baseband(prog, gen, pulse_name, gain_val, prog.soccfg)


def init_point_multisweep(
    prog: asm_v2.QickProgramV2,
    cfg: spam_models.DefaultSpamDac,
    sweep_step: str,
    gx_gens: list[int],
    gy_gens: list[int],
    x_sweeps: list[asm_v2.QickParam],
    y_sweeps: list[asm_v2.QickParam],
    pulse_length: float | None = None,
):
    """Declare generators and program waveforms for a 2D sweep of a spam point.  This is used for
    the measurement window scan code, as it allows you to sweep more than one spamstep.

    :param sweep_step: the name of the spam step which will be swept.
    :param gx_gens: the generators whose voltages are being swept on the x-axis
    :param gy_gens: the generators whose voltages are being swept on the y-axis
    :param x_sweeps: QickParam objects describing the x-axis sweeps
    :param y_sweeps: QickParam objects describing the y-axis sweeps
    :param pulse_length: specify a pulse duration to store the whole pulse in waveform memory.
        Otherwise, plays a pulse and waits until the user pulses to idle.
    """

    # now program the sweep step
    sweep_step_cfg = getattr(cfg, sweep_step)
    gate_list = sweep_step_cfg.gate_list
    for gate in gate_list:
        pulse_gain = gate_list[gate].coordinate
        gen = gate_list[gate].gen
        if gen in gx_gens:
            ind = [i for i in range(len(gx_gens)) if gx_gens[i] == gen]
            gain_val = x_sweeps[ind[0]]
        elif gen in gy_gens:
            ind = [i for i in range(len(gy_gens)) if gy_gens[i] == gen]
            gain_val = y_sweeps[ind[0]]
        else:
            gain_val = pulse_gain
        pulse_name = gate + "_" + sweep_step
        if pulse_length is not None:
            awg_pulse.add_short_baseband(
                prog, gen, pulse_name, gain_val, pulse_length, prog.soccfg
            )
        else:
            awg_pulse.add_long_baseband(prog, gen, pulse_name, gain_val, prog.soccfg)


def init_psb(prog: asm_v2.QickProgramV2, cfg: spam_models.DefaultSpamDac):
    """Initialize pauli spin blockade readout in a qickprogram."""
    setup_spam_gens(prog, cfg)
    program_spam_waveforms(prog, cfg)


# replaces the pt1-pt3 functionality
def psb_fm(
    prog: asm_v2.QickProgramV2,
    spam_cfg: spam_models.DefaultSpamDac,
    dcs_cfg: dcs_model.DcsConfig,
):
    """Plays flush and measure spam steps."""
    for step in ["flush", "entry_20", "meas"]:
        play_spam_step(prog, spam_cfg, step)
    readout_dcs(prog, dcs_cfg)
    prog.delay(dcs_cfg.length)
    prog.wait_auto(gens=True, ros=True)  # wait until readout is complete


def psb_f11(prog: asm_v2.QickProgramV2, spam_cfg: spam_models.DefaultSpamDac):
    """Plays flush, entry_20 and exit_11 steps."""
    for step in ["flush", "entry_20", "exit_11"]:
        play_spam_step(prog, spam_cfg, step)


def psb_fe(prog: asm_v2.QickProgramV2, spam_cfg: spam_models.DefaultSpamDac):
    """Plays flush, entry_20 and exit_11 steps, then pulse to idle."""
    psb_f11(prog, spam_cfg)
    play_spam_step(prog, spam_cfg, "idle")


def psb_em(
    prog: asm_v2.QickProgramV2,
    spam_cfg: spam_models.DefaultSpamDac,
    dcs_cfg: dcs_model.DcsConfig,
):
    """Plays entry_11 and measurement coordinates.

    Then plays readout_dcs, and waits for the readout to end.
    """
    play_spam_step(prog, spam_cfg, "entry_11")
    play_spam_step(prog, spam_cfg, "meas")
    readout_dcs(prog, dcs_cfg)
    prog.wait_auto(gens=True, ros=True)  # wait until readout is complete
