"""Functions for interfacing with crosstalk parameter registers."""

from spinqick.helper_functions.spinqick_enums import GateNames
from spinqick.models import hardware_config_models


def gen_to_xtalk_ch(gen: int):
    """Convert generator number to crosstalk matrix index.

    Channel 5 is the first xtalk channel.
    """
    ch = gen - 5
    if ch < 0:
        raise ValueError("not a crosstalk channel")
    return ch


def perp_gen_to_perp_ch(perp_gen: int, victim_gen: int):
    """Calculated the correct register number for the victim and perpetrator channels.

    :param perp_gen: qick generator number for the perpetrator channel
    :param victim_gen: qick generator number for the victim channel. This channel will compensate
        when the perpetrator channel output changes, and not the other way around.
    """
    victim_ch = gen_to_xtalk_ch(victim_gen)
    perp_ch = gen_to_xtalk_ch(perp_gen)
    if perp_ch < victim_ch:
        kval = perp_ch + 1
    elif perp_ch > victim_ch:
        kval = perp_ch
    else:
        raise ValueError("channels are identical")
    return victim_ch, kval


def set_xtalk_element(
    perp_gate: GateNames,
    victim_gate: GateNames,
    xtalk: float,
    soc,
    hw_cfg: hardware_config_models.HardwareConfig,
):
    """Set crosstalk matrix element for a perpetrator, victim pair.  The victim gate will respond to
    changes in the perpetrator gates voltage by xtalk*perpetrator voltage.

    :param perp_gate: name of perpetrator gate
    :param victim_gate: name of victim gate
    :param xtalk: A crosstalk matrix element value between -1 and 1.
    :param soc: XtalkSoc object. You need to initialize the qick board with the XtalkSoc class
        instead of the QickSoc class to be able to use this.
    """
    perp_cfg = hw_cfg.channels[perp_gate]
    victim_cfg = hw_cfg.channels[victim_gate]
    assert isinstance(perp_cfg, hardware_config_models.FastGate)
    assert isinstance(victim_cfg, hardware_config_models.FastGate)
    v_ch, k_val = perp_gen_to_perp_ch(perp_cfg.qick_gen, victim_cfg.qick_gen)
    soc.set_xtalk(v_ch, k_val, xtalk)
