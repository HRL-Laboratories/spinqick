"""
Qick code for noise experiments
"""

from qick import asm_v1

from spinqick.models import config_models
from spinqick.qick_code.qick_utils import Mode, Outsel, Waveform


def grab_noise(
    soccfg,
    config: config_models.ReadoutConfig,
    pulse_length: float,
    demodulate: bool,
    readout_freq: float,
    readout_tone: bool = True,
    continuous_tone: bool = False,
) -> asm_v1.QickProgram:
    gain = config.DCS_cfg.pulse_gain_readout
    gen_ch = config.DCS_cfg.res_ch
    readout_ch = config.DCS_cfg.ro_ch
    freq = soccfg.freq2reg(readout_freq)
    pulse_time = pulse_length
    if continuous_tone:
        pulse_mode = Mode.PERIODIC
    else:
        pulse_mode = Mode.ONESHOT

    noiseprogram = asm_v1.QickProgram(soccfg)
    if readout_tone:
        noiseprogram.declare_gen(ch=gen_ch, nqz=1)
        # turn on a tone which runs for the full acquisition
        noiseprogram.set_pulse_registers(
            ch=gen_ch,
            freq=soccfg.freq2reg(freq, gen_ch=gen_ch, ro_ch=readout_ch),
            phase=0,
            gain=gain,
            mode=pulse_mode,
            style=Waveform.CONSTANT,
            length=pulse_time,
        )

    if demodulate:
        noiseprogram.declare_readout(
            ch=readout_ch,
            freq=freq,
            length=pulse_time,
            gen_ch=gen_ch,
            sel=Outsel.PRODUCT,
        )
    else:
        noiseprogram.declare_readout(
            ch=readout_ch, freq=0, length=pulse_time, gen_ch=gen_ch, sel=Outsel.INPUT
        )
    if readout_tone:
        noiseprogram.pulse(ch=gen_ch)
    noiseprogram.trigger(ddr4=True, mr=True, pins=[0])
    noiseprogram.end()

    return noiseprogram
