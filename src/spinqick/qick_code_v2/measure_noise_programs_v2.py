"""
Qick code for noise experiments
"""

from qick import asm_v2, QickConfig

from spinqick.models import dcs_model
from spinqick.core.qick_utils import Mode, Waveform


def grab_noise(
    soccfg: QickConfig,
    config: dcs_model.DcsConfig,
    demodulate: bool | None = None,
    readout_tone: bool | None = None,
    continuous_tone: bool | None = None,
) -> asm_v2.QickProgramV2:
    """This function creates a simple qick program which utilizes the ddr4 buffer to be able to grab long time traces of data.
    The user may apply a tone while grabbing this data or turn on demodulation.

    :param demodulate: demodulate output at readout_freq frequency
    :param readout_tone: optionally play a tone on the readout channel
    :param continuous_tone: play readout tone continuously for the entire experiment
    """

    gain = config.pulse_gain_readout
    gen_ch = config.sd_gen
    readout_ch = config.ro_chs
    freq = config.dds_freq
    pulse_time = config.length
    if continuous_tone:
        pulse_mode = Mode.PERIODIC
    else:
        pulse_mode = Mode.ONESHOT

    noiseprogram = asm_v2.QickProgramV2(soccfg)
    if readout_tone:
        noiseprogram.declare_gen(ch=gen_ch, nqz=1)
        # turn on a tone which runs for the full acquisition
        noiseprogram.add_pulse(
            ch=gen_ch,
            name="sourcedrain",
            freq=freq,
            phase=0,
            gain=gain,
            mode=pulse_mode,
            style=Waveform.CONSTANT,
            length=pulse_time,
            phrst=1,
            ro_ch=readout_ch[0],
        )
    # for ro_ch in readout_ch:
    ro_ch = readout_ch[0]
    noiseprogram.declare_readout(
        ch=ro_ch,
        length=pulse_time,  # the ddr4 will grab longer time trace automatically
    )
    ro_name = "sd" + str(ro_ch)
    if demodulate:
        noiseprogram.add_readoutconfig(
            ch=ro_ch,
            name=ro_name,
            freq=freq,
            gen_ch=gen_ch,
            outsel="product",
        )
    else:
        if readout_tone:
            noiseprogram.add_readoutconfig(
                ch=ro_ch,
                name=ro_name,
                freq=freq,
                gen_ch=gen_ch,
                outsel="input",
            )
        else:
            noiseprogram.add_readoutconfig(
                ch=ro_ch,
                name=ro_name,
                freq=freq,
                outsel="input",
            )
    noiseprogram.send_readoutconfig(ch=ro_ch, name="sd" + str(ro_ch))
    noiseprogram.delay(100)
    if readout_tone:
        noiseprogram.pulse(ch=gen_ch, name="sourcedrain", t=0)
    noiseprogram.trigger(ros=[ro_ch], ddr4=True, mr=True, t=config.adc_trig_offset)
    noiseprogram.end()

    return noiseprogram
