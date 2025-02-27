"""
Storing the actual qick class code for the experiments in this file
"""

from qick import averager_program
from spinqick.qick_code import readout
from spinqick.qick_code.qick_utils import Outsel, Mode, Stdysel, Waveform
from spinqick.helper_functions import dac_pulses


### System Calibrations ###


class BasebandVoltageCalibration(averager_program.RAveragerProgram, readout.Readout):
    """Use a loading line to calibrate baseband pulse voltages at the device, given known DC bias voltage."""

    def initialize(self):
        cfg = self.cfg
        self.declare_gen(
            ch=cfg.calibrate_cfg.x_gate.gen, nqz=1
        )  # assuming SDAC sweep on p gate.  measure trans with x
        phase = self.deg2reg(
            cfg.calibrate_cfg.res_phase,
            gen_ch=cfg.DCS_cfg.res_ch,
        )
        freq_ro = self.soccfg.adcfreq(
            cfg.DCS_cfg.dds_freq / 2, cfg.calibrate_cfg.x_gate.gen, cfg.DCS_cfg.ro_ch
        )

        # set up x gate.  Hard coded to modulate at half the SD gate frequency
        phase = self.deg2reg(
            cfg.calibrate_cfg.res_phase + cfg.calibrate_cfg.res_phase_diff,
            gen_ch=cfg.calibrate_cfg.x_gate.gen,
        )
        freq_x = self.freq2reg(
            freq_ro, cfg.calibrate_cfg.x_gate.gen, ro_ch=cfg.DCS_cfg.ro_ch
        )
        self.set_pulse_registers(
            ch=cfg.calibrate_cfg.x_gate.gen,
            style=Waveform.CONSTANT,
            freq=freq_x,
            gain=cfg.calibrate_cfg.x_gate.gain,
            phase=phase,
            length=cfg.DCS_cfg.readout_length,
            mode=Mode.ONESHOT,
        )

        # Declare the ADC channel
        self.declare_readout(
            ch=cfg.DCS_cfg.ro_ch,
            length=cfg.DCS_cfg.readout_length,
            freq=freq_ro,
            gen_ch=cfg.calibrate_cfg.x_gate.gen,
            sel=Outsel.PRODUCT,
        )

        # set up SD channel
        self.declare_gen(ch=cfg.DCS_cfg.res_ch, nqz=1)
        freq_dac = self.soccfg.adcfreq(cfg.DCS_cfg.dds_freq, cfg.DCS_cfg.res_ch)
        freq = self.freq2reg(freq_dac, gen_ch=cfg.DCS_cfg.res_ch)

        self.set_pulse_registers(
            ch=cfg.DCS_cfg.res_ch,
            style=Waveform.CONSTANT,
            freq=freq,
            gain=cfg.DCS_cfg.pulse_gain_readout,
            phase=phase,
            length=cfg.DCS_cfg.readout_length,
            mode=Mode.ONESHOT,
        )

        # set up P gate. This chops p gate at a much faster rate than the readout frequency
        self.declare_gen(
            ch=cfg.calibrate_cfg.p_gate.gen, nqz=1
        )  # assuming SDAC sweep on P
        self.add_pulse(
            ch=cfg.calibrate_cfg.p_gate.gen,
            name="chop",
            idata=dac_pulses.chop(
                length=cfg.calibrate_cfg.p_gate.chop_time,
                maxv=cfg.calibrate_cfg.p_gate.gain,
            ),
        )
        self.add_pulse(
            ch=cfg.calibrate_cfg.p_gate.gen,
            name="baseband",
            idata=dac_pulses.baseband(),
        )  # initialize

        self.set_pulse_registers(
            ch=cfg.calibrate_cfg.p_gate.gen,
            freq=self.freq2reg(0, gen_ch=cfg.calibrate_cfg.p_gate.gen),
            phase=self.deg2reg(0, gen_ch=cfg.calibrate_cfg.p_gate.gen),
            style=Waveform.ARB,
            mode=Mode.PERIODIC,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.LAST,
            waveform="chop",
            gain=cfg.calibrate_cfg.p_gate.gain,
        )
        self.sync_all()
        self.trigger(pins=[0], t=0, width=100)

    def body(self):
        self.pulse(self.cfg.calibrate_cfg.x_gate.gen, t=0)
        self.set_pulse_registers(
            ch=self.cfg.calibrate_cfg.p_gate.gen,
            freq=self.freq2reg(0, gen_ch=self.cfg.calibrate_cfg.p_gate.gen),
            phase=self.deg2reg(0, gen_ch=self.cfg.calibrate_cfg.p_gate.gen),
            style=Waveform.ARB,
            mode=Mode.PERIODIC,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.LAST,
            waveform="chop",
            gain=self.cfg.calibrate_cfg.p_gate.gain,
        )
        self.pulse(self.cfg.calibrate_cfg.p_gate.gen, t=0)
        self.measure(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            pulse_ch=self.cfg.DCS_cfg.res_ch,
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            t=0,
            wait=True,
            syncdelay=0,
        )
        self.set_pulse_registers(
            ch=self.cfg.calibrate_cfg.p_gate.gen,
            freq=self.freq2reg(0, gen_ch=self.cfg.calibrate_cfg.p_gate.gen),
            phase=self.deg2reg(0, gen_ch=self.cfg.calibrate_cfg.p_gate.gen),
            style=Waveform.ARB,
            mode=Mode.PERIODIC,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.LAST,
            waveform="baseband",
            gain=0,
        )
        self.pulse(self.cfg.calibrate_cfg.p_gate.gen, t=0)
        self.sync_all(self.cfg.measure_delay)


class HSATune(averager_program.RAveragerProgram, readout.Readout):
    """baseband pulse 'tune_gate' for a given amount of time, and measure after the pulse ends.
    designed for tuning the HSA
    """

    def initialize(self):
        cfg = self.cfg
        self.init_dcs()
        self.declare_gen(
            ch=cfg.calibrate_cfg.tune_gate.gen, nqz=1
        )  # assuming SDAC sweep on P5
        self.add_pulse(
            ch=cfg.calibrate_cfg.tune_gate.gen,
            name="baseband",
            idata=dac_pulses.baseband(),
        )  # initialize
        self.default_pulse_registers(
            ch=cfg.calibrate_cfg.tune_gate.gen,
            freq=self.freq2reg(
                0, gen_ch=cfg.calibrate_cfg.tune_gate.gen, ro_ch=cfg.DCS_cfg.ro_ch
            ),
            phase=self.deg2reg(0, gen_ch=cfg.calibrate_cfg.tune_gate.gen),
            style=Waveform.ARB,
            mode=Mode.ONESHOT,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.LAST,
            waveform="baseband",
        )
        self.sync_all(100)

    def body(self):
        self.set_pulse_registers(
            ch=self.cfg.calibrate_cfg.tune_gate.gen,
            gain=self.cfg.calibrate_cfg.tune_gate.gain,
        )
        self.pulse(self.cfg.calibrate_cfg.tune_gate.gen, t=0)
        self.sync_all(self.cfg.calibrate_cfg.tune_gate.pulse_time)
        self.set_pulse_registers(ch=self.cfg.calibrate_cfg.tune_gate.gen, gain=0)
        self.pulse(self.cfg.gates.P5.gen, t=0)

        self.measure(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            pulse_ch=self.cfg.DCS_cfg.res_ch,
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            t=1,  # how long to wait after pulse before measuring! May want to change this
            wait=True,
            syncdelay=self.cfg.measure_delay,
        )


class PulseAndMeasure(averager_program.AveragerProgram):
    """simple loopback program"""

    def initialize(self):
        gain = self.cfg.DCS_cfg.pulse_gain_readout
        gen_ch = self.cfg.DCS_cfg.res_ch
        readout_ch = self.cfg.DCS_cfg.ro_ch
        freq = self.soccfg.freq2reg(self.cfg.DCS_cfg.dds_freq)
        pulse_time = self.cfg.DCS_cfg.length
        readout_length = self.cfg.DCS_cfg.readout_length
        pulse_mode = Mode.ONESHOT

        self.declare_gen(ch=gen_ch, nqz=1)
        self.set_pulse_registers(
            ch=gen_ch,
            freq=freq,
            phase=0,
            gain=gain,
            mode=pulse_mode,
            style=Waveform.CONSTANT,
            length=pulse_time,
        )
        freq_round = self.soccfg.adcfreq(
            self.cfg.DCS_cfg.readout_freq,
            self.cfg.DCS_cfg.res_ch,
            self.cfg.DCS_cfg.ro_ch,
        )  # pick out a frequency that works for both the ADC and DAC
        self.declare_readout(
            ch=readout_ch,
            freq=freq_round,
            length=readout_length,
            gen_ch=gen_ch,
            sel=Outsel.PRODUCT,
        )
        self.sync_all(100)  # pause before executing the measure function

    def body(self):
        self.measure(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            pulse_ch=self.cfg.DCS_cfg.res_ch,
            pins=[0],
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            t=0,
            wait=True,
            syncdelay=100,
        )
