"""
AveragerProgram -type qick code for the tune_electrostatics module
"""

from qick import averager_program
from spinqick.qick_code import readout
from spinqick.qick_code.qick_utils import Outsel, Mode, Stdysel, Waveform, Defaults
from spinqick.helper_functions import dac_pulses


class BasebandPulseGvG(averager_program.NDAveragerProgram, readout.Readout):
    """QICK class to sweep two gate voltages against each other, using QICK not slow DAC"""

    def initialize(self):
        # Define shortcuts for more convenient access of experiment parameters
        cfg = self.cfg.gvg_expt

        # Declare the DAC channel
        self.init_dcs()
        self.declare_gen(ch=cfg.gates.gy.gen, nqz=1)
        self.declare_gen(ch=cfg.gates.gx.gen, nqz=1)

        if cfg.add_pat:
            if cfg.pat_freq < 3000:
                nqz = 1
            else:
                nqz = 2
            self.declare_gen(cfg.pat_gen, nqz=nqz)
            freq = self.freq2reg(cfg.pat_freq, gen_ch=cfg.pat_gen)
            self.set_pulse_registers(
                ch=cfg.pat_gen,
                style=Waveform.CONSTANT,
                freq=freq,
                phase=0,
                mode=Mode.ONESHOT,
                gain=cfg.pat_gain,
                stdysel=Stdysel.ZERO,
                length=self.us2cycles(self.cfg.gvg_expt.measure_delay)
                + self.cfg.DCS_cfg.length,
            )

        # Set the default pulse registers to be used in the body when calling the pulse
        self.add_pulse(
            ch=cfg.gates.gy.gen,
            name="baseband",
            idata=dac_pulses.baseband(),
        )
        self.default_pulse_registers(
            ch=cfg.gates.gy.gen,
            waveform="baseband",
            freq=self.freq2reg(
                0, gen_ch=cfg.gates.gy.gen, ro_ch=self.cfg.DCS_cfg.ro_ch
            ),
            phase=self.deg2reg(0, gen_ch=cfg.gates.gy.gen),
            style=Waveform.ARB,
            mode=Mode.ONESHOT,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.LAST,
            gain=Defaults.GAIN,
        )
        self.add_pulse(
            ch=cfg.gates.gx.gen,
            name="baseband",
            idata=dac_pulses.baseband(),
        )
        self.default_pulse_registers(
            ch=cfg.gates.gx.gen,
            waveform="baseband",
            freq=self.freq2reg(
                0, gen_ch=cfg.gates.gx.gen, ro_ch=self.cfg.DCS_cfg.ro_ch
            ),
            phase=self.deg2reg(0, gen_ch=cfg.gates.gx.gen),
            style=Waveform.ARB,
            mode=Mode.ONESHOT,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.LAST,
            gain=Defaults.GAIN,
        )

        self.res_r_gain_gy = self.get_gen_reg(cfg.gates.gy.gen, "gain")
        self.res_r_gain_gx = self.get_gen_reg(cfg.gates.gx.gen, "gain")

        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_gy,
                cfg.gates.gy.start,
                cfg.gates.gy.stop,
                cfg.gates.gy.expts,
            )
        )
        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_gx,
                cfg.gates.gx.start,
                cfg.gates.gx.stop,
                cfg.gates.gx.expts,
            )
        )

        self.synci(200)

    def body(self):
        self.set_pulse_registers(
            ch=self.cfg.gvg_expt.gates.gy.gen,
        )
        self.pulse(ch=self.cfg.gvg_expt.gates.gy.gen, t=0)
        self.set_pulse_registers(
            ch=self.cfg.gvg_expt.gates.gx.gen,
        )
        self.pulse(ch=self.cfg.gvg_expt.gates.gx.gen, t=0)
        measure_delay_cycles = self.us2cycles(self.cfg.gvg_expt.measure_delay)
        if self.cfg.gvg_expt.add_pat:
            self.pulse(ch=self.cfg.gvg_expt.pat_gen, t=0)
        self.sync_all(measure_delay_cycles)

        ### readout
        self.measure(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            pulse_ch=self.cfg.DCS_cfg.res_ch,
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            pins=[self.cfg.gvg_expt.trig_pin],
            t=0,
            wait=True,
            syncdelay=measure_delay_cycles,
        )


class GvG(averager_program.RAveragerProgram, readout.Readout):
    """Time this carefully with your DC voltage source.  This runs once per dac ramp, based on a trigger"""

    def initialize(self):
        self.init_dcs()
        self.trigger(pins=[self.cfg.gvg_expt.trig_pin], width=100)
        meas_delay = self.soccfg.us2cycles(self.cfg.gvg_expt.measure_delay)
        self.sync_all()
        self.synci(meas_delay)

    def body(self):
        # Plays our readout pulse at t = measure_delay, waits again for measure_delay after pulse. This is just creating a buffer around the readout window
        self.trigger(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            t=0,
        )
        self.pulse(ch=self.cfg.DCS_cfg.res_ch, t=0)

        self.synci(
            self.soccfg.us2cycles(self.cfg.gvg_expt.measure_delay * 2)
            + self.cfg.DCS_cfg.length
        )

        self.wait_all()
