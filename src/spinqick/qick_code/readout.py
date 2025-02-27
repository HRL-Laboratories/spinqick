"""Module to implement dot charge sensor readout with qick. Includes routines that incorporate pauli spin blockade as well."""

from qick import asm_v1

from spinqick.qick_code.qick_utils import Outsel, Mode, Stdysel, Waveform, Defaults

from spinqick.helper_functions import dac_pulses


class Readout(asm_v1.QickProgram):
    def init_dcs(self):
        """Initialize DCS generator and readout channel."""
        cfg = self.cfg.DCS_cfg
        self.declare_gen(
            ch=cfg.res_ch,
            nqz=1,
        )
        freq_round = self.soccfg.adcfreq(
            cfg.dds_freq, cfg.res_ch, cfg.ro_ch
        )  # pick out a frequency that works for both the ADC and DAC
        freq = self.freq2reg(
            freq_round, gen_ch=cfg.res_ch, ro_ch=cfg.ro_ch
        )  # convert frequency to dac frequency

        # Declare the ADC channel, setup readout channel
        self.declare_readout(
            ch=cfg.ro_ch,
            length=cfg.readout_length,
            freq=freq_round,
            gen_ch=cfg.res_ch,
            sel=Outsel.PRODUCT,
        )
        self.set_pulse_registers(
            ch=cfg.res_ch,
            style=Waveform.CONSTANT,
            freq=freq,
            gain=cfg.pulse_gain_readout,
            phase=0,
            length=cfg.length,
            mode=Mode.ONESHOT,
            phrst=1,
        )

    def init_psb(self):
        """Initialize Pauli spin blockade experiment.  Right now we are using this for meashist()."""
        cfg = self.cfg.PSB_cfg
        self.init_dcs()
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.declare_gen(ch=cfg.gates[str(gatename)].gen, nqz=1)
            self.add_pulse(
                ch=cfg.gates[str(gatename)].gen,
                name="baseband",
                idata=dac_pulses.baseband(),
            )
            self.default_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                freq=self.freq2reg(
                    0, gen_ch=cfg.gates[str(gatename)].gen, ro_ch=self.cfg.DCS_cfg.ro_ch
                ),
                phase=self.deg2reg(0, gen_ch=cfg.gates[str(gatename)].gen),
                style=Waveform.ARB,
                mode=Mode.ONESHOT,
                outsel=Outsel.INPUT,
                stdysel=Stdysel.LAST,
            )
            self.add_pulse(
                ch=cfg.gates[str(gatename)].gen,
                name="ramp1",
                idata=dac_pulses.ramp(
                    ramp_length=cfg.times.ramp1_time,
                    startv=cfg.gates[str(gatename)].gains.entry1_gain,
                    stopv=cfg.gates[str(gatename)].gains.window1_gain,
                ),
            )
            self.add_pulse(
                ch=cfg.gates[str(gatename)].gen,
                name="ramp2",
                idata=dac_pulses.ramp(
                    ramp_length=cfg.times.ramp2_time,
                    startv=cfg.gates[str(gatename)].gains.entry2_gain,
                    stopv=cfg.gates[str(gatename)].gains.window1_gain,
                ),
            )

    def init_psb_expt(self):
        """Initialize a PSB readout for an experiment that involves ramping through measurement window."""

        cfg = self.cfg.PSB_cfg
        self.init_dcs()
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.declare_gen(ch=cfg.gates[str(gatename)].gen, nqz=1)
            self.add_pulse(
                ch=cfg.gates[str(gatename)].gen,
                name="baseband",
                idata=dac_pulses.baseband(),
            )
            self.default_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                freq=self.freq2reg(
                    0, gen_ch=cfg.gates[str(gatename)].gen, ro_ch=self.cfg.DCS_cfg.ro_ch
                ),
                phase=self.deg2reg(0, gen_ch=cfg.gates[str(gatename)].gen),
                style=Waveform.ARB,
                mode=Mode.ONESHOT,
                outsel=Outsel.INPUT,
                stdysel=Stdysel.LAST,
            )
            self.add_pulse(
                ch=cfg.gates[str(gatename)].gen,
                name="ramp1",
                idata=dac_pulses.ramp(
                    ramp_length=cfg.times.ramp1_time,
                    startv=cfg.gates[str(gatename)].gains.entry1_gain,
                    stopv=cfg.gates[str(gatename)].gains.window1_gain,
                ),
            )
            self.add_pulse(
                ch=cfg.gates[str(gatename)].gen,
                name="ramp2",
                idata=dac_pulses.ramp(
                    ramp_length=cfg.times.ramp2_time,
                    startv=cfg.gates[str(gatename)].gains.entry2_gain,
                    stopv=cfg.gates[str(gatename)].gains.window1_gain,
                ),
            )
            self.add_pulse(
                ch=cfg.gates[str(gatename)].gen,
                name="ramp3",
                idata=dac_pulses.ramp(
                    ramp_length=cfg.times.ramp2_time,
                    startv=cfg.gates[str(gatename)].gains.window1_gain,
                    stopv=cfg.gates[str(gatename)].gains.entry2_gain,
                ),
            )

    def readout_psb(self):
        """Simple readout for meashist"""
        cfg = self.cfg.PSB_cfg
        relax_delay = self.soccfg.us2cycles(cfg["relax_delay"])
        self.sync_all(relax_delay)

        # pulse to flush/initialize
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.init_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.init_time)

        # ramp to measurement window
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen, waveform="ramp1", gain=Defaults.GAIN
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.ramp1_time + cfg.times.settle_time)

        # readout
        self.measure(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            pulse_ch=self.cfg.DCS_cfg.res_ch,
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            t=0,
            wait=True,
            syncdelay=100,
        )

        # pulse to 1,2 charge cell flush (or somewhere else defined by init2_gain) to initialize a mix of singlets and triplets
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.init2_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.init_time)

        # ramp back to measurement window
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen, waveform="ramp2", gain=Defaults.GAIN
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.ramp2_time + cfg.times.settle_time)

        # second readout
        self.measure(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            pulse_ch=self.cfg.DCS_cfg.res_ch,
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            t=0,
            wait=True,
            syncdelay=100,
        )

        # turn DAC to "home" position
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.home_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)

        self.sync_all(relax_delay)

    def readout_psb_pt1(self):
        """Initialize and perform a reference measurement"""

        cfg = self.cfg.PSB_cfg
        relax_delay = self.soccfg.us2cycles(cfg["relax_delay"])
        self.sync_all(relax_delay)

        # pulse to flush/initialize
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.init_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.init_time)

        # ramp to measurement window
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen, waveform="ramp1", gain=Defaults.GAIN
            )  # gain has to be 32000 for the ramp to work ?
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.ramp1_time + cfg.times.settle_time)

        # readout
        self.measure(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            pulse_ch=self.cfg.DCS_cfg.res_ch,
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            t=0,
            wait=True,
            syncdelay=100,
        )

    def readout_psb_pt2(self):
        """Initialize, ramp through measurement window, go to idle point"""

        cfg = self.cfg.PSB_cfg
        relax_delay = self.soccfg.us2cycles(cfg["relax_delay"])
        self.sync_all(relax_delay)

        # pulse to flush/initialize
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.init_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.init_time)

        # ramp through measurement window
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen, waveform="ramp1", gain=Defaults.GAIN
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen, waveform="ramp3", gain=Defaults.GAIN
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=cfg.times.ramp1_time)

        self.sync_all()

        # go to idle
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.idle_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.idle_time)

    def readout_psb_pt3(self):
        """End of experiment.  Start at idle, ramp to measurement window, measure"""
        cfg = self.cfg.PSB_cfg
        # go to idle
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.idle_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all()

        # ramp back to measurement window
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen, waveform="ramp2", gain=Defaults.GAIN
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.ramp2_time + cfg.times.settle_time)

        # readout again
        self.measure(
            adcs=[self.cfg.DCS_cfg.ro_ch],
            pulse_ch=self.cfg.DCS_cfg.res_ch,
            adc_trig_offset=self.cfg.DCS_cfg.adc_trig_offset,
            t=0,
            wait=True,
            syncdelay=100,
        )

        # return DAC to "home" position
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.home_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all()

    def readout_psb_pt2_a(self):
        """Initialize and ramp through measurement window, but stop at end of ramp"""

        cfg = self.cfg.PSB_cfg
        relax_delay = self.soccfg.us2cycles(cfg["relax_delay"])
        self.sync_all(relax_delay)

        # pulse to flush/initialize
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen,
                waveform="baseband",
                gain=cfg.gates[str(gatename)].gains.init_gain,
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        self.sync_all(cfg.times.init_time)

        # ramp through measurement window
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen, waveform="ramp1", gain=Defaults.GAIN
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=0)
        for gate_label in cfg.active_gates:
            gatename = cfg.active_gates[str(gate_label)]
            self.set_pulse_registers(
                ch=cfg.gates[str(gatename)].gen, waveform="ramp3", gain=Defaults.GAIN
            )
            self.pulse(ch=cfg.gates[str(gatename)].gen, t=cfg.times.ramp1_time)

        self.sync_all()
