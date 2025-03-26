"""qick programs for single Loss-DiVincenzo qubit experiments"""

from qick import averager_program
from spinqick.qick_code import readout
from spinqick.qick_code.qick_utils import Mode, Stdysel, Waveform, Time
from spinqick.qick_code import spin_averager


class ScanRFFrequency(averager_program.RAveragerProgram, readout.Readout):
    """Scan RF frequency to look for EDSR signal"""

    def initialize(self):
        cfg = self.cfg.rf_expt

        self.init_psb_expt()

        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )

        # Declare the ADC channel, setup readout channel
        self.f_start = self.freq2reg(cfg.start, gen_ch=cfg.rf_gen)
        self.f_step = self.freq2reg(self.cfg.step, gen_ch=cfg.rf_gen)
        self.set_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=self.f_start,
            phase=0,
            length=cfg.rf_length,
            mode=Mode.ONESHOT,
            gain=cfg.rf_gain,
        )

        self.q_rp = self.ch_page(cfg.rf_gen)  # get register page for qubit_ch
        self.r_freq = self.sreg(
            cfg.rf_gen, "freq"
        )  # get frequency register for qubit_ch

        self.sync_all()

    def body(self):
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.sync_all()
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_length,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=0)
        self.sync_all()
        self.readout_psb_pt3()

    def update(self):
        self.mathi(
            self.q_rp, self.r_freq, self.r_freq, "+", self.f_step
        )  # update frequency list index


class RabiChevron(averager_program.NDAveragerProgram, readout.Readout):
    """2D scan of RF pulse time and frequency.  Here we use an outer python loop to sweep time"""

    def initialize(self):
        cfg = self.cfg.rf_expt
        freq = cfg.rf_freq

        self.init_psb_expt()

        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )

        freq = self.freq2reg(cfg.start, gen_ch=cfg.rf_gen)

        self.default_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=freq,
            phase=0,
            length=cfg.rf_time,
            mode=Mode.ONESHOT,
        )

        self.res_r_freq = self.get_gen_reg(cfg.rf_gen, "freq")
        self.add_sweep(
            averager_program.QickSweep(
                self, self.res_r_freq, cfg.start, cfg.stop, cfg.expts
            )
        )

        if self.cfg.expts > 1:
            # dummy sweep for repeated inner loop measurement
            self.dummy_reg = self.new_reg(1)
            self.add_sweep(
                averager_program.QickSweep(
                    self, self.dummy_reg, self.cfg.start, self.cfg.stop, self.cfg.expts
                )
            )
        self.sync_all()

    def body(self):
        trig_offset = self.cfg.rf_expt.trig_offset
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.set_pulse_registers(
            ch=self.cfg.rf_expt.rf_gen,
            gain=self.cfg.rf_expt.rf_gain,
            phrst=0,
            stdysel=Stdysel.ZERO,
        )
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)

        self.sync_all(
            self.cfg.rf_expt.rf_time + trig_offset + self.cfg.rf_expt.rf_cooldown
        )

        self.readout_psb_pt3()
        self.sync_all(self.cfg.rf_expt.rf_cooldown)


class RabiChevronV2(spin_averager.FlexyPSBAveragerProgram, readout.Readout):
    """2D scan of RF pulse time and frequency. This sweeps both time and frequency on board."""

    def initialize(self):
        cfg = self.cfg.rf_expt
        self.init_psb_expt()
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )

        ### pulse length is stored in the bottom 16 bits of the mode register.
        self.q_rf = self.ch_page(self.cfg.rf_expt.rf_gen)
        self.t_addr = self.sreg(self.cfg.rf_expt.rf_gen, "mode")
        self.freq_addr = self.sreg(self.cfg.rf_expt.rf_gen, "freq")
        self.r_wait = 3
        self.regwi(self.q_rf, self.r_wait, self.cfg.start)
        self.t_step = self.cfg.step
        self.f_step = self.cfg.step_outer

        ### supply necessary register information
        self.add_outer_reg_sweep(self.q_rf, self.freq_addr)
        self.add_inner_reg_sweep(self.q_rf, self.t_addr)
        self.add_inner_reg_sweep(self.q_rf, self.r_wait)

        # Declare the ADC channel, setup readout channel
        self.set_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=self.cfg.start_outer,
            phase=0,
            length=self.cfg.start,
            mode=Mode.ONESHOT,
            gain=cfg.rf_gain,
        )
        self.sync_all()

    def body(self):
        trig_offset = self.soccfg.us2cycles(0.1)
        full_width = self.cfg.start + self.cfg.step * self.cfg.expts
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.trigger(pins=[self.cfg.rf_expt.rf_trig_pin], t=0, width=full_width)
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)
        if self.cfg.rf_expt.trig_ignore:
            self.sync(self.q_rf, self.r_wait)
        else:
            self.sync_all(full_width)
        self.sync_all(self.soccfg.us2cycles(self.cfg.rf_expt.rf_cooldown))
        self.readout_psb_pt3()
        self.wait_all()

    def update(self):
        self.mathi(self.q_rf, self.t_addr, self.t_addr, "+", self.t_step)
        if self.cfg.rf_expt.trig_ignore:
            self.mathi(self.q_rf, self.r_wait, self.r_wait, "+", self.t_step)

    def update2(self):
        self.mathi(self.q_rf, self.freq_addr, self.freq_addr, "+", self.f_step)


class TimeRabi(averager_program.NDAveragerProgram, readout.Readout):
    """1D scan of RF pulse time. Tested.  This requires an outer python loop to run the time sweep."""

    def initialize(self):
        cfg = self.cfg.rf_expt

        self.init_psb_expt()

        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )
        self.default_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            phase=0,
            mode=Mode.ONESHOT,
        )

        if self.cfg.expts > 1:
            # dummy sweep for repeated inner loop measurement
            self.dummy_reg = self.new_reg(1)
            self.add_sweep(
                averager_program.QickSweep(
                    self, self.dummy_reg, self.cfg.start, self.cfg.stop, self.cfg.expts
                )
            )
        self.sync_all()

    def body(self):
        cfg = self.cfg.rf_expt
        trig_offset = self.soccfg.us2cycles(0.1)
        freq_prepulse = self.freq2reg(cfg.off_resonant_frequency, gen_ch=cfg.rf_gen)
        freq = self.freq2reg(cfg.rf_freq, gen_ch=cfg.rf_gen)
        time_prepulse = cfg.off_resonant_time
        time_rfpulse = cfg.rf_time

        self.readout_psb_pt1()
        self.readout_psb_pt2()

        self.set_pulse_registers(
            ch=self.cfg.rf_expt.rf_gen,
            gain=self.cfg.rf_expt.rf_gain,
            freq=freq_prepulse,
            length=time_prepulse,
            stdysel=Stdysel.ZERO,
        )
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=cfg.time_prepulse + 2 * trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)

        # currently no wait time between prepulse and rfpulse
        self.set_pulse_registers(
            ch=self.cfg.rf_expt.rf_gen,
            gain=self.cfg.rf_expt.rf_gain,
            freq=freq,
            length=time_rfpulse,
            stdysel=Stdysel.ZERO,
        )
        self.sync_all(time_prepulse)
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=cfg.time_rfpulse + 2 * trig_offset,
        )
        if time_rfpulse > 3:
            self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)

        self.synci(time_rfpulse)
        self.sync_all(self.cfg.rf_expt.rf_cooldown)
        self.readout_psb_pt3()


class TimeRabiV2(spin_averager.FlexyPSBAveragerProgram, readout.Readout):
    """1D scan of RF pulse time. Sweeps pulse time on RFSoC
    This works but we aren't sweeping trigger length on the FPGA so the trigger is just open for the max time every loop.
    """

    def initialize(self):
        cfg = self.cfg.rf_expt
        self.init_psb_expt()
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )
        freq = self.freq2reg(cfg.rf_freq, gen_ch=cfg.rf_gen)
        plength = self.cfg.start

        self.set_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=freq,
            phase=0,
            length=plength,
            mode=Mode.ONESHOT,
            gain=cfg.rf_gain,
            stdysel=Stdysel.ZERO,
        )

        ### pulse length is stored in the bottom 16 bits of the mode register.
        self.q_rf = self.ch_page(self.cfg.rf_expt.rf_gen)
        self.rf_addr = self.sreg(self.cfg.rf_expt.rf_gen, "mode")
        self.r_wait = 3
        self.regwi(self.q_rf, self.r_wait, self.cfg.start)
        self.step = self.cfg.step
        self.sync_all()

    def body(self):
        trig_offset = self.soccfg.us2cycles(0.1)
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.trigger(pins=[self.cfg.rf_expt.rf_trig_pin], t=0, width=self.cfg.stop)
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)
        if self.cfg.rf_expt.trig_ignore:
            self.sync(self.q_rf, self.r_wait)
        else:
            self.sync_all(self.cfg.stop)
        self.sync_all(self.soccfg.us2cycles(self.cfg.rf_expt.rf_cooldown))
        self.readout_psb_pt3()
        self.sync_all(self.soccfg.us2cycles(self.cfg.rf_expt.rf_cooldown))

    def update(self):
        self.mathi(self.q_rf, self.rf_addr, self.rf_addr, "+", self.step)
        if self.cfg.rf_expt.trig_ignore:
            self.mathi(self.q_rf, self.r_wait, self.r_wait, "+", self.step)


class AmplitudeRabi(averager_program.RAveragerProgram, readout.Readout):
    """1D scan of RF gain"""

    def initialize(self):
        cfg = self.cfg.rf_expt
        self.init_psb_expt()
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )
        freq = self.freq2reg(cfg.rf_freq, gen_ch=cfg.rf_gen)
        self.set_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=freq,
            phase=0,
            length=cfg.rf_time,
            gain=self.cfg.start,
            mode=Mode.ONESHOT,
        )

        self.q_rp = self.ch_page(cfg.rf_gen)  # get register page for rf_gen
        self.r_gain = self.sreg(cfg.rf_gen, "gain")  # get gain register for rf_gen

        self.sync_all(self.us2cycles(500))

    def body(self):
        trig_offset = self.cfg.rf_expt.trig_offset
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)
        self.sync_all(self.cfg.rf_expt.rf_cooldown)
        self.readout_psb_pt3()
        self.sync_all(self.soccfg.us2cycles(10))

    def update(self):
        self.mathi(self.q_rp, self.r_gain, self.r_gain, "+", self.cfg.step)


class RamseyFringes(spin_averager.FlexyPSBAveragerProgram, readout.Readout):
    """Play pi/2 pulse, wait for a time tau and advance phase,
    play another pi/2 pulse, measure.  Set ramsey_freq to zero to skip the phase
    incrementing.
    """

    def initialize(self):
        cfg = self.cfg.rf_expt
        self.init_psb_expt()
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )

        self.set_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=self.freq2reg(cfg.rf_freq, gen_ch=cfg.rf_gen),
            phase=0,
            length=cfg.rf_time,
            mode=Mode.ONESHOT,
            gain=cfg.rf_gain,
        )

        self.omega = cfg.ramsey_freq * 360
        self.q_rf = self.ch_page(self.cfg.rf_expt.rf_gen)
        self.r_phase2 = 4
        self.r_phase = self.sreg(self.cfg.rf_expt.rf_gen, "phase")
        self.regwi(self.q_rf, self.r_phase2, 0)
        self.r_wait = 3
        self.regwi(self.q_rf, self.r_wait, self.cfg.start)
        self.step = self.cfg.step

        self.add_inner_reg_sweep(self.q_rf, self.r_wait)
        self.add_inner_reg_sweep(self.q_rf, self.r_phase2)
        self.sync_all()

    def body(self):
        trig_offset = self.soccfg.us2cycles(0.1)
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + 2 * trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)
        self.mathi(self.q_rf, self.r_phase, self.r_phase2, "+", 0)
        self.sync_all()
        self.sync(self.q_rf, self.r_wait)

        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            width=self.cfg.rf_expt.rf_time + 2 * trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen)
        self.sync_all(self.cfg.rf_expt.rf_cooldown)
        self.readout_psb_pt3()
        self.sync_all(self.cfg.rf_expt.rf_cooldown)

    def update(self):
        self.mathi(
            self.q_rf, self.r_wait, self.r_wait, "+", self.cfg.step
        )  # update the time between two π/2 pulses
        self.mathi(
            self.q_rf,
            self.r_phase2,
            self.r_phase2,
            "+",
            self.soccfg.deg2reg(self.soccfg.cycles2us(self.cfg.step) * self.omega),
        )  # advance the phase of the LO for the second π/2 pulse


class Ramsey2D(spin_averager.FlexyPSBAveragerProgram, readout.Readout):
    """Play pi/2 pulse, wait for a time tau and play another pi/2 pulse, measure. Sweeps tau and frequency"""

    def initialize(self):
        cfg = self.cfg.rf_expt
        self.init_psb_expt()
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )

        self.set_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=self.cfg.start_outer,
            phase=0,
            length=cfg.rf_time,
            mode=Mode.ONESHOT,
            gain=cfg.rf_gain,
        )

        self.p_rf = self.ch_page(self.cfg.rf_expt.rf_gen)
        self.r_freq = self.sreg(self.cfg.rf_expt.rf_gen, "freq")
        self.r_wait = 3
        self.regwi(self.p_rf, self.r_wait, self.cfg.start)
        self.t_step = self.cfg.step
        self.f_step = self.cfg.step_outer

        self.add_inner_reg_sweep(self.p_rf, self.r_wait)
        self.add_outer_reg_sweep(self.p_rf, self.r_freq)
        self.sync_all()

    def body(self):
        trig_offset = self.soccfg.us2cycles(0.1)
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.trigger(
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + 2 * trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)
        self.sync_all()
        self.sync(self.p_rf, self.r_wait)
        self.trigger(
            pins=[self.cfg.rf_expt.rf_trig_pin],
            width=self.cfg.rf_expt.rf_time + 2 * trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen)
        self.sync_all(self.cfg.rf_expt.rf_cooldown)
        self.readout_psb_pt3()
        self.sync_all(self.cfg.rf_expt.rf_cooldown)

    def update(self):
        self.mathi(
            self.p_rf, self.r_wait, self.r_wait, "+", self.t_step
        )  # update the time between two π/2 pulses

    def update2(self):
        self.mathi(
            self.p_rf, self.r_freq, self.r_freq, "+", self.f_step
        )  # update the frequency


class AllXY(averager_program.NDAveragerProgram, readout.Readout):
    """QICK code to perform an AllXY experiment on a single spin"""

    def phase_reset(self):
        """play a 3 clock cycle pulse to reset phase"""
        rfphase = self.soccfg.deg2reg(0, gen_ch=self.cfg.rf_expt.rf_gen)
        self.set_pulse_registers(
            ch=self.cfg.rf_expt.rf_gen, phase=rfphase, phrst=1, length=3
        )
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=3 + self.cfg.rf_expt.trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=self.cfg.rf_expt.trig_offset)
        self.sync_all(self.cfg.rf_expt.pulse_delay)

    def pulse_id(self):
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + self.cfg.rf_expt.trig_offset,
        )
        self.sync_all(self.cfg.rf_expt.rf_time + self.cfg.rf_expt.pulse_delay)

    def pulse_xy(self, phi, xy):
        self.sync_all()
        if phi == 90:
            if xy == "X":
                rfphase = self.soccfg.deg2reg(0, gen_ch=self.cfg.rf_expt.rf_gen)
            else:
                rfphase = self.soccfg.deg2reg(90, gen_ch=self.cfg.rf_expt.rf_gen)
            self.set_pulse_registers(
                ch=self.cfg.rf_expt.rf_gen,
                phase=rfphase,
                length=self.cfg.rf_expt.rf_time,
                phrst=0,
            )
            self.trigger(
                adcs=None,
                pins=[self.cfg.rf_expt.rf_trig_pin],
                t=0,
                width=self.cfg.rf_expt.rf_time + self.cfg.rf_expt.trig_offset,
            )
            self.pulse(ch=self.cfg.rf_expt.rf_gen, t=self.cfg.rf_expt.trig_offset)
            self.sync_all(self.cfg.rf_expt.pulse_delay + self.cfg.rf_expt.trig_offset)

        if phi == 180:
            if xy == "X":
                rfphase = self.soccfg.deg2reg(0, gen_ch=self.cfg.rf_expt.rf_gen)
            else:
                rfphase = self.soccfg.deg2reg(90, gen_ch=self.cfg.rf_expt.rf_gen)
            self.set_pulse_registers(
                ch=self.cfg.rf_expt.rf_gen,
                phase=rfphase,
                length=self.cfg.rf_expt.rf_time * 2,
                phrst=0,
            )
            self.trigger(
                adcs=None,
                pins=[self.cfg.rf_expt.rf_trig_pin],
                t=0,
                width=self.cfg.rf_expt.rf_time * 2 + self.cfg.rf_expt.trig_offset,
            )
            self.pulse(ch=self.cfg.rf_expt.rf_gen, t=self.cfg.rf_expt.trig_offset)
            self.sync_all(self.cfg.rf_expt.pulse_delay + self.cfg.rf_expt.trig_offset)

    def xy_parser(self, m):
        gates = m.split(",")
        for gate in gates:
            if gate[0] == "X":
                phi = int(gate[1:])
                self.pulse_xy(phi, "X")
            if gate[0] == "Y":
                phi = int(gate[1:])
                self.pulse_xy(phi, "Y")
            if gate[0] == "I":
                self.pulse_id()

    def initialize(self):
        cfg = self.cfg.rf_expt
        self.init_psb_expt()
        if cfg.rf_freq > 3000:
            nqz = 2
        else:
            nqz = 1
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=nqz,
        )
        freq = self.freq2reg(cfg.rf_freq, gen_ch=cfg.rf_gen)
        self.default_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=freq,
            mode=Mode.ONESHOT,
            gain=cfg.rf_gain,
            stdysel=Stdysel.ZERO,
        )

        # dummy sweep for repeated inner loop measurement
        self.dummy_reg = self.new_reg(1)
        self.add_sweep(
            averager_program.QickSweep(
                self, self.dummy_reg, self.cfg.start, self.cfg.stop, self.cfg.expts
            )
        )
        self.sync_all()

    def body(self):
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.phase_reset()
        self.xy_parser(self.cfg.rf_expt.gate_set)
        self.sync_all(self.cfg.rf_expt.rf_cooldown)
        self.readout_psb_pt3()


class SweepPhase(averager_program.NDAveragerProgram, readout.Readout):
    """demonstrate phase control by applying two pi/2 pulses, sweep relative phase of the RF tone of the second pulse"""

    def initialize(self):
        cfg = self.cfg.rf_expt

        self.init_psb_expt()

        # TODO: Functionalize the NQZ check and place max NQZ=1 frequency in settings
        if cfg.rf_freq > 3000:
            nqz = 2
        else:
            nqz = 1
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=nqz,
        )

        freq = self.freq2reg(cfg.rf_freq, gen_ch=cfg.rf_gen)

        self.default_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=freq,
            mode=Mode.ONESHOT,
            gain=cfg.rf_gain,
            length=self.cfg.rf_expt.rf_time,
            stdysel=Stdysel.ZERO,
        )

        # dummy sweep for repeated inner loop measurement
        self.dummy_reg = self.new_reg(1)
        self.add_sweep(
            averager_program.QickSweep(
                self, self.dummy_reg, self.cfg.start, self.cfg.stop, self.cfg.expts
            )
        )
        self.sync_all()

    def body(self):
        self.readout_psb_pt1()
        self.readout_psb_pt2()

        rfphase = self.soccfg.deg2reg(
            self.cfg.rf_expt.phase_pulse_1, gen_ch=self.cfg.rf_expt.rf_gen
        )
        self.set_pulse_registers(
            ch=self.cfg.rf_expt.rf_gen,
            phase=rfphase,
            phrst=1,
        )
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + self.cfg.rf_expt.trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=self.cfg.rf_expt.trig_offset)
        self.sync_all(self.cfg.rf_expt.pulse_delay)

        rfphase = self.soccfg.deg2reg(
            self.cfg.rf_expt.phase_sweep, gen_ch=self.cfg.rf_expt.rf_gen
        )
        self.set_pulse_registers(
            ch=self.cfg.rf_expt.rf_gen,
            phase=rfphase,
            phrst=0,
        )
        self.trigger(
            adcs=None,
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + self.cfg.rf_expt.trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=self.cfg.rf_expt.trig_offset)
        self.sync_all(self.cfg.rf_expt.rf_cooldown)
        self.readout_psb_pt3()


class SpinEcho(spin_averager.FlexyPSBAveragerProgram, readout.Readout):
    """CPMG pulse sequence for measuring T2.  Pi/2 (x) pulse followed by a variable delay and a train
    of pi (x) pulses, each with time 2tau between them.  Follow up with another Pi/2 (x) pulse and measure.
    """

    def initialize(self):
        cfg = self.cfg.rf_expt
        self.init_psb_expt()
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=cfg.nqz,
        )

        freq = self.freq2reg(cfg.rf_freq, gen_ch=cfg.rf_gen)
        plength = cfg.rf_time
        self.default_pulse_registers(
            ch=cfg.rf_gen,
            style=Waveform.CONSTANT,
            freq=freq,
            length=plength,
            mode=Mode.ONESHOT,
            gain=cfg.rf_gain,
            stdysel=Stdysel.ZERO,
            phase=0,
        )

        self.p_rf = self.ch_page(self.cfg.rf_expt.rf_gen)
        self.r_wait = 3
        self.regwi(self.p_rf, self.r_wait, self.cfg.start)
        self.step = self.cfg.step
        self.add_inner_reg_sweep(self.p_rf, self.r_wait)

    def body(self):
        trig_offset = self.soccfg.us2cycles(0.1)
        self.readout_psb_pt1()
        self.readout_psb_pt2()
        self.set_pulse_registers(
            ch=self.cfg.rf_expt.rf_gen,
        )
        self.trigger(
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + 2 * trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)
        self.sync_all()  # need this?

        # pi pulse for specified number of pulses
        for n in range(self.cfg.rf_expt.n_echoes):
            # wait for tau cycles
            self.sync(self.p_rf, self.r_wait)
            self.set_pulse_registers(
                ch=self.cfg.rf_expt.rf_gen,
            )
            self.trigger(
                pins=[self.cfg.rf_expt.rf_trig_pin],
                t=0,
                width=2 * self.cfg.rf_expt.rf_time + 2 * trig_offset,
            )
            # play pi pulse (2 pi/2 pulses)
            self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)
            self.pulse(ch=self.cfg.rf_expt.rf_gen, t=Time.AUTO)

            # wait for tau cycles
            self.sync_all()
            self.sync(self.p_rf, self.r_wait)

        # pi/2 pulse
        self.set_pulse_registers(
            ch=self.cfg.rf_expt.rf_gen,
        )
        self.trigger(
            pins=[self.cfg.rf_expt.rf_trig_pin],
            t=0,
            width=self.cfg.rf_expt.rf_time + 2 * trig_offset,
        )
        self.pulse(ch=self.cfg.rf_expt.rf_gen, t=trig_offset)

        # readout
        self.sync_all(self.cfg.rf_expt.rf_cooldown)
        self.readout_psb_pt3()
        self.sync_all(self.cfg.rf_expt.rf_cooldown)

    def update(self):
        self.mathi(
            self.p_rf, self.r_wait, self.r_wait, "+", self.cfg.step
        )  # update the time between two π/2 pulses
