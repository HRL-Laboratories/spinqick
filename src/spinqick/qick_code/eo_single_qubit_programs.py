"""qick programs for exchange only experiments"""

from qick import averager_program
from spinqick.qick_code import readout
from spinqick.qick_code.qick_utils import Outsel, Mode, Stdysel, Waveform
from spinqick.helper_functions import dac_pulses
from spinqick.qick_code import spin_averager


class DoNonEquilibriumCell(spin_averager.PSBAveragerProgram, readout.Readout):
    """
    Turn on exchange and scan the non-equilibrium cell. Right now this is set up to just do dots 2 and 3, with psb on dots 1 and 2
    """

    def initialize(self):
        cfg = self.cfg
        self.init_psb_expt()

        # setup pulses on p3
        p3_gen = cfg.EO_cfg.gates.p3.gen
        x_gen = cfg.EO_cfg.gates.x.gen
        self.declare_gen(ch=x_gen, nqz=1)
        self.declare_gen(ch=p3_gen, nqz=1)

        self.add_pulse(
            ch=p3_gen,
            name="baseband",
            idata=dac_pulses.baseband(),
        )

        self.add_pulse(
            ch=x_gen,
            name="baseband",
            idata=dac_pulses.baseband(length=cfg.EO_cfg.gates.x.pulse_time),
        )
        self.set_pulse_registers(
            ch=x_gen,
            waveform="baseband",
            freq=self.freq2reg(0, gen_ch=x_gen, ro_ch=cfg.DCS_cfg.ro_ch),
            phase=self.deg2reg(0, gen_ch=x_gen),
            style=Waveform.ARB,
            mode=Mode.ONESHOT,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.ZERO,
            gain=cfg.EO_cfg.gates.x.gain,
        )

        self.set_pulse_registers(
            ch=p3_gen,
            waveform="baseband",
            freq=self.freq2reg(0, gen_ch=p3_gen, ro_ch=cfg.DCS_cfg.ro_ch),
            phase=self.deg2reg(0, gen_ch=p3_gen),
            style=Waveform.ARB,
            mode=Mode.ONESHOT,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.LAST,
            gain=cfg.EO_cfg.gates.p3.start,
        )
        self.res_r_gain_p3 = self.get_gen_reg(p3_gen, "gain")
        p2_gen = cfg.PSB_cfg.gates[str(cfg.EO_cfg.gates.p2.gate)].gen
        self.res_r_gain_p2 = self.get_gen_reg(p2_gen, "gain")
        # declare a new register in the res_ch register page that keeps the sweep value for P2, since this gate is used in the psb sequence too
        self.res_r_gain_update_p2 = self.new_gen_reg(
            p2_gen, init_val=cfg.EO_cfg.gates.p2.start, name="gain_update_p2"
        )

        # create Qicksweeps for p2 and p3
        # first sweep is outermost axis according to QICK documentation
        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_p3,
                cfg.EO_cfg.gates.p3.start,
                cfg.EO_cfg.gates.p3.stop,
                cfg.EO_cfg.gates.p3.expts,
            )
        )
        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_update_p2,
                cfg.EO_cfg.gates.p2.start,
                cfg.EO_cfg.gates.p2.stop,
                cfg.EO_cfg.gates.p2.expts,
            )
        )

        self.sync_all()

    def body(self):
        ### define some useful variables
        relax_delay = self.soccfg.us2cycles(self.cfg.PSB_cfg.relax_delay)

        self.readout_psb_pt1()  # get a reference measurement
        self.sync_all(relax_delay)
        self.readout_psb_pt2_a()

        # pulse to p3 value
        self.pulse(ch=self.cfg.EO_cfg.gates.p3.gen, t=0)
        # pulse to p2 value
        self.set_pulse_registers(
            ch=self.cfg.EO_cfg.gates.p2.gen,
            waveform="baseband",
            gain=0,
        )
        self.res_r_gain_p2.set_to(self.res_r_gain_update_p2)
        self.pulse(ch=self.cfg.EO_cfg.gates.p2.gen, t=0)
        # pulse x gate after idle_time
        self.pulse(ch=self.cfg.EO_cfg.gates.x.gen, t=self.cfg.PSB_cfg.times.idle_time)
        self.sync_all()
        # measure
        self.readout_psb_pt3()


class DoFingerprint(spin_averager.PSBAveragerProgram, readout.Readout):
    """
    Scan detuning axis and exchange gate amplitude
    """

    def initialize(self):
        cfg = self.cfg
        self.init_psb_expt()

        # setup pulses on p3
        p3_gen = cfg.EO_cfg.gates.p3.gen
        x_gen = cfg.EO_cfg.gates.x.gen
        self.declare_gen(ch=x_gen, nqz=1)
        self.declare_gen(ch=p3_gen, nqz=1)

        self.add_pulse(
            ch=p3_gen,
            name="baseband",
            idata=dac_pulses.baseband(),
        )

        self.add_pulse(
            ch=x_gen,
            name="baseband",
            idata=dac_pulses.baseband(length=cfg.EO_cfg.gates.x.pulse_time),
        )
        self.set_pulse_registers(
            ch=x_gen,
            waveform="baseband",
            freq=self.freq2reg(0, gen_ch=x_gen, ro_ch=cfg.DCS_cfg.ro_ch),
            phase=self.deg2reg(0, gen_ch=x_gen),
            style=Waveform.ARB,
            mode=Mode.ONESHOT,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.ZERO,
            gain=cfg.EO_cfg.gates.x.start,
        )

        self.set_pulse_registers(
            ch=p3_gen,
            waveform="baseband",
            freq=self.freq2reg(0, gen_ch=p3_gen, ro_ch=cfg.DCS_cfg.ro_ch),
            phase=self.deg2reg(0, gen_ch=p3_gen),
            style=Waveform.ARB,
            mode=Mode.ONESHOT,
            outsel=Outsel.INPUT,
            stdysel=Stdysel.LAST,
            gain=cfg.EO_cfg.gates.p3.start,
        )
        self.res_r_gain_p3 = self.get_gen_reg(p3_gen, "gain")
        p2_gen = cfg.PSB_cfg.gates[str(cfg.EO_cfg.gates.p2.gate)].gen
        self.res_r_gain_p2 = self.get_gen_reg(p2_gen, "gain")
        # declare a new register in the res_ch register page that keeps the sweep value for P2, since this gate is used in the psb sequence too
        self.res_r_gain_update_p2 = self.new_gen_reg(
            p2_gen, init_val=cfg.EO_cfg.gates.p2.start, name="gain_update_p2"
        )
        self.res_r_gain_x = self.get_gen_reg(x_gen, "gain")

        # create Qicksweeps for p2 and p3
        # first sweep is outermost axis according to QICK documentation
        p3_sweep = averager_program.QickSweep(
            self,
            self.res_r_gain_p3,
            cfg.EO_cfg.gates.p3.start,
            cfg.EO_cfg.gates.p3.stop,
            cfg.EO_cfg.gates.p3.expts,
        )
        p2_sweep = averager_program.QickSweep(
            self,
            self.res_r_gain_update_p2,
            cfg.EO_cfg.gates.p2.start,
            cfg.EO_cfg.gates.p2.stop,
            cfg.EO_cfg.gates.p2.expts,
        )

        self.add_sweep(averager_program.merge_sweeps([p3_sweep, p2_sweep]))
        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_x,
                cfg.EO_cfg.gates.x.start,
                cfg.EO_cfg.gates.x.stop,
                cfg.EO_cfg.gates.x.expts,
            )
        )

        self.sync_all()

    def body(self):
        ### define some useful variables
        relax_delay = self.soccfg.us2cycles(self.cfg.PSB_cfg.relax_delay)

        self.readout_psb_pt1()  # get a reference measurement
        self.sync_all(relax_delay)
        self.readout_psb_pt2_a()

        # pulse to p3 value
        self.pulse(ch=self.cfg.EO_cfg.gates.p3.gen, t=0)
        # pulse to p2 value
        self.set_pulse_registers(
            ch=self.cfg.EO_cfg.gates.p2.gen,
            waveform="baseband",
            gain=0,
        )
        self.res_r_gain_p2.set_to(self.res_r_gain_update_p2)
        self.pulse(ch=self.cfg.EO_cfg.gates.p2.gen, t=0)
        # pulse x gate after idle_time
        self.pulse(ch=self.cfg.EO_cfg.gates.x.gen, t=self.cfg.PSB_cfg.times.idle_time)
        self.sync_all()
        # measure
        self.readout_psb_pt3()
