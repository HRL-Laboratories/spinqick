"""qick programs for setting up pauli spin blockade"""

from qick import averager_program
from spinqick.qick_code import readout
from spinqick.qick_code.qick_utils import Outsel, Mode, Stdysel, Waveform, Defaults
from spinqick.helper_functions import dac_pulses
import numpy as np


class PSBExperiment(averager_program.NDAveragerProgram, readout.Readout):
    """This is a bare bones class to make a PSB measurement histogram"""

    def initialize(self):
        cfg = self.cfg
        if self.cfg.flush_2:
            self.init_psb()
        else:
            self.init_psb_expt()
        # make a dummy register to sweep over 'expts' times
        self.dummy_reg = self.new_reg(1)
        self.add_sweep(
            averager_program.QickSweep(
                self, self.dummy_reg, cfg.start, cfg.stop, cfg.expts
            )
        )
        self.sync_all()

    def body(self):
        if self.cfg.flush_2:
            self.readout_psb()
        else:
            self.readout_psb_pt1()
            self.readout_psb_pt2()
            self.readout_psb_pt3()


class IdleScan(averager_program.NDAveragerProgram, readout.Readout):
    def initialize(self):
        self.init_psb_expt()
        cfg = self.cfg
        sweep_cfg = cfg.psb_sweep_cfg

        if sweep_cfg.add_rf:
            self.declare_gen(
                ch=sweep_cfg.rf_gen,
                nqz=sweep_cfg.nqz,
            )
            freq = self.freq2reg(sweep_cfg.rf_freq, gen_ch=sweep_cfg.rf_gen)
            self.set_pulse_registers(
                ch=sweep_cfg.rf_gen,
                style=Waveform.CONSTANT,
                freq=freq,
                phase=0,
                mode=Mode.ONESHOT,
                gain=sweep_cfg.rf_gain,
                stdysel=Stdysel.ZERO,
                length=cfg.psb_cfg.times.idle_time,
            )

        ### now define useful qick registers
        # create QICKregister objects to keep track of the gain registers
        py_gen = cfg.psb_cfg.gates[str(sweep_cfg.gates.py.gate)].gen
        self.res_r_gain_py = self.get_gen_reg(py_gen, "gain")
        px_gen = cfg.psb_cfg.gates[str(sweep_cfg.gates.px.gate)].gen
        self.res_r_gain_px = self.get_gen_reg(px_gen, "gain")
        # declare a new register in the res_ch register page that keeps the sweep value
        self.res_r_gain_update_py = self.new_gen_reg(
            py_gen, init_val=sweep_cfg.gates.py.start, name="gain_update_py"
        )
        self.res_r_gain_update_px = self.new_gen_reg(
            px_gen, init_val=sweep_cfg.gates.px.start, name="gain_update_px"
        )
        # create Qicksweeps
        # first sweep is outermost axis according to QICK documentation
        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_update_py,
                cfg.psb_sweep_cfg.gates.py.start,
                cfg.psb_sweep_cfg.gates.py.stop,
                cfg.psb_sweep_cfg.gates.py.expts,
            )
        )
        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_update_px,
                cfg.psb_sweep_cfg.gates.px.start,
                cfg.psb_sweep_cfg.gates.px.stop,
                cfg.psb_sweep_cfg.gates.px.expts,
            )
        )

        # add a dummy register for averaging
        if cfg.expts != 1:
            self.dummy_reg = self.new_reg(1)
            self.add_sweep(
                averager_program.QickSweep(
                    self, self.dummy_reg, cfg["start"], cfg["stop"], cfg["expts"]
                )
            )

    def body(self):
        cfg = self.cfg.psb_cfg
        py_gate = self.cfg.psb_cfg.active_gates.py
        px_gate = self.cfg.psb_cfg.active_gates.px
        py_gen = self.cfg.psb_cfg.gates[str(py_gate)].gen
        px_gen = self.cfg.psb_cfg.gates[str(px_gate)].gen
        py_cfg = self.cfg.psb_cfg.gates[str(py_gate)]
        px_cfg = self.cfg.psb_cfg.gates[str(px_gate)]

        # reference measurement
        self.readout_psb_pt1()

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
            self.pulse(
                ch=cfg.gates[str(gatename)].gen, t="auto"
            )  # t='auto' should work

        self.sync_all()

        # go to idle
        self.set_pulse_registers(
            ch=px_gen, waveform="baseband", gain=px_cfg.gains.idle_gain
        )
        self.res_r_gain_px.set_to(self.res_r_gain_update_px)
        self.set_pulse_registers(
            ch=py_gen, waveform="baseband", gain=py_cfg.gains.idle_gain
        )
        self.res_r_gain_py.set_to(self.res_r_gain_update_py)
        self.pulse(ch=px_gen, t=0)
        self.pulse(ch=py_gen, t=0)

        if self.cfg.psb_sweep_cfg.add_rf:
            self.pulse(self.cfg.psb_sweep_cfg.rf_gen, t=0)
        self.sync_all(cfg.times.idle_time)

        # ramp to measurement window and measure
        self.readout_psb_pt3()


class PSBScanGeneral(averager_program.NDAveragerProgram, readout.Readout):
    """
    PSB setup script. Does a measurement window, idle or flush window scan.  We don't have enough waveform memory to save the
    whole baseband pulse for all the steps plus all the ramps, so we set stdysel to 'last' and just define the first few clock cycles of the baseband pulse.
    """

    def initialize(self):
        cfg = self.cfg
        sweep_cfg = cfg.psb_sweep_cfg
        scan_type = sweep_cfg.scan_type

        self.init_dcs()
        ### for the P and X gates in psb_cfg, loop through and setup each channel for ramping and baseband pulsing
        set_up_gates = [
            str(sweep_cfg.gates.px.gate),
            str(sweep_cfg.gates.py.gate),
            str(sweep_cfg.gates.x.gate),
        ]
        for gate_label in set_up_gates:
            self.declare_gen(ch=cfg.psb_cfg.gates[gate_label].gen, nqz=1)
            self.add_pulse(
                ch=cfg.psb_cfg.gates[str(gate_label)].gen,
                name="baseband",
                idata=dac_pulses.baseband(),
            )

            if scan_type in ["flush", "flush_2", "idle"]:
                if gate_label in [
                    str(sweep_cfg.gates.px.gate),
                    str(sweep_cfg.gates.py.gate),
                ]:
                    self.add_pulse(
                        ch=cfg.psb_cfg.gates[str(gate_label)].gen,
                        name="ramp1",
                        idata=dac_pulses.ramp(
                            ramp_length=cfg.psb_cfg.times.ramp1_time,
                            startv=cfg.psb_cfg.gates[str(gate_label)].gains.entry1_gain,
                            stopv=cfg.psb_cfg.gates[str(gate_label)].gains.window1_gain,
                        ),
                    )  # ramp to measurement window from (2,0)

                    self.add_pulse(
                        ch=cfg.psb_cfg.gates[str(gate_label)].gen,
                        name="ramp2",
                        idata=dac_pulses.ramp(
                            ramp_length=cfg.psb_cfg.times.ramp2_time,
                            startv=cfg.psb_cfg.gates[str(gate_label)].gains.entry2_gain,
                            stopv=cfg.psb_cfg.gates[str(gate_label)].gains.window1_gain,
                        ),
                    )  # ramp to measurement window from (1,1)
            if np.logical_and(
                self.cfg.psb_sweep_cfg.x_init, gate_label == str(sweep_cfg.gates.x.gate)
            ):
                self.add_pulse(
                    ch=cfg.psb_cfg.gates[str(sweep_cfg.gates.x.gate)].gen,
                    name="rampx",
                    idata=dac_pulses.ramp(
                        ramp_length=cfg.psb_cfg.times.rampx_time,
                        startv=cfg.psb_cfg.gates[
                            str(sweep_cfg.gates.x.gate)
                        ].gains.init2_gain,
                        stopv=cfg.psb_cfg.gates[
                            str(sweep_cfg.gates.x.gate)
                        ].gains.entry2_gain,
                    ),
                )  # ramp to (1,1) from (2,1) to initialize a random spin
            self.default_pulse_registers(
                ch=cfg.psb_cfg.gates[str(gate_label)].gen,
                style=Waveform.ARB,
                stdysel=Stdysel.LAST,
                outsel=Outsel.INPUT,
                mode=Mode.ONESHOT,
                freq=self.freq2reg(
                    0,
                    gen_ch=cfg.psb_cfg.gates[str(gate_label)].gen,
                    ro_ch=cfg.dcs_cfg.ro_ch,
                ),
                phase=self.deg2reg(0, gen_ch=cfg.psb_cfg.gates[str(gate_label)].gen),
            )

        ### now define useful qick registers
        # create QICKregister objects to keep track of the gain registers
        py_gen = cfg.psb_cfg.gates[str(sweep_cfg.gates.py.gate)].gen
        self.res_r_gain_py = self.get_gen_reg(py_gen, "gain")
        px_gen = cfg.psb_cfg.gates[str(sweep_cfg.gates.px.gate)].gen
        self.res_r_gain_px = self.get_gen_reg(px_gen, "gain")
        # declare a new register in the res_ch register page that keeps the sweep value
        self.res_r_gain_update_py = self.new_gen_reg(
            py_gen, init_val=sweep_cfg.gates.py.start, name="gain_update_py"
        )
        self.res_r_gain_update_px = self.new_gen_reg(
            px_gen, init_val=sweep_cfg.gates.px.start, name="gain_update_px"
        )

        # create Qicksweeps
        # first sweep is outermost axis according to QICK documentation
        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_update_py,
                cfg.psb_sweep_cfg.gates.py.start,
                cfg.psb_sweep_cfg.gates.py.stop,
                cfg.psb_sweep_cfg.gates.py.expts,
            )
        )
        self.add_sweep(
            averager_program.QickSweep(
                self,
                self.res_r_gain_update_px,
                cfg.psb_sweep_cfg.gates.px.start,
                cfg.psb_sweep_cfg.gates.px.stop,
                cfg.psb_sweep_cfg.gates.px.expts,
            )
        )
        # add a dummy register for averaging
        if cfg.expts != 1:
            self.dummy_reg = self.new_reg(1)
            self.add_sweep(
                averager_program.QickSweep(
                    self, self.dummy_reg, cfg["start"], cfg["stop"], cfg["expts"]
                )
            )
        self.sync_all()

    def body(self):
        ### define some useful variables
        relax_delay = self.soccfg.us2cycles(self.cfg.psb_cfg.relax_delay)
        scan_type = self.cfg.psb_sweep_cfg.scan_type
        py_gate = self.cfg.psb_sweep_cfg.gates.py.gate
        px_gate = self.cfg.psb_sweep_cfg.gates.px.gate
        x_gate = self.cfg.psb_sweep_cfg.gates.x.gate
        py_gen = self.cfg.psb_cfg.gates[str(py_gate)].gen
        px_gen = self.cfg.psb_cfg.gates[str(px_gate)].gen
        x_gen = self.cfg.psb_cfg.gates[str(x_gate)].gen
        py_cfg = self.cfg.psb_cfg.gates[str(py_gate)]
        px_cfg = self.cfg.psb_cfg.gates[str(px_gate)]
        x_cfg = self.cfg.psb_cfg.gates[str(x_gate)]

        ### pulse to flush/initialize.  If your scan type is flush, it will sweep this point
        self.set_pulse_registers(
            ch=py_gen, waveform="baseband", gain=py_cfg.gains.init_gain
        )
        if scan_type == "flush":
            self.res_r_gain_py.set_to(self.res_r_gain_update_py)
        self.set_pulse_registers(
            ch=px_gen, waveform="baseband", gain=px_cfg.gains.init_gain
        )
        if scan_type == "flush":
            self.res_r_gain_px.set_to(self.res_r_gain_update_px)
        self.pulse(ch=py_gen, t=0)
        self.pulse(ch=px_gen, t=0)
        self.sync_all(self.cfg.psb_cfg.times.init_time)

        ### go to measurement window via the entry point.
        if scan_type == "meas":
            ### for executing this with no ramp. This worked for us for testing,
            # but in the future we really want a ramp here.

            # go to entry point for ramp1_time duration
            self.set_pulse_registers(
                ch=px_gen, waveform="baseband", gain=px_cfg.gains.entry1_gain
            )
            self.res_r_gain_px.set_to(
                self.res_r_gain_update_px,
                "-",
                px_cfg.gains.window1_gain - px_cfg.gains.entry1_gain,
            )  # double check this
            self.set_pulse_registers(
                ch=py_gen, waveform="baseband", gain=py_cfg.gains.entry1_gain
            )
            self.res_r_gain_py.set_to(
                self.res_r_gain_update_py,
                "-",
                py_cfg.gains.window1_gain - py_cfg.gains.entry1_gain,
            )
            self.pulse(ch=px_gen, t=0)
            self.pulse(ch=py_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.ramp1_time)

            # go to window for settle_time duration
            self.set_pulse_registers(
                ch=px_gen, waveform="baseband", gain=px_cfg.gains.window1_gain
            )
            self.res_r_gain_px.set_to(self.res_r_gain_update_px)
            self.set_pulse_registers(
                ch=py_gen, waveform="baseband", gain=py_cfg.gains.window1_gain
            )
            self.res_r_gain_py.set_to(self.res_r_gain_update_py)
            self.pulse(ch=px_gen, t=0)
            self.pulse(ch=py_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.settle_time)

        ### ramp to measurement window if not sweeping meas window
        if scan_type in ["flush", "flush_2", "idle"]:
            for pgen in [px_gen, py_gen]:
                self.set_pulse_registers(ch=pgen, waveform="ramp1", gain=Defaults.GAIN)
                self.pulse(ch=pgen, t=0)
            self.sync_all(
                self.cfg.psb_cfg.times.ramp1_time + self.cfg.psb_cfg.times.settle_time
            )

        ### readout
        self.measure(
            adcs=[self.cfg.dcs_cfg.ro_ch],
            pulse_ch=self.cfg.dcs_cfg.res_ch,
            adc_trig_offset=self.cfg.dcs_cfg.adc_trig_offset,
            t=0,
            wait=True,
            syncdelay=0,
        )

        ### pulse to 1,2 flush if x_init is false.  Initialize a random pair of electrons
        if self.cfg.psb_sweep_cfg.x_init is False:
            self.set_pulse_registers(
                ch=px_gen, waveform="baseband", gain=px_cfg.gains.init2_gain
            )
            self.set_pulse_registers(
                ch=py_gen, waveform="baseband", gain=py_cfg.gains.init2_gain
            )
            if scan_type == "flush_2":
                self.res_r_gain_px.set_to(self.res_r_gain_update_px)
                self.res_r_gain_py.set_to(self.res_r_gain_update_py)
            self.pulse(ch=px_gen, t=0)
            self.pulse(ch=py_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.init2_time)

        # if initialization is involving the x-gate, pulse x to init2 and ramp back down
        if self.cfg.psb_sweep_cfg.x_init:
            # pulse into (1,2)
            self.set_pulse_registers(
                ch=x_gen, waveform="baseband", gain=x_cfg.gains.init2_gain
            )
            self.pulse(ch=x_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.init2_time)
            self.set_pulse_registers(ch=x_gen, waveform="rampx", gain=Defaults.GAIN)
            self.pulse(ch=x_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.rampx_time)
            # go to entry point for init2_time duration
            self.set_pulse_registers(
                ch=px_gen, waveform="baseband", gain=px_cfg.gains.entry2_gain
            )
            self.set_pulse_registers(
                ch=py_gen, waveform="baseband", gain=py_cfg.gains.entry2_gain
            )
            if scan_type == "meas":
                self.res_r_gain_px.set_to(
                    self.res_r_gain_update_px,
                    "-",
                    px_cfg.gains.window1_gain - px_cfg.gains.entry2_gain,
                )
                self.res_r_gain_py.set_to(
                    self.res_r_gain_update_py,
                    "-",
                    py_cfg.gains.window1_gain - py_cfg.gains.entry2_gain,
                )
            self.pulse(ch=px_gen, t=0)
            self.pulse(ch=py_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.init2_time)
            self.set_pulse_registers(ch=x_gen, waveform="baseband", gain=0)
            self.pulse(ch=x_gen, t=0)

        if scan_type == "meas":
            ### for executing this with no ramp. This worked for us for testing, but in the future we really want a ramp here.

            # go to entry point for ramp1_time duration
            self.set_pulse_registers(
                ch=px_gen, waveform="baseband", gain=px_cfg.gains.entry2_gain
            )
            self.res_r_gain_px.set_to(
                self.res_r_gain_update_px,
                "-",
                px_cfg.gains.window1_gain - px_cfg.gains.entry2_gain,
            )  # double check this
            self.set_pulse_registers(
                ch=py_gen, waveform="baseband", gain=py_cfg.gains.entry2_gain
            )
            self.res_r_gain_py.set_to(
                self.res_r_gain_update_py,
                "-",
                py_cfg.gains.window1_gain - py_cfg.gains.entry2_gain,
            )

            self.pulse(ch=px_gen, t=0)
            self.pulse(ch=py_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.ramp2_time)

            # go to window for settle_time duration
            self.set_pulse_registers(
                ch=px_gen, waveform="baseband", gain=px_cfg.gains.window1_gain
            )
            self.set_pulse_registers(
                ch=py_gen, waveform="baseband", gain=py_cfg.gains.window1_gain
            )
            self.res_r_gain_px.set_to(self.res_r_gain_update_px)
            self.res_r_gain_py.set_to(self.res_r_gain_update_py)
            self.pulse(ch=px_gen, t=0)
            self.pulse(ch=py_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.settle_time)

        ### ramp to measurement window
        if scan_type in ["flush", "flush_2", "idle"]:
            for p_gen in [px_gen, py_gen]:
                self.set_pulse_registers(ch=p_gen, waveform="ramp2", gain=Defaults.GAIN)
                self.pulse(ch=p_gen, t=0)
            self.sync_all(self.cfg.psb_cfg.times.ramp2_time)

        ### wait for settle_time and then readout
        self.measure(
            adcs=[self.cfg.dcs_cfg.ro_ch],
            pulse_ch=self.cfg.dcs_cfg.res_ch,
            adc_trig_offset=self.cfg.dcs_cfg.adc_trig_offset,
            t=self.cfg.psb_cfg.times.settle_time,
            wait=True,
            syncdelay=0,
        )

        ### turn DAC to "home" position
        self.set_pulse_registers(
            ch=px_gen, waveform="baseband", gain=px_cfg.gains.home_gain
        )
        self.set_pulse_registers(
            ch=py_gen, waveform="baseband", gain=py_cfg.gains.home_gain
        )
        self.pulse(ch=px_gen, t=0)
        self.pulse(ch=py_gen, t=0)

        self.sync_all(relax_delay)
