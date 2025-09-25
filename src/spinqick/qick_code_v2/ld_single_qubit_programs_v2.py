"""
Single LD qubit setup and control experiments
"""

from qick import asm_v2
from spinqick.core import readout_v2
from spinqick.models import experiment_models
from spinqick.core import qick_utils
from spinqick.core import ld_pulses


class IdleScanWithRf(asm_v2.AveragerProgramV2):
    """Perform a 2D sweep of the idle point with Rf turned on"""

    def _initialize(self, cfg: experiment_models.IdleScanConfig):
        idle_x_gains = asm_v2.QickSweep1D("x_sweep", cfg.gx_start, cfg.gx_stop)
        idle_y_gains = asm_v2.QickSweep1D("y_sweep", cfg.gy_start, cfg.gy_stop)
        readout_v2.init_dcs(self, cfg.dcs_cfg)
        readout_v2.setup_spam_gens(self, cfg.psb_cfg)
        readout_v2.init_spam_point_sweep(
            self,
            cfg.psb_cfg,
            "idle",
            cfg.gx_gen,
            cfg.gy_gen,
            idle_x_gains,
            idle_y_gains,
        )
        self.declare_gen(
            ch=cfg.rf_gen,
            nqz=qick_utils.check_nyquist(cfg.rf_freq, cfg.rf_gen, self.soccfg),
        )
        self.add_pulse(
            ch=cfg.rf_gen,
            name="rf",
            style="const",
            freq=cfg.rf_freq,
            phase=0,
            gain=cfg.rf_gain,
            length=cfg.rf_length,
        )
        # self.trigger(pins=[0])  # for testing only

        self.add_loop("full_avgs", cfg.full_avgs)  # add a loop
        self.add_loop("x_sweep", cfg.gx_expts)
        self.add_loop("y_sweep", cfg.gy_expts)
        self.add_loop("shots", cfg.point_avgs)  # add a loop

    def _body(self, cfg: experiment_models.IdleScanConfig):
        if cfg.reference:
            readout_v2.psb_fm(self, cfg.psb_cfg, cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.psb_cfg)
        self.pulse(cfg.rf_gen, "rf")
        self.delay_auto()
        readout_v2.psb_em(self, cfg.psb_cfg, cfg.dcs_cfg)


class ScanRfFrequency(asm_v2.AveragerProgramV2):
    """Scan RF frequency to look for EDSR signal"""

    def _initialize(self, cfg: experiment_models.RfSweep):
        readout_v2.init_dcs(self, cfg.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.ro_cfg.psb_cfg)
        nqz = qick_utils.check_nyquist(cfg.start, cfg.gen, self.soccfg)
        self.declare_gen(cfg.gen, nqz=nqz)
        freq_sweep = asm_v2.QickSweep1D("freq", cfg.start, cfg.stop)
        self.add_pulse(
            ch=cfg.gen,
            name="rf_drive",
            style="const",
            freq=freq_sweep,
            phase=0,
            gain=cfg.pulse_gain,
            length=cfg.pulse_length,
        )

        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("freq", cfg.expts)
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.RfSweep):
        if cfg.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.ro_cfg.psb_cfg, cfg.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.ro_cfg.psb_cfg)
        self.delay_auto()
        self.pulse(cfg.gen, "rf_drive", t=0)
        self.delay_auto()
        readout_v2.psb_em(self, cfg.ro_cfg.psb_cfg, cfg.ro_cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)

#replaces RabiChevronV2 from spinQICK V1
class RabiChevron(asm_v2.AveragerProgramV2):
    """perform a 2D sweep of frequency and time"""

    def _initialize(self, cfg: experiment_models.RfSweepTwo):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        nqz = qick_utils.check_nyquist(cfg.gx_start, cfg.qubit.rf_gen, self.soccfg)
        self.declare_gen(cfg.qubit.rf_gen, nqz=nqz)
        pulse_time_sweep = asm_v2.QickSweep1D("time", cfg.gy_start, cfg.gy_stop)
        pulse_freq_sweep = asm_v2.QickSweep1D("freq", cfg.gx_start, cfg.gx_stop)
        self.add_pulse(
            cfg.qubit.rf_gen,
            "rf_drive",
            style="const",
            freq=pulse_freq_sweep,
            phase=0,
            gain=cfg.gain,
            length=pulse_time_sweep,
        )

        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("freq", cfg.gx_expts)
        self.add_loop("time", cfg.gy_expts)
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.RfSweepTwo):
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto()
        self.pulse(cfg.qubit.rf_gen, "rf_drive", t=0)
        self.delay_auto()
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)


class TimeRabi(asm_v2.AveragerProgramV2):
    """perform an rf pulse time sweep"""

    def _initialize(self, cfg: experiment_models.TimeRabi):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        nqz = qick_utils.check_nyquist(cfg.qubit.f0, cfg.qubit.rf_gen, self.soccfg)
        self.declare_gen(cfg.qubit.rf_gen, nqz=nqz)
        pulse_time_sweep = asm_v2.QickSweep1D("time", cfg.start, cfg.stop)
        self.add_pulse(
            cfg.qubit.rf_gen,
            "rf_drive",
            style="const",
            freq=cfg.qubit.f0,
            phase=0,
            gain=cfg.gain,
            length=pulse_time_sweep,
        )

        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("time", cfg.expts)
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.TimeRabi):
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto()
        self.pulse(cfg.qubit.rf_gen, "rf_drive", t=0)
        self.delay_auto()
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)


class AmplitudeRabi(asm_v2.AveragerProgramV2):
    """perform a rf pulse gain sweep"""

    def _initialize(self, cfg: experiment_models.AmplitudeRabi):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        nqz = qick_utils.check_nyquist(cfg.qubit.f0, cfg.qubit.rf_gen, self.soccfg)
        self.declare_gen(cfg.qubit.rf_gen, nqz=nqz)
        pulse_gain_sweep = asm_v2.QickSweep1D("gain", cfg.start, cfg.stop)
        self.add_pulse(
            cfg.qubit.rf_gen,
            "rf_drive",
            style="const",
            freq=cfg.qubit.f0,
            phase=0,
            gain=pulse_gain_sweep,
            length=cfg.time,
        )

        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("gain", cfg.expts)
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.RfSweepTwo):
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto()
        self.pulse(cfg.qubit.rf_gen, "rf_drive", t=0)
        self.delay_auto()
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)


class RamseyFringe(asm_v2.AveragerProgramV2):
    """Play pi/2 pulse, wait for a time tau,
    play another pi/2 pulse, measure.
    """

    def _initialize(self, cfg: experiment_models.LdSweepOne):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        nqz = qick_utils.check_nyquist(cfg.qubit.f0, cfg.qubit.rf_gen, self.soccfg)
        self.declare_gen(cfg.qubit.rf_gen, nqz=nqz)
        self.add_pulse(
            cfg.qubit.rf_gen,
            "rf_90",
            style="const",
            freq=cfg.qubit.f0,
            phase=0,
            gain=cfg.qubit.pulses.gain_90,
            length=cfg.qubit.pulses.time_90,
        )
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("time_delay", cfg.expts)
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.LdSweepOne):
        time_delay_sweep = asm_v2.QickSweep1D("time_delay", cfg.start, cfg.stop)
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto()
        self.pulse(cfg.qubit.rf_gen, "rf_90", t=0)
        self.delay_auto(t=time_delay_sweep)
        self.pulse(cfg.qubit.rf_gen, "rf_90", t=0)
        self.delay_auto()
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)


class Ramsey2D(asm_v2.AveragerProgramV2):
    """Play pi/2 pulse, wait for a time tau and play another pi/2 pulse, measure. Sweeps tau and frequency"""

    def _initialize(self, cfg: experiment_models.LdSweepTwo):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        nqz = qick_utils.check_nyquist(cfg.qubit.f0, cfg.qubit.rf_gen, self.soccfg)
        self.declare_gen(cfg.qubit.rf_gen, nqz=nqz)
        freq_sweep = asm_v2.QickSweep1D(
            "freq", cfg.gx_start + cfg.qubit.f0, cfg.gx_stop + cfg.qubit.f0
        )  #
        self.add_pulse(
            cfg.qubit.rf_gen,
            "rf_90",
            style="const",
            freq=freq_sweep,
            phase=0,
            gain=cfg.qubit.pulses.gain_90,
            length=cfg.qubit.pulses.time_90,
        )
        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("freq", cfg.gx_expts)
        self.add_loop("time_delay", cfg.gy_expts)
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.LdSweepTwo):
        time_delay_sweep = asm_v2.QickSweep1D("time_delay", cfg.gy_start, cfg.gy_stop)
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto()
        self.pulse(cfg.qubit.rf_gen, "rf_90", t=0)
        self.delay_auto(t=time_delay_sweep)
        self.pulse(cfg.qubit.rf_gen, "rf_90", t=0)
        self.delay_auto()
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)


class SpinEcho(asm_v2.AveragerProgramV2):
    """CPMG pulse sequence for measuring T2.  Pi/2 pulse followed by a variable delay and a train
    of pi pulses, each with time 2tau between them.  Follow up with another Pi/2 pulse and measure.
    """

    def _initialize(self, cfg: experiment_models.SpinEcho):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        nqz = qick_utils.check_nyquist(cfg.qubit.f0, cfg.qubit.rf_gen, self.soccfg)
        self.declare_gen(cfg.qubit.rf_gen, nqz=nqz)
        self.add_pulse(
            cfg.qubit.rf_gen,
            "rf_90",
            style="const",
            freq=cfg.qubit.f0,
            phase=0,
            gain=cfg.qubit.pulses.gain_90,
            length=cfg.qubit.pulses.time_90,
        )

        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("time_delay", cfg.expts)
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.SpinEcho):
        time_delay_sweep = asm_v2.QickSweep1D("time_delay", cfg.start, cfg.stop)
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto()
        self.pulse(cfg.qubit.rf_gen, "rf_90", t=0)

        for n in range(cfg.n_echoes):
            self.delay_auto(t=time_delay_sweep)
            self.pulse(cfg.qubit.rf_gen, "rf_90")
            self.pulse(cfg.qubit.rf_gen, "rf_90")  # specify timing more carefully?
            self.delay_auto(t=time_delay_sweep)

        self.pulse(cfg.qubit.rf_gen, "rf_90", t=0)
        self.delay_auto()
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)


class SweepPhase(asm_v2.AveragerProgramV2):
    """demonstrate phase control by applying two pi/2 pulses, sweep relative phase of the RF tone of the second pulse"""

    def _initialize(self, cfg: experiment_models.LdSweepOne):
        readout_v2.init_dcs(self, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.init_psb(self, cfg.qubit.ro_cfg.psb_cfg)
        nqz = qick_utils.check_nyquist(cfg.qubit.f0, cfg.qubit.rf_gen, self.soccfg)
        phase_sweep = asm_v2.QickSweep1D("phase", cfg.start, cfg.stop)
        self.declare_gen(cfg.qubit.rf_gen, nqz=nqz)
        self.add_pulse(
            cfg.qubit.rf_gen,
            "rf_90",
            style="const",
            freq=cfg.qubit.f0,
            phase=0,
            gain=cfg.qubit.pulses.gain_90,
            length=cfg.qubit.pulses.time_90,
        )
        self.add_pulse(
            cfg.qubit.rf_gen,
            "rf_90_phase_off",
            style="const",
            freq=cfg.qubit.f0,
            phase=phase_sweep,
            gain=cfg.qubit.pulses.gain_90,
            length=cfg.qubit.pulses.time_90,
        )

        self.add_loop("full_avgs", cfg.full_avgs)
        self.add_loop("phase", cfg.expts)
        self.add_loop("point_avgs", cfg.point_avgs)

    def _body(self, cfg: experiment_models.LdSweepOne):
        if cfg.qubit.ro_cfg.reference:
            readout_v2.psb_fm(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        readout_v2.psb_fe(self, cfg.qubit.ro_cfg.psb_cfg)
        self.delay_auto()
        self.pulse(cfg.qubit.rf_gen, "rf_90", t=0)
        self.pulse(cfg.qubit.rf_gen, "rf_90_phase_off")
        self.delay_auto()
        readout_v2.psb_em(self, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
        self.wait_auto(gens=True, ros=True)

#replaces AllXY
def play_xy(soccfg, cfg: experiment_models.PlayXY):
    """QICK code to perform an AllXY experiment or a series of x, y and z gates on a single spin"""
    gate_set_len = len(cfg.gate_set)
    prog = asm_v2.AcquireProgramV2(soccfg)
    prog.set_ext_counter(addr=1, val=0)
    readout_v2.init_dcs(prog, cfg.qubit.ro_cfg.dcs_cfg)
    readout_v2.init_psb(prog, cfg.qubit.ro_cfg.psb_cfg)
    ld_pulses.setup_1_ld_qubit(prog, cfg.qubit)
    ro_class_1 = asm_v2.QickProgramV2(soccfg)
    if cfg.qubit.ro_cfg.reference:
        readout_v2.psb_fm(
            ro_class_1, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg
        )
    readout_v2.psb_fe(ro_class_1, cfg.qubit.ro_cfg.psb_cfg)
    asm_1 = ro_class_1.asm()
    prog.label("readout_1")
    prog.extend_macros(asm_1)
    prog.ret()
    ro_class_2 = asm_v2.QickProgramV2(soccfg)
    readout_v2.psb_em(ro_class_2, cfg.qubit.ro_cfg.psb_cfg, cfg.qubit.ro_cfg.dcs_cfg)
    asm_2 = ro_class_2.asm()
    prog.label("readout_2")
    prog.extend_macros(asm_2)
    prog.ret()
    prog.delay(1)
    prog.open_loop(1, "reps")  # need a reps axis to average over
    prog.open_loop(cfg.full_avgs, "full_avgs")
    # test the script as is, then add another loop,
    for gate_list in cfg.gate_set:
        prog.open_loop(cfg.point_avgs)
        # add spam wrapper!
        prog.call("readout_1")
        ld_pulses.parse_and_play_1q(prog, cfg.qubit, gate_list)
        prog.delay_auto()
        # second spam wrapper
        prog.call("readout_2")
        prog.close_loop()

    prog.inc_ext_counter(addr=1, val=1)
    prog.close_loop()
    prog.close_loop()
    prog.end()
    prog.setup_acquire(
        counter_addr=1,
        loop_dims=[1, cfg.full_avgs, gate_set_len, cfg.point_avgs],
        avg_level=0,
    )
    prog.compile()

    return prog
