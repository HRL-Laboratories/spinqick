"""Modified qick code copied directly from qick's NDAveragerProgram, RAverager and AcquireMixin.
Methods in the acquire classes were overwritten to process data more efficiently and facilitate more straightforward averaging
in experiments that involve a reference measurement and averaging over an inner loop.

These functions are only partially implemented in the rest of spinqick and haven't been used with actual experiments yet.  They are
a work in progress and will be implemented more widely in the future.
"""

import logging
from typing import Any, List

import numpy as np
from qick.asm_v1 import QickRegisterManagerMixin
from qick.averager_program import AbsQickSweep, QickProgram
from qick.qick_asm import AcquireMixin

logger = logging.getLogger(__name__)


class PSBAcquireMixin(AcquireMixin):
    def _average_buf(
        self,
        d_reps: np.ndarray,
        reads_per_shot: list,
        length_norm: bool = True,
        remove_offset: bool = True,
    ) -> list[np.ndarray]:
        """
        calculate averaged data in a data acquire round. This function should be overwritten in the child qick program
        if the data is created in a different shape.

        :param d_reps: buffer data acquired in a round
        :param reads_per_shot: readouts per experiment
        :param length_norm: normalize by readout window length (should use False if thresholding; this setting is ignored and False is used for readouts where edge-counting is enabled)
        :param remove_offset: if normalizing by length, also subtract the readout's IQ offset if any
        :return: averaged iq data after each round.
        """
        avg_d = []
        for i_ch, (ch, ro) in enumerate(self.ro_chs.items()):
            # remove this averaging step
            # average over the avg_level
            avg = d_reps[i_ch]
            if length_norm and not ro["edge_counting"]:
                avg = avg / ro["length"]
                if remove_offset:
                    avg = avg - self._ro_offset(ch, ro.get("ro_config"))
            # the reads_per_shot axis should be the first one
            avg_d.append(np.moveaxis(avg, -2, 0))

        return avg_d


class PSBAcquire(PSBAcquireMixin, QickProgram):
    """this is how the inheritance is done in the QICK api"""

    pass


class PSBAveragerProgram(QickRegisterManagerMixin, PSBAcquire):
    """
    Based on qick.averager.NDAveragerProgram, modified for spin qubit experiments that specifically make use of a reference
    measurement for Pauli Spin Blockade.  Include reps and shots in your config dictionary.

    NDAveragerProgram class, for experiments that sweep over multiple variables in qick. The order of experiment runs
    follow outer->inner: reps, sweep_n,... sweep_0, shots

    :param cfg: Configuration dictionary
    :type cfg: dict
    """

    COUNTER_ADDR = 1

    def __init__(self, soccfg, cfg):
        """
        Constructor for the NDAveragerProgram. Make the ND sweep asm commands.
        """
        super().__init__(soccfg)
        self.cfg = cfg
        self.qick_sweeps: List[AbsQickSweep] = []
        self.sweep_axes: list[Any] = []
        self.make_program()
        self.soft_avgs = 1
        if "soft_avgs" in cfg:
            self.soft_avgs = cfg["soft_avgs"]
        if "rounds" in cfg:
            self.soft_avgs = cfg["rounds"]
        self.reps_avg = cfg["reps"]
        self.shots_avg = cfg["shots"]
        # reps loop is the outer loop, first-added sweep is innermost loop
        if np.logical_and(self.reps_avg == 1, self.shots_avg == 1):
            loop_dims = [*self.sweep_axes[::-1]]
        if np.logical_and(self.reps_avg > 1, self.shots_avg == 1):
            loop_dims = [self.reps_avg, *self.sweep_axes[::-1]]
        if np.logical_and(self.reps_avg == 1, self.shots_avg > 1):
            loop_dims = [*self.sweep_axes[::-1], self.shots_avg]
        else:
            loop_dims = [self.reps_avg, *self.sweep_axes[::-1], self.shots_avg]

        # acquiremixin class averages over the reps axis -- we changed that analysis though so this does nothing rn!
        self.setup_acquire(
            counter_addr=self.COUNTER_ADDR, loop_dims=loop_dims, avg_level=None
        )

    def initialize(self):
        """
        Abstract method for initializing the program. Should include the instructions that will be executed once at the
        beginning of the qick program.
        """
        pass

    def body(self):
        """
        Abstract method for the body of the program.
        """
        pass

    def add_sweep(self, sweep: AbsQickSweep):
        """
        Add a layer of register sweep to the qick asm program. The order of sweeping will follow first added first sweep.
        :param sweep:
        :return:
        """
        self.qick_sweeps.append(sweep)
        self.sweep_axes.append(sweep.expts)

    def make_program(self):
        """
        Make the N dimensional sweep program. The program will run initialize once at the beginning, then iterate over
        all the sweep parameters and run the body. The whole sweep will repeat for cfg["reps"] number of times.
        """
        p = self

        p.initialize()  # initialize only run once at the very beginning

        rcount = 13  # total run counter
        rep_count = 14  # repetition counter
        shot_count = 15  # shot (repeat only the body) counter

        n_sweeps = len(self.qick_sweeps)
        if (
            n_sweeps > 5
        ):  # to be safe, only register 17-21 in page 0 can be used as sweep counters
            raise OverflowError(
                f"too many qick inner loops ({n_sweeps}), run out of counter registers"
            )
        counter_regs = (
            np.arange(n_sweeps) + 17
        ).tolist()  # not sure why this has to be a list (np.array doesn't work)

        p.regwi(0, rcount, 0)  # reset total run count

        # set repetition counter and tag
        p.regwi(0, rep_count, self.cfg["reps"] - 1)
        p.label("LOOP_rep")

        # add reset and start tags for each sweep
        for creg, swp in zip(counter_regs[::-1], self.qick_sweeps[::-1]):
            swp.reset()
            p.regwi(0, creg, swp.expts - 1)
            p.label(f"LOOP_{swp.label if swp.label is not None else creg}")

        # set shot counter and tag
        p.regwi(0, shot_count, self.cfg["shots"] - 1)
        p.label("LOOP_shot")

        # run body and total_run_counter++
        p.body()
        p.mathi(0, rcount, rcount, "+", 1)
        p.memwi(0, rcount, self.COUNTER_ADDR)

        # stop condition for shot
        p.loopnz(0, shot_count, "LOOP_shot")

        # add update and stop condition for each sweep
        for creg, swp in zip(counter_regs, self.qick_sweeps):
            swp.update()
            p.loopnz(0, creg, f"LOOP_{swp.label if swp.label is not None else creg}")

        # stop condition for repetition
        p.loopnz(0, rep_count, "LOOP_rep")

        p.end()

    def get_expt_pts(self):
        """
        :return:
        """
        sweep_pts = []
        for swp in self.qick_sweeps:
            sweep_pts.append(swp.get_sweep_pts())
        return sweep_pts

    def acquire(
        self,
        soc,
        threshold: int | None = None,
        angle: List | None = None,
        load_pulses=True,
        readouts_per_experiment=None,
        save_experiments: List | None = None,
        start_src: str = "internal",
        progress=False,
        remove_offset=False,
    ):
        """
        This method optionally loads pulses on to the SoC, configures the ADC readouts, loads the machine code
        representation of the AveragerProgram onto the SoC, starts the program and streams the data into the Python,
        returning it as a set of numpy arrays.
        Note here the buf data has "reps" as the outermost axis, and the first swept parameter corresponds to the
        innermost axis.

        config requirements:
        "reps" = number of repetitions;

        :param soc: Qick object
        :param threshold: threshold
        :param angle: rotation angle
        :param readouts_per_experiment: readouts per experiment
        :param save_experiments: saved readouts (by default, save all readouts)
        :param load_pulses: If true, loads pulses into the tProc
        :param start_src: "internal" (tProc starts immediately) or "external" (each round waits for an external trigger)
        :param progress: If true, displays progress bar
        :returns:
            - expt_pts (:py:class:`list`) - list of experiment points
            - avg_data (:py:class:`list`) - list of lists of averaged accumulated and optionally thresholded data ADCs 0 and 1
        """

        if readouts_per_experiment is not None:
            self.set_reads_per_shot(readouts_per_experiment)

        self._set_up_thresholding(
            thresholding=self.cfg["PSB_cfg"]["thresholding"],
            threshold_val=self.cfg["PSB_cfg"]["thresh"],
            reference_meas=self.cfg["PSB_cfg"]["reference_meas"],
        )

        avg_d = super().acquire(
            soc,
            soft_avgs=self.soft_avgs,
            load_pulses=load_pulses,
            start_src=start_src,
            threshold=threshold,
            angle=angle,
            progress=progress,
            remove_offset=remove_offset,
        )

        expt_pts = self.get_expt_pts()
        n_ro = len(self.ro_chs)

        # average over shots and reps axes!
        for i_ch in range(n_ro):
            avg_loops = avg_d[i_ch]
            avg_loops = np.mean(avg_loops, axis=-1)
            if self.cfg.shots == 1:
                avg_loops = np.mean(avg_loops, axis=0)
            if self.cfg.reps > 1:
                avg_loops = np.mean(avg_loops, axis=0)
            if i_ch == 0:
                avg_d_final = avg_loops
            elif i_ch > 0:
                avg_d_final = np.vstack(avg_d_final, avg_loops)  # type: ignore

        return expt_pts, avg_d_final


class FlexyPSBAveragerProgram(PSBAcquire):
    """
    This is largely copied from RAveragerProgram, modified to work better for our experiments.  This adds an outer loop to sweep another variable and averaging on the innermost (shots) and outermost (reps) loops

    :param cfg: Configuration dictionary
    :type cfg: dict
    """

    COUNTER_ADDR = 1

    def __init__(self, soccfg, cfg):
        """
        Constructor for the RAveragerProgram, calls make program at the end so for classes that inherit from this if you want it to do something before the program is made and compiled either do it before calling this __init__ or put it in the initialize method.
        """
        super().__init__(soccfg)
        self.cfg = cfg
        # add some attributes to help track the registers you're sweeping
        self.outer_pages = []
        self.outer_addrs = []
        self.inner_pages = []
        self.inner_addrs = []
        self.outer_sweep = False
        self.inner_sweep = False
        self.psb_thresholding = self.cfg["PSB_cfg"]["thresholding"]
        self.psb_threshold = self.cfg["PSB_cfg"]["thresh"]
        self.psb_reference = self.cfg["PSB_cfg"]["reference_meas"]
        self.make_program()
        self.soft_avgs = 1
        if "rounds" in cfg:
            self.soft_avgs = cfg["rounds"]
        # reps is outer loop, sweepable is inner loop
        loop_dims = [
            self.cfg["reps"],
            self.cfg["expts_outer"],
            self.cfg["expts"],
            self.cfg["shots"],
        ]
        # must give it an average level, although we removed that feature
        # TODO remove this feature
        self.setup_acquire(
            counter_addr=self.COUNTER_ADDR, loop_dims=loop_dims, avg_level=0
        )

    def initialize(self):
        """
        Abstract method for initializing the program and can include any instructions that are executed once at the beginning of the program.
        """
        pass

    def body(self):
        """
        Abstract method for the body of the program
        """
        pass

    def update(self):
        """
        Abstract method for updating the program
        """
        pass

    def update2(self):
        """
        Abstract method for updating the program
        """
        pass

    def make_program(self):
        """
        A template program which repeats the instructions defined in the body() method the number of times specified in self.cfg["reps"].
        """
        p = self

        rcount = 13
        rii = 14
        rjj = 19
        rkk = 17
        rmm = 18

        p.initialize()

        p.regwi(0, rcount, 0)
        p.regwi(0, rkk, self.cfg["reps"] - 1)
        p.label("LOOP_rep")

        p.regwi(0, rmm, self.cfg["expts_outer"] - 1)

        p.label("LOOP_M")

        p.regwi(0, rii, self.cfg["expts"] - 1)

        p.label("LOOP_I")

        p.regwi(0, rjj, self.cfg["shots"] - 1)
        p.label("LOOP_shot")

        p.body()

        p.mathi(0, rcount, rcount, "+", 1)

        p.memwi(0, rcount, self.COUNTER_ADDR)

        p.loopnz(0, rjj, "LOOP_shot")

        p.update()

        p.loopnz(0, rii, "LOOP_I")
        # reset inner loop registers
        if self.inner_sweep:
            for i, reg_addr in enumerate(self.inner_addrs):
                p.mathi(
                    self.inner_pages[i],
                    reg_addr,
                    reg_addr,
                    "-",
                    self.cfg["step"] * self.cfg["expts"],
                )

        p.update2()

        p.loopnz(0, rmm, "LOOP_M")
        # reset outer loop registers back to start value
        # todo: modify this so you can reset sweeps that aren't linear!
        if self.outer_sweep:
            for i, reg_addr in enumerate(self.outer_addrs):
                p.mathi(
                    self.outer_pages[i],
                    reg_addr,
                    reg_addr,
                    "-",
                    self.cfg["expts_outer"] * self.cfg["step_outer"],
                )

        p.loopnz(0, rkk, "LOOP_rep")

        p.end()

    def get_expt_pts(self):
        """
        Method for calculating experiment points (for x-axis of plots) based on the config.

        :return: Numpy array of experiment points
        :rtype: array
        """
        pts_inner = self.cfg["start"] + np.arange(self.cfg["expts"]) * self.cfg["step"]
        pts_outer = (
            self.cfg["start_outer"]
            + np.arange(self.cfg["expts_outer"]) * self.cfg["step_outer"]
        )
        expt_pts = [pts_inner, pts_outer]

        return expt_pts

    def add_outer_reg_sweep(self, ch_page, addr):
        # this just stores info for the outer sweep
        self.outer_sweep = True
        self.outer_pages.append(ch_page)
        self.outer_addrs.append(addr)

    def add_inner_reg_sweep(self, ch_page, addr):
        self.inner_sweep = True
        self.inner_pages.append(ch_page)
        self.inner_addrs.append(addr)

    def acquire(self, soc, readouts_per_experiment=None, **kwargs):
        """
        This method optionally loads pulses on to the SoC, configures the ADC readouts, loads the machine code representation of the AveragerProgram onto the SoC, starts the program and streams the data into the Python, returning it as a set of numpy arrays.
        config requirements:
        "reps" = number of repetitions;
        "shots" = number of shots of the innermost loop
        Right now this assumes the user is collecting data with only one ADC.

        :param soc: Qick object
        :type soc: Qick object
        :param threshold: threshold
        :type threshold: int
        :param angle: rotation angle
        :type angle: list
        :param readouts_per_experiment: readouts per experiment
        :type readouts_per_experiment: int
        :param save_experiments: saved readouts (by default, save all readouts)
        :type save_experiments: list
        :param load_pulses: If true, loads pulses into the tProc
        :type load_pulses: bool
        :param start_src: "internal" (tProc starts immediately) or "external" (each round waits for an external trigger)
        :type start_src: string
        :param progress: If true, displays progress bar
        :type progress: bool
        :returns:
            - expt_pts (:py:class:`list`) - list of experiment points
            - avg_di (:py:class:`list`) - list of lists of averaged accumulated I data for ADCs 0 and 1
            - avg_dq (:py:class:`list`) - list of lists of averaged accumulated Q data for ADCs 0 and 1
        """
        if readouts_per_experiment is not None:
            self.set_reads_per_shot(readouts_per_experiment)

        avg_d = super().acquire(soc, soft_avgs=self.soft_avgs, **kwargs)

        # reformat the data into separate I and Q arrays
        # save results to class in case you want to look at it later or for analysis
        raw = [d.reshape((-1, 2)) for d in self.get_raw()]
        self.di_buf = [d[:, 0] for d in raw]
        self.dq_buf = [d[:, 1] for d in raw]

        expt_pts = self.get_expt_pts()

        # if save_experiments is None:
        avg_di = [d[..., 0] for d in avg_d]
        avg_dq = [d[..., 1] for d in avg_d]
        # assume one adc for now
        if self.psb_reference:
            mag_diff = np.sqrt(avg_di[0][0] ** 2 + avg_dq[0][0] ** 2) - np.sqrt(
                avg_di[0][1] ** 2 + avg_dq[0][1] ** 2
            )
        else:
            mag_diff = np.sqrt(
                avg_di[0][1] ** 2 + avg_dq[0][1] ** 2
            )  # if no reference measurement subtraction is desired, return the second measurement

        if self.psb_thresholding:
            mag_thresh = np.where(np.abs(mag_diff) > self.psb_threshold, 0, 1.0)
        else:
            mag_thresh = mag_diff

        avgd_shots = np.mean(mag_thresh, axis=-1)
        avgd_reps = np.mean(avgd_shots, axis=0)

        return expt_pts, avgd_reps
