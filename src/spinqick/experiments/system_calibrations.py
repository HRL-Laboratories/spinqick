"""This module holds scripts which can be useful when setting up a new device or system."""

import logging
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from spinqick.core import dot_experiment, spinqick_data
from spinqick.helper_functions import (
    analysis,
    hardware_manager,
    plot_tools,
    spinqick_enums,
)
from spinqick.models import experiment_models
from spinqick.qick_code_v2 import system_calibrations_programs_v2

logger = logging.getLogger(__name__)


class SystemCalibrations(dot_experiment.DotExperiment):
    """This class holds functions that wrap the QICK classes for hardware calibrations.Initialize
    with information about your rfsoc and your experimental setup.

    :param soccfg: QickConfig object
    :param soc: Qick object
    :param datadir: data directory where all data is being stored. Experiment will make a folder
        here with today's date.
    """

    def __init__(
        self, soccfg, soc, voltage_source: hardware_manager.VoltageSource, **kwargs
    ):
        super().__init__(**kwargs)
        self.soccfg = soccfg
        self.soc = soc
        self.vdc = hardware_manager.DCSource(voltage_source=voltage_source)

    @dot_experiment.updater
    def sweep_adc_trig_offset(
        self,
        times: Tuple[float, float, int],
        point_avgs: int,
        full_avgs: int,
        loop_slack: float,
    ) -> spinqick_data.SpinqickData:
        """Sweep the adc trigger offset parameter, which sets the offset between when a pulse is
        fired and when readout is turned on.

        :param times: (start offset time, end offset time, number of points)
        """
        delay_start, delay_stop, points = times
        delay_sweep = np.linspace(delay_start, delay_stop, points)
        sweep_cfg = experiment_models.SweepDelay(
            point_avgs=point_avgs,
            full_avgs=full_avgs,
            delay_start=delay_start,
            delay_stop=delay_stop,
            delay_points=points,
            dcs_cfg=self.dcs_config,
            loop_slack=loop_slack,
        )
        prog = system_calibrations_programs_v2.SweepAdcDelay(
            self.soccfg, 1, 1, cfg=sweep_cfg, final_wait=1
        )
        raw_data = prog.acquire(self.soc, progress=True)
        assert raw_data
        data_obj = spinqick_data.SpinqickData(
            raw_data,
            sweep_cfg,
            1,
            1,
            "_adc_sweep",
        )

        analysis.calculate_conductance(
            data_obj,
            [1 for i in range(len(raw_data))],
            average_level=spinqick_enums.AverageLevel.BOTH,
        )
        data_obj.add_axis(
            [delay_sweep],
            "time",
            ["adc_trig_delay"],
            points,
            units=["microseconds"],
            loop_no=1,
        )
        data_obj.add_full_average(full_avgs)
        data_obj.add_point_average(point_avgs, 2)
        if self.plot:
            fignums = []
            for adc_idx in range(len(data_obj.raw_data)):
                if data_obj.analyzed_data is not None:
                    fig = plot_tools.plot1_simple(
                        delay_sweep,
                        data_obj.analyzed_data[adc_idx][0],
                        data_obj.timestamp,
                    )
                    plt.title("adc %d" % adc_idx)
                    plt.ylabel("conductance (arbs)")
                    plt.xlabel("adc trigger delay (us)")
                    fignums.append(fig.number)
        if self.save_data:
            nc_file = data_obj.save_data()
            if self.plot:
                for num in fignums:
                    nc_file.save_last_plot(fignum=num)
            nc_file.close()
        return data_obj
