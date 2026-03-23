"""Work in progress, reorganizing models for use in tprocv2 experiments."""

# from typing import Dict, Optional
from typing import List

import pydantic


class DcsConfigParams(pydantic.BaseModel):
    """Config parameters specifically for readout chain."""

    adc_trig_offset: float  # time delay of adc measurement
    dds_freq: float  # frequency of readout pulse (MHz)
    readout_freq: float  # demodulation frequency (MHz)
    length: float  # pulse length in us
    readout_length: float  # readout length in us
    pulse_gain_readout: float  # readout pulse gain, between -1 and 1
    slack_delay: float  # delay time in program after measurement is taken
    ac_gate_gain: float | None = (
        None  # readout pulse for transconductance gain, between -1 and 1
    )


class DcsConfig(DcsConfigParams):
    """DcsConfig which includes information about dac and adc channels."""

    sd_gen: int  # readout pulse dac channel
    ac_gate_gen: int | None = None  # optional gate for transconductance
    ro_chs: List[int]  # adc channel
