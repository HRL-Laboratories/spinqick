"""work in progress, creating a template for each type of config that we use in our experiments.
Right now, just using these to validate the readout config
TODO incorporate pydantic

"""

from typing import Dict, Optional

import pydantic


class DcsConfig(pydantic.BaseModel):
    """config parameters specifically for readout chain"""

    adc_trig_offset: int  # time delay of adc measurement
    dds_freq: float  # frequency of readout pulse
    readout_freq: float  # demodulation frequency
    length: int  # pulse length in clock cycles
    readout_length: int  # readout length in clock cycles
    pulse_gain_readout: int  # readout pulse gain in dac units
    res_ch: int  # readout pulse dac channel
    ro_ch: int  # adc channel


class ActivePsbGates(pydantic.BaseModel):
    """Specify which gates to use for PSB.  Gates here must have associated PsbGates in the config"""

    Px: str
    Py: str
    X: str


class PsbGains(pydantic.BaseModel):
    """gain in DAC units for each time step of PSB routine"""

    entry1_gain: int
    entry2_gain: int
    home_gain: int
    idle_gain: int
    init2_gain: int
    init_gain: int
    window1_gain: int


class PsbTimes(pydantic.BaseModel):
    """time in clock cycles for each step of PSB routine"""

    idle_time: int
    init2_time: int
    init_time: int
    ramp1_time: int
    ramp2_time: int
    settle_time: int


class PsbGate(pydantic.BaseModel):
    """generalized template for each gate in PSB_cfg"""

    gen: int  # baseband pulse channel corresponding to the gate in use
    gains: PsbGains


class PsbConfig(pydantic.BaseModel):
    """model for PSB_cfg"""

    active_gates: ActivePsbGates | None
    gates: Dict[str, PsbGate] | None
    times: PsbTimes | None
    reference_meas: bool | None  # optionally take a reference measurement
    relax_delay: int  # time between measurements in clock cycles
    thresh: float | None  # singlet/triplet threshold value
    thresholding: bool | None  # turn on or off thresholding


class ReadoutConfig(pydantic.BaseModel):
    """model for a full config dictionary"""

    DCS_cfg: DcsConfig
    PSB_cfg: PsbConfig | None


class NDAveragerConfig(ReadoutConfig):
    """Add on necessary parameters for NDAverager programs"""

    reps: int
    rounds: Optional[int] = None
    soft_avgs: Optional[int] = None


class RAveragerConfig(ReadoutConfig):
    """Add on necessary parameters for RAverager programs"""

    reps: int
    expts: int
    start: int
    step: int
    rounds: Optional[int] = None


class FlexyPSBAveragerConfig(ReadoutConfig):
    """Add on necessary parameters for FlexyPSBAverager programs"""

    reps: int
    shots: int
    start: float
    step: float
    expts: int
    start_outer: float
    step_outer: float
    expts_outer: int
    rounds: Optional[int] = None


class PSBAveragerConfig(ReadoutConfig):
    """Add on necessary parameters for PSBAverager programs"""

    reps: int
    shots: int
    expts: int
    rounds: Optional[int] = None
    soft_avgs: Optional[int] = None
