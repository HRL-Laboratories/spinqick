"""pydantic templates for hardware config dictionary"""

import enum
from typing import Dict, Union

import pydantic


class VoltageSourceType(str, enum.Enum):
    slow_dac = "slow_dac"
    test = "test"


class HemtGate(pydantic.BaseModel):
    slow_dac_address: str
    slow_dac_channel: int
    dc_conversion_factor: float
    max_v: float


class SlowGate(pydantic.BaseModel):
    dc_conversion_factor: float
    # slow dac information for DCSource class
    slow_dac_address: str
    slow_dac_channel: int
    max_v: float
    # crosscoupling between gates
    crosscoupling: Dict[str, float] | None = None


class FastGate(SlowGate):
    # dac units to volts conversion
    dac_conversion_factor: float | None
    # qick channel associated with gate
    qick_gen: int | None


class SourceDrainIn(pydantic.BaseModel):
    qick_gen: int  # readout pulse channel


class SourceDrainOut(pydantic.BaseModel):
    qick_adc: int  # adc channel


class HardwareConfig(pydantic.BaseModel):
    sd_in: SourceDrainIn
    sd_out: SourceDrainOut
    rf_gen: int | None = None
    slow_dac_trig_pin: int  # trigger on PMOD header for triggering dac sweeps
    rf_trig_pin: int | None = None  # trigger pin for the RF switch
    channels: Dict[str, Union[FastGate, SlowGate, HemtGate]]
    voltage_source: VoltageSourceType = (
        VoltageSourceType.test  # specify the type of dc supply you're using for DCSource class
    )
