"""pydantic templates for hardware config dictionary"""

import enum
from typing import Dict, Union

import pydantic


class VoltageSourceType(str, enum.Enum):
    slow_dac = "slow_dac"
    test = "test"


class SlowGate(pydantic.BaseModel):
    dc_conversion_factor: float
    # slow dac information for DCSource class
    slow_dac_address: str
    slow_dac_channel: int
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
    SD_in: SourceDrainIn
    SD_out: SourceDrainOut
    channels: Dict[str, Union[FastGate, SlowGate]]
    voltage_source: VoltageSourceType = (
        VoltageSourceType.test  # specify the type of dc supply you're using for DCSource class
    )
