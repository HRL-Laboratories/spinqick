"""pydantic templates for hardware config dictionary"""

from typing import Dict, Union, Literal, List
import pydantic
from spinqick import settings


class VoltageSourceType:
    slow_dac = "slow_dac"
    test = "test"


class AuxGate(pydantic.BaseModel):
    slow_dac_address: str
    slow_dac_channel: int


class HemtGate(AuxGate):
    dc_conversion_factor: float
    max_v: float
    sd_out: int  # specify index of SourceDrainOut list


class SlowGate(pydantic.BaseModel):
    dc_conversion_factor: float
    # slow dac information for DCSource class
    slow_dac_address: str
    slow_dac_channel: int
    max_v: float
    gate_type: settings.GateTypes
    # crosscoupling between gates
    crosscoupling: Dict[settings.GateNames, float] | None = None


class FastGate(SlowGate):
    # dac units to volts conversion
    dac_conversion_factor: float
    # qick channel associated with gate
    qick_gen: int


class SourceDrainIn(pydantic.BaseModel):
    qick_gen: int  # readout pulse channel
    unit_conversion: float
    sd_units: str


class SourceDrainOut(pydantic.BaseModel):
    qick_adc: int  # adc channel
    unit_conversion: float
    adc_units: str


class HardwareConfig(pydantic.BaseModel):
    sd_in: SourceDrainIn
    m1_readout: List[SourceDrainOut]
    m2_readout: List[SourceDrainOut]
    rf_gen: int | None = None
    rf_trig_pin: int | None = None  # trigger pin for the RF switch
    ac_gate: SourceDrainIn | None = (
        None  # gate used to apply ac signal for transconductance
    )
    channels: Dict[settings.GateNames, Union[FastGate, SlowGate, HemtGate, AuxGate]]
    voltage_source: Literal["test", "slow_dac"] = (
        "test"  # specify the type of dc supply you're using for DCSource class
    )
