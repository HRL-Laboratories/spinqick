"""Pydantic templates for hardware config parameters."""

from typing import Dict, List, Literal, Union

import pydantic

import spinqick.helper_functions.spinqick_enums


class VoltageSourceType:
    slow_dac = "slow_dac"
    test = "test"


class AuxGate(pydantic.BaseModel):
    """Auxiliary gate model."""

    slow_dac_address: str
    slow_dac_channel: int


class HemtGate(AuxGate):
    """Auxiliary gate model specific for powering HEMTs."""

    dc_conversion_factor: float
    max_v: float
    sd_out: int  # specify index of SourceDrainOut list


class SlowGate(pydantic.BaseModel):
    """Model for gates with a DC voltage source channel."""

    dc_conversion_factor: float  # output voles to gate voltage conversion
    slow_dac_address: str
    slow_dac_channel: int
    max_v: float
    gate_type: spinqick.helper_functions.spinqick_enums.GateTypes
    crosscoupling: Dict[spinqick.helper_functions.spinqick_enums.GateNames, float] | None = (
        None  # crosscoupling between gates
    )


class FastGate(SlowGate):
    """Model for gates associated with both a DC source channel and qick generator."""

    dac_conversion_factor: float  # dac units to volts conversion
    qick_gen: int  # qick channel associated with gate


class SourceDrainIn(pydantic.BaseModel):
    """Model describing the source-drain AC signal into the device."""

    qick_gen: int  # readout pulse channel
    unit_conversion: float
    sd_units: str


class SourceDrainOut(pydantic.BaseModel):
    """Model describign the output readout signal from the device."""

    qick_adc: int  # adc channel
    unit_conversion: float
    adc_units: str


class HardwareConfig(pydantic.BaseModel):
    """Model for the full hardware config."""

    sd_in: SourceDrainIn
    m1_readout: List[SourceDrainOut]
    m2_readout: List[SourceDrainOut]
    rf_gen: int | None = None
    rf_trig_pin: int | None = None  # trigger pin for the RF switch
    ac_gate: SourceDrainIn | None = None  # gate used to apply ac signal for transconductance
    channels: Dict[
        spinqick.helper_functions.spinqick_enums.GateNames,
        Union[FastGate, SlowGate, HemtGate, AuxGate],
    ]
    voltage_source: Literal["test", "slow_dac"] | None = (
        "test"  # specify the type of dc supply you're using for DCSource class
    )
