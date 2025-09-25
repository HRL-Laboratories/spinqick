"""Pydantic models for SPAM in spinqick v2"""

from typing import Union, Mapping
import pydantic

from spinqick import settings
from spinqick.models import dcs_model


class SpamPulse(pydantic.BaseModel):
    voltage: float


class SpamRamp(SpamPulse):
    voltage_2: float


class SpamPulseDac(pydantic.BaseModel):
    gen: int
    coordinate: float  # DAC units


class SpamRampDac(SpamPulseDac):
    coordinate_2: float  # DAC units


class SpamStep(pydantic.BaseModel):
    duration: float  # pulse durations in microseconds
    # gate_list: Mapping[
    #     settings.GateNames, Union[SpamPulse, SpamRamp]]  # supply a list of SpamPulse objects
    gate_list: Mapping[
        settings.GateNames, Union[SpamPulse, SpamRamp]
    ]  # supply a dict of SpamPulse objects


class SpamStepDac(pydantic.BaseModel):
    duration: float  # pulse durations in microseconds
    # gate_list: Dict[
    #     settings.GateNames, Union[SpamPulseDac, SpamRampDac]
    # ]  # supply a list of SpamPulse objects
    gate_list: Mapping[settings.GateNames, Union[SpamPulseDac, SpamRampDac]]


class DefaultSpam(pydantic.BaseModel):
    flush: SpamStep
    entry_20: SpamStep
    exit_11: SpamStep
    idle: SpamStep
    entry_11: SpamStep
    meas: SpamStep


class DefaultSpamDac(pydantic.BaseModel):
    flush: SpamStepDac
    entry_20: SpamStepDac
    exit_11: SpamStepDac
    idle: SpamStepDac  # idle duration defines settle time at idle
    entry_11: SpamStepDac
    meas: SpamStepDac  # meas duration defines settle time at meas window


class ReadoutConfig(pydantic.BaseModel):
    dcs_cfg: dcs_model.DcsConfig
    psb_cfg: DefaultSpamDac
    reference: bool  # whether to take a reference measurement
    thresh: bool  # whether to threshold PSB data
    threshold: float | None  # value of threshold
