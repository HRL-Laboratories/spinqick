"""Pydantic models to hold SPAM parameters."""

from typing import Mapping, Union

import pydantic

from spinqick.helper_functions import spinqick_enums
from spinqick.models import dcs_model


class SpamPulse(pydantic.BaseModel):
    """Defines a single voltage for a spam step."""

    voltage: float


class SpamRamp(SpamPulse):
    """Defines two voltages for a ramp-type spam step."""

    voltage_2: float


class SpamPulseDac(pydantic.BaseModel):
    """Generator and gain definition for a spam step."""

    gen: int
    coordinate: float  # DAC units


class SpamRampDac(SpamPulseDac):
    """Generator and gain definitions for a ramp-type spam step."""

    coordinate_2: float  # DAC units


class SpamStep(pydantic.BaseModel):
    """Defines a step within a spam sequence."""

    duration: float  # pulse durations in microseconds
    gate_list: Mapping[
        spinqick_enums.GateNames, Union[SpamPulse, SpamRamp]
    ]  # supply a dict of SpamPulse objects and corresponding gate names


class SpamStepDac(pydantic.BaseModel):
    """Spam step definition in DAC units."""

    duration: float  # pulse durations in microseconds
    gate_list: Mapping[spinqick_enums.GateNames, Union[SpamPulseDac, SpamRampDac]]


class DefaultSpam(pydantic.BaseModel):
    """Defines a default spam sequence as a series of spam steps."""

    flush: SpamStep  # flush duration defines duration at flush point
    entry_20: SpamStep  # this can be a ramp to measurement point from (2,0) cell
    exit_11: SpamStep  # exit ramp from measurement window to (1,1) charge cell
    idle: SpamStep  # idle duration defines settle time at idle
    entry_11: SpamStep  # entry to measurement window from (1,1) cell
    meas: SpamStep  # meas duration defines settle time at meas window


class DefaultSpamDac(pydantic.BaseModel):
    """Defines a default spam sequence as a series of spam steps in DAC units."""

    flush: SpamStepDac
    entry_20: SpamStepDac
    exit_11: SpamStepDac
    idle: SpamStepDac
    entry_11: SpamStepDac
    meas: SpamStepDac


class ReadoutConfig(pydantic.BaseModel):
    """Complete description of a qubit's readout in qick-friendly units."""

    dcs_cfg: dcs_model.DcsConfig
    psb_cfg: DefaultSpamDac
    reference: bool  # whether to take a reference measurement
    thresh: bool  # whether to threshold PSB data
    threshold: float | None  # value of threshold
