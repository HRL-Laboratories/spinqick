from typing import List, Optional

import pydantic

from spinqick.helper_functions import spinqick_enums
from spinqick.models import spam_models


class ExchangeGains(pydantic.BaseModel):
    """Gains in DAC units."""

    idle_gain: float
    exchange_gain: float


class ExchangeVoltages(pydantic.BaseModel):
    """Gains in voltage units."""

    idle_voltage: float
    exchange_voltage: float


class ExchangeTimes(pydantic.BaseModel):
    """Times in microseconds."""

    idle_time: float
    exchange_time: float


class ExchangeGateParams(pydantic.BaseModel):
    """Generalized template for each gate in ExchangeConfig."""

    name: spinqick_enums.GateNames  # name of gate, e.g. P1
    gate_voltages: ExchangeVoltages  # associated coordinates for common sequencing


class ExchangeGate(pydantic.BaseModel):
    """Generalized template for each gate in ExchangeConfig."""

    name: spinqick_enums.GateNames  # name of gate, e.g. P1
    gains: ExchangeGains  # associated coordinates for common sequencing
    gen: int  # should be chosen off hardware_cfg.yml file.


class ExchangeGateMap(pydantic.BaseModel):
    """Specify gates used for an exchange axis.

    Gates must have an associated ExchangeGate in the axis config.
    """

    px: ExchangeGate
    py: ExchangeGate
    x: ExchangeGate


class ExchangeGateMapParams(pydantic.BaseModel):
    """Specify gates used for an exchange axis.

    Gates must have an associated ExchangeGate in the axis config.
    """

    px: ExchangeGateParams
    py: ExchangeGateParams
    x: ExchangeGateParams


class CalParameters(pydantic.BaseModel):
    """Model to hold calibration parameters.

    A, B and theta_max specify parameters from the course calibration fit. theta_list and
    voltage_list hold a list of phase and voltage calibration values from the fine calibration.
    """

    A: float
    B: float
    theta_max: float
    theta_list: List[float] | None  # TODO make this a file path
    volt_list: List[float] | None


class ExchangeCalibration(pydantic.BaseModel):
    """Parameters from each calibration are stored here."""

    rough_num_pulses: int | None = None
    fine_num_pulses: int | None = None
    cal_parameters: CalParameters


class ExchangeAxisConfig(pydantic.BaseModel):
    """Configuration parameters for a single exchange axis."""

    gates: ExchangeGateMap | ExchangeGateMapParams
    detuning_vector: Optional[List] = None
    exchange_vector: Optional[List] = None
    symmetric_vector: Optional[List] = None
    times: ExchangeTimes
    exchange_cal: Optional[ExchangeCalibration] = None


class Eo1QubitAxes(pydantic.BaseModel):
    """Exchange parameters for each axis, in units that qick can use."""

    z: ExchangeAxisConfig
    n: ExchangeAxisConfig
    m: ExchangeAxisConfig | None = None


class Eo1Qubit(Eo1QubitAxes):
    """All parameters needed for a single qubit experiment."""

    ro_cfg: spam_models.ReadoutConfig


class Eo2Qubit(pydantic.BaseModel):
    """All parameters needed for a two qubit experiment."""

    q1: Eo1Qubit
    q2: Eo1Qubit
