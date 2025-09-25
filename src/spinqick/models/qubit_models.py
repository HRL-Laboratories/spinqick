import pydantic
from typing import List, Optional
from spinqick.models import spam_models
from spinqick import settings


class ExchangeGains(pydantic.BaseModel):
    """gains in DAC units"""

    idle_gain: float
    exchange_gain: float


class ExchangeVoltages(pydantic.BaseModel):
    """gains in voltage units"""

    idle_voltage: float
    exchange_voltage: float


class ExchangeTimes(pydantic.BaseModel):
    """times in microseconds"""

    idle_time: float
    exchange_time: float
    # t1j_time: float | None = None  # few microseconds


class ExchangeGateParams(pydantic.BaseModel):
    """generalized template for each gate in ExchangeConfig"""

    name: settings.GateNames  # name of gate, e.g. P1
    gate_voltages: ExchangeVoltages  # associated coordinates for common sequencing


class ExchangeGate(pydantic.BaseModel):
    """generalized template for each gate in ExchangeConfig"""

    name: settings.GateNames  # name of gate, e.g. P1
    gains: ExchangeGains  # associated coordinates for common sequencing
    gen: int  # should be chosen off hardware_cfg.yml file.


class ExchangeGateMap(pydantic.BaseModel):
    """specify gates used for an exchange axis. Gates must have an associated ExchangeGate in the axis config."""

    px: ExchangeGate
    py: ExchangeGate
    x: ExchangeGate


class ExchangeGateMapParams(pydantic.BaseModel):
    """specify gates used for an exchange axis. Gates must have an associated ExchangeGate in the axis config."""

    px: ExchangeGateParams
    py: ExchangeGateParams
    x: ExchangeGateParams


class CalParameters(pydantic.BaseModel):
    A: float
    B: float
    theta_max: float
    theta_list: List[float] | None
    volt_list: List[float] | None


class ExchangeCalibration(pydantic.BaseModel):
    """parameters from each calibration are stored here"""

    # cal_model: str  # = "log_poly" , make a list of acceptable types
    rough_num_pulses: int
    fine_num_pulses: int
    cal_parameters: CalParameters  # maybe this should be an object containing the cal curves etc instead of just a parameter list


class ExchangeAxisConfig(pydantic.BaseModel):
    """configuration parameters for a single exchange axis"""

    gates: ExchangeGateMap | ExchangeGateMapParams
    detuning_vector: Optional[List] = None
    exchange_vector: Optional[List] = None
    symmetric_vector: Optional[List] = None
    times: ExchangeTimes
    exchange_cal: Optional[ExchangeCalibration] = None


class Eo1QubitAxes(pydantic.BaseModel):
    """Exchange parameters for each axis, in units that qick can use"""

    z: ExchangeAxisConfig
    n: ExchangeAxisConfig
    m: ExchangeAxisConfig | None = None


class Eo1Qubit(Eo1QubitAxes):
    """All parameters needed for a single qubit experiment"""

    ro_cfg: spam_models.ReadoutConfig


class Eo2Qubit(pydantic.BaseModel):
    """All parameters needed for a two qubit experiment"""

    q1: Eo1Qubit
    q2: Eo1Qubit
