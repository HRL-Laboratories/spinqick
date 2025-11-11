"""Pydantic models for passing in to qick_code classes."""

from typing import List, Literal

import pydantic

from spinqick import settings
from spinqick.helper_functions import spinqick_enums
from spinqick.models import dcs_model, ld_qubit_models, qubit_models, spam_models


class SweepTwoConfig(pydantic.BaseModel):
    gx_gen: int
    gx_gate: settings.GateNames
    gy_gen: int
    gy_gate: settings.GateNames
    gx_start: float  # specify range (-1,1) on these
    gx_stop: float
    gx_expts: int
    gy_start: float
    gy_stop: float
    gy_expts: int


class PatConfig(pydantic.BaseModel):
    pat_freq: float = 0
    pat_gen: int = 0
    pat_gain: float = 0


class GvgBasebandConfig(SweepTwoConfig):
    measure_buffer: float  # delay in microseconds between changing voltage points and measurement
    dcs_cfg: dcs_model.DcsConfig


class GvgDcConfig(pydantic.BaseModel):
    trig_pin: int
    measure_buffer: float
    points: int
    dcs_cfg: dcs_model.DcsConfig
    trig_length: float  # length of trigger pulse
    mode: Literal["transdc", "sd_chop"] = "sd_chop"


class GvgPatConfig(GvgDcConfig):
    pat_cfg: PatConfig


class AvgedExperiment(pydantic.BaseModel):
    point_avgs: int
    full_avgs: int


class AvgedReadout(AvgedExperiment):
    dcs_cfg: dcs_model.DcsConfig


class SweepDelay(AvgedReadout):
    delay_start: float
    delay_stop: float
    delay_points: int
    loop_slack: float


class LineSplitting(GvgDcConfig):
    differential_channel: int
    differential_ac_gain: float
    differential_ac_freq: float
    differential_step_gain: float


class HsaTune(pydantic.BaseModel):
    dcs_cfg: dcs_model.DcsConfig
    point_avgs: int
    tune_gate: str
    tune_gate_gen: int
    pulse_time: float
    pulse_gain: float
    measure_buffer: float


class MeashistConfig(spam_models.ReadoutConfig):
    num_measurements: int


class PsbScanConfig(SweepTwoConfig, spam_models.ReadoutConfig, AvgedExperiment): ...


class T2StarConfig(AvgedExperiment):
    dcs_cfg: dcs_model.DcsConfig
    psb_cfg: spam_models.DefaultSpamDac
    reference: bool
    start: float
    stop: float
    expts: int
    axis: spinqick_enums.ExchangeAxis
    qubit: qubit_models.Eo1Qubit


class MeasScanConfig(PsbScanConfig):
    step_time: float


class IdleScanConfig(PsbScanConfig):
    add_rf: bool = False
    rf_gen: int
    rf_freq: float
    rf_length: float
    rf_gain: float


class FingerprintConfig(SweepTwoConfig, AvgedExperiment):
    qubit: qubit_models.Eo1Qubit
    axis: spinqick_enums.ExchangeAxis
    n_pulses: int = 1


class NonEquilibriumConfig(FingerprintConfig):
    t1j: bool


class SweepOneConfig(pydantic.BaseModel):
    gen: int
    start: float
    stop: float
    expts: int


class NoscConfig(AvgedExperiment):
    qubit: qubit_models.Eo1Qubit
    axis: spinqick_enums.ExchangeAxis
    start: float
    stop: float
    expts: int


class CourseCalConfig(NoscConfig):
    n_pulses: int


class FineCalConfig(pydantic.BaseModel):
    qubit: qubit_models.Eo1Qubit
    n_pulses: int
    point_avgs: int
    axis: spinqick_enums.ExchangeAxis
    exchange_gain: float
    t_res: Literal["fs", "fabric"]


class RfSweep(SweepOneConfig, AvgedExperiment):
    pulse_gain: float
    pulse_length: float
    ro_cfg: spam_models.ReadoutConfig


class RfSweepTwo(AvgedExperiment):
    qubit: ld_qubit_models.Ld1Qubit
    gain: float
    gx_start: float
    gx_stop: float
    gx_expts: int
    gy_start: float
    gy_stop: float
    gy_expts: int


class TimeRabi(AvgedExperiment):
    qubit: ld_qubit_models.Ld1Qubit
    gain: float
    start: float
    stop: float
    expts: int


class AmplitudeRabi(AvgedExperiment):
    qubit: ld_qubit_models.Ld1Qubit
    time: float
    start: float
    stop: float
    expts: int


class LdSweepOne(AvgedExperiment):
    qubit: ld_qubit_models.Ld1Qubit
    start: float
    stop: float
    expts: int


class LdSweepTwo(AvgedExperiment):
    qubit: ld_qubit_models.Ld1Qubit
    gx_start: float
    gx_stop: float
    gx_expts: int
    gy_start: float
    gy_stop: float
    gy_expts: int


class SpinEcho(LdSweepOne):
    n_echoes: int


class PlayXY(AvgedExperiment):
    qubit: ld_qubit_models.Ld1Qubit
    gate_set: List[List[str]]
