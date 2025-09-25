import pydantic
from spinqick.models import spam_models


class RfPulses(pydantic.BaseModel):
    gain_90: float
    time_90: float


class Ld1QubitParams(pydantic.BaseModel):
    gate: str
    f0: float
    pulses: RfPulses
    rf_gen: int


class Ld1Qubit(Ld1QubitParams):
    ro_cfg: spam_models.ReadoutConfig
