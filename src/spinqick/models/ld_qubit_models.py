import pydantic

from spinqick.models import spam_models


class RfPulses(pydantic.BaseModel):
    """Definitions for pi/2 pulses."""

    gain_90: float
    time_90: float


class Ld1QubitParams(pydantic.BaseModel):
    """Full single-qubit definition."""

    gate: str
    f0: float  # frequency in MHz
    pulses: RfPulses
    rf_gen: int


class Ld1Qubit(Ld1QubitParams):
    """Full single-qubit definition with readout."""

    ro_cfg: spam_models.ReadoutConfig
