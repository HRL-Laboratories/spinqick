from spinqick.models import dcs_model, spam_models, qubit_models, ld_qubit_models
import pydantic
from typing import Union, Literal, Mapping


class ReadoutParams(pydantic.BaseModel):
    """psb-related parameters"""

    psb_cfg: spam_models.DefaultSpam
    measure_dot: Literal["M1", "M2"]  # specify which readout to use
    reference: bool  # whether to take a reference measurement
    thresh: bool  # whether to threshold PSB data
    threshold: float  # value of threshold


class QubitParams(pydantic.BaseModel):
    """spam settings and qubit parameters for one qubit"""

    ro_cfg: ReadoutParams
    qubit_params: Union[
        qubit_models.Eo1QubitAxes, ld_qubit_models.Ld1QubitParams, None
    ] = None


class ExperimentConfig(pydantic.BaseModel):
    """full experiment config model for up to two qubits"""

    m1_readout: dcs_model.DcsConfigParams
    m2_readout: dcs_model.DcsConfigParams
    qubit_configs: Mapping[str, QubitParams] | None = None


class Ro1Qubit(pydantic.BaseModel):
    """Just readout, in rfsoc units"""

    ro_cfg: spam_models.ReadoutConfig


class ExperimentConfigRfsoc(pydantic.BaseModel):
    """full experiment config in rfsoc units"""

    m1_readout: dcs_model.DcsConfig
    m2_readout: dcs_model.DcsConfig
    qubit_configs: Mapping[
        str, Union[qubit_models.Eo1Qubit, ld_qubit_models.Ld1Qubit, Ro1Qubit]
    ] = {}
