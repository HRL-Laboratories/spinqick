"""User specific settings"""

import pydantic_settings
from typing import List, Tuple, Literal
from enum import StrEnum, auto


class GateTypes(StrEnum):
    MEASURE = auto()
    TUNNEL = auto()
    EXCHANGE = auto()
    PLUNGER = auto()
    AUX = auto()


class GateNames(StrEnum):
    """modify this to suit the way you label your system"""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    P6 = "P6"
    X1 = "X1"
    X2 = "X2"
    X3 = "X3"
    X4 = "X4"
    X5 = "X5"
    T1 = "T1"
    T6 = "T6"
    T3 = "T3"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    Z1 = "Z1"
    Z2 = "Z2"
    Z3 = "Z3"
    Z4 = "Z4"
    M1 = "M1"
    M2 = "M2"
    SG = "SG"
    IFG = "IFG"
    OFG = "OFG"
    HEMT1 = "HEMT1"
    HEMT2 = "HEMT2"
    SD = "SD"
    DVDD = "DVDD"
    AVDD = "AVDD"
    AVSS = "AVSS"
    DVSS = "DVSS"
    SW_LGC = "SW_LGC"
    TEST = "test"


class FileSettings(pydantic_settings.BaseSettings):
    """specify the locations of each config file on your machine.
    TODO combine these into a single json file
    """

    data_directory: str = "C:/Data/QICK/eo_demo_2025/data/"
    hardware_config: str = "C:/Data/QICK/eo_demo_2025/hardware_cfg.json"
    dot_experiment_config: str = "C:/Data/QICK/eo_demo_2025/full_config.json"


class FilterSettings(pydantic_settings.BaseSettings):
    """optional settings for implementing pulse predistortion. Filtering is implemented with lfilter
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter
    """

    iir_taps: Tuple[List[float], List[float]] | None = (
        [1.0076777, -1.00368251],
        [1.0, -0.996],
    )
    iir_2_taps: Tuple[List[float], List[float]] | None = (
        [0.06151177, 0.06151177],
        [1.0, -0.87697646],
    )
    apply_filter: Literal["both", "iir_1"] | None = None


class DacSettings(pydantic_settings.BaseSettings):
    """specific settings pertaining to slow speed DACs"""

    t_min_slow_dac: float = 3.0  # minimum slow dac sweep step
    trig_length: float = 0.2  # time that dac trigger is set high
    trig_pin: int = 0  # trigger pin to use when calling the trigger function


file_settings = FileSettings()
filter_settings = FilterSettings()
dac_settings = DacSettings()
