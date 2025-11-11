"""User specific settings."""

import os
from enum import StrEnum, auto
from typing import List, Literal, Tuple

import numpy as np
import pydantic_settings


class GateTypes(StrEnum):
    """Strenum for labeling the purpose of each gate."""

    MEASURE = auto()
    TUNNEL = auto()
    EXCHANGE = auto()
    PLUNGER = auto()
    AUX = auto()


class GateNames(StrEnum):
    """Modify this to suit the way you label your system."""

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


def make_default_config_path(filename: str):
    """Makes a default path to the config file within the spinqick package."""
    settings_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(settings_path)
    full_path = os.path.join(current_dir, filename)
    return full_path


class FileSettings(pydantic_settings.BaseSettings):
    """Specify the locations of each config file on your machine, and the directory to save data to.

    By default, the files are located in the spinqick repo, but its a good idea to save them
    elsewhere on your machine.
    """

    data_directory: str = "C:/Users/alwessels/data"
    hardware_config: str = make_default_config_path("config/hardware_config.json")
    dot_experiment_config: str = make_default_config_path("config/full_config.json")


class FilterSettings(pydantic_settings.BaseSettings):
    """Optional settings for implementing pulse predistortion.

    Filtering is implemented with lfilter
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter
    """

    iir_taps: Tuple[List[float], List[float]] | None = (
        [1.0, -1.99081583, 0.99082555],
        [0.9685454, -1.92799398, 0.95945829],
    )
    iir_2_taps: Tuple[List[float], List[float]] | None = (
        [1.0, -0.99050165],
        [0.98835501, -0.97885945],
    )

    fir_taps: np.ndarray | None = None
    apply_filter: Literal["both", "iir_1", "fir"] | None = None


class DacSettings(pydantic_settings.BaseSettings):
    """Specific settings pertaining to slow speed DACs."""

    t_min_slow_dac: float = 3.0  # minimum slow dac sweep step in microseconds
    trig_length: float = 0.2  # time that dac trigger is set high in microseconds
    trig_pin: int = 0  # trigger pin to use when calling the trigger function


file_settings = FileSettings()
filter_settings = FilterSettings()
dac_settings = DacSettings()
