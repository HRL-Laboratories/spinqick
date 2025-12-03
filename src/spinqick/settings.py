"""User specific settings."""

import logging
import os
from typing import List, Literal, Tuple

import numpy as np
import yaml

from spinqick.make_config import FSETTING_PATH

logger = logging.getLogger(__name__)


class FileSettings:
    """Specify the locations of each config file on your machine, and the directory to save data
    to."""

    def __init__(self):
        try:
            self.load_settings()
        except FileNotFoundError as exc:
            print(f"config or settings files not found: {exc}")

    def load_settings(self):
        with open(FSETTING_PATH, "r") as file:
            self._file_dict = yaml.safe_load(file)
        self.data_directory: str = self._file_dict["file_settings"]["data_directory_path"]
        self.hardware_config: str = self._file_dict["file_settings"]["hardware_config_path"]
        self.dot_experiment_config: str = self._file_dict["file_settings"][
            "dot_experiment_config_path"
        ]
        if not os.path.isdir(self.data_directory):
            logger.warning("data directory does not exist")
        if not os.path.isfile(self.hardware_config):
            logger.warning("hardware config file does not exist")
        if not os.path.isfile(self.dot_experiment_config):
            logger.warning("experiment config file does not exist")

    def save_settings(self):
        self._file_dict["file_settings"]["data_directory_path"] = self.data_directory
        self._file_dict["file_settings"]["hardware_config_path"] = self.hardware_config
        self._file_dict["file_settings"]["dot_experiment_config_path"] = self.dot_experiment_config
        with open(FSETTING_PATH, "w") as file:
            yaml.safe_dump(self._file_dict, file)


class FilterSettings:
    """Optional settings for implementing pulse predistortion.

    Filtering is implemented with lfilter
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter
    """

    def __init__(self):
        self.load_settings()

    def load_settings(self):
        with open(FSETTING_PATH, "r") as file:
            self._file_dict = yaml.safe_load(file)
        self.iir_taps: Tuple[List[float], List[float]] = self._file_dict["filter_settings"][
            "iir_taps"
        ]
        self.iir_2_taps: Tuple[List[float], List[float]] = self._file_dict["filter_settings"][
            "iir_2_taps"
        ]
        self.fir_taps: np.ndarray | None = self._file_dict["filter_settings"]["fir_taps"]
        self.apply_filter: Literal["both", "iir_1", "fir"] | None = self._file_dict[
            "filter_settings"
        ]["apply_filter"]

    def save_settings(self):
        self._file_dict["filter_settings"]["iir_taps"] = self.iir_taps
        self._file_dict["filter_settings"]["iir_2_taps"] = self.iir_2_taps
        self._file_dict["filter_settings"]["fir_taps"] = self.fir_taps
        self._file_dict["filter_settings"]["apply_filter"] = self.apply_filter
        with open(FSETTING_PATH, "w") as file:
            yaml.safe_dump(self._file_dict, file)


class DacSettings:
    def __init__(self):
        self.load_settings()

    def load_settings(self):
        with open(FSETTING_PATH, "r") as file:
            self._file_dict = yaml.safe_load(file)
        self.t_min_slow_dac: float = self._file_dict["dac_settings"]["t_min_slow_dac"]
        self.trig_length: float = self._file_dict["dac_settings"]["trig_length"]
        self.trig_pin: int = self._file_dict["dac_settings"]["trig_pin"]

    def save_settings(self):
        self._file_dict["dac_settings"]["t_min_slow_dac"] = self.t_min_slow_dac
        self._file_dict["dac_settings"]["trig_length"] = self.trig_length
        self._file_dict["dac_settings"]["trig_pin"] = self.trig_pin
        with open(FSETTING_PATH, "w") as file:
            yaml.safe_dump(self._file_dict, file)


file_settings = FileSettings()
filter_settings = FilterSettings()
dac_settings = DacSettings()
