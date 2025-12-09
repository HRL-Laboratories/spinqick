"""User specific settings."""

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pydantic_settings

logger = logging.getLogger(__name__)


class FileSettings(pydantic_settings.BaseSettings):
    """Specify the locations of each config file on your machine, and the directory to save data to.

    By default, the files are located in the spinqick repo, but its a good idea to save them
    elsewhere on your machine.
    """

    model_config = pydantic_settings.SettingsConfigDict(env_prefix="SPINQICK_")
    data_directory: str
    hardware_config: str
    dot_experiment_config: str
    filter_config: Optional[str]


class FilterSettings(pydantic_settings.BaseSettings):
    iir_taps: Tuple[List[float], List[float]] | None = None
    iir_2_taps: Tuple[List[float], List[float]] | None = None
    fir_taps: np.ndarray | None = None
    apply_filter: Literal["both", "iir_1", "fir"] | None = None


file_settings = FileSettings()  # type: ignore
if file_settings.filter_config is not None:
    json_string = Path(file_settings.filter_config).read_text()
    filter_settings = FilterSettings.model_validate_json(json_string)
else:
    filter_settings = FilterSettings()
