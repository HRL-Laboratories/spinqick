"""User specific settings."""

import logging
from typing import Optional

import pydantic_settings

logger = logging.getLogger(__name__)


class FileSettings(pydantic_settings.BaseSettings):
    """Specify the locations of each config file on your machine, and the directory to save data to.

    By default, the files are located in the spinqick repo, but its a good idea to save them
    elsewhere on your machine.
    """

    model_config = pydantic_settings.SettingsConfigDict(env_prefix="SPINQICK_")
    data_directory: str = ""
    hardware_config: str = ""
    dot_experiment_config: str = ""
    filter_config: Optional[str] = None


file_settings = FileSettings()
