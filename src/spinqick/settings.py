"""User specific settings"""

import pydantic_settings


class FileSettings(pydantic_settings.BaseSettings):
    data_directory: str = "C:/Data/QICK/"
    hardware_config: str = "C:/Data/QICK/hardware_cfg.yaml"
    readout_config: str = "C:/Data/QICK/readout_cfg.yaml"


file_settings = FileSettings()
