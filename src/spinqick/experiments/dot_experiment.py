import addict
import logging

from spinqick.helper_functions import file_manager
from spinqick.settings import file_settings
from spinqick.models import hardware_config_models, config_models

logger = logging.getLogger(__name__)


def updater(func):
    """Call this as a decorator to update config before and after a dot experiment"""

    def wrapper(dot_expt: DotExperiment, *args, **kwargs):
        # pull the parameters from the yaml file and assign to the local config object
        dot_expt.update_local()
        dot_expt.validate_hardware_config()
        dot_expt.validate_readout_config()
        result = func(dot_expt, *args, **kwargs)
        return result

    return wrapper


class DotExperiment:
    """Manages the readout and hardware configs, and updates them appropriately"""

    def __init__(self, datadir: str = file_settings.data_directory):
        self.data_path = file_manager.get_data_dir(datadir=datadir)
        file_manager.check_configs_exist()
        self.config_path = file_settings.readout_config
        self.hardware_path = file_settings.hardware_config
        self.config = addict.Dict(file_manager.load_config(self.config_path))
        self.hardware_config = addict.Dict(file_manager.load_config(self.hardware_path))
        self.config = file_manager.sync_configs(self.config, self.hardware_config)

    def update_config(self, updates: dict):
        """pass in updates to readout_cfg in the form of a dictionary

        :param updates: dictionary containing the updated parameters
        """
        self.config.update(updates)
        file_manager.save_config(self.config, self.config_path)

    def update_hardware_dict(self, updates: dict):
        """pass in updates in the form of a dictionary

        :param updates: dictionary containing the updated parameters
        """
        self.hardware_config.update(updates)
        file_manager.save_config(self.hardware_config, self.hardware_path)

    def update_yaml(self):
        """take just the PSB and DCS readout parameters and overwrite them in the yaml"""
        config = file_manager.load_config(self.config_path)
        config["DCS_cfg"] = self.config["DCS_cfg"]
        config["PSB_cfg"] = self.config["PSB_cfg"]
        if isinstance(config, addict.Dict):
            file_manager.save_config(config.to_dict(), self.config_path)
        else:
            file_manager.save_config(config, self.config_path)
        logger.info("updated yaml file")

    def update_local(self):
        """update local config dict from readout and hardware yaml files"""
        config = addict.Dict(file_manager.load_config(self.config_path))
        self.hardware_config = addict.Dict(file_manager.load_config(self.hardware_path))
        DCS = config["DCS_cfg"]
        PSB = config["PSB_cfg"]
        self.config["DCS_cfg"].update(DCS)
        self.config["PSB_cfg"].update(PSB)
        self.config = file_manager.sync_configs(self.config, self.hardware_config)
        logger.info("updated local params")

    def validate_hardware_config(self):
        """use pydantic models to validate the hardware config.  Returns pydantic model object"""

        try:
            config_object = hardware_config_models.HardwareConfig.model_validate(
                self.hardware_config
            )
        except Exception as exc:
            logger.error("Couldn't validate hardware config, %s", exc, exc_info=True)
        return config_object

    def validate_readout_config(self):
        """use pydantic models to validate the readout config.  Returns pydantic model object"""

        try:
            config_object = config_models.ReadoutConfig.model_validate(self.config)
        except Exception as exc:
            logger.error("Couldn't validate readout config, %s", exc, exc_info=True)
        return config_object

    def volts2dac(self, volts: float, gate: str) -> int:
        """
        convert voltage out of qick frontend to dac units

        :param volts: voltage to convert
        :return: RFSoC output in DAC units
        """

        conversion = self.hardware_config["channels"][gate]["dac_conversion_factor"]
        return int(volts * conversion)

    def dac2volts(self, dacunits: int, gate: str) -> float:
        """
        convert dac units to voltage out of RFSoC front end

        :param dacunits: DAC units
        :return: voltage
        """
        conversion = self.hardware_config["channels"][gate]["dac_conversion_factor"]
        return dacunits / conversion
