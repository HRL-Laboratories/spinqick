"""
Helper functions for managing config dictionaries and file saving structure
We use a library called addict that
TODO: update the default hardware and readout configs in the file folder
"""

import datetime
import glob
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import addict
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import yaml

from spinqick.settings import file_settings

logger = logging.getLogger(__name__)
DATA_DIRECTORY = file_settings.data_directory


def get_data_dir(datadir: str = DATA_DIRECTORY) -> str:
    """get the data directory for today's date. If it doesn't exist, make the directory.

    :param datadir: general data directory where you want to store
    data from your experiments.  Also where you store the config files for spinqick.
    :return: path to the directory
    """
    date = datetime.date.today()
    current_dir = str(date.month) + "_" + str(date.day) + "_" + str(date.year)
    data_path = os.path.join(datadir, current_dir)

    if not os.path.isdir(data_path):
        try:
            os.mkdir(data_path)
        except FileExistsError:
            logger.info("directory already exists")
        except Exception:
            logger.warning("couldn't create today's folder")

    return data_path


def check_configs_exist():
    """Check if readout and hardware configs exist."""
    ro_cfg_path = file_settings.readout_config
    if os.path.isfile(ro_cfg_path):
        logger.info("readout config exists, using this file: %s" % ro_cfg_path)
    else:
        logger.error("no readout config found")

    hw_cfg_path = file_settings.hardware_config
    if os.path.isfile(hw_cfg_path):
        logger.info("harwdare config exists, using this file: %s" % hw_cfg_path)
    else:
        logger.error("no hardware config found")


def get_new_timestamp(datadir: str = DATA_DIRECTORY) -> tuple[str, float]:
    """Set up a data folder for today's date and get a current timestamp

    :param datadir: general data directory where you want to store
        data from your experiments.  Also where you store the config files for spinqick.
    :return:
            -path to the directory created
            -timestamp to uniquely identify a file in the directory
    """
    data_path = get_data_dir(datadir)
    stamp = int(time.time())
    return data_path, stamp


def get_new_filename(suffix: str, data_path: str, timestamp: int) -> str:
    """Create a new filename from data path and timestamp

    :param datapath: file directory
    :param timestamp: time in seconds
    :return: full file path
    """
    fname = os.path.join(data_path, str(timestamp) + suffix)
    return fname


def load_config(filename: str) -> dict:
    """load a config from yaml file format

    :param filename: config file name
    :return: config dictionary
    """
    try:
        with open(filename, "r") as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        logger.info("Loaded config file", filename)
        return yaml_cfg
    except Exception:
        logger.exception("unable to load config")
        cfg: dict[str, Any] = {}
        return cfg


def save_config(config: dict, filename: str):
    """save a config to a yaml file

    :param config: dictionary
    :param filename: config file name
    """
    with open(filename, "w") as file:
        yaml.dump(config, file)
    logger.info("saved config file", filename)


def load_hardware_config() -> dict:
    """load hardware config file

    :param datadir: general data directory where you want to store
        data from your experiments.  Also where you store the config files for spinqick.
    :return: hardware config dictionary
    """
    return load_config(file_settings.hardware_config)


def sync_configs(readout_cfg: Dict, hardware_cfg: Dict) -> Dict:
    """pulls readout config parameters from the hardware cfg, update and return readout config.

    :param readout_cfg: readout config dictionary
    :param hardware_cfg: hardware config dictionary
    :return: readout config dictionary
    """
    config = readout_cfg
    config["DCS_cfg"]["ro_ch"] = hardware_cfg["SD_out"]["qick_adc"]
    config["DCS_cfg"]["res_ch"] = hardware_cfg["SD_in"]["qick_gen"]
    return config


class SaveData(netCDF4.Dataset):
    """Save data in netcdf4 format. Check out their documentation for information at
    https://unidata.github.io/netcdf4-python/.  Don't forget to run .close()
    on the Dataset object after you've finished adding data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_axis(
        self,
        label: str,
        data: np.ndarray | None = None,
        units: str | None = None,
        dtype=np.float32,
    ):
        """provide axes for the data you want to store. If the axis doesn't
        have data associated with it (ie I and Q axis) then leave data=None.
        This sets the axis dimension to unlimited.

        :param label: A label for the axis you're providing
        :param data: If the axis has data associated with it (i.e. time in seconds)
            you can provide it here
        :param units: Provide a string that describes the units of this axis
        :param dtype: datatype of the data provided.  This is in a numpy dtype form.
        """

        if isinstance(data, np.ndarray):
            self.createDimension(label + "_dim", len(data))
            var_obj = self.createVariable(label, dtype, (label + "_dim"))
            var_obj[:] = data
            if units is not None:
                var_obj.units = units
        else:
            self.createDimension(label + "_dim", None)

    def add_dataset(
        self,
        label: str,
        axes: list[str],
        data: np.ndarray,
        units: str | None = None,
        dtype=np.float32,
    ):
        """add your data array.  Axes needs the be a list of the values

        :param label: A label for the data you're providing
        :param axes: Provide a list of the axes you made which reflects the shape of the data array
        :param data: Data in array form
        :param units: Provide a string that describes the units of this axis
        :param dtype: datatype of the data provided.  This is in a numpy dtype form.

        """

        ax_labels = (axis + "_dim" for axis in axes)
        dataset = self.createVariable(label, dtype, ax_labels)
        dataset[:] = data
        if units is not None:
            dataset.units = units

    def save_last_plot(self):
        """saves your figure as a .png with the same timestamp
        and name as the data"""

        filename = Path(self.filepath())
        # end = filename.suffix
        filename_stripped = str(filename).split(".")
        plotname = filename_stripped[0] + ".png"
        if Path(plotname).is_file():
            prefix = str(filename.stem).split("_")[0]
            location = os.path.dirname(plotname)
            base = os.path.join(location, prefix)
            dirlist = glob.glob(str(base) + "*.png")
            dirlist.sort()
            try:
                ii = int(str(dirlist[-1]).split("_")[-1].strip(".png")) + 1
            except Exception:
                ii = 0
            plotname = filename_stripped[0] + "_" + str(ii) + ".png"
        logger.info("saved plot at %s" % plotname)
        plt.gcf()
        plt.savefig(plotname)

    def save_config(self, full_config):
        """saves your config dictionary as a yaml"""

        filename = Path(self.filepath())
        # end = filename.suffix
        filename_stripped = str(filename).split(".")
        config_file = filename_stripped[0] + ".yaml"
        if isinstance(full_config, addict.Dict):
            save_config(full_config.to_dict(), config_file)
        else:
            save_config(full_config, config_file)
        logger.info("saved config at %s" % config_file)
        logger.info("saved config at %s" % config_file)
