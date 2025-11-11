"""Helper functions for managing configs and file saving."""

import datetime
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pydantic
import yaml
from qick import QickConfig, asm_v2, helpers
from qick.qick_asm import AbsQickProgram

from spinqick.models import hardware_config_models
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)
DATA_DIRECTORY = file_settings.data_directory


def get_data_dir(datadir: str = DATA_DIRECTORY) -> str:
    """Get the data directory for today's date. If it doesn't exist, make the directory.

    :param datadir: general data directory where you want to store data from your experiments. Also
        where you store the config files for spinqick.
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

    ro_cfg_path = file_settings.dot_experiment_config
    if os.path.isfile(ro_cfg_path):
        logger.info("experiment config exists, using this file: %s", ro_cfg_path)
    else:
        logger.error("no experiment config found")

    hw_cfg_path = file_settings.hardware_config
    if os.path.isfile(hw_cfg_path):
        logger.info("harwdare config exists, using this file: %s", hw_cfg_path)
    else:
        logger.error("no hardware config found")


def get_new_timestamp(datadir: str = DATA_DIRECTORY) -> tuple[str, int]:
    """Set up a data folder for today's date and get a current timestamp.

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
    """Create a new filename from data path and timestamp.

    :param datapath: file directory
    :param timestamp: time in seconds
    :return: full file path
    """
    fname = os.path.join(data_path, str(timestamp) + suffix)
    return fname


def load_config_yaml(filename: str) -> dict:
    """Load a config from yaml file format.

    :param filename: config file name
    :return: config dictionary
    """
    try:
        with open(filename, "r", encoding="utf-8") as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        logger.info("Loaded config file %s", filename)
        return yaml_cfg
    except Exception:
        logger.exception("unable to load config")
        cfg: dict[str, Any] = {}
        return cfg


def load_config_json(filename: str, config_model):
    """Load config from json."""
    json_string = Path(filename).read_text()
    config = config_model.model_validate_json(json_string)
    return config


def save_config_yaml(config: dict, filename: str):
    """Save a config to a yaml file.

    :param config: dictionary
    :param filename: config file name
    """
    cfg_dict = config
    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(cfg_dict, file)
    logger.info("saved config file as %s", filename)


def save_config_json(config: pydantic.BaseModel, filename: str):
    """Save pydantic model object to json file."""
    json_string = config.model_dump(mode="json")
    with open(filename, "w", encoding="utf8") as f:
        json.dump(json_string, f, ensure_ascii=False, indent=4)


def load_hardware_config() -> hardware_config_models.HardwareConfig:
    """Load hardware config file.

    :param datadir: general data directory where you want to store data from your experiments. Also
        where you store the config files for spinqick.
    :return: hardware config dictionary
    """
    return load_config_json(
        file_settings.hardware_config, hardware_config_models.HardwareConfig
    )


def save_prog(prog: AbsQickProgram, filename: str):
    """Save a qickprogram to json."""
    prog_dict = prog.dump_prog()
    prog_json = helpers.progs2json(prog_dict)
    with open(filename, "w", encoding="utf8") as f:
        json.dump(prog_json, f, ensure_ascii=False, indent=4)


def load_qickprogram(fname: str, soccfg: QickConfig):
    """Loads a qickprogram from json.

    This requires the soccfg used to run the program.
    """
    with open(fname, "r", encoding="utf8") as f:
        prog_json_string = json.load(f)
    prog_dict = helpers.json2progs(prog_json_string)
    prog = asm_v2.QickProgramV2(soccfg)
    prog.load_prog(prog_dict)
    return prog


# pylint: disable = no-member
class SaveData(netCDF4.Dataset):
    """Save data in netcdf4 format.

    Check out their documentation for information at
    https://unidata.github.io/netcdf4-python/.
    Don't forget to run .close()
    on the Dataset object after you've finished adding data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename_prefix = self.get_filename_prefix()

    def get_filename_prefix(self):
        """Returns the filename prefix for the given netcdf object."""
        filename = Path(self.filepath())
        filename_stripped = str(filename).split(".")
        return filename_stripped[0]

    def add_axis(
        self,
        label: str,
        data: np.ndarray | None = None,
        units: str | None = None,
        dtype=np.float32,
    ):
        """Provide axes for the data you want to store. If the axis doesn't have data associated
        with it (ie I and Q axis) then leave data=None. This sets the axis dimension to unlimited.

        :param label: A label for the axis you're providing
        :param data: If the axis has data associated with it (i.e. time in seconds) you can provide
            it here
        :param units: Provide a string that describes the units of this axis
        :param dtype: datatype of the data provided. This is in a numpy dtype form.
        """

        if isinstance(data, np.ndarray):
            self.createDimension(label + "_dim", len(data))
            var_obj = self.createVariable(label, dtype, (label + "_dim"))
            var_obj[:] = data
            if units is not None:
                var_obj.units = units
        else:
            self.createDimension(label + "_dim", None)

    def add_multivariable_axis(
        self,
        dim_label: str,
        sweep_dict: dict,
        group_path: str | None = None,
        dtype=np.float32,
    ):
        """Adds a dimension to netCDF object which corresponds to multiple swept variables.

        :param dim_label: name of the dimension associate with this axis
        :param sweep_dict: provide the axis dictionary. This needs to be in the format that is used
            for spinqickdata axis dictionaries.
        :param group_path: if the dataset is in a specific folder within the netcdf file, specify
            the name of this folder.
        :param dtype: datatype of the data provided. This is in a numpy dtype form.
        """
        dim = dim_label + "_dim"
        if dim not in self.dimensions:
            self.createDimension(dim, sweep_dict["size"])
        swept_vars = sweep_dict["sweeps"]
        for sweep_var in list(swept_vars.keys()):
            if group_path is None:
                var_obj = self.createVariable(sweep_var, dtype, (dim))
            else:
                group_obj: netCDF4.Group = self[group_path]
                var_obj = group_obj.createVariable(sweep_var, dtype, (dim))
            var_obj[:] = swept_vars[sweep_var]["data"]
            if swept_vars[sweep_var]["units"] is not None:
                var_obj.units = swept_vars[sweep_var]["units"]

    def add_dataset(
        self,
        label: str,
        axes: list[str],
        data: np.ndarray,
        units: str | None = None,
        group_path: str | None = None,
        dtype=np.float32,
    ):
        """Adds a data array to the netcdf object.  Axes needs the be a list of axis values.

        :param label: A label for the data you're providing
        :param axes: Provide a list of the axes you made which reflects the shape of the data array
        :param data: Data in array form
        :param units: Provide a string that describes the units of this axis
        :param group_path: if the dataset is in a specific folder within the netcdf file, specify
            the name of this folder.
        :param dtype: datatype of the data provided. This is in a numpy dtype form.
        """

        ax_labels = (axis + "_dim" for axis in axes)
        if group_path is None:
            variable = self.createVariable(label, dtype, ax_labels)
        else:
            group_obj: netCDF4.Group = self[group_path]
            variable = group_obj.createVariable(label, dtype, ax_labels)
        variable[:] = data
        if units is not None:
            variable.units = units

    def save_last_plot(self, fignum=None):
        """Saves a figure as a .png with the same filename as the data."""
        if fignum is None:
            plt.gcf()
        else:
            plt.figure(fignum)
        plotname = self.filename_prefix + ".png"
        if Path(plotname).is_file():
            location = os.path.dirname(plotname)
            base = os.path.join(location, self.filename_prefix)
            dirlist = glob.glob(str(base) + "*.png")
            dirlist.sort()
            try:
                ii = int(str(dirlist[-1]).split("_")[-1].strip(".png")) + 1
            except Exception:
                ii = 2
            plotname = self.filename_prefix + "_" + str(ii) + ".png"
        logger.info("saved plot at %s", plotname)
        plt.savefig(plotname)

    def save_config_json(self, full_config: pydantic.BaseModel):
        """Saves a pydantic model as a json file."""
        config_file = self.filename_prefix + "_cfg.json"
        save_config_json(full_config, config_file)
        logger.info("saved config at %s", config_file)

    def save_prog_json(self, prog: AbsQickProgram):
        """Saves a qick program to a json file."""
        prog_file = self.filename_prefix + "_prog.json"
        save_prog(prog, prog_file)
        logger.info("saved program at %s", prog_file)
