"""Helper functions for managing configs and file saving."""

import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

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
        except OSError:
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


def json_to_qickprog(data_obj, soccfg):
    """Load a JSON-encoded program stored on a data object into a QickProgramV2.

    :param data_obj: Any object with a ``prog`` attribute containing a JSON string.
    :param soccfg: The QICK system-on-chip configuration.
    """
    try:
        prog_dict = helpers.json2progs(data_obj.prog)
        qick_prog = asm_v2.QickProgramV2(soccfg)
        qick_prog.load_prog(prog_dict)
        data_obj.prog = qick_prog
    except TypeError:
        logger.warning("program attribute is not in json format")
    except Exception:
        logger.exception("failed to load qick program from json")
