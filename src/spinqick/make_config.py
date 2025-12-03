import os
import shutil
from importlib import resources

import yaml

FSETTING_PATH = os.path.expanduser("~/.spinqick/settings.yaml")
FSETTING_DIR = os.path.dirname(FSETTING_PATH)


def make_default_filesettings(
    cfg_dir=FSETTING_DIR,
    data_dir=os.path.join(FSETTING_DIR, "data"),
    make_hw_cfg=True,
    make_experiment_cfg=True,
):
    """Makes default file settings file and entries. Also makes default hardware_cfg and
    experiment_cfg files This will create a settings.yaml file in a ~/.spinqick directory. By
    default, this copies the hardware config and experiment config into the .spinqick folder as
    well.

    :param cfg_dir:
    """

    hw_cfg = "hardware_config.json"
    exp_cfg = "full_config.json"
    hw_path = os.path.join(cfg_dir, hw_cfg)
    exp_path = os.path.join(cfg_dir, exp_cfg)
    fsetting_default = resources.files("spinqick") / "config/settings.yaml"
    with fsetting_default.open() as file:
        fset_dict = yaml.safe_load(file)
    fset_dict["file_settings"]["hardware_config_path"] = hw_path
    fset_dict["file_settings"]["dot_experiment_config_path"] = exp_path
    fset_dict["file_settings"]["data_directory_path"] = data_dir
    os.makedirs(FSETTING_DIR, exist_ok=True)
    with open(FSETTING_PATH, "w") as file:
        yaml.safe_dump(fset_dict, file)
    ### copy the default files into the specified config directory
    os.makedirs(data_dir, exist_ok=True)
    r_hw = resources.files("spinqick") / "config/hardware_config.json"
    r_exp = resources.files("spinqick") / "config/experiment_cfg.json"
    if make_hw_cfg:
        with resources.as_file(r_hw) as file_path:
            shutil.copy(file_path, hw_path)
    if make_experiment_cfg:
        with resources.as_file(r_exp) as file_path:
            shutil.copy(file_path, exp_path)
