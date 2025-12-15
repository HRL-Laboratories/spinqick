import os
import shutil
from importlib import resources


def make_default_configs(
    make_hw_cfg=True, make_experiment_cfg=True, make_filter_cfg=True, overwrite=False
):
    """Makes default file settings file and entries. Also makes default hardware_cfg, filter_config
    and experiment_cfg files. By default, this copies the hardware config and experiment config into
    the .spinqick folder as well.

    :param cfg_dir:
    """

    hw_path = os.environ["SPINQICK_HARDWARE_CONFIG"]
    exp_path = os.environ["SPINQICK_DOT_EXPERIMENT_CONFIG"]
    data_dir = os.environ["SPINQICK_DATA_DIRECTORY"]
    if os.environ["SPINQICK_FILTER_CONFIG"]:
        filter_path = os.environ["SPINQICK_DATA_DIRECTORY"]
    else:
        filter_path = None
    ### copy the default files into the specified config directory
    os.makedirs(data_dir, exist_ok=True)
    r_hw = resources.files("spinqick") / "config/hardware_config.json"
    r_exp = resources.files("spinqick") / "config/experiment_config.json"
    r_filt = resources.files("spinqick") / "config/filter_config.json"
    if make_hw_cfg:
        if not os.path.isfile(hw_path) or overwrite:
            with resources.as_file(r_hw) as file_path:
                shutil.copy(file_path, hw_path)
    if make_experiment_cfg:
        if not os.path.isfile(exp_path) or overwrite:
            with resources.as_file(r_exp) as file_path:
                shutil.copy(file_path, exp_path)
    if make_filter_cfg and filter_path is not None:
        if not os.path.isfile(filter_path) or overwrite:
            with resources.as_file(r_filt) as file_path:
                shutil.copy(file_path, filter_path)
