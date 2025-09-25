"""
data handling and saving for spinqick experiments
"""

import logging
import os
from typing import List, Dict, Sequence
import time
import json
import pydantic
import netCDF4
import importlib_metadata
from qick import qick_asm, helpers, asm_v2
import numpy as np
from spinqick.helper_functions import file_manager
from spinqick.models import experiment_models
from spinqick.core import spinqick_utils

logger = logging.getLogger(__name__)


def load_analysis_data(
    nc_file: netCDF4.Dataset, data_desc: str, attr_name: str | None = None
):
    """load data stored in the 'analyzed_data' folder of the netcdf file"""
    processed = []
    attrs = []
    ana = nc_file["analyzed_data"]
    ana_data_keys = list(ana.variables.keys())
    analysis_type = "None"
    for d in ana_data_keys:
        if data_desc in d:
            data_array = np.asarray(ana[d][:])
            processed.append(data_array)
            if hasattr(ana[d], "units"):
                analysis_type = ana[d].units
            if attr_name is not None:
                if hasattr(ana[d], attr_name):
                    var_attr = getattr(ana[d], attr_name)
                    attrs.append(var_attr)
    return processed, analysis_type, attrs


def _assign_timestamp():
    return int(time.time())


def _get_filename(timestamp, experiment_name):
    data_path = file_manager.get_data_dir()
    data_file = os.path.join(
        data_path,
        str(timestamp) + experiment_name + ".nc",
    )
    return data_file


class SpinqickData:
    """Self describing data object to handle data from a QICK AcquireProgramv2 output"""

    def __init__(
        self,
        raw_data: List[np.ndarray],
        cfg: pydantic.BaseModel,
        triggers: int,
        reps: int,
        experiment_name: str,
        analyzed_data: List[np.ndarray] | None = None,
        timestamp: int | None = None,
        filename: str | None = None,
        prog: qick_asm.AbsQickProgram | str | None = None,
        voltage_state: Dict[str, float] | None = None,
    ):
        self.raw_data = raw_data
        self.analyzed_data = analyzed_data
        self.analysis_type = ""
        self.analysis_averaged: spinqick_utils.AverageLevel | None = None
        self.prog = prog
        self.cfg = cfg
        self.timestamp = _assign_timestamp() if timestamp is None else timestamp
        self.axes: dict = {}
        self.experiment_name = experiment_name
        self.data_file = (
            _get_filename(self.timestamp, self.experiment_name)
            if filename is None
            else filename
        )
        self.triggers = triggers
        self.reps = reps
        self.cfg_class = cfg.__class__.__name__
        self.fit_param_dict: dict = {}
        self.best_fit: np.ndarray = np.array([])
        self.fit_axis: str = ""
        self.spinqick_version: str = importlib_metadata.version("spinqick")
        self._cfg = self.cfg.model_dump_json()
        self.voltage_state = voltage_state

    def add_full_average(self, avgs: int):
        """data contains an outer loop for averaging"""
        self.axes["full_avgs"] = {"size": avgs, "loop_no": 0}

    def add_point_average(self, avgs: int, loop_no: int):
        """data contains a point average loop. User supplies the loop number, where outermost loop is zero"""
        self.axes["point_avgs"] = {"size": avgs, "loop_no": loop_no}

    def add_axis(
        self,
        data: List[np.ndarray],
        axis_name: str,
        sweep_names: Sequence[str],
        dim_size: int,
        loop_no: int = 0,
        units: List[str] | None = None,
    ):
        """Add information describing a swept variable/ set of variables in your dataset"""
        ax_dict = {}
        for i, sweep in enumerate(sweep_names):
            if units is None:
                unit_val = "NA"
            else:
                unit_val = units[i]
            ax_dict[sweep] = {"data": data[i], "units": unit_val}
        self.axes[axis_name] = {"sweeps": ax_dict, "size": dim_size, "loop_no": loop_no}
        # TODO add a check that all sweeps added are the correct size

    def add_fit_params(self, param_dict: dict, best_fit: np.ndarray, fit_axis: str):
        """add fit parameter attributes to the spinqick data object"""
        self.fit_param_dict = param_dict
        self.best_fit = best_fit
        self.fit_axis = fit_axis

    def json_to_qickprog(self, soccfg):
        """load json string program into qick program using a known soccfg"""
        try:
            prog_dict = helpers.json2progs(self.prog)
            qick_prog = asm_v2.QickProgramV2(soccfg)
            qick_prog.load_prog(prog_dict)
            self.prog = qick_prog
        except TypeError:
            print("program attribute is not in json format")
        except Exception as exc:
            print("An unexpected error occurred: %s", exc)

    def save_fit_params(self, nc_file: file_manager.SaveData):
        """save parameters from a fit into a dict"""
        fit_grp = nc_file.createGroup("fits")
        if self.fit_param_dict is not None:
            nc_file.groups["fits"].setncatts(self.fit_param_dict)
        nc_file.add_dataset(
            "best_fit",
            axes=[self.fit_axis],
            data=self.best_fit,
            group_path=fit_grp.path,
        )

    def save_voltage_data(
        self,
        nc_file: file_manager.SaveData,
        nest_in_group: None | str = None,
    ):
        """saves the all_voltages output from the hardware manager as json string"""
        vstate_json = json.dumps(self.voltage_state)
        if nest_in_group is not None:
            nc_file[nest_in_group].voltage_state = vstate_json
        else:
            nc_file.voltage_state = vstate_json
        return nc_file

    def save_data(self):
        """save all information to a netcdf file"""
        nc_file = file_manager.SaveData(self.data_file, "a", format="NETCDF4")
        ncdf = self.basic_save(nc_file)
        logger.info("data saved at %s", self.data_file)
        return ncdf

    def basic_save(
        self, nc_file: file_manager.SaveData, nest_in_group: None | str = None
    ):
        """Save data from an instantiated SpinqickData object"""

        if nest_in_group is None:
            nc_group: netCDF4.Group | file_manager.SaveData = nc_file
        else:
            nc_group = nc_file.createGroup(nest_in_group)
        nc_group.timestamp = self.timestamp
        nc_group.cfg_type = self.cfg_class
        nc_group.experiment_name = self.experiment_name
        self.spinqick_version = importlib_metadata.version("spinqick")
        nc_group.spinqick_version = self.spinqick_version
        ### start by saving raw data
        if self.axes:
            axis_group = "swept_variables"
            sweep_grp = nc_group.createGroup(axis_group)
            for axis, axis_dict in self.axes.items():
                dim_name = axis + "_dim"
                if dim_name in nc_file.dimensions:
                    continue
                else:
                    ax_grp = sweep_grp.createGroup(axis)
                    ax_grp.ax_dim = axis_dict["size"]
                    ax_grp.loop_no = axis_dict["loop_no"]
                    if axis == "full_avgs":
                        nc_group.createDimension("full_avgs_dim", axis_dict["size"])
                    elif axis == "point_avgs":
                        nc_group.createDimension("point_avgs_dim", axis_dict["size"])
                    else:
                        nc_file.add_multivariable_axis(
                            axis, axis_dict, group_path=ax_grp.path
                        )
        nc_group.createDimension("reps_dim", self.reps)
        nc_group.createDimension("triggers_dim", self.triggers)
        nc_group.createDimension("IQ_dim", 2)
        adc_ind = 0
        for array in self.raw_data:
            axes_names = ["reps", "triggers"]
            sweep_ax_order = []
            sweep_ax_label = []
            # get the axes in the correct order before assigning
            for axis, axis_dict in self.axes.items():
                sweep_ax_order.append(axis_dict["loop_no"])
                sweep_ax_label.append(axis)
            ordered_ax_list = sweep_ax_label.copy()
            for k, i in enumerate(sweep_ax_order):
                ordered_ax_list[i] = sweep_ax_label[k]
            for ax_label in ordered_ax_list:
                axes_names.append(ax_label)
            axes_names.append("IQ")
            if nest_in_group is None:
                gp = None
            else:
                gp = nc_group.path
            nc_file.add_dataset(
                "raw_data_" + str(adc_ind),
                axes_names,
                array,
                units="adc_raw",
                group_path=gp,
            )
            adc_ind += 1
        if self.analyzed_data:
            ana_group = nc_group.createGroup("analyzed_data")
            adc_ind = 0
            if self.analysis_averaged is not None:
                ana_group.analysis_avged = self.analysis_averaged
            for array in self.analyzed_data:
                axes_names_analyzed = []
                for axis in axes_names:
                    # these axes need to be in the correct order
                    if axis == "point_avgs":
                        if self.analysis_averaged not in ["inner", "both"]:
                            axes_names_analyzed.append("point_avgs")
                    elif axis == "full_avgs":
                        if self.analysis_averaged not in ["outer", "both"]:
                            axes_names_analyzed.append("full_avgs")
                    else:
                        if axis != "IQ":
                            axes_names_analyzed.append(axis)
                nc_file.add_dataset(
                    "analyzed_" + str(adc_ind),
                    axes_names_analyzed,
                    array,
                    group_path=ana_group.path,
                    units=self.analysis_type,
                )
                adc_ind += 1
        if self.fit_param_dict:
            self.save_fit_params(nc_file)
        nc_group.cfg = self._cfg

        if isinstance(self.prog, str):
            nc_file.prog = self.prog
        elif isinstance(self.prog, qick_asm.AbsQickProgram):
            prog_dict = self.prog.dump_prog()
            prog_json = helpers.progs2json(prog_dict)
            nc_file.prog = prog_json
        if self.voltage_state is not None:
            nc_file = self.save_voltage_data(nc_file, nest_in_group=nest_in_group)
        return nc_file

    @classmethod
    def load_spinqick_data(cls, nc_file: netCDF4.Dataset, **kwargs):
        """load data from netcdf dataset to spinqickdata"""
        cfg_type = nc_file.cfg_type
        cfg_model = getattr(experiment_models, cfg_type)
        try:
            cfg = cfg_model.model_validate_json(nc_file.cfg)
        except pydantic.ValidationError:
            logger.warning("unable to load config into pydantic model")
        data_keys = list(nc_file.variables.keys())
        raw = []
        analysis_type = ""
        axes = {}
        for d in data_keys:
            if "raw_data" in d:
                data_array = np.asarray(nc_file[d][:])
                raw.append(data_array)
        if "analyzed_data" in nc_file.groups.keys():
            processed, analysis_type, ana_avg = load_analysis_data(
                nc_file, "analyzed", "analysis_avged"
            )
        else:
            processed = None
            ana_avg = None
        if "swept_variables" in nc_file.groups.keys():
            sweep = nc_file["swept_variables"]
            sweep_axis_keys = list(sweep.groups.keys())
            for sw in sweep_axis_keys:
                ax_dict = {}
                ax = sweep[sw]
                sweep_var_keys = list(ax.variables.keys())
                for sv in sweep_var_keys:
                    data_array = np.asarray(ax[sv][:])
                    ax_dict[sv] = {"data": data_array, "units": ax[sv].units}
                ax_dict["size"] = ax.ax_dim
                ax_dict["loop_no"] = ax.loop_no
                axes[sw] = ax_dict

        if hasattr(nc_file, "prog"):
            prog = nc_file.prog
        else:
            prog = None
        data_obj = cls(
            raw_data=raw,
            cfg=cfg,
            experiment_name=nc_file.experiment_name,
            reps=nc_file.dimensions["reps_dim"].size,
            triggers=nc_file.dimensions["triggers_dim"].size,
            analyzed_data=processed,
            filename=nc_file.filepath(),
            timestamp=nc_file.timestamp,
            prog=prog,
            **kwargs,
        )
        data_obj.analysis_type = analysis_type
        if ana_avg is not None:
            data_obj.analysis_averaged = ana_avg
        data_obj.spinqick_version = nc_file.spinqick_version
        data_obj.axes = axes
        if hasattr(nc_file, "voltage_state"):
            v_json = nc_file.voltage_state
            data_obj.voltage_state = json.loads(v_json)
        if "fits" in nc_file.groups.keys():
            fit_grp = nc_file["fits"]
            best_fit = fit_grp["best_fit"]
            best_fit_dim_label = best_fit.dimensions
            best_fit_dim = best_fit_dim_label[0][:-4]
            fitparams = fit_grp.ncattrs()
            fit_dict = {}
            for p in fitparams:
                fit_dict[p] = getattr(fit_grp, p)
            data_obj.add_fit_params(fit_dict, np.asarray(best_fit), best_fit_dim)
        return data_obj


class PsbData(SpinqickData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.difference_data: List[np.ndarray] | None = None
        self.difference_avged: spinqick_utils.AverageLevel | None = None
        self.thresh_avged: spinqick_utils.AverageLevel | None = None
        self.threshed_data: List[np.ndarray] | None = None
        self.threshold: List[float] | None

    def save_difference_data(self, ncdf: file_manager.SaveData):
        """save data from measurements with a reference measurement"""
        adc_ind = 0
        ana_group = ncdf["analyzed_data"]
        if self.difference_avged is not None:
            ana_group.difference_avged = self.difference_avged
        ana_dims = ana_group.variables["analyzed_" + str(adc_ind)].dimensions
        if self.difference_data is not None:
            for array in self.difference_data:
                axes_names_diff = []
                for axis in ana_dims:
                    ax = axis.strip("_dim")
                    if ax == "point_avgs":
                        if self.difference_avged not in ["inner", "both"]:
                            axes_names_diff.append("point_avgs")
                    elif ax == "full_avgs":
                        if self.difference_avged not in ["outer", "both"]:
                            axes_names_diff.append("full_avgs")
                    else:
                        if ax != "triggers":
                            axes_names_diff.append(ax)

                ncdf.add_dataset(
                    "difference_" + str(adc_ind),
                    axes_names_diff,
                    array,
                    group_path=ana_group.path,
                )
                adc_ind += 1

    def save_threshed_data(self, ncdf: file_manager.SaveData):
        """save thresholded data"""
        adc_ind = 0
        ana_group = ncdf["analyzed_data"]
        if self.thresh_avged is not None:
            ana_group.thresh_avged = self.thresh_avged
        if self.difference_data is not None:
            ana_dims = ana_group.variables["difference_" + str(adc_ind)].dimensions
        else:
            ana_dims = ana_group.variables["analyzed_" + str(adc_ind)].dimensions
        if self.threshed_data is not None:
            for array in self.threshed_data:
                axes_names = []
                for axis in ana_dims:
                    ax = axis.strip("_dim")
                    if ax == "point_avgs":
                        if self.thresh_avged not in ["inner", "both"]:
                            axes_names.append("point_avgs")
                    elif ax == "full_avgs":
                        if self.thresh_avged not in ["outer", "both"]:
                            axes_names.append("full_avgs")
                    else:
                        if ax != "triggers":
                            axes_names.append(ax)
                dset_name = "threshed_" + str(adc_ind)
                ncdf.add_dataset(
                    "threshed_" + str(adc_ind),
                    axes_names,
                    array,
                    group_path=ana_group.path,
                )
                assert self.threshold
                ana_group[dset_name].threshold = self.threshold[adc_ind]
                adc_ind += 1

    def save_data(self):
        ncdf = super().save_data()
        if self.difference_data is not None:
            self.save_difference_data(ncdf)
        if self.threshed_data is not None:
            self.save_threshed_data(ncdf)
        return ncdf

    @classmethod
    def load_spinqick_data(cls, nc_file: netCDF4.Dataset, **kwargs):
        sqd = super().load_spinqick_data(nc_file, **kwargs)
        if "analyzed_data" in nc_file.groups.keys():
            processed_diff, _, _ = load_analysis_data(nc_file, "difference")
            if processed_diff:
                sqd.difference_data = processed_diff
                sqd.difference_avged = (
                    nc_file["analyzed_data"].difference_avged
                    if hasattr(nc_file["analyzed_data"], "difference_avged")
                    else None
                )
            threshed, _, thresholds = load_analysis_data(
                nc_file, "threshed", "threshold"
            )
            if threshed:
                sqd.threshed_data = threshed
                sqd.threshold = thresholds
                sqd.thresh_avged = (
                    nc_file["analyzed_data"].thresh_avged
                    if hasattr(nc_file["analyzed_data"], "thresh_avged")
                    else None
                )
        return sqd


class CompositeSpinqickData:
    """Stores a list of SpinqickData objects.  This is designed for datasets which include
    sweeps"""

    def __init__(
        self,
        qdata_array: Sequence[SpinqickData | PsbData],
        dset_labels: List[str],
        experiment_name: str,
        dset_coordinates: np.ndarray | None = None,
        dset_coordinate_units: str | None = None,
        analyzed_data: np.ndarray | None = None,
        timestamp: int | None = None,
        filename: str | None = None,
    ):
        self.qdata_array = qdata_array
        self.experiment_name = experiment_name
        self.spinqick_version: str = importlib_metadata.version("spinqick")
        self.analyzed_data = analyzed_data
        self.dset_coordinates = dset_coordinates
        self.dset_labels = dset_labels
        self.dset_coordinate_units = dset_coordinate_units
        self.timestamp = _assign_timestamp() if timestamp is None else timestamp
        self.data_file = (
            _get_filename(self.timestamp, self.experiment_name)
            if filename is None
            else filename
        )
        self.fit_param_dict: dict = {}
        self.best_fit: np.ndarray = np.array([])
        self.fit_axis: str = ""

    def basic_composite_save(self):
        """Save all SpinqickData objects to a single file."""

        nc_file = file_manager.SaveData(self.data_file, "a", format="NETCDF4")
        for i, d in enumerate(self.qdata_array):
            d.basic_save(nc_file, self.dset_labels[i])
        if self.dset_coordinates is not None:
            nc_file.add_axis(
                "outer_sweep", self.dset_coordinates, units=self.dset_coordinate_units
            )
        if self.analyzed_data is not None:
            nc_file.add_dataset("analyzed_data", ["outer_sweep"], self.analyzed_data)
        nc_file.experiment_name = self.experiment_name
        nc_file.timestamp = self.timestamp
        logger.info("data saved at %s", self.data_file)
        return nc_file

    @classmethod
    def load_composite(
        cls,
        nc_file: netCDF4.Dataset,
        load_psb: bool = False,
        **kwargs,
    ):
        """create a composite data object from a netcdf file"""
        sqd_instances = []
        dset_labels = []
        for grp in nc_file.groups:
            dset = nc_file[grp]
            if load_psb:
                sqd = PsbData.load_spinqick_data(dset, **kwargs)
            else:
                sqd = SpinqickData.load_spinqick_data(dset, **kwargs)
            sqd_instances.append(sqd)
            dset_labels.append(grp)
        if "outer_sweep" in nc_file.variables:
            coordinates = nc_file["outer_sweep"][:]
            units = nc_file["outer_sweep"].units
        else:
            coordinates = None
            units = None
        if "analyzed_data" in nc_file.variables:
            analyzed = nc_file["analyzed_data"]
        else:
            analyzed = None
        data_obj = cls(
            qdata_array=sqd_instances,
            dset_labels=dset_labels,
            experiment_name=nc_file.experiment_name,
            dset_coordinates=coordinates,
            dset_coordinate_units=units,
            analyzed_data=analyzed,
            filename=nc_file.filepath(),
            timestamp=nc_file.timestamp,
        )
        return data_obj
