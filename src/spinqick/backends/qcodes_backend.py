"""QCoDeS backend for SpinQICK data persistence.

This module contains the ``QCoDesHandler`` which stores SpinQICK experiment
data in a QCoDeS SQLite database.  Each :meth:`save` call creates one QCoDeS
*run* whose ``run_id`` is returned inside a :class:`QCoDesHandle`.

Data is stored in a **plottr-friendly** two-tier format:

* **Plottr-visible parameters** (dependencies) — rep/trigger-averaged
  analyzed data stored as rows that ``plottr-inspectr`` can grid and plot.
  For 2D sweeps: *N* outer rows with numeric outer setpoints, array inner
  setpoints, and array data slices.  For 1D sweeps: one row with array
  setpoints and array data.

* **Standalone blob parameters** — full raw arrays (with reps, triggers,
  IQ) and full analyzed arrays stored as single-row blobs for lossless
  round-trip loading via :meth:`load`.

Data is written using the lower-level ``new_data_set`` /
``InterDependencies_`` / ``add_results`` API rather than the ``Measurement``
context manager, because the measurement has already occurred — we are only
persisting pre-collected data.

Requirements:
    ``qcodes`` must be installed (``pip install qcodes``).

Usage::

    from spinqick.backends import set_backend

    set_backend("qcodes", db_path="my_data.db", sample_name="device_A")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from spinqick.backends import register_backend
from spinqick.backends.data_protocols import DataHandler

if TYPE_CHECKING:
    from qcodes.dataset.data_set_protocol import DataSetProtocol
    from qcodes.dataset.experiment_container import Experiment

    from spinqick.core.spinqick_data import CompositeSpinqickData, SpinqickData

logger = logging.getLogger(__name__)


@dataclass
class QCoDesHandle:
    """Lightweight handle returned by :meth:`QCoDesHandler.save`.

    Carries the QCoDeS dataset reference and run identifier.
    """

    dataset: DataSetProtocol
    run_id: int
    _closed: bool = field(default=False, repr=False)


def _extract_blob(
    param_data: dict[str, dict[str, Any]], name: str
) -> np.ndarray | None:
    """Extract a standalone blob array from ``get_parameter_data()`` output.

    Standalone parameters appear as top-level keys.  The leading "rows"
    dimension (always length 1 for blobs) is stripped via ``[0]``.
    Returns ``None`` if the parameter is not present.
    """
    if name in param_data:
        return np.asarray(param_data[name][name])[0]
    return None


class QCoDesHandler(DataHandler):
    """DataHandler implementation that saves data to a QCoDeS SQLite database.

    Each :meth:`save` call creates a QCoDeS *run*.
    Raw data arrays and sweep coordinates are stored as ``paramtype="array"``
    parameters.  All SpinQICK metadata is attached via ``dataset.add_metadata``.

    :param db_path: Path to the SQLite database file.  Created if it does not
        exist.  Defaults to ``"<data_directory>/spinqick.db"``.
    :param sample_name: Sample identifier written into the QCoDeS experiment
        container.  Defaults to ``"default_sample"``.
    """

    def __init__(
        self,
        db_path: str | None = None,
        sample_name: str = "default_sample",
    ) -> None:
        from qcodes.dataset import initialise_or_create_database_at

        if db_path is None:
            from spinqick.settings import file_settings

            db_path = os.path.join(file_settings.data_directory, "spinqick.db")
        self.db_path = os.path.realpath(db_path)
        self.sample_name = sample_name
        initialise_or_create_database_at(self.db_path)

    # -- helpers --------------------------------------------------------------

    def _get_experiment(self, experiment_name: str) -> Experiment:
        """Return (or create) a QCoDeS ``Experiment`` for *experiment_name*."""
        from qcodes.dataset import load_or_create_experiment

        return load_or_create_experiment(
            experiment_name=experiment_name,
            sample_name=self.sample_name,
        )

    # -- public save methods --------------------------------------------------

    def save(self, data: SpinqickData | CompositeSpinqickData) -> QCoDesHandle:
        """Save data to the QCoDeS database.

        Dispatches to :meth:`_save_single` for :class:`SpinqickData` /
        :class:`PsbData` or :meth:`_save_composite` for
        :class:`CompositeSpinqickData`.

        Returns a :class:`QCoDesHandle` for subsequent :meth:`close` calls.
        """
        from spinqick.core.spinqick_data import CompositeSpinqickData

        if isinstance(data, CompositeSpinqickData):
            return self._save_composite(data)
        return self._save_single(data)

    def _save_single(self, data: SpinqickData) -> QCoDesHandle:
        """Save :class:`SpinqickData` (or :class:`PsbData`) to the QCoDeS database.

        Analyzed data is stored in a **plottr-friendly** row format:

        * **1D sweep** — one row with array setpoints and array data.
        * **2D sweep** — *N* outer rows with numeric outer setpoints, array
          inner setpoints, and array data slices.

        The rep/trigger-averaged analyzed data populates the plottr-visible
        dependency parameters.  Full raw arrays (with reps, triggers, IQ) and
        full analyzed arrays are stored as standalone blob parameters for
        lossless round-trip loading.

        Returns a :class:`QCoDesHandle` for subsequent :meth:`close` calls.
        """
        from qcodes.dataset import new_data_set
        from qcodes.dataset.descriptions.dependencies import InterDependencies_
        from qcodes.parameters import ParamSpecBase

        from spinqick.core.spinqick_data import PsbData

        is_psb = isinstance(data, PsbData)
        psb = data if isinstance(data, PsbData) else None
        meta = data.metadata
        exp = self._get_experiment(data.experiment_name)

        # --- Identify sweep layout -------------------------------------------
        # Only axes with actual sweep variables (exclude averaging-only axes).
        sweep_axes = sorted(
            [(name, d) for name, d in data.axes.items() if self.get_sweep_vars(d)],
            key=lambda x: x[1]["loop_no"],
        )
        outer_ax = sweep_axes[0] if len(sweep_axes) >= 2 else None
        inner_ax = sweep_axes[-1] if sweep_axes else None
        if len(sweep_axes) == 1:
            outer_ax = None  # single sweep axis → 1D (inner only)

        n_outer = outer_ax[1]["size"] if outer_ax else None
        n_sweep = 2 if outer_ax else 1 if inner_ax else 0

        # --- Build ParamSpecBase objects -------------------------------------
        dependencies: dict[ParamSpecBase, tuple[ParamSpecBase, ...]] = {}
        standalone_specs: list[ParamSpecBase] = []

        # Coordinate specs: outer = numeric per row, inner = array per row
        outer_coord_specs: list[ParamSpecBase] = []
        inner_coord_specs: list[ParamSpecBase] = []

        if outer_ax:
            for var_name, var_info in self.get_sweep_vars(outer_ax[1]).items():
                outer_coord_specs.append(
                    ParamSpecBase(
                        _safe_param_name(var_name),
                        "numeric",
                        label=var_name,
                        unit=var_info.get("units", ""),
                    )
                )

        if inner_ax:
            for var_name, var_info in self.get_sweep_vars(inner_ax[1]).items():
                inner_coord_specs.append(
                    ParamSpecBase(
                        _safe_param_name(var_name),
                        "array",
                        label=var_name,
                        unit=var_info.get("units", ""),
                    )
                )

        setpoint_tuple = tuple(outer_coord_specs + inner_coord_specs)

        # Plottr-visible: analyzed data (rep/trigger-averaged)
        if data.analyzed_data and setpoint_tuple:
            for i in range(len(data.analyzed_data)):
                spec = ParamSpecBase(
                    f"analyzed_{i}",
                    "array",
                    label=f"analyzed_{i}",
                    unit=data.analysis_type or "",
                )
                dependencies[spec] = setpoint_tuple

        # Plottr-visible: PSB arrays
        if psb is not None and setpoint_tuple:
            if psb.difference_data is not None:
                for i in range(len(psb.difference_data)):
                    spec = ParamSpecBase(
                        f"difference_{i}",
                        "array",
                        label=f"difference_{i}",
                    )
                    dependencies[spec] = setpoint_tuple
            if psb.threshed_data is not None:
                for i in range(len(psb.threshed_data)):
                    spec = ParamSpecBase(
                        f"threshed_{i}",
                        "array",
                        label=f"threshed_{i}",
                    )
                    dependencies[spec] = setpoint_tuple

        # Standalone blobs (archival, full fidelity)
        for i in range(len(data.raw_data)):
            standalone_specs.append(
                ParamSpecBase(
                    f"raw_data_{i}", "array", label=f"raw_data_{i}", unit="adc_raw"
                )
            )
        if data.analyzed_data:
            for i in range(len(data.analyzed_data)):
                standalone_specs.append(
                    ParamSpecBase(
                        f"analyzed_full_{i}", "array", label=f"analyzed_full_{i}"
                    )
                )
        if psb is not None:
            if psb.difference_data is not None:
                for i in range(len(psb.difference_data)):
                    standalone_specs.append(
                        ParamSpecBase(
                            f"difference_full_{i}",
                            "array",
                            label=f"difference_full_{i}",
                        )
                    )
            if psb.threshed_data is not None:
                for i in range(len(psb.threshed_data)):
                    standalone_specs.append(
                        ParamSpecBase(
                            f"threshed_full_{i}", "array", label=f"threshed_full_{i}"
                        )
                    )
        if "fit_params" in meta:
            standalone_specs.append(
                ParamSpecBase("best_fit", "array", label="best_fit")
            )

        # --- Create dataset --------------------------------------------------
        dataset = new_data_set(name=data.experiment_name, exp_id=exp.exp_id)
        interdeps = InterDependencies_(
            dependencies=dependencies if dependencies else None,
            standalones=tuple(standalone_specs) if standalone_specs else (),
        )
        dataset.set_interdependencies(interdeps)
        dataset.mark_started()
        self._add_station_snapshot(dataset)

        # --- Helpers: collapse non-sweep dims for plottr ---------------------
        # Identify which loop_no positions are sweep dimensions.
        sweep_loop_nos: set[int] = set()
        if inner_ax:
            sweep_loop_nos.add(inner_ax[1]["loop_no"])
        if outer_ax:
            sweep_loop_nos.add(outer_ax[1]["loop_no"])

        def _avg_over_non_sweep(arr: np.ndarray) -> np.ndarray:
            """Average over all non-sweep axes.

            After the trigger dim has been removed the remaining dims
            correspond to loop_no values from data.axes.  We average
            over every axis whose loop_no is NOT a sweep dimension
            (e.g. full_avgs, point_avgs / shots).
            """
            if arr.ndim <= n_sweep:
                return arr
            axes_to_avg = tuple(i for i in range(arr.ndim) if i not in sweep_loop_nos)
            if axes_to_avg:
                return np.mean(arr, axis=axes_to_avg)
            return arr

        def _plottr_avg_analyzed(arr: np.ndarray) -> np.ndarray:
            """Prepare analyzed_data for plottr.

            analyzed_data shape is (triggers, *loop_dims) when
            average_level=None.  We select the signal trigger and
            then average the remaining non-sweep dims.

            Convention: trigger ordering is [reference, signal] so the
            signal (measurement) window is the last trigger index.
            This matches calculate_difference which does
            conductance_data[1] - conductance_data[0].
            """
            # Select the signal (measurement) trigger.
            # By convention trigger[-1] is the signal, trigger[0] is the
            # reference.  For single-trigger data this is a no-op squeeze.
            arr = arr[data.triggers - 1]
            return _avg_over_non_sweep(arr)

        def _plottr_avg(arr: np.ndarray) -> np.ndarray:
            """Average non-sweep dims for difference/thresholded data.

            These arrays have already had the trigger dim removed
            (via subtraction) and may already be fully averaged, so
            the dim positions map directly to loop_no values.
            """
            return _avg_over_non_sweep(arr)

        # --- Write plottr-visible rows ---------------------------------------
        if setpoint_tuple and data.analyzed_data:
            if outer_ax and inner_ax and n_outer is not None:
                # 2D: one row per outer sweep point
                outer_svs = self.get_sweep_vars(outer_ax[1])
                inner_svs = self.get_sweep_vars(inner_ax[1])

                avg_analyzed = [_plottr_avg_analyzed(a) for a in data.analyzed_data]
                avg_diff = (
                    [_plottr_avg(d) for d in psb.difference_data]
                    if psb is not None and psb.difference_data
                    else []
                )
                avg_thresh = (
                    [_plottr_avg(t) for t in psb.threshed_data]
                    if psb is not None and psb.threshed_data
                    else []
                )

                rows: list[dict[str, Any]] = []
                for j in range(n_outer):
                    row: dict[str, Any] = {}
                    for vn, vi in outer_svs.items():
                        row[_safe_param_name(vn)] = float(vi["data"][j])
                    for vn, vi in inner_svs.items():
                        row[_safe_param_name(vn)] = np.asarray(vi["data"])
                    for i, avg in enumerate(avg_analyzed):
                        row[f"analyzed_{i}"] = avg[j]
                    for i, avg in enumerate(avg_diff):
                        row[f"difference_{i}"] = avg[j]
                    for i, avg in enumerate(avg_thresh):
                        row[f"threshed_{i}"] = avg[j]
                    rows.append(row)
                dataset.add_results(rows)
            elif inner_ax:
                # 1D: single row
                inner_svs = self.get_sweep_vars(inner_ax[1])
                row_1d: dict[str, Any] = {}
                for vn, vi in inner_svs.items():
                    row_1d[_safe_param_name(vn)] = np.asarray(vi["data"])
                for i, ana in enumerate(data.analyzed_data):
                    row_1d[f"analyzed_{i}"] = _plottr_avg_analyzed(ana)
                if psb is not None:
                    if psb.difference_data:
                        for i, d in enumerate(psb.difference_data):
                            row_1d[f"difference_{i}"] = _plottr_avg(d)
                    if psb.threshed_data:
                        for i, t in enumerate(psb.threshed_data):
                            row_1d[f"threshed_{i}"] = _plottr_avg(t)
                dataset.add_results([row_1d])

        # --- Write standalone blobs ------------------------------------------
        blob_row: dict[str, Any] = {}
        for i, raw in enumerate(data.raw_data):
            blob_row[f"raw_data_{i}"] = np.asarray(raw)
        if data.analyzed_data:
            for i, ana in enumerate(data.analyzed_data):
                blob_row[f"analyzed_full_{i}"] = np.asarray(ana)
        if psb is not None:
            if psb.difference_data is not None:
                for i, d in enumerate(psb.difference_data):
                    blob_row[f"difference_full_{i}"] = np.asarray(d)
            if psb.threshed_data is not None:
                for i, t in enumerate(psb.threshed_data):
                    blob_row[f"threshed_full_{i}"] = np.asarray(t)
        if "fit_params" in meta:
            blob_row["best_fit"] = np.asarray(meta["fit_params"]["best_fit"])
        if blob_row:
            dataset.add_results([blob_row])

        dataset.mark_completed()

        # --- Metadata --------------------------------------------------------
        self._write_metadata(dataset, meta)
        self._write_axes_metadata(dataset, data.axes)
        dataset.add_metadata("reps", str(data.reps))
        dataset.add_metadata("triggers", str(data.triggers))
        if data.analysis_type:
            dataset.add_metadata("analysis_type", data.analysis_type)
        if is_psb:
            psb_meta = data.metadata
            if "threshold" in psb_meta:
                dataset.add_metadata("threshold", json.dumps(psb_meta["threshold"]))
            if "difference_avged" in psb_meta:
                dataset.add_metadata(
                    "difference_avged", str(psb_meta["difference_avged"])
                )
            if "thresh_avged" in psb_meta:
                dataset.add_metadata("thresh_avged", str(psb_meta["thresh_avged"]))

        dataset.add_metadata("data_type", type(data).__name__)

        handle = QCoDesHandle(dataset=dataset, run_id=dataset.run_id)
        logger.info(
            "data saved to QCoDeS database %s, run_id=%d",
            self.db_path,
            dataset.run_id,
        )
        return handle

    def _save_composite(self, data: CompositeSpinqickData) -> QCoDesHandle:
        """Save a :class:`CompositeSpinqickData` to the QCoDeS database.

        When sub-datasets share a common inner sweep axis, the analyzed data
        is stored in a plottr-friendly grid: ``outer_sweep`` (numeric, one
        row per composite step) × sub inner sweep (array).  Raw data and
        full analyzed arrays are stored as standalone blob parameters.
        """
        from qcodes.dataset import new_data_set
        from qcodes.dataset.descriptions.dependencies import InterDependencies_
        from qcodes.parameters import ParamSpecBase

        exp = self._get_experiment(data.experiment_name)
        n_outer = len(data.qdata_array)

        # --- Detect common inner sweep across subs ---------------------------
        inner_axis_info: dict[str, Any] | None = None
        if data.dset_coordinates is not None and n_outer > 0:
            first_sub = data.qdata_array[0]
            sub_sweep_axes = sorted(
                [(n, d) for n, d in first_sub.axes.items() if self.get_sweep_vars(d)],
                key=lambda x: x[1]["loop_no"],
            )
            if sub_sweep_axes:
                _inner_name, inner_axis_info = sub_sweep_axes[-1]

        # --- Build ParamSpecBase objects -------------------------------------
        dependencies: dict[ParamSpecBase, tuple[ParamSpecBase, ...]] = {}
        standalone_specs: list[ParamSpecBase] = []

        outer_spec: ParamSpecBase | None = None
        inner_coord_specs: list[ParamSpecBase] = []

        if data.dset_coordinates is not None:
            outer_spec = ParamSpecBase(
                "outer_sweep",
                "numeric",
                label="outer_sweep",
                unit=data.dset_coordinate_units or "",
            )

        if inner_axis_info is not None:
            for var_name, var_info in self.get_sweep_vars(inner_axis_info).items():
                inner_coord_specs.append(
                    ParamSpecBase(
                        _safe_param_name(var_name),
                        "array",
                        label=var_name,
                        unit=var_info.get("units", ""),
                    )
                )

        setpoint_tuple = tuple(
            ([outer_spec] if outer_spec else []) + inner_coord_specs,
        )

        # Plottr-visible: per-sub analyzed data (averaged over reps/triggers)
        has_plottr = bool(
            setpoint_tuple and n_outer > 0 and data.qdata_array[0].analyzed_data
        )
        if has_plottr:
            first_sub = data.qdata_array[0]
            assert first_sub.analyzed_data is not None
            for i in range(len(first_sub.analyzed_data)):
                spec = ParamSpecBase(
                    f"analyzed_{i}",
                    "array",
                    label=f"analyzed_{i}",
                )
                dependencies[spec] = setpoint_tuple

        # Standalone: composite_analyzed
        if data.analyzed_data is not None:
            standalone_specs.append(
                ParamSpecBase("composite_analyzed", "array", label="composite_analyzed")
            )

        # Standalone: per-sub raw + full analyzed
        for label, sub in zip(data.dset_labels, data.qdata_array):
            safe_label = _safe_param_name(label)
            for i in range(len(sub.raw_data)):
                standalone_specs.append(
                    ParamSpecBase(
                        f"{safe_label}__raw_data_{i}",
                        "array",
                        label=f"{label}/raw_data_{i}",
                        unit="adc_raw",
                    )
                )
            if sub.analyzed_data:
                for i in range(len(sub.analyzed_data)):
                    standalone_specs.append(
                        ParamSpecBase(
                            f"{safe_label}__analyzed_full_{i}",
                            "array",
                            label=f"{label}/analyzed_full_{i}",
                        )
                    )

        # --- Create dataset --------------------------------------------------
        dataset = new_data_set(name=data.experiment_name, exp_id=exp.exp_id)
        interdeps = InterDependencies_(
            dependencies=dependencies if dependencies else None,
            standalones=tuple(standalone_specs) if standalone_specs else (),
        )
        dataset.set_interdependencies(interdeps)
        dataset.mark_started()
        self._add_station_snapshot(dataset)

        # --- Write plottr-visible rows ---------------------------------------
        if has_plottr:
            n_inner_sweep = 1 if inner_axis_info else 0

            rows: list[dict[str, Any]] = []
            for j in range(n_outer):
                sub = data.qdata_array[j]
                row: dict[str, Any] = {}
                if outer_spec is not None and data.dset_coordinates is not None:
                    row["outer_sweep"] = float(data.dset_coordinates[j])
                if inner_axis_info is not None:
                    for vn, vi in self.get_sweep_vars(inner_axis_info).items():
                        row[_safe_param_name(vn)] = np.asarray(vi["data"])
                if sub.analyzed_data:
                    for i, ana in enumerate(sub.analyzed_data):
                        n_leading = ana.ndim - n_inner_sweep
                        avg = (
                            np.mean(ana, axis=tuple(range(n_leading)))
                            if n_leading > 0
                            else ana
                        )
                        row[f"analyzed_{i}"] = avg
                rows.append(row)
            dataset.add_results(rows)

        # --- Write standalone blobs ------------------------------------------
        blob_row: dict[str, Any] = {}
        if data.analyzed_data is not None:
            blob_row["composite_analyzed"] = np.asarray(data.analyzed_data)
        for label, sub in zip(data.dset_labels, data.qdata_array):
            safe_label = _safe_param_name(label)
            for i, raw in enumerate(sub.raw_data):
                blob_row[f"{safe_label}__raw_data_{i}"] = np.asarray(raw)
            if sub.analyzed_data:
                for i, ana in enumerate(sub.analyzed_data):
                    blob_row[f"{safe_label}__analyzed_full_{i}"] = np.asarray(ana)
        if blob_row:
            dataset.add_results([blob_row])

        dataset.mark_completed()

        # --- Metadata --------------------------------------------------------
        dataset.add_metadata("experiment_name", data.experiment_name)
        dataset.add_metadata("timestamp", str(data.timestamp))
        dataset.add_metadata("spinqick_version", data.spinqick_version)
        dataset.add_metadata("dset_labels", json.dumps(data.dset_labels))
        if data.dset_coordinates is not None:
            dataset.add_metadata(
                "dset_coordinates",
                json.dumps(data.dset_coordinates.tolist()),
            )
        if data.dset_coordinate_units is not None:
            dataset.add_metadata("dset_coordinate_units", data.dset_coordinate_units)

        for i, sub in enumerate(data.qdata_array):
            sub_meta = sub.metadata
            dataset.add_metadata(f"sub_{i}_cfg_json", sub_meta["cfg_json"])
            dataset.add_metadata(f"sub_{i}_cfg_type", sub_meta["cfg_type"])
            dataset.add_metadata(
                f"sub_{i}_experiment_name", sub_meta["experiment_name"]
            )
            dataset.add_metadata(f"sub_{i}_reps", str(sub.reps))
            dataset.add_metadata(f"sub_{i}_triggers", str(sub.triggers))
            self._write_axes_metadata(
                dataset,
                sub.axes,
                prefix=f"sub_{i}_",
            )

        dataset.add_metadata("data_type", "CompositeSpinqickData")

        handle = QCoDesHandle(dataset=dataset, run_id=dataset.run_id)
        logger.info(
            "composite data saved to QCoDeS database %s, run_id=%d",
            self.db_path,
            dataset.run_id,
        )
        return handle

    def save_plot(self, handle: QCoDesHandle, fignum: int | str | None = None) -> None:
        """No-op for QCoDeS backend.

        QCoDeS users visualise data on-demand from the database using
        ``plottr-inspectr`` or :func:`qcodes.dataset.plotting.plot_dataset`.
        Static PNG export is not part of the typical QCoDeS workflow.
        """

    def close(self, handle: QCoDesHandle) -> None:
        """Mark the handle as closed.

        The QCoDeS database connection is managed globally, so there is
        nothing to explicitly close per-run.  This satisfies the
        :class:`DataHandler` interface contract.
        """
        handle._closed = True

    # -- public load methods --------------------------------------------------

    def load(
        self, identifier: str, load_psb: bool | None = None
    ) -> SpinqickData | CompositeSpinqickData:
        """Load data by QCoDeS ``run_id``, auto-detecting the type.

        Auto-detection logic:

        1. If ``dset_labels`` is in metadata → :class:`CompositeSpinqickData`.
        2. If ``threshed_full_0`` is in param data → :class:`PsbData`.
        3. Otherwise → :class:`SpinqickData`.

        The ``data_type`` metadata key (written by :meth:`save`) is checked
        first as a fast path; probing for markers is the fallback for data
        saved before this change.

        :param identifier: The ``run_id`` as a string (e.g. ``"42"``).
        :param load_psb: Override auto-detection.  ``None`` (default) =
            auto-detect.  ``True`` = force PsbData.  ``False`` = force
            SpinqickData even when PSB markers are present.
        """
        from qcodes.dataset import load_by_id

        run_id = int(identifier)
        dataset = load_by_id(run_id)
        ds_metadata = dataset.metadata

        # Auto-detect composite
        data_type = ds_metadata.get("data_type", "")
        if data_type == "CompositeSpinqickData" or "dset_labels" in ds_metadata:
            return self._load_composite(dataset)

        # Auto-detect PSB
        param_data = dataset.get_parameter_data()
        if load_psb is None:
            is_psb = data_type == "PsbData" or "threshed_full_0" in param_data
        else:
            is_psb = load_psb

        return self._load_single(dataset, param_data, ds_metadata, load_psb=is_psb)

    def _load_single(
        self,
        dataset: DataSetProtocol,
        param_data: dict | None = None,
        ds_metadata: dict | None = None,
        load_psb: bool = False,
    ) -> SpinqickData:
        """Load a :class:`SpinqickData` from a pre-loaded QCoDeS dataset.

        Full-fidelity data is reconstructed from the standalone blob
        parameters.  Axes are restored from the JSON metadata (which
        embeds the coordinate data arrays).

        :param dataset: Already-loaded QCoDeS dataset.
        :param param_data: Pre-fetched parameter data (or ``None`` to fetch).
        :param ds_metadata: Pre-fetched metadata dict (or ``None`` to fetch).
        :param load_psb: If ``True``, return a :class:`PsbData` and load
            difference/thresholded arrays.
        """
        import pydantic

        from spinqick.core.spinqick_data import PsbData, SpinqickData
        from spinqick.models import experiment_models

        if param_data is None:
            param_data = dataset.get_parameter_data()
        if ds_metadata is None:
            ds_metadata = dataset.metadata
        run_id = dataset.run_id

        # --- reconstruct config ---
        cfg_type = ds_metadata.get("cfg_type", "")
        cfg_json = ds_metadata.get("cfg_json", "{}")
        try:
            cfg_model = getattr(experiment_models, cfg_type)
            cfg = cfg_model.model_validate_json(cfg_json)
        except (AttributeError, pydantic.ValidationError):
            logger.warning("unable to load config model %s, using fake model", cfg_type)
            DynamicFakeConfig = pydantic.create_model("DynamicFakeConfig", cfg=dict)
            cfg = DynamicFakeConfig(cfg=json.loads(cfg_json))

        # --- extract raw data (standalone blobs) ---
        raw: list[np.ndarray] = []
        i = 0
        while (arr := _extract_blob(param_data, f"raw_data_{i}")) is not None:
            raw.append(arr)
            i += 1

        # --- extract analyzed data (full blobs) ---
        analyzed: list[np.ndarray] | None = None
        i = 0
        ana_list: list[np.ndarray] = []
        while (arr := _extract_blob(param_data, f"analyzed_full_{i}")) is not None:
            ana_list.append(arr)
            i += 1
        if ana_list:
            analyzed = ana_list

        # --- reconstruct axes from JSON metadata ---
        axes: dict[str, Any] = {}
        axes_meta_raw = ds_metadata.get("axes_meta", "")
        if axes_meta_raw:
            axes_meta: dict[str, dict] = json.loads(axes_meta_raw)
            for axis_name, axis_info in axes_meta.items():
                ax_dict: dict[str, Any] = {
                    "size": axis_info["size"],
                    "loop_no": axis_info["loop_no"],
                }
                for var_name, var_info in axis_info.get("sweep_vars", {}).items():
                    ax_dict[var_name] = {
                        "data": np.asarray(var_info["data"]),
                        "units": var_info["units"],
                    }
                axes[axis_name] = ax_dict

        # --- determine reps / triggers from raw shape or metadata ---
        reps = int(ds_metadata.get("reps") or (raw[0].shape[0] if raw else 1))
        triggers = int(
            ds_metadata.get("triggers")
            or (raw[0].shape[1] if raw and raw[0].ndim > 1 else 1)
        )

        cls = PsbData if load_psb else SpinqickData
        data_obj = cls(
            raw_data=raw,
            cfg=cfg,
            experiment_name=ds_metadata.get("experiment_name", ""),
            reps=reps,
            triggers=triggers,
            analyzed_data=analyzed,
            filename=f"qcodes://run_id/{run_id}",
            timestamp=int(ds_metadata.get("timestamp", 0)),
            prog=ds_metadata.get("prog"),
        )
        data_obj.axes = axes
        analysis_type = ds_metadata.get("analysis_type", "")
        if analysis_type:
            data_obj.analysis_type = analysis_type
        analysis_avged = ds_metadata.get("analysis_avged")
        if analysis_avged is not None:
            data_obj.analysis_averaged = analysis_avged
        spinqick_version = ds_metadata.get("spinqick_version", "")
        if spinqick_version:
            data_obj.spinqick_version = spinqick_version
        voltage_state_raw = ds_metadata.get("voltage_state")
        if voltage_state_raw:
            data_obj.voltage_state = json.loads(voltage_state_raw)

        # fit params (standalone blob)
        fit_param_dict_raw = ds_metadata.get("fit_param_dict")
        if fit_param_dict_raw:
            fit_dict = json.loads(fit_param_dict_raw)
            best_fit_arr = _extract_blob(param_data, "best_fit")
            fit_axis = ds_metadata.get("fit_axis", "")
            if best_fit_arr is not None:
                data_obj.add_fit_params(fit_dict, best_fit_arr, fit_axis)

        # PSB-specific loading (standalone blobs)
        if load_psb and isinstance(data_obj, PsbData):
            diff_list: list[np.ndarray] = []
            j = 0
            while (
                arr := _extract_blob(param_data, f"difference_full_{j}")
            ) is not None:
                diff_list.append(arr)
                j += 1
            if diff_list:
                data_obj.difference_data = diff_list
            diff_avged = ds_metadata.get("difference_avged")
            if diff_avged is not None:
                data_obj.difference_avged = diff_avged

            thresh_list: list[np.ndarray] = []
            j = 0
            while (arr := _extract_blob(param_data, f"threshed_full_{j}")) is not None:
                thresh_list.append(arr)
                j += 1
            if thresh_list:
                data_obj.threshed_data = thresh_list
            thresh_avged = ds_metadata.get("thresh_avged")
            if thresh_avged is not None:
                data_obj.thresh_avged = thresh_avged
            threshold_raw = ds_metadata.get("threshold")
            if threshold_raw:
                data_obj.threshold = json.loads(threshold_raw)

        return data_obj

    def _load_composite(
        self,
        dataset_or_id: DataSetProtocol | str,
    ) -> CompositeSpinqickData:
        """Load a :class:`CompositeSpinqickData` from a QCoDeS dataset.

        Sub-dataset raw and analyzed data are reconstructed from standalone
        blob parameters.  Axes are restored from per-sub JSON metadata.

        :param dataset_or_id: An already-loaded QCoDeS dataset, or a
            ``run_id`` string for backward compatibility.
        """
        from qcodes.dataset import load_by_id

        from spinqick.core.spinqick_data import CompositeSpinqickData, SpinqickData

        if isinstance(dataset_or_id, str):
            dataset = load_by_id(int(dataset_or_id))
        else:
            dataset = dataset_or_id
        param_data = dataset.get_parameter_data()
        ds_metadata = dataset.metadata
        run_id = dataset.run_id

        dset_labels: list[str] = json.loads(ds_metadata.get("dset_labels", "[]"))

        # outer sweep coordinates (from JSON metadata)
        coordinates: np.ndarray | None = None
        coords_raw = ds_metadata.get("dset_coordinates")
        if coords_raw:
            coordinates = np.asarray(json.loads(coords_raw))
        coord_units = ds_metadata.get("dset_coordinate_units")

        # composite analyzed (standalone blob)
        analyzed = _extract_blob(param_data, "composite_analyzed")

        # reconstruct sub-datasets
        sqd_list: list[SpinqickData] = []
        for i, label in enumerate(dset_labels):
            safe_label = _safe_param_name(label)

            # raw data (standalone blobs)
            sub_raw: list[np.ndarray] = []
            j = 0
            while (
                arr := _extract_blob(
                    param_data,
                    f"{safe_label}__raw_data_{j}",
                )
            ) is not None:
                sub_raw.append(arr)
                j += 1

            # analyzed data (standalone blobs)
            sub_ana: list[np.ndarray] = []
            j = 0
            while (
                arr := _extract_blob(
                    param_data,
                    f"{safe_label}__analyzed_full_{j}",
                )
            ) is not None:
                sub_ana.append(arr)
                j += 1

            # config
            import pydantic

            from spinqick.models import experiment_models

            cfg_type = ds_metadata.get(f"sub_{i}_cfg_type", "")
            cfg_json = ds_metadata.get(f"sub_{i}_cfg_json", "{}")
            try:
                cfg_model = getattr(experiment_models, cfg_type)
                cfg = cfg_model.model_validate_json(cfg_json)
            except (AttributeError, pydantic.ValidationError):
                DynamicFakeConfig = pydantic.create_model("DynamicFakeConfig", cfg=dict)
                cfg = DynamicFakeConfig(cfg=json.loads(cfg_json))

            sub_experiment_name = ds_metadata.get(
                f"sub_{i}_experiment_name",
                ds_metadata.get("experiment_name", ""),
            )

            reps = int(
                ds_metadata.get(f"sub_{i}_reps")
                or (sub_raw[0].shape[0] if sub_raw and sub_raw[0].ndim > 0 else 1)
            )
            triggers = int(
                ds_metadata.get(f"sub_{i}_triggers")
                or (sub_raw[0].shape[1] if sub_raw and sub_raw[0].ndim > 1 else 1)
            )

            sub_data = SpinqickData(
                raw_data=sub_raw,
                cfg=cfg,
                experiment_name=sub_experiment_name,
                reps=reps,
                triggers=triggers,
                analyzed_data=sub_ana or None,
                timestamp=int(ds_metadata.get("timestamp", 0)),
            )

            # Reconstruct sub axes from JSON metadata
            sub_axes_raw = ds_metadata.get(f"sub_{i}_axes_meta", "")
            if sub_axes_raw:
                sub_axes_meta = json.loads(sub_axes_raw)
                for axis_name, axis_info in sub_axes_meta.items():
                    ax_dict: dict[str, Any] = {
                        "size": axis_info["size"],
                        "loop_no": axis_info["loop_no"],
                    }
                    for var_name, var_info in axis_info.get(
                        "sweep_vars",
                        {},
                    ).items():
                        ax_dict[var_name] = {
                            "data": np.asarray(var_info["data"]),
                            "units": var_info["units"],
                        }
                    sub_data.axes[axis_name] = ax_dict

            sqd_list.append(sub_data)

        return CompositeSpinqickData(
            qdata_array=sqd_list,
            dset_labels=dset_labels,
            experiment_name=ds_metadata.get("experiment_name", ""),
            dset_coordinates=coordinates,
            dset_coordinate_units=coord_units,
            analyzed_data=analyzed,
            filename=f"qcodes://run_id/{run_id}",
            timestamp=int(ds_metadata.get("timestamp", 0)),
        )

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _add_station_snapshot(dataset: DataSetProtocol) -> None:
        """Attach the QCoDeS Station snapshot to the dataset, if available."""
        from qcodes.station import Station

        station = Station.default
        if station is not None:
            dataset.add_snapshot(json.dumps(station.snapshot()))

    @staticmethod
    def _write_metadata(dataset: DataSetProtocol, meta: dict[str, Any]) -> None:
        """Persist SpinqickData.metadata fields as QCoDeS dataset metadata."""
        dataset.add_metadata("timestamp", str(meta["timestamp"]))
        dataset.add_metadata("experiment_name", meta["experiment_name"])
        dataset.add_metadata("cfg_json", meta["cfg_json"])
        dataset.add_metadata("cfg_type", meta["cfg_type"])
        dataset.add_metadata("spinqick_version", meta["spinqick_version"])

        if "prog" in meta:
            dataset.add_metadata("prog", meta["prog"])
        if "voltage_state" in meta:
            dataset.add_metadata("voltage_state", json.dumps(meta["voltage_state"]))
        if "analysis_avged" in meta:
            dataset.add_metadata("analysis_avged", str(meta["analysis_avged"]))

        if "fit_params" in meta:
            dataset.add_metadata(
                "fit_param_dict",
                json.dumps(meta["fit_params"]["fit_param_dict"]),
            )
            dataset.add_metadata("fit_axis", meta["fit_params"]["fit_axis"])

    def _write_axes_metadata(
        self,
        dataset: DataSetProtocol,
        axes: dict[str, Any],
        prefix: str = "",
    ) -> None:
        """Serialize the axes structure as JSON metadata for round-trip loading.

        Coordinate data arrays are embedded as JSON lists so that ``load()``
        can reconstruct axes entirely from metadata, without needing to
        extract setpoint arrays from the dependency chain.

        :param prefix: Optional key prefix (e.g. ``"sub_0_"``) for
            composite sub-dataset axes.
        """
        axes_meta: dict[str, Any] = {}
        for axis_name, axis_dict in axes.items():
            entry: dict[str, Any] = {
                "size": axis_dict["size"],
                "loop_no": axis_dict["loop_no"],
                "sweep_vars": {},
            }
            sweep_vars = self.get_sweep_vars(axis_dict)
            for var_name, var_info in sweep_vars.items():
                entry["sweep_vars"][var_name] = {
                    "units": var_info.get("units", ""),
                    "data": np.asarray(var_info["data"]).tolist(),
                }
            axes_meta[axis_name] = entry
        dataset.add_metadata(f"{prefix}axes_meta", json.dumps(axes_meta))


def _safe_param_name(name: str) -> str:
    """Sanitise a sweep variable name for use as a QCoDeS parameter name.

    QCoDeS parameter names must be valid Python identifiers.  This replaces
    common problematic characters with underscores.
    """
    safe = name.replace(" ", "_").replace("-", "_").replace(".", "_")
    if safe and safe[0].isdigit():
        safe = f"p_{safe}"
    return safe


"Register QCoDeS handler as a usable data backend in SpinQICK."
register_backend("qcodes", QCoDesHandler)
