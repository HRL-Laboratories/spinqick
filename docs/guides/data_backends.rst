.. _data-backends:

Data Backends
=============

spinQICK uses a **pluggable backend system** to persist experiment data.
Every experiment that inherits from
:class:`~spinqick.core.dot_experiment.DotExperiment` automatically saves its
results through whichever backend is active.  You can switch backends with a
single call at the top of your notebook — no experiment code changes needed.

Two backends ship with spinQICK:

========  ===============================================  ===============
Backend   Storage format                                   Default?
========  ===============================================  ===============
netcdf4   One ``.nc`` file per run in ``data_directory``   Yes
qcodes    SQLite database (QCoDeS DataSet), viewable in    No
          plottr-inspectr
========  ===============================================  ===============


Quick start
-----------

The default backend (netCDF4) works out of the box once you set
the ``SPINQICK_DATA_DIRECTORY`` environment variable
(see :ref:`environment-variables`).  To use QCoDeS instead:

.. code-block:: python

   from spinqick.backends import set_backend

   set_backend("qcodes", sample_name="device_A")

All subsequent experiments will save to an SQLite database in the
``SPINQICK_DATA_DIRECTORY`` configured during
:ref:`installation <environment-variables>`.  You can customize the database
name and location at runtime with the ``db_name`` and ``db_path`` parameters.
See :ref:`qcodes-backend` for details.


Default backend: netCDF4
------------------------

The :class:`~spinqick.backends.netcdf4_backend.NetCDF4Handler` stores each
experiment run as a self-contained ``.nc`` file.  No configuration is
required beyond setting the ``SPINQICK_DATA_DIRECTORY`` environment variable
(see :ref:`environment-variables`).

**When to use it:**

- You want simple file-per-run storage that is easy to browse and share.
- You do not need the plottr-inspectr GUI.
- You want to open results in standard netCDF tools (``ncdump``, ``xarray``,
  ``h5py``).

**File layout:**

Each ``.nc`` file has this structure:

.. code-block:: none

   my_experiment_20260408_120000.nc
   │
   ├── Dimensions:
   │   ├── reps_dim        = 5
   │   ├── triggers_dim    = 1
   │   ├── IQ_dim          = 2
   │   ├── gate_y_dim      = 7
   │   └── gate_x_dim      = 11
   │
   ├── Variables (raw data):
   │   ├── raw_data_0  [reps, triggers, gate_y, gate_x, IQ]
   │   └── raw_data_1  [reps, triggers, gate_y, gate_x, IQ]
   │
   ├── Group: swept_variables/
   │   ├── Group: gate_y/
   │   │   └── Variable: gate_y  [gate_y_dim]  (units="V")
   │   └── Group: gate_x/
   │       └── Variable: gate_x  [gate_x_dim]  (units="V")
   │
   ├── Group: analyzed_data/
   │   ├── analyzed_0  [gate_y_dim, gate_x_dim]
   │   └── analyzed_1  [gate_y_dim, gate_x_dim]
   │
   ├── Group: fits/              (optional)
   │   └── Variable: best_fit
   │
   └── Root Attributes:
       ├── timestamp, experiment_name, cfg_type, cfg (JSON)
       ├── spinqick_version, voltage_state (JSON)
       └── prog (JSON, optional)

Dimensions are named and shared, so tools like ``xarray.open_dataset``
can reconstruct labelled arrays automatically.  Raw data preserves all
dimensions (reps, triggers, sweep dims, IQ).  Analyzed data only includes
the dimensions that survived averaging.


.. _qcodes-backend:

QCoDeS backend
--------------

The :class:`~spinqick.backends.qcodes_backend.QCoDesHandler` stores data in a
QCoDeS SQLite database.  This enables browsing results in
`plottr-inspectr <https://github.com/plottr/plottr>`__, which provides an
interactive GUI for filtering and plotting past runs.

**When to use it:**

- You want a single database of all runs, searchable by experiment name,
  sample, and run ID.
- You want to browse results in plottr-inspectr without writing any code.
- You are already using QCoDeS for instrument drivers and want station
  snapshots saved with your data.

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   from spinqick.backends import set_backend

   handler = set_backend(
       "qcodes",
       db_path="C:/data",              # directory; created if needed
       db_name="spinqick.db",           # database filename (default)
       sample_name="device_A",           # label for the QCoDeS experiment
   )

=========== ================================================================
Parameter   Description
=========== ================================================================
db_path     Directory for the SQLite file.  Defaults to
            ``<data_directory>``.
db_name     Name of the SQLite database file.  Defaults to
            ``"spinqick.db"``.
sample_name Identifier written to the QCoDeS experiment container.
            Defaults to ``"default_sample"``.
=========== ================================================================

Two-tier storage design
^^^^^^^^^^^^^^^^^^^^^^^

The QCoDeS backend uses a **two-tier** layout inside each run to serve both
the plottr GUI and full-fidelity round-trip loading:

.. code-block:: none

   ┌──────────────────────────────────────────────────────────┐
   │                    QCoDeS Run (SQLite)                   │
   │                                                          │
   │  ┌────────────────────────────────────────────────────┐  │
   │  │  TIER 1: Plottr-Visible (dependency parameters)    │  │
   │  │                                                    │  │
   │  │  Written as N_outer rows so plottr can grid/plot.  │  │
   │  │                                                    │  │
   │  │  Setpoints:                                        │  │
   │  │    gate_y  (numeric) ← one scalar per row          │  │
   │  │    gate_x  (array)   ← full inner sweep per row    │  │
   │  │                                                    │  │
   │  │  Dependents:                                       │  │
   │  │    analyzed_0 (array) ← inner slice per row        │  │
   │  │    analyzed_1 (array) ← inner slice per row        │  │
   │  └────────────────────────────────────────────────────┘  │
   │                                                          │
   │  ┌────────────────────────────────────────────────────┐  │
   │  │  TIER 2: Standalone Blobs (full-fidelity archive)  │  │
   │  │                                                    │  │
   │  │  Written as ONE row of complete arrays.            │  │
   │  │                                                    │  │
   │  │    raw_data_0       (5, 1, 7, 11, 2)               │  │
   │  │    raw_data_1       (5, 1, 7, 11, 2)               │  │
   │  │    analyzed_full_0  (7, 11)                        │  │
   │  │    analyzed_full_1  (7, 11)                        │  │
   │  │    best_fit         (optional)                     │  │
   │  └────────────────────────────────────────────────────┘  │
   │                                                          │
   │  Metadata: timestamp, experiment_name, cfg_json,         │
   │    axes_meta (JSON), voltage_state, fit_params, ...      │
   │                                                          │
   │  Station Snapshot (JSON, from QCoDeS Station)            │
   └──────────────────────────────────────────────────────────┘

**Tier 1** feeds plottr-inspectr.  For a 7×11 gate sweep, it writes 7 rows
(one per outer sweep point) with array inner data, producing a heatmap in
the GUI.

**Tier 2** holds the full raw and analyzed arrays exactly as they were in
memory, enabling lossless ``load()`` round-trips.

**Why two tiers?**  plottr-inspectr can only visualize row-based dependency
parameters.  But full-fidelity round-trip loading needs the unsummarized
arrays.  Tier 2 serves ``load()``, while Tier 1 serves the plottr GUI.


Data structures
---------------

All backends persist the same in-memory data structures.  Here is what each
object contains:

SpinqickData
^^^^^^^^^^^^

.. code-block:: none

   SpinqickData
   ├── raw_data: List[ndarray]          # One array per ADC channel
   │   └── shape: (reps, triggers, *sweep_dims, 2)    ← IQ pair
   │
   ├── analyzed_data: List[ndarray]     # Post-analysis (magnitude, etc.)
   │   └── shape: (*remaining_dims)     ← IQ collapsed, some dims averaged
   │
   ├── axes: dict                       # Swept dimensions
   │   ├── "gate_y": {size, loop_no, "gate_y": {data, units}}
   │   └── "gate_x": {size, loop_no, "gate_x": {data, units}}
   │
   ├── cfg: pydantic.BaseModel          # Experiment configuration
   ├── experiment_name, reps, triggers, timestamp
   ├── analysis_type, voltage_state
   └── fit_param_dict, best_fit, fit_axis

PsbData (extends SpinqickData)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   PsbData
   ├── [all SpinqickData fields]
   ├── difference_data: List[ndarray]   # I − Q difference signal
   ├── threshed_data: List[ndarray]     # Binary classification
   └── threshold: List[float]           # Per-ADC threshold values

CompositeSpinqickData
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   CompositeSpinqickData
   ├── qdata_array: List[SpinqickData]  # Sub-datasets
   ├── dset_labels: List[str]           # ["step_0", "step_1", ...]
   ├── dset_coordinates: ndarray        # Outer sweep values
   ├── dset_coordinate_units: str
   └── analyzed_data: ndarray | None    # Composite-level analysis


Saving and loading
------------------

Regardless of which backend is active, the save/load pattern is the same.
You typically do not call these directly — ``DotExperiment.finalize()``
handles it — but you can use them for manual saves or reloading old data.

**Save:**

.. code-block:: python

   from spinqick.backends import get_backend

   backend = get_backend()
   handle = backend.save(my_data)        # SpinqickData or CompositeSpinqickData
   backend.close(handle)

**Load:**

.. code-block:: python

   # netCDF4: identifier is a file path
   data = backend.load("/path/to/experiment.nc")

   # QCoDeS: identifier is a run_id (as string)
   data = backend.load("42")

``save()`` accepts both ``SpinqickData`` and ``CompositeSpinqickData`` and
dispatches internally.  ``load()`` auto-detects the stored type from metadata
and returns the appropriate object.


Per-experiment backend override
-------------------------------

If you want most experiments on one backend but a specific experiment on
another, pass ``backend=`` to the experiment constructor:

.. code-block:: python

   from spinqick.backends.qcodes_backend import QCoDesHandler

   qcodes = QCoDesHandler(db_path="C:/data/special", db_name="calibration.db", sample_name="calibration")
   cal_exp = SystemCalibrations(soccfg, soc, voltage_source=vs, backend=qcodes)

This overrides the global backend for that experiment only.


Further reading
---------------

- :ref:`custom-backends` — How to write your own backend
- :class:`~spinqick.backends.data_protocols.DataHandler` — The abstract interface
- :class:`~spinqick.backends.netcdf4_backend.NetCDF4Handler` — netCDF4 implementation
- :class:`~spinqick.backends.qcodes_backend.QCoDesHandler` — QCoDeS implementation
