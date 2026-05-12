.. _custom-backends:

Writing a Custom Backend
========================

spinQICK's backend system is extensible.  If netCDF4 and QCoDeS do not fit
your workflow, you can write a custom backend by implementing the
:class:`~spinqick.backends.data_protocols.DataHandler` interface and
registering it with the backend registry.

Custom backends **do not need to live inside the spinQICK repository**.  You
can implement a backend in any Python package — your own lab utilities, a
pip-installable library, or even a single-file module.  As long as the module
is importable and calls ``register_backend()`` at import time, spinQICK will
recognise it.


The DataHandler interface
-------------------------

All backends inherit from the abstract base class
:class:`~spinqick.backends.data_protocols.DataHandler`:

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import Any

   class DataHandler(ABC):

       @abstractmethod
       def save(self, data: SpinqickData | CompositeSpinqickData) -> Any:
           """Persist data.  Dispatches internally for composite vs single.

           Returns a backend-specific handle."""
           ...

       @abstractmethod
       def load(self, identifier: str) -> SpinqickData | CompositeSpinqickData:
           """Load data by identifier.  Auto-detects type from stored metadata."""
           ...

       @abstractmethod
       def save_plot(self, handle: Any, fignum: int | str | None = None) -> None:
           """Save a matplotlib figure to the dataset identified by *handle*."""
           ...

       @abstractmethod
       def close(self, handle: Any) -> None:
           """Finalise / close the dataset identified by *handle*."""
           ...

       @staticmethod
       def get_sweep_vars(axis_dict: dict) -> dict:
           """Extract sweep variables from a flat axis dict.

           Returns only entries that are dicts containing a ``"data"`` key,
           excluding metadata keys like ``"size"`` and ``"loop_no"``.
           """
           return {
               k: v for k, v in axis_dict.items()
               if isinstance(v, dict) and "data" in v
           }


The metadata contract
---------------------

Every ``SpinqickData`` object exposes a ``.metadata`` property that returns a
canonical ``dict`` for backends to persist.  Your backend should save all keys
present in this dict and restore them on ``load()``.

**Always-present keys:**

=========== ===================================
Key         Type / Description
=========== ===================================
timestamp   ``int`` — Unix epoch seconds
experiment_name  ``str``
cfg         ``dict`` — config as flat dict
cfg_json    ``str`` — raw Pydantic JSON string
cfg_type    ``str`` — Pydantic model class name
spinqick_version  ``str``
=========== ===================================

**Conditionally present keys** (only when the field is set):

================ ===================================
Key              Type / Description
================ ===================================
prog             ``str`` — QICK program JSON
voltage_state    ``dict`` — DC voltage snapshot
analysis_avged   ``str`` — averaging level applied
fit_params       ``dict`` — contains ``fit_param_dict``, ``best_fit``, ``fit_axis``
================ ===================================

**PsbData adds** (when set):

================ ===================================
Key              Type / Description
================ ===================================
threshold        ``list[float]`` — per-ADC thresholds
difference_avged ``str``
thresh_avged     ``str``
================ ===================================


Registering your backend
------------------------

Backends self-register at import time by calling ``register_backend()`` at
module level.  This works from **any** Python module — it does not need to
be part of the spinQICK package:

.. code-block:: python

   # my_backend.py
   from spinqick.backends import register_backend
   from spinqick.backends.data_protocols import DataHandler

   class MyHandler(DataHandler):
       def __init__(self, connection_string: str = "default"):
           self.connection_string = connection_string

       def save(self, data):
           ...  # your implementation

       def load(self, identifier):
           ...

       def save_plot(self, handle, fignum=None):
           ...

       def close(self, handle):
           ...

   register_backend("mybackend", MyHandler)

Then users can activate it from a notebook or script by importing the module
(which triggers registration) and calling ``set_backend``:

.. code-block:: python

   from spinqick.backends import set_backend
   import my_backend  # import triggers register_backend() at module level

   set_backend("mybackend", connection_string="...")


Example: minimal skeleton
-------------------------

Here is a complete (non-functional) skeleton showing the expected structure:

.. code-block:: python

   """Minimal custom backend example."""

   import json
   from typing import Any

   from spinqick.backends import register_backend
   from spinqick.backends.data_protocols import DataHandler
   from spinqick.core.spinqick_data import (
       CompositeSpinqickData,
       SpinqickData,
   )


   class MinimalHandle:
       """Return value from save(), passed to close() and save_plot()."""

       def __init__(self, record_id: str) -> None:
           self.record_id = record_id


   class MinimalHandler(DataHandler):

       def __init__(self, storage_path: str = "/tmp/spinqick") -> None:
           self.storage_path = storage_path

       def save(self, data: SpinqickData | CompositeSpinqickData) -> MinimalHandle:
           if isinstance(data, CompositeSpinqickData):
               # Iterate data.qdata_array for sub-datasets
               return MinimalHandle(str(data.timestamp))
           meta = data.metadata
           # Persist meta, data.raw_data, data.analyzed_data, data.axes, ...
           record_id = str(meta["timestamp"])
           return MinimalHandle(record_id)

       def load(self, identifier: str) -> SpinqickData | CompositeSpinqickData:
           # Auto-detect type from stored metadata and reconstruct
           raise NotImplementedError

       def save_plot(self, handle: Any, fignum: int | str | None = None) -> None:
           # Save current matplotlib figure alongside the dataset
           pass

       def close(self, handle: Any) -> None:
           # Finalise the record (flush buffers, release locks, etc.)
           pass


   register_backend("minimal", MinimalHandler)


Tips for implementers
---------------------

- **Use** ``data.metadata`` as the canonical source of metadata to persist.
  Do not reach into private attributes.
- **Use** ``DataHandler.get_sweep_vars(axis_dict)`` to separate sweep
  variables from axis metadata (``size``, ``loop_no``).
- **Test round-trips**: ``save()`` → ``load()`` should reproduce all fields.
- **Handle PsbData**: Check ``isinstance(data, PsbData)`` to save the extra
  fields (``difference_data``, ``threshed_data``, ``threshold``).
- **Auto-detect on load**: Store a ``data_type`` marker (e.g. the class name)
  when saving, and use it in ``load()`` to reconstruct the correct type
  (``SpinqickData``, ``PsbData``, or ``CompositeSpinqickData``).
- **Composite data**: Each sub-dataset in ``data.qdata_array`` is a full
  ``SpinqickData`` with its own axes, config, and raw/analyzed arrays.

See :class:`~spinqick.backends.netcdf4_backend.NetCDF4Handler` and
:class:`~spinqick.backends.qcodes_backend.QCoDesHandler` for complete
reference implementations.
