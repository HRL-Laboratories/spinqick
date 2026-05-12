"""Pluggable data backend registry for SpinQICK.

The default backend is netCDF4.  To use a non-default backend, call
:func:`set_backend` at the top of a notebook or script::

    from spinqick.backends import set_backend

    set_backend("qcodes", db_path="C:/data", db_name="spinqick.db", sample_name="device_A")

All subsequent :class:`~spinqick.core.dot_experiment.DotExperiment` instances
will use the configured backend automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spinqick.backends.data_protocols import DataHandler

_BACKENDS: dict[str, type[DataHandler]] = {}
_ACTIVE_BACKEND: DataHandler | None = None

_BUILTIN_IMPORTS: dict[str, str] = {
    "netcdf4": "spinqick.backends.netcdf4_backend",
    "qcodes": "spinqick.backends.qcodes_backend",
}


def register_backend(name: str, backend_cls: type[DataHandler]) -> None:
    """Register a backend class by name (case-insensitive).

    Anyone can call this function to add a new backend following the pattern in netcdf4_backend.py.
    Backends must implement the DataHandler interface. Backends do not need to be located in
    spinqick repository.
    """
    _BACKENDS[name.lower()] = backend_cls


def set_backend(name: str, **kwargs: Any) -> DataHandler:
    """Configure and activate a non-default data backend.

    Call this once at the top of a notebook or script.  All subsequent
    :func:`get_backend` calls (including those inside
    :class:`~spinqick.core.dot_experiment.DotExperiment`) will return the
    instance created here.

    :param name: Backend name (case-insensitive), e.g. ``"qcodes"``,
        ``"netcdf4"``, ``"quantify"``.
    :param kwargs: Backend-specific keyword arguments forwarded to the
        backend constructor (e.g. ``db_path``, ``sample_name`` for QCoDeS).
    :returns: The configured :class:`DataHandler` instance.

    Example::

        from spinqick.backends import set_backend

        set_backend("qcodes", sample_name="device_A")
    """
    global _ACTIVE_BACKEND
    key = name.lower()
    _ensure_registered(key)
    _ACTIVE_BACKEND = _BACKENDS[key](**kwargs)
    return _ACTIVE_BACKEND


def get_backend() -> DataHandler:
    """Return the active data handler.

    If :func:`set_backend` has been called, returns that instance.
    Otherwise returns a default ``NetCDF4Handler()``.
    """
    if _ACTIVE_BACKEND is not None:
        return _ACTIVE_BACKEND
    _ensure_registered("netcdf4")
    return _BACKENDS["netcdf4"]()


def _ensure_registered(key: str) -> None:
    """Lazy-import a built-in backend so it self-registers."""
    if key not in _BACKENDS:
        if key in _BUILTIN_IMPORTS:
            import importlib

            importlib.import_module(_BUILTIN_IMPORTS[key])
        else:
            import spinqick.backends.netcdf4_backend  # noqa: F401
    if key not in _BACKENDS:
        raise KeyError(
            f"Unknown data backend {key!r}. Available: {list(_BACKENDS.keys())}"
        )
