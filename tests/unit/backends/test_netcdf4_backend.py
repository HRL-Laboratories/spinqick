"""Hardware-free tests for the NetCDF4 persistence wrapper."""

import netCDF4
import numpy as np

from spinqick.backends.netcdf4_backend import SaveData


def test_savedata_round_trip(tmp_path):
    """SaveData preserves a synthetic axis, dataset, and units."""
    filename = tmp_path / "synthetic_measurement.nc"
    axis = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with SaveData(filename, "w", format="NETCDF4") as dataset:
        dataset.add_axis("time", axis, units="us")
        dataset.add_dataset("signal", ["time"], values, units="V")

    with netCDF4.Dataset(filename, "r") as dataset:
        np.testing.assert_allclose(dataset.variables["time"][:], axis)
        np.testing.assert_allclose(dataset.variables["signal"][:], values)
        assert dataset.variables["time"].units == "us"
        assert dataset.variables["signal"].units == "V"
        assert len(dataset.dimensions["time_dim"]) == len(axis)
