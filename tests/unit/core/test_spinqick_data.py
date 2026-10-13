"""Unit tests for SpinQICK data objects using synthetic data only."""

import json

import numpy as np
import pytest
from pydantic import BaseModel

from spinqick.backends.netcdf4_backend import NetCDF4Handler
from spinqick.core.spinqick_data import SpinqickData


class ExampleConfig(BaseModel):
    amplitude: float
    enabled: bool


@pytest.fixture
def sample_data(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "spinqick.core.spinqick_data.importlib_metadata.version",
        lambda _: "test-version",
    )
    config = ExampleConfig(amplitude=1.25, enabled=True)
    return SpinqickData(
        raw_data=[np.arange(12, dtype=np.float32).reshape(2, 3, 2)],
        cfg=config,
        triggers=2,
        reps=1,
        experiment_name="synthetic",
        timestamp=123,
        filename=str(tmp_path / "synthetic.nc"),
        voltage_state={"P1": 0.1},
    )


def test_data_constructor_stores_identity_and_config(sample_data):
    assert sample_data.timestamp == 123
    assert sample_data.experiment_name == "synthetic"
    assert sample_data.triggers == 2
    assert sample_data.reps == 1
    assert sample_data.get_config_dict() == {
        "amplitude": 1.25,
        "enabled": True,
    }
    assert sample_data.data_file.endswith("synthetic.nc")


def test_axis_and_fit_metadata_are_retained(sample_data):
    axis = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    best_fit = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    sample_data.add_axis(
        data=[axis],
        axis_name="time",
        sweep_names=["delay"],
        dim_size=3,
        loop_no=0,
        units=["us"],
    )
    sample_data.add_full_average(4)
    sample_data.add_point_average(2, 1)
    sample_data.add_fit_params({"amplitude": 2.0}, best_fit, "time")

    assert sample_data.axes["time"]["size"] == 3
    assert sample_data.axes["time"]["delay"]["units"] == "us"
    np.testing.assert_array_equal(sample_data.axes["time"]["delay"]["data"], axis)
    assert sample_data.axes["full_avgs"] == {"size": 4, "loop_no": 0}
    assert sample_data.axes["point_avgs"] == {"size": 2, "loop_no": 1}
    assert sample_data.fit_param_dict == {"amplitude": 2.0}
    np.testing.assert_array_equal(sample_data.best_fit, best_fit)
    assert sample_data.fit_axis == "time"


def test_metadata_contains_serializable_measurement_metadata(sample_data):
    metadata = sample_data.metadata

    assert metadata["timestamp"] == 123
    assert metadata["experiment_name"] == "synthetic"
    assert metadata["cfg_type"] == "ExampleConfig"
    assert metadata["voltage_state"] == {"P1": 0.1}
    assert json.loads(metadata["cfg_json"]) == sample_data.get_config_dict()


def test_load_to_fake_config_preserves_json_values(sample_data):
    fake_config = sample_data.load_to_fake_config(sample_data._cfg)

    assert fake_config.cfg == {
        "amplitude": 1.25,
        "enabled": True,
    }


def test_sweep_variable_extraction_excludes_axis_metadata(sample_data):
    sample_data.add_axis(
        data=[np.array([1.0, 2.0])],
        axis_name="gate",
        sweep_names=["P1"],
        dim_size=2,
    )

    sweep_vars = sample_data.get_sweep_vars(sample_data.axes["gate"])

    assert set(sweep_vars) == {"P1"}


def test_saved_data_can_be_loaded_as_spinqick_data(monkeypatch, tmp_path):
    """A saved synthetic dataset loads back with its data and metadata intact."""
    monkeypatch.setattr(
        "spinqick.core.spinqick_data.importlib_metadata.version",
        lambda _: "test-version",
    )
    config = ExampleConfig(amplitude=1.25, enabled=True)
    original = SpinqickData(
        # Shape is (reps, triggers, IQ), matching the metadata below.
        raw_data=[np.arange(4, dtype=np.float32).reshape(1, 2, 2)],
        cfg=config,
        triggers=2,
        reps=1,
        experiment_name="load-test",
        timestamp=456,
        filename=str(tmp_path / "load-test.nc"),
        voltage_state={"P1": 0.1},
    )

    handler = NetCDF4Handler()
    handle = handler.save(original)
    handler.close(handle)

    loaded = handler.load(original.data_file)

    assert isinstance(loaded, SpinqickData)
    np.testing.assert_array_equal(loaded.raw_data[0], original.raw_data[0])
    assert loaded.triggers == original.triggers
    assert loaded.reps == original.reps
    assert loaded.timestamp == original.timestamp
    assert loaded.experiment_name == original.experiment_name
    assert loaded.get_config_dict() == original.get_config_dict()
    assert loaded.voltage_state == original.voltage_state
