"""Temporary-database tests for the optional QCoDeS backend."""

import numpy as np
import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.qcodes


class ExampleConfig(BaseModel):
    amplitude: float
    enabled: bool


@pytest.fixture
def qcodes_handler(monkeypatch, tmp_path):
    pytest.importorskip("qcodes")

    from spinqick.backends.qcodes_backend import QCoDesHandler

    monkeypatch.setattr(
        "spinqick.core.spinqick_data.importlib_metadata.version",
        lambda _: "test-version",
    )
    return QCoDesHandler(
        db_path=str(tmp_path),
        db_name="synthetic-qcodes.db",
        sample_name="synthetic-sample",
    )


@pytest.fixture
def sample_data(monkeypatch, tmp_path):
    from spinqick.core.spinqick_data import SpinqickData

    monkeypatch.setattr(
        "spinqick.core.spinqick_data.importlib_metadata.version",
        lambda _: "test-version",
    )
    return SpinqickData(
        # Shape is (reps, triggers, IQ).
        raw_data=[np.arange(4, dtype=np.float32).reshape(1, 2, 2)],
        cfg=ExampleConfig(amplitude=1.25, enabled=True),
        triggers=2,
        reps=1,
        experiment_name="synthetic-qcodes-test",
        timestamp=123,
        filename=str(tmp_path / "unused-by-qcodes.nc"),
        voltage_state={"P1": 0.1},
    )


def test_handler_creates_database_in_requested_directory(qcodes_handler, tmp_path):
    assert qcodes_handler.db_path == str(tmp_path / "synthetic-qcodes.db")
    assert (tmp_path / "synthetic-qcodes.db").exists()


def test_save_returns_open_handle_and_creates_run(qcodes_handler, sample_data):
    handle = qcodes_handler.save(sample_data)

    assert handle.run_id > 0
    assert handle.dataset.run_id == handle.run_id
    assert handle._closed is False

    qcodes_handler.close(handle)


def test_save_load_round_trip_preserves_raw_data_and_metadata(
    qcodes_handler, sample_data
):
    handle = qcodes_handler.save(sample_data)
    loaded = qcodes_handler.load(str(handle.run_id))

    np.testing.assert_array_equal(loaded.raw_data[0], sample_data.raw_data[0])
    assert loaded.experiment_name == sample_data.experiment_name
    assert loaded.timestamp == sample_data.timestamp
    assert loaded.triggers == sample_data.triggers
    assert loaded.reps == sample_data.reps
    assert loaded.get_config_dict() == sample_data.get_config_dict()
    assert loaded.voltage_state == sample_data.voltage_state

    qcodes_handler.close(handle)


def test_close_marks_handle_closed(qcodes_handler, sample_data):
    handle = qcodes_handler.save(sample_data)

    qcodes_handler.close(handle)

    assert handle._closed is True


def test_save_plot_is_a_safe_no_op(qcodes_handler, sample_data):
    handle = qcodes_handler.save(sample_data)

    qcodes_handler.save_plot(handle, fignum=1)

    assert handle._closed is False
    qcodes_handler.close(handle)


def test_load_rejects_non_numeric_run_id(qcodes_handler):
    with pytest.raises(ValueError):
        qcodes_handler.load("not-a-run-id")
