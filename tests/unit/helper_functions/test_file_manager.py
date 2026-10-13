"""Unit tests for configuration and path helpers."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from spinqick.helper_functions import file_manager


class ExampleConfig(BaseModel):
    amplitude: float
    enabled: bool


def test_get_data_dir_creates_date_directory(tmp_path):
    data_dir = file_manager.get_data_dir(str(tmp_path))

    path = Path(data_dir)
    assert path.is_dir()
    assert path.parent == tmp_path


def test_get_new_timestamp_returns_directory_and_integer_timestamp(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(file_manager.time, "time", lambda: 1234.9)

    data_dir, timestamp = file_manager.get_new_timestamp(str(tmp_path))

    assert Path(data_dir).is_dir()
    assert timestamp == 1234


def test_get_new_filename_joins_path_and_suffix(tmp_path):
    result = file_manager.get_new_filename("_data.json", str(tmp_path), 123)

    assert result == str(tmp_path / "123_data.json")


def test_json_config_round_trip(tmp_path):
    filename = tmp_path / "config.json"
    config = ExampleConfig(amplitude=1.25, enabled=True)

    file_manager.save_config_json(config, str(filename))
    loaded = file_manager.load_config_json(str(filename), ExampleConfig)

    assert loaded == config
    assert json.loads(filename.read_text(encoding="utf-8")) == {
        "amplitude": 1.25,
        "enabled": True,
    }


def test_load_config_json_rejects_invalid_model_data(tmp_path):
    filename = tmp_path / "invalid.json"
    filename.write_text('{"amplitude": "not-a-number"}', encoding="utf-8")

    with pytest.raises(ValueError):
        file_manager.load_config_json(str(filename), ExampleConfig)


def test_load_config_yaml_uses_safe_loading(tmp_path):
    filename = tmp_path / "config.yaml"
    filename.write_text("amplitude: 1.5\nenabled: true\n", encoding="utf-8")

    loaded = file_manager.load_config_yaml(str(filename))

    assert loaded == {"amplitude": 1.5, "enabled": True}


def test_load_config_yaml_returns_empty_dict_for_malformed_yaml(tmp_path):
    filename = tmp_path / "invalid.yaml"
    filename.write_text("enabled: [", encoding="utf-8")

    assert file_manager.load_config_yaml(str(filename)) == {}


def test_load_config_yaml_does_not_construct_python_objects(tmp_path):
    filename = tmp_path / "unsafe.yaml"
    filename.write_text(
        "!!python/object/apply:os.system ['echo unsafe']\n",
        encoding="utf-8",
    )

    assert file_manager.load_config_yaml(str(filename)) == {}


def test_load_config_json_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_manager.load_config_json(str(tmp_path / "missing.json"), ExampleConfig)


def test_check_configs_exist_logs_missing_files(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(
        file_manager.file_settings,
        "dot_experiment_config",
        str(tmp_path / "experiment.json"),
    )
    monkeypatch.setattr(
        file_manager.file_settings,
        "hardware_config",
        str(tmp_path / "hardware.json"),
    )

    with caplog.at_level("ERROR"):
        file_manager.check_configs_exist()

    assert "no experiment config found" in caplog.text
    assert "no hardware config found" in caplog.text
