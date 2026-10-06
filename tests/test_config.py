"""Tests for iiot.config: loading and validating config/settings.yaml."""

from __future__ import annotations

from datetime import date

import pytest

from iiot.config import PROJECT_ROOT, get_settings, load_settings


def test_default_settings_load():
    settings = load_settings()
    assert set(settings.sensors) == {"volt", "rotate", "pressure", "vibration"}
    assert settings.ml.prediction_horizon_hours == 24
    assert settings.ml.train_end_date == date(2015, 10, 1)
    assert settings.business.currency == "INR"
    assert settings.dataset.files["telemetry"] == "PdM_telemetry.csv"


def test_paths_are_absolute_and_inside_project():
    paths = load_settings().paths
    for path in vars(paths).values():
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_sensor_limits_are_ordered():
    for name, limit in load_settings().sensors.items():
        assert limit.min < limit.max, name


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_iiot_config_env_var_overrides_path(write_config, monkeypatch):
    path = write_config({"ml": {"prediction_horizon_hours": 48}})
    monkeypatch.setenv("IIOT_CONFIG", str(path))
    assert load_settings().ml.prediction_horizon_hours == 48


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sensors": {"vibration": {"min": 100.0, "max": 0.0}}}, "min"),
        ({"data_quality_injection": {"missing_rate": 1.5}}, "missing_rate"),
        ({"data_quality_injection": {"spike_rate": -0.1}}, "spike_rate"),
        ({"ml": {"prediction_horizon_hours": 0}}, "prediction_horizon_hours"),
        ({"logging": {"level": "LOUD"}}, "log level"),
    ],
)
def test_invalid_settings_are_rejected(write_config, overrides, message):
    with pytest.raises(ValueError, match=message):
        load_settings(write_config(overrides))
