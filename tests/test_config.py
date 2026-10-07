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
    assert [p.plant_id for p in settings.plant_layout.plants] == ["PUNE", "CHENNAI"]
    assert settings.plant_layout.total_lines == 5


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
        ({"data_quality_injection": {"stuck_episodes": -1}}, "stuck_episodes"),
        ({"data_quality_injection": {"stuck_min_hours": 1}}, "stuck_min_hours"),
        ({"data_quality_injection": {"stuck_min_hours": 20}}, "stuck_min_hours"),
        ({"ml": {"prediction_horizon_hours": 0}}, "prediction_horizon_hours"),
        ({"logging": {"level": "LOUD"}}, "log level"),
        ({"silver": {"stuck_min_run": 1}}, "stuck_min_run"),
        ({"gold": {"feature_step_hours": 0}}, "feature_step_hours"),
        ({"gold": {"window_hours": [24]}}, "at least two windows"),
        ({"gold": {"window_hours": [1, 24]}}, "at least two windows"),
        ({"gold": {"window_hours": [24, 3]}}, "increasing order"),
        ({"silver": {"max_fill_hours": -1}}, "max_fill_hours"),
        ({"silver": {"fill_window_hours": 1}}, "fill_window_hours"),
        ({"business": {"emergency_repair_premium": 0.5}}, "emergency_repair_premium"),
        ({"business": {"downtime_cost_per_hour": -1}}, "downtime_cost_per_hour"),
        (
            {
                "business": {
                    "components": {
                        "comp1": {
                            "repair_cost": 1,
                            "unplanned_downtime_hours": 2,
                            "planned_downtime_hours": 5,
                        }
                    }
                }
            },
            "planned_downtime_hours",
        ),
        (
            {
                "business": {
                    "components": {
                        "comp1": {
                            "repair_cost": -1,
                            "unplanned_downtime_hours": 2,
                            "planned_downtime_hours": 1,
                        }
                    }
                }
            },
            "repair_cost",
        ),
        ({"plant_layout": {"plants": []}}, "at least one plant"),
        (
            {
                "plant_layout": {
                    "plants": [
                        {"plant_id": "A", "name": "A", "city": "X", "lines": 1},
                        {"plant_id": "A", "name": "B", "city": "Y", "lines": 1},
                    ]
                }
            },
            "unique",
        ),
        (
            {"plant_layout": {"plants": [{"plant_id": "A", "name": "A", "city": "X", "lines": 0}]}},
            "at least 1 production line",
        ),
    ],
)
def test_invalid_settings_are_rejected(write_config, overrides, message):
    with pytest.raises(ValueError, match=message):
        load_settings(write_config(overrides))
