"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import yaml

from iiot.config import DEFAULT_CONFIG_PATH
from iiot.model.warehouse import build_warehouse


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., Path]:
    """Write a copy of the real settings.yaml with selected values overridden.

    Example:
        path = write_config({"sensors": {"volt": {"min": 10, "max": 5}}})
    """

    def _write(overrides: dict | None = None) -> Path:
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        for section, values in (overrides or {}).items():
            if isinstance(values, dict):
                raw[section].update(values)
            else:
                raw[section] = values
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    return _write


# --- Warehouse: a tiny but complete Silver + Gold layer, and the warehouse built from it


def _ts(values: list[str]) -> pd.Series:
    return pd.to_datetime(pd.Series(values)).astype("datetime64[us]")


@pytest.fixture
def silver(tmp_path):
    """A tiny but complete Silver layer."""
    d = tmp_path / "silver"
    d.mkdir()
    pd.DataFrame(
        {
            "machine_id": pd.array([1, 2], dtype="Int64"),
            "model": ["model3", "model4"],
            "age": pd.array([18, 7], dtype="Int64"),
            "plant_id": ["PUNE", "CHENNAI"],
            "plant_name": ["Pune Plant", "Chennai Plant"],
            "city": ["Pune", "Chennai"],
            "line_id": ["PUNE-L1", "CHENNAI-L1"],
        }
    ).to_parquet(d / "machines.parquet")
    pd.DataFrame(
        {
            "component": ["comp1", "comp2"],
            "repair_cost": [40000.0, 60000.0],
            "unplanned_repair_cost": [60000.0, 90000.0],
            "unplanned_downtime_hours": [8.0, 10.0],
            "planned_downtime_hours": [2.0, 3.0],
            "downtime_cost_per_hour": [50000.0, 50000.0],
            "unplanned_failure_cost": [460000.0, 590000.0],
            "planned_maintenance_cost": [140000.0, 210000.0],
            "saving_if_prevented": [320000.0, 380000.0],
            "currency": ["INR", "INR"],
        }
    ).to_parquet(d / "component_costs.parquet")
    quality = ["ok", "filled", "missing"]
    telemetry = pd.DataFrame(
        {
            "machine_id": pd.array([1, 2], dtype="Int64"),
            "timestamp": _ts(["2015-01-01 06:00", "2015-01-10 06:00"]),
            "volt": [170.0, None],
            "rotate": [450.0, 440.0],
            "pressure": [100.0, 101.0],
            "vibration": [40.0, 41.0],
        }
    )
    for sensor in ("volt", "rotate", "pressure", "vibration"):
        telemetry[f"{sensor}_quality"] = pd.Categorical(["ok", "ok"], quality)
    telemetry["volt_quality"] = pd.Categorical(["filled", "missing"], quality)
    telemetry.to_parquet(d / "telemetry.parquet")
    _events(["2015-01-03 07:00"], [2], "error_id", ["error3"]).to_parquet(d / "errors.parquet")
    _events(
        ["2014-12-30 06:00", "2015-01-05 06:00"], [1, 1], "component", ["comp2", "comp1"]
    ).to_parquet(d / "maintenance.parquet")
    _events(
        ["2015-01-05 06:00", "2015-01-05 06:00"], [1, 1], "component", ["comp1", "comp2"]
    ).to_parquet(d / "failures.parquet")

    g = tmp_path / "gold"
    g.mkdir()
    pd.DataFrame(
        {
            "machine_id": pd.array([1, 2], dtype="Int64"),
            "model": ["model3", "model4"],
            "plant_id": ["PUNE", "CHENNAI"],
            "line_id": ["PUNE-L1", "CHENNAI-L1"],
            "month": ["2015-01", "2015-01"],
            "period_hours": [738.0, 738.0],
            "failures": [1, 0],
            "components_failed": [2, 0],
            "planned_maintenances": [0, 0],
            "unplanned_downtime_h": [10.0, 0.0],
            "planned_downtime_h": [0.0, 0.0],
            "downtime_h": [10.0, 0.0],
            "failure_cost": [1050000.0, 0.0],
            "maintenance_cost": [0.0, 0.0],
            "total_cost": [1050000.0, 0.0],
            "availability": [1 - 10 / 738, 1.0],
            "mtbf_h": [728.0, None],
            "mttr_h": [10.0, None],
        }
    ).to_parquet(g / "kpi_machine_monthly.parquet")
    return d


def _events(times, machines, value_col, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": pd.array(machines, dtype="Int64"),
            "timestamp": _ts(times),
            value_col: values,
        }
    )


@pytest.fixture
def built(silver, tmp_path):
    path = tmp_path / "warehouse" / "iiot.duckdb"
    counts = build_warehouse(silver, path, gold_dir=silver.parent / "gold")
    return path, counts
