"""Tests for iiot.dashboard.data (the tiny warehouse from conftest.py, plus predictions)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from iiot.dashboard import data
from iiot.model.warehouse import build_warehouse

T0 = pd.Timestamp("2015-01-02 00:00")  # inside the fixture calendar


def ts(*values: str) -> pd.Series:
    return pd.to_datetime(pd.Series(values)).astype("datetime64[us]")


def write_predictions(gold_dir) -> None:
    """Two machines scored at two times; machine 1 is alerted (comp2) at the later time."""
    times = ts("2015-01-01 21:00", "2015-01-01 21:00", "2015-01-02 00:00", "2015-01-02 00:00")
    pd.DataFrame(
        {
            "machine_id": pd.array([1, 2, 1, 2], dtype="Int64"),
            "timestamp": times,
            "failure_risk": [0.2, 0.1, 0.95, 0.05],
            "alert": [False, False, True, False],
            "predicted_component": ["comp2", "comp1", "comp2", "comp1"],
            "comp1_risk": [0.0, 0.1, 0.0, 0.05],
            "comp2_risk": [0.2, 0.0, 0.95, 0.0],
            "comp3_risk": [0.0] * 4,
            "comp4_risk": [0.0] * 4,
            "anomaly_score": [0.4, 0.4, 0.7, 0.6],
            "anomaly_flag": [False, False, True, True],
            "reason_1": [None, None, "rotation 24 h mean 380.0 vs normal 450.0 (-16%)", None],
            "reason_2": [None] * 4,
            "reason_3": [None] * 4,
            "out_of_sample": [True] * 4,
            "failure_model_version": ["1"] * 4,
            "component_model_version": ["1"] * 4,
            "anomaly_model_version": ["1"] * 4,
        }
    ).to_parquet(gold_dir / "predictions.parquet")


@pytest.fixture
def warehouse(built):
    path, _ = built
    return path


@pytest.fixture
def scored(silver, tmp_path):
    write_predictions(silver.parent / "gold")
    path = tmp_path / "scored.duckdb"
    build_warehouse(silver, path, gold_dir=silver.parent / "gold")
    return path


def test_status_without_predictions(warehouse):
    s = data.status(warehouse)
    assert s["tables"]["dim_machine"] == 2
    assert not s["has_predictions"] and s["predictions_from"] is None
    assert "v_machine_risk_now" in s["views"]
    assert s["data_from"] == pd.Timestamp("2015-01-01 06:00")


def test_fleet_kpis(warehouse):
    k = data.fleet_kpis(warehouse)
    assert (k["failures"], k["machines"]) == (1, 2)
    assert k["availability"] == pytest.approx(1 - 10 / (2 * 738))
    assert k["total_cost"] == 1_050_000
    assert isinstance(k["failures"], int)


def test_plant_filter(warehouse):
    assert list(data.plants(warehouse)["plant_id"]) == ["CHENNAI", "PUNE"]
    chennai = data.fleet_kpis(warehouse, plant_id="CHENNAI")  # machine 2: no failures
    assert (chennai["failures"], chennai["machines"], chennai["availability"]) == (0, 1, 1.0)
    pune = data.fleet_monthly(warehouse, plant_id="PUNE")
    assert pune[["failures", "total_cost"]].values.tolist() == [[1, 1_050_000.0]]
    both = data.fleet_monthly(warehouse)
    assert both.loc[0, "availability"] == pytest.approx(1 - 10 / (2 * 738))


def test_fleet_views_are_returned(warehouse):
    assert len(data.fleet_monthly(warehouse)) == 1
    assert list(data.line_performance(warehouse)["availability_rank"]) == [1, 2]
    assert len(data.component_reliability(warehouse)) == 2
    assert list(data.machine_health(warehouse)["machine_id"]) == [1, 2]
    assert list(data.machines(warehouse)["line_id"]) == ["PUNE-L1", "CHENNAI-L1"]


def test_risk_as_of_uses_the_latest_scores_at_or_before_the_time(scored):
    later = data.risk_as_of(T0, scored)
    assert list(later["machine_id"]) == [1, 2]  # riskiest first
    assert list(later["risk_rank"]) == [1, 2]
    assert later.loc[0, "alert"] and later.loc[0, "reason_1"].startswith("rotation")
    assert later.loc[0, "line_id"] == "PUNE-L1"
    earlier = data.risk_as_of(T0 - pd.Timedelta(hours=1), scored)
    assert set(earlier["scored_at"]) == {pd.Timestamp("2015-01-01 21:00")}
    assert data.prediction_times(scored).tolist() == [T0 - pd.Timedelta(hours=3), T0]


def test_maintenance_plan_puts_alerts_first_with_their_saving(scored):
    plan = data.maintenance_plan(T0, scored)
    assert list(plan["priority"]) == [1, 2]
    first, second = plan.iloc[0], plan.iloc[1]
    assert (first["machine_id"], first["action"]) == (1, "replace comp2")
    assert first["saving_if_acted"] == 380_000  # comp2's saving_if_prevented
    assert first["alert_since"] == T0
    assert (second["machine_id"], second["action"], second["saving_if_acted"]) == (
        2,
        "inspect sensors",
        0.0,
    )


def test_maintenance_plan_is_empty_when_nothing_is_flagged(scored):
    plan = data.maintenance_plan(T0 - pd.Timedelta(hours=1), scored)
    assert plan.empty


def test_one_machine(scored):
    start, end = pd.Timestamp("2014-12-01"), pd.Timestamp("2016-01-01")
    sensors = data.machine_sensors(1, start, end, scored)
    assert len(sensors) == 4 and set(sensors["sensor"]) == {
        "volt",
        "rotate",
        "pressure",
        "vibration",
    }
    events = data.machine_events(1, start, end, scored)
    assert list(zip(events["kind"], events["detail"], strict=True)) == [
        ("maintenance", "planned: comp2"),
        ("failure", "comp1"),
        ("failure", "comp2"),
        ("maintenance", "failure_repair: comp1"),
    ]
    assert len(data.machine_predictions(1, start, end, scored)) == 2


def test_report_status_lists_missing_reports(tmp_path, monkeypatch):
    present = tmp_path / "gold_report.json"
    present.write_text(json.dumps({"passed": True, "generated_at": "2026-01-01T00:00:00"}))
    monkeypatch.setattr(
        data, "report_files", lambda: {"gold": present, "ml": tmp_path / "missing.json"}
    )
    table = data.report_status()
    assert table.set_index("layer")["passed"].to_dict() == {"gold": True, "ml": None}


def test_model_cards_read_what_exists(tmp_path):
    (tmp_path / "failure_24h_model_card.json").write_text(json.dumps({"version": "3"}))
    assert data.model_cards(tmp_path) == {"failure": {"version": "3"}}


def test_missing_warehouse(tmp_path):
    assert not data.warehouse_exists(tmp_path / "none.duckdb")
    with pytest.raises(FileNotFoundError, match="iiot model build"):
        data.status(tmp_path / "none.duckdb")
