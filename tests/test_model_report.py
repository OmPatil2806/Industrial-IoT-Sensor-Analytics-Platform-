"""Tests for iiot.model.report: the warehouse checks and model_report.json.

Every check must pass on the clean warehouse from conftest.py, and fail when the
warehouse is corrupted in the way that check is meant to catch.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from iiot.model.report import REPORT_FILE_NAME, build_model, check_model, show_report


@pytest.fixture
def check(built, silver, tmp_path):
    """Run check_model on the test warehouse, optionally after corrupting it with SQL."""
    path, _ = built

    def _check(*statements: str) -> dict:
        with duckdb.connect(str(path)) as con:
            for statement in statements:
                con.execute(statement)
        return check_model(path, silver, silver.parent / "gold", out_dir=tmp_path)

    return _check


def test_clean_warehouse_passes_and_report_is_written(check, tmp_path):
    report = check()
    assert report["passed"], {n: c for n, c in report["checks"].items() if not c["passed"]}
    assert report["tables"]["fact_sensor_reading"] == 8
    assert report["checks"]["silver_reconciliation"]["sensor_readings"] == 8
    assert report["checks"]["gold_reconciliation"]["failure_events"] == 1
    assert report["checks"]["gold_reconciliation"]["planned_maintenance_events"] == 0
    saved = json.loads((tmp_path / REPORT_FILE_NAME).read_text(encoding="utf-8"))
    assert saved["passed"] and set(saved["checks"]) == set(report["checks"])


def failed(report: dict) -> list[str]:
    return [name for name, c in report["checks"].items() if not c["passed"]]


def test_duplicate_sensor_reading_is_caught(check):
    report = check("INSERT INTO fact_sensor_reading SELECT * FROM fact_sensor_reading LIMIT 1")
    assert "unique_keys" in failed(report)
    assert report["checks"]["unique_keys"]["problems"]["fact_sensor_reading"] == {
        "duplicate_keys": 1,
        "empty_keys": 0,
    }


@pytest.mark.parametrize(
    ("statement", "orphan"),
    [
        (
            "UPDATE fact_sensor_reading SET machine_id = 99 WHERE machine_id = 1",
            "fact_sensor_reading.machine_id -> dim_machine",
        ),
        (
            "UPDATE fact_sensor_reading SET sensor = 'temperature' WHERE sensor = 'volt'",
            "fact_sensor_reading.sensor -> dim_sensor",
        ),
        (
            "UPDATE fact_sensor_reading SET date_key = 20990101 WHERE machine_id = 1",
            "fact_sensor_reading.date_key -> dim_date",
        ),
    ],
    ids=["unknown machine", "unknown sensor", "unknown date"],
)
def test_orphan_sensor_readings_are_caught(check, statement, orphan):
    report = check(statement)
    assert "references" in failed(report)
    assert report["checks"]["references"]["orphans"][orphan] > 0


def test_date_key_that_does_not_match_the_timestamp_is_caught(check):
    report = check("UPDATE fact_error SET date_key = 20150101")  # the error is on 01-03
    assert failed(report) == ["date_keys"]
    assert report["checks"]["date_keys"]["mismatched_date_keys"] == {"fact_error": 1}


def test_gap_in_the_calendar_is_caught(check):
    report = check("DELETE FROM dim_date WHERE date_key = 20150107")
    assert failed(report) == ["date_keys"]
    assert report["checks"]["date_keys"]["missing_days"] == 1


def test_changed_sensor_values_are_caught(check):
    report = check("UPDATE fact_sensor_reading SET value = value + 1 WHERE sensor = 'rotate'")
    assert failed(report) == ["silver_reconciliation"]
    assert list(report["checks"]["silver_reconciliation"]["differences"]) == [
        "fact_sensor_reading.rotate.sum"
    ]


def test_lost_rows_are_caught(check):
    report = check("DELETE FROM fact_error", "DELETE FROM fact_sensor_reading WHERE value IS NULL")
    differences = report["checks"]["silver_reconciliation"]["differences"]
    assert {"fact_error.rows", "fact_sensor_reading.volt.missing"} <= set(differences)


def test_changed_kpi_is_caught(check):
    report = check("UPDATE fact_machine_month SET total_cost = total_cost + 1 WHERE machine_id = 1")
    assert failed(report) == ["gold_reconciliation"]
    assert "fact_machine_month.total_cost" in report["checks"]["gold_reconciliation"]["differences"]


def test_event_missing_from_the_kpis_is_caught(check):
    report = check(
        "INSERT INTO fact_maintenance "
        "VALUES (1, 20150107, TIMESTAMP '2015-01-07 06:00', 'comp1', 'planned', 2, 140000)"
    )
    assert report["checks"]["gold_reconciliation"]["differences"]["planned_maintenance_events"] == {
        "fact_tables": 1,
        "fact_machine_month": 0,
    }


@pytest.mark.parametrize(
    ("statements", "view", "problem"),
    [
        (["DROP VIEW v_line_performance"], "v_line_performance", "missing"),
        (
            ["CREATE OR REPLACE VIEW v_fleet_monthly AS SELECT 1 AS failures WHERE false"],
            "v_fleet_monthly",
            "returns no rows",
        ),
        (
            ["CREATE OR REPLACE VIEW v_machine_health AS SELECT 0 AS failures, 0.0 AS total_cost"],
            "v_machine_health",
            "totals",
        ),
        (
            [
                "CREATE TABLE scratch AS SELECT 1 AS x",
                "CREATE VIEW v_broken AS SELECT * FROM scratch",
                "DROP TABLE scratch",
            ],
            "v_broken",
            "fails",
        ),
    ],
    ids=["missing", "empty", "wrong totals", "broken"],
)
def test_view_problems_are_caught(check, statements, view, problem):
    report = check(*statements)
    assert failed(report) == ["views"]
    assert report["checks"]["views"]["problems"][view].startswith(problem)


def test_build_model_builds_checks_and_shows_the_report(silver, tmp_path):
    path = tmp_path / "warehouse" / "iiot.duckdb"
    report = build_model(silver, silver.parent / "gold", path)
    assert report["passed"]
    assert show_report(path.parent)["generated_at"] == report["generated_at"]


def test_show_report_without_a_report_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot model build"):
        show_report(tmp_path)


def test_missing_gold_kpis_raise(check, silver):
    (silver.parent / "gold" / "kpi_machine_monthly.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="iiot gold build"):
        check()
