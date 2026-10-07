"""Tests for iiot.silver.report: both proofs, end to end and in isolation."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from iiot.bronze.ingest import ingest_file
from iiot.config import get_settings
from iiot.ingestion.inject_issues import MANIFEST_FILE_NAME, inject
from iiot.silver import telemetry
from iiot.silver.cleaning import CleaningStats
from iiot.silver.report import (
    REPORT_FILE_NAME,
    build_silver,
    check_against_ground_truth,
    check_against_manifest,
    load_ground_truth,
    show_report,
)
from iiot.silver.tables import COST_NUMERIC_COLUMNS

SETTINGS = get_settings()
SENSORS = list(SETTINGS.sensors)


def make_clean(n_machines: int = 4, hours: int = 600, seed: int = 1) -> pd.DataFrame:
    """Clean telemetry in the raw CSV layout (like PdM_telemetry.csv)."""
    rng = np.random.default_rng(seed)
    n = n_machines * hours
    times = pd.date_range("2015-01-01 06:00", periods=hours, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    return pd.DataFrame(
        {
            "datetime": np.tile(times, n_machines),
            "machineID": np.repeat(np.arange(1, n_machines + 1), hours),
            "volt": rng.normal(170, 15, n),
            "rotate": rng.normal(446, 52, n),
            "pressure": rng.normal(100, 11, n),
            "vibration": rng.normal(40, 5, n),
        }
    )


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """clean -> inject issues -> Bronze -> Silver telemetry, all on synthetic data."""
    root = tmp_path_factory.mktemp("lake")
    raw, bronze, silver = root / "raw", root / "bronze", root / "silver"
    raw.mkdir()
    clean = make_clean()
    params = replace(SETTINGS.data_quality_injection, stuck_episodes=15)
    dirty, manifest = inject(clean, params, SETTINGS.sensors)
    clean.to_csv(raw / "PdM_telemetry.csv", index=False)
    dirty.to_csv(raw / "dirty.csv", index=False)
    ingest_file(raw / "dirty.csv", "telemetry", bronze)
    cleaned, _, stats = telemetry.build(bronze, silver)
    return {"raw": raw, "clean": clean, "manifest": manifest, "cleaned": cleaned, "stats": stats}


# --- proof 1: manifest -----------------------------------------------------------


def test_end_to_end_cleaning_matches_manifest(pipeline_run):
    result = check_against_manifest(pipeline_run["stats"], pipeline_run["manifest"])
    assert result["passed"], result["checks"]
    assert result["checks"]["stuck_runs"]["found"] == 15
    assert result["checks"]["duplicates"]["found"] == pipeline_run["manifest"]["duplicate_rows"]


def test_manifest_mismatch_is_reported():
    manifest = {
        "duplicate_rows": 5,
        "spikes": {"volt": 2},
        "stuck_sensor": {
            "episodes": 1,
            "cells_per_sensor": {"volt": 4},
            "episode_list": [{"sensor": "volt"}],
        },
    }
    stats = CleaningStats(
        duplicates_removed=4,  # one missed
        out_of_range_nulled={"volt": 2},
        stuck_runs={"volt": 1},
        stuck_values_nulled={"volt": 3},
    )
    result = check_against_manifest(stats, manifest)
    assert not result["passed"]
    assert result["checks"]["duplicates"]["match"] is False
    assert result["checks"]["spikes"]["match"] is True
    assert result["checks"]["stuck_values"]["injected"] == {"volt": 3}  # 4 cells - 1 first value
    assert result["checks"]["stuck_values"]["match"] is True


# --- proof 2: ground truth -------------------------------------------------------


def test_end_to_end_ok_values_equal_ground_truth(pipeline_run, tmp_path):
    truth = load_ground_truth(pipeline_run["raw"] / "PdM_telemetry.csv")
    result = check_against_ground_truth(pipeline_run["cleaned"], truth, SENSORS)
    assert result["passed"]
    assert result["rows_compared"] == len(pipeline_run["clean"])
    for sensor, v in result["sensors"].items():
        assert v["ok_mismatches"] == 0, sensor
        assert v["filled_values"] > 0
        assert v["filled_mae"] is not None and v["linear_interpolation_mae"] is not None


def test_changed_ok_value_is_detected(pipeline_run):
    truth = load_ground_truth(pipeline_run["raw"] / "PdM_telemetry.csv")
    tampered = pipeline_run["cleaned"].copy()
    first_ok = tampered.index[tampered["volt_quality"] == "ok"][0]
    tampered.loc[first_ok, "volt"] += 1.0
    result = check_against_ground_truth(tampered, truth, SENSORS)
    assert not result["passed"]
    assert result["sensors"]["volt"]["ok_mismatches"] == 1


def test_filled_error_is_measured_correctly():
    times = pd.date_range("2015-01-01", periods=3, freq="h").astype("datetime64[ns]")
    silver = pd.DataFrame(
        {
            "machine_id": pd.array([1, 1, 1], dtype="Int64"),
            "timestamp": times,
            "volt": [10.0, 13.0, 10.0],
            "volt_quality": ["ok", "filled", "ok"],
        }
    )
    truth = silver[["machine_id", "timestamp"]].assign(volt=[10.0, 11.0, 10.0])
    result = check_against_ground_truth(silver, truth, ["volt"])
    assert result["sensors"]["volt"]["filled_mae"] == pytest.approx(2.0)  # |13 - 11|
    assert result["sensors"]["volt"]["linear_interpolation_mae"] == pytest.approx(1.0)  # |10 - 11|


# --- full Silver build ----------------------------------------------------------


def text_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).astype("string")
    df["_batch_id"] = "b"
    df["_source_line"] = range(2, len(df) + 2)
    return df


def write_minimal_bronze(bronze, pipeline_run):
    ingest_file(pipeline_run["raw"] / "dirty.csv", "telemetry", bronze)
    machines = [str(m) for m in range(1, 5)]
    text_frame([{"machineID": m, "model": "model1", "age": "5"} for m in machines]).to_parquet(
        bronze / "machines.parquet"
    )
    text_frame(
        [
            {"machineID": m, "plant_id": "P", "plant_name": "P", "city": "C", "line_id": "P-L1"}
            for m in machines
        ]
    ).to_parquet(bronze / "machine_location.parquet")
    event = {"datetime": "2015-01-02 06:00:00", "machineID": "1"}
    text_frame([{**event, "errorID": "error1"}]).to_parquet(bronze / "errors.parquet")
    text_frame([{**event, "comp": "comp1"}]).to_parquet(bronze / "maintenance.parquet")
    text_frame([{**event, "failure": "comp1"}]).to_parquet(bronze / "failures.parquet")
    costs = dict.fromkeys(COST_NUMERIC_COLUMNS, "1")
    text_frame([{"component": "comp1", **costs, "currency": "INR"}]).to_parquet(
        bronze / "component_costs.parquet"
    )


def test_build_silver_writes_report_with_both_proofs(pipeline_run, tmp_path):
    bronze, silver, raw = tmp_path / "bronze", tmp_path / "silver", pipeline_run["raw"]
    write_minimal_bronze(bronze, pipeline_run)
    (raw / MANIFEST_FILE_NAME).write_text(json.dumps(pipeline_run["manifest"]), encoding="utf-8")

    report = build_silver(bronze, silver, raw)

    assert report["passed"]
    assert report["manifest_check"]["status"] == "checked"
    assert report["ground_truth_check"]["status"] == "checked"
    saved = json.loads((silver / REPORT_FILE_NAME).read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert set(saved["tables"]) == {
        "machines",
        "errors",
        "maintenance",
        "failures",
        "component_costs",
    }
    assert set(saved["telemetry"]["quality"]["volt"]) <= {"ok", "filled", "missing"}


def test_proofs_are_skipped_without_their_inputs(pipeline_run, tmp_path):
    bronze, silver, raw = tmp_path / "bronze", tmp_path / "silver", tmp_path / "empty_raw"
    raw.mkdir()
    write_minimal_bronze(bronze, pipeline_run)
    report = build_silver(bronze, silver, raw)
    assert report["manifest_check"]["status"] == "skipped"
    assert report["ground_truth_check"]["status"] == "skipped"
    assert report["passed"]  # skipped proofs do not fail the build


def test_show_report_reads_latest_report(tmp_path):
    (tmp_path / REPORT_FILE_NAME).write_text(
        json.dumps(
            {
                "generated_at": "2026-10-07T00:00:00+00:00",
                "passed": True,
                "manifest_check": {"status": "skipped", "reason": "x"},
                "ground_truth_check": {"status": "skipped", "reason": "y"},
            }
        ),
        encoding="utf-8",
    )
    assert show_report(tmp_path)["passed"] is True


def test_show_report_without_build_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot silver build"):
        show_report(tmp_path)
