"""Tests for iiot.ingestion.validate_raw using a tiny synthetic dataset."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from iiot.config import get_settings
from iiot.ingestion.validate_raw import CONTRACTS, REPORT_FILE_NAME, validate_raw

FILES = get_settings().dataset.files
N_MACHINES = 3

# Same rules as the real contract, but sized for 3 machines and a few rows.
SMALL_CONTRACTS = {
    name: replace(
        c,
        min_rows=1,
        max_rows=10_000,
        machine_count=N_MACHINES if c.machine_count else None,
    )
    for name, c in CONTRACTS.items()
}


def make_tables() -> dict[str, pd.DataFrame]:
    times = pd.date_range("2015-01-01 06:00", periods=4, freq="h").astype(str)
    telemetry = pd.DataFrame(
        [
            {
                "datetime": t,
                "machineID": m,
                "volt": 170.0,
                "rotate": 450.0,
                "pressure": 100.0,
                "vibration": 40.0,
            }
            for m in range(1, N_MACHINES + 1)
            for t in times
        ]
    )
    return {
        "telemetry": telemetry,
        "errors": pd.DataFrame(
            {"datetime": ["2015-01-03 07:00:00"], "machineID": [1], "errorID": ["error1"]}
        ),
        "maintenance": pd.DataFrame(
            {"datetime": ["2014-06-01 06:00:00"], "machineID": [2], "comp": ["comp2"]}
        ),
        "failures": pd.DataFrame(
            {"datetime": ["2015-01-05 06:00:00"], "machineID": [3], "failure": ["comp4"]}
        ),
        "machines": pd.DataFrame(
            {"machineID": [1, 2, 3], "model": ["model1", "model3", "model4"], "age": [18, 7, 8]}
        ),
    }


def write_tables(raw_dir, tables) -> None:
    for name, df in tables.items():
        df.to_csv(raw_dir / FILES[name], index=False)


def run(raw_dir, tables=None, **kwargs):
    write_tables(raw_dir, tables or make_tables())
    return validate_raw(raw_dir=raw_dir, contracts=SMALL_CONTRACTS, **kwargs)


def failed_checks(report) -> set[tuple[str, str]]:
    return {(r.table, r.check) for r in report.failures}


def test_valid_dataset_passes(tmp_path):
    report = run(tmp_path)
    assert report.passed, report.failures
    assert len(report.results) > 30


def test_missing_file_fails(tmp_path):
    write_tables(tmp_path, make_tables())
    (tmp_path / FILES["errors"]).unlink()
    report = validate_raw(raw_dir=tmp_path, contracts=SMALL_CONTRACTS)
    assert failed_checks(report) == {("errors", "file_exists")}


def test_missing_column_fails(tmp_path):
    tables = make_tables()
    tables["telemetry"] = tables["telemetry"].drop(columns="vibration")
    assert ("telemetry", "columns") in failed_checks(run(tmp_path, tables))


@pytest.mark.parametrize(
    ("table", "column", "value", "check"),
    [
        ("telemetry", "volt", "not-a-number", "numeric:volt"),
        ("telemetry", "datetime", "yesterday", "dates_parse"),
        ("telemetry", "datetime", "2017-01-01 00:00:00", "date_range"),
        ("telemetry", "machineID", -1, "machine_id_valid"),
        ("errors", "errorID", "error99", "allowed_values:errorID"),
        ("failures", "failure", "engine", "allowed_values:failure"),
        ("machines", "model", "model9", "allowed_values:model"),
    ],
)
def test_bad_values_fail(tmp_path, table, column, value, check):
    tables = make_tables()
    tables[table][column] = tables[table][column].astype(object)
    tables[table].loc[0, column] = value
    assert (table, check) in failed_checks(run(tmp_path, tables))


def test_us_style_timestamps_are_accepted(tmp_path):
    tables = make_tables()
    tables["errors"]["datetime"] = ["1/3/2015 7:00:00 AM"]
    assert run(tmp_path, tables).passed


def test_nulls_fail(tmp_path):
    tables = make_tables()
    tables["telemetry"].loc[0, "pressure"] = None
    assert ("telemetry", "no_nulls") in failed_checks(run(tmp_path, tables))


def test_duplicate_machine_fails(tmp_path):
    tables = make_tables()
    tables["machines"].loc[2, "machineID"] = 1
    failed = failed_checks(run(tmp_path, tables))
    assert ("machines", "unique:machineID") in failed
    assert ("machines", "machine_count") in failed


def test_row_count_out_of_range_fails(tmp_path):
    contracts = {**SMALL_CONTRACTS, "errors": replace(SMALL_CONTRACTS["errors"], min_rows=5)}
    write_tables(tmp_path, make_tables())
    report = validate_raw(raw_dir=tmp_path, contracts=contracts)
    assert failed_checks(report) == {("errors", "row_count")}


def test_report_file_is_written(tmp_path):
    run(tmp_path)
    data = json.loads((tmp_path / REPORT_FILE_NAME).read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["checks_failed"] == 0
    assert data["checks_run"] == len(data["results"])


def test_report_can_be_skipped(tmp_path):
    run(tmp_path, write_report=False)
    assert not (tmp_path / REPORT_FILE_NAME).exists()
