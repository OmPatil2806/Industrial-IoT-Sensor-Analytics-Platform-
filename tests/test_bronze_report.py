"""Tests for iiot.bronze.report (DuckDB verification of the Bronze layer)."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iiot.bronze.ingest import ingest_file
from iiot.bronze.pipeline import LOG_FILE_NAME, ingest_all
from iiot.bronze.report import REPORT_FILE_NAME, build_report

TABLES = {"sensors": "sensors.csv", "events": "events.csv"}


@pytest.fixture
def lake(tmp_path):
    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    raw.mkdir()
    (raw / "sensors.csv").write_text(
        'datetime,volt,"odd ""name"""\n2015-01-01 06:00:00,170.5,x\n2015-01-01 07:00:00,,\n'
        "2015-01-01 08:00:00,9999.0,y\n",
        encoding="utf-8",
    )
    (raw / "events.csv").write_text("datetime,event\n2015-01-02 06:00:00,error1\n", "utf-8")
    ingest_all(raw_dir=raw, bronze_dir=bronze, tables=TABLES)
    return raw, bronze


def report(lake, **kwargs):
    raw, bronze = lake
    return build_report(bronze_dir=bronze, raw_dir=raw, tables=list(TABLES), **kwargs)


def by_table(rep):
    return {t.table: t for t in rep.tables}


def test_healthy_lake_passes(lake):
    rep = report(lake)
    assert rep.passed
    sensors = by_table(rep)["sensors"]
    assert sensors.rows == 3
    assert sensors.columns[:3] == ["datetime", "volt", 'odd "name"']
    assert sensors.stale is False
    assert sensors.compression_ratio > 0


def test_empty_cells_counted_per_column(lake):
    sensors = by_table(report(lake))["sensors"]
    assert sensors.empty_cells == {"datetime": 0, "volt": 1, 'odd "name"': 1}


def test_missing_table_fails(lake):
    (lake[1] / "events.parquet").unlink()
    events = by_table(report(lake))["events"]
    assert events.checks["exists"] is False
    assert not events.passed
    assert "not found" in events.error


def test_corrupted_table_fails(lake):
    (lake[1] / "events.parquet").write_bytes(b"this is not parquet")
    rep = report(lake)
    events = by_table(rep)["events"]
    assert events.checks["readable"] is False
    assert not rep.passed
    assert by_table(rep)["sensors"].passed  # other tables are still checked


def test_row_count_mismatch_with_log_fails(lake):
    log_path = lake[1] / LOG_FILE_NAME
    log = json.loads(log_path.read_text(encoding="utf-8"))
    log["tables"]["sensors"]["rows"] = 99
    log_path.write_text(json.dumps(log), encoding="utf-8")
    sensors = by_table(report(lake))["sensors"]
    assert sensors.checks["rows_match_log"] is False
    assert not sensors.passed


def test_table_missing_from_log_fails(lake):
    log_path = lake[1] / LOG_FILE_NAME
    log = json.loads(log_path.read_text(encoding="utf-8"))
    del log["tables"]["events"]
    log_path.write_text(json.dumps(log), encoding="utf-8")
    events = by_table(report(lake))["events"]
    assert not events.passed
    assert "ingestion log" in events.error


def test_mixed_batches_fail(lake):
    raw, bronze = lake
    first = pq.read_table(bronze / "sensors.parquet")
    ingest_file(raw / "sensors.csv", "other", bronze, batch_id="another-batch")
    second = pq.read_table(bronze / "other.parquet")
    pq.write_table(pa.concat_tables([first, second]), bronze / "sensors.parquet")
    sensors = by_table(report(lake))["sensors"]
    assert sensors.checks["single_batch"] is False


def test_changed_source_is_reported_stale_but_passes(lake):
    raw, _ = lake
    (raw / "events.csv").write_text("datetime,event\n2015-01-03 06:00:00,error2\n", "utf-8")
    events = by_table(report(lake))["events"]
    assert events.stale is True
    assert events.passed  # stale is a warning: Bronze itself is still consistent


def test_report_file_written(lake):
    rep = report(lake)
    saved = json.loads((lake[1] / REPORT_FILE_NAME).read_text(encoding="utf-8"))
    assert saved["passed"] is rep.passed is True
    assert set(saved["tables"]) == set(TABLES)
    assert saved["tables"]["sensors"]["rows"] == 3


def test_report_can_skip_writing(lake):
    report(lake, write=False)
    assert not (lake[1] / REPORT_FILE_NAME).exists()
