"""Tests for iiot.bronze.pipeline (batch ingestion with idempotent reloads)."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from iiot.bronze import pipeline
from iiot.bronze.ingest import BronzeError
from iiot.bronze.pipeline import (
    LOG_FILE_NAME,
    bronze_tables,
    file_sha256,
    ingest_all,
    read_log,
)

TABLES = {"alpha": "alpha.csv", "beta": "beta.csv", "gamma": "gamma.csv"}


@pytest.fixture
def dirs(tmp_path):
    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    raw.mkdir()
    for i, name in enumerate(TABLES.values(), 1):
        (raw / name).write_text(f"id,value\n{i},a\n{i + 1},b\n", encoding="utf-8")
    return raw, bronze


def run(dirs, **kwargs):
    raw, bronze = dirs
    return ingest_all(raw_dir=raw, bronze_dir=bronze, tables=TABLES, **kwargs)


def test_registry_covers_all_seven_sources():
    tables = bronze_tables()
    assert list(tables) == [
        "telemetry",
        "errors",
        "maintenance",
        "failures",
        "machines",
        "machine_location",
        "component_costs",
    ]
    assert tables["telemetry"] == "PdM_telemetry_dirty.csv"  # the dirty sensor feed


def test_first_run_loads_every_table_in_one_batch(dirs):
    batch = run(dirs)
    assert [r.table for r in batch.loaded] == list(TABLES)
    assert batch.skipped == []
    _, bronze = dirs
    for table in TABLES:
        df = pq.read_table(bronze / f"{table}.parquet").to_pandas()
        assert (df["_batch_id"] == batch.batch_id).all()


def test_second_run_skips_unchanged_files(dirs):
    first = run(dirs)
    second = run(dirs)
    assert second.loaded == []
    assert second.skipped == list(TABLES)
    _, bronze = dirs
    df = pq.read_table(bronze / "alpha.parquet").to_pandas()
    assert (df["_batch_id"] == first.batch_id).all()  # table was not rewritten


def test_changed_file_is_reloaded(dirs):
    run(dirs)
    raw, _ = dirs
    (raw / "beta.csv").write_text("id,value\n9,z\n", encoding="utf-8")
    batch = run(dirs)
    assert [r.table for r in batch.loaded] == ["beta"]
    assert batch.skipped == ["alpha", "gamma"]
    assert read_log(dirs[1])["tables"]["beta"]["rows"] == 1


def test_deleted_parquet_is_reloaded(dirs):
    run(dirs)
    (dirs[1] / "gamma.parquet").unlink()
    assert [r.table for r in run(dirs).loaded] == ["gamma"]


def test_force_reloads_everything(dirs):
    run(dirs)
    batch = run(dirs, force=True)
    assert [r.table for r in batch.loaded] == list(TABLES)
    assert batch.skipped == []


def test_log_records_hash_rows_and_history(dirs):
    raw, bronze = dirs
    first = run(dirs)
    run(dirs)
    log = read_log(bronze)
    entry = log["tables"]["alpha"]
    assert entry["source_file"] == "alpha.csv"
    assert entry["source_sha256"] == file_sha256(raw / "alpha.csv")
    assert entry["rows"] == 2
    assert entry["columns"] == ["id", "value"]
    assert entry["batch_id"] == first.batch_id
    assert [h["loaded"] for h in log["history"]] == [list(TABLES), []]
    assert [h["skipped"] for h in log["history"]] == [[], list(TABLES)]


def test_history_is_capped(dirs, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_HISTORY", 3)
    for _ in range(5):
        run(dirs)
    history = read_log(dirs[1])["history"]
    assert len(history) == 3


def test_missing_source_fails_before_loading_anything(dirs):
    raw, bronze = dirs
    (raw / "gamma.csv").unlink()
    with pytest.raises(FileNotFoundError, match="gamma.csv.*iiot data prepare"):
        run(dirs)
    assert not list(bronze.glob("*.parquet"))
    assert not (bronze / LOG_FILE_NAME).exists()


def test_failure_keeps_progress_of_earlier_tables(dirs, monkeypatch):
    real_ingest = pipeline.ingest_file

    def flaky(source, table, *args, **kwargs):
        if table == "gamma":
            raise BronzeError("simulated failure")
        return real_ingest(source, table, *args, **kwargs)

    monkeypatch.setattr(pipeline, "ingest_file", flaky)
    with pytest.raises(BronzeError):
        run(dirs)
    log = read_log(dirs[1])
    assert set(log["tables"]) == {"alpha", "beta"}  # recorded before the failure

    monkeypatch.setattr(pipeline, "ingest_file", real_ingest)
    batch = run(dirs)
    assert [r.table for r in batch.loaded] == ["gamma"]  # only the missing one is loaded


def test_log_file_location(dirs):
    run(dirs)
    assert (dirs[1] / LOG_FILE_NAME).exists()
    assert not list(dirs[1].glob(".*.tmp"))
