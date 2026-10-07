"""Tests for iiot.bronze.ingest."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iiot.bronze import ingest
from iiot.bronze.ingest import (
    METADATA_COLUMNS,
    BronzeError,
    count_data_lines,
    ingest_file,
    new_batch_id,
)

CSV = (
    "datetime,machineID,volt,comp\n"
    "2015-01-01 06:00:00,1,176.217853015625,comp1\n"
    "yesterday,2,,comp2\n"
    '2015-01-01 08:00:00,007,-999.0,"comp3"\n'
    "2015-01-01 09:00:00,3,9999.0,NA\n"
)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "raw" / "sample.csv"
    path.parent.mkdir()
    path.write_text(CSV, encoding="utf-8")
    return path


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "bronze"


def read(result):
    return pq.read_table(result.output).to_pandas()


def test_values_are_kept_exactly_as_text(source, out_dir):
    df = read(ingest_file(source, "sample", out_dir))
    assert df["datetime"].tolist()[1] == "yesterday"  # unparseable timestamp kept
    assert df["machineID"].tolist()[2] == "007"  # leading zeros kept
    assert df["volt"].tolist() == ["176.217853015625", "", "-999.0", "9999.0"]  # exact text
    assert df["comp"].tolist() == ["comp1", "comp2", "comp3", "NA"]  # "NA" is not null


def test_source_columns_are_strings(source, out_dir):
    schema = pq.read_schema(ingest_file(source, "sample", out_dir).output)
    for column in ("datetime", "machineID", "volt", "comp"):
        assert schema.field(column).type == pa.string()


def test_metadata_columns(source, out_dir):
    ts = datetime(2026, 10, 7, 10, 15, 2, tzinfo=UTC)
    result = ingest_file(source, "sample", out_dir, batch_id="batch-1", ingested_at=ts)
    df = read(result)
    assert list(df.columns[-4:]) == list(METADATA_COLUMNS)
    assert (df["_source_file"] == "sample.csv").all()
    assert df["_source_line"].tolist() == [2, 3, 4, 5]  # header is line 1
    assert (df["_batch_id"] == "batch-1").all()
    assert (df["_ingested_at"] == ts).all()


def test_result_summary(source, out_dir):
    result = ingest_file(source, "sample", out_dir)
    assert result.table == "sample"
    assert result.output == out_dir / "sample.parquet"
    assert result.rows == 4
    assert result.columns == ("datetime", "machineID", "volt", "comp")
    assert result.csv_bytes == source.stat().st_size
    assert result.parquet_bytes == result.output.stat().st_size


def test_reingest_overwrites_and_leaves_no_temp_files(source, out_dir):
    ingest_file(source, "sample", out_dir)
    source.write_text(CSV + "2015-01-01 10:00:00,4,170.0,comp4\n", encoding="utf-8")
    assert ingest_file(source, "sample", out_dir).rows == 5
    assert sorted(p.name for p in out_dir.iterdir()) == ["sample.parquet"]


def test_row_count_mismatch_fails_and_keeps_old_table(source, out_dir, monkeypatch):
    first = ingest_file(source, "sample", out_dir)
    monkeypatch.setattr(ingest, "count_data_lines", lambda path: 999)
    with pytest.raises(BronzeError, match="row count mismatch"):
        ingest_file(source, "sample", out_dir)
    assert read(first)["volt"].tolist()[0] == "176.217853015625"  # previous table intact
    assert sorted(p.name for p in out_dir.iterdir()) == ["sample.parquet"]


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_file(tmp_path / "nope.csv", "nope", tmp_path)


def test_reserved_column_names_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,_batch_id\n1,2\n", encoding="utf-8")
    with pytest.raises(BronzeError, match="must not start with '_'"):
        ingest_file(path, "bad", tmp_path / "bronze")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"h\n1\n2\n", 2),
        (b"h\n1\n2", 2),  # no trailing newline
        (b"h\r\n1\r\n2\r\n", 2),  # Windows line endings
        (b"h\n", 0),  # header only
    ],
)
def test_count_data_lines(tmp_path, content, expected):
    path = tmp_path / "f.csv"
    path.write_bytes(content)
    assert count_data_lines(path) == expected


def test_batch_id_format():
    batch = new_batch_id(datetime(2026, 10, 7, 10, 15, 2, tzinfo=UTC))
    assert batch.startswith("20261007T101502Z-")
    assert len(batch) == len("20261007T101502Z-") + 6
    assert new_batch_id() != new_batch_id()
