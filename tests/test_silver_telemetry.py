"""Tests for iiot.silver.telemetry (typing and standardisation)."""

from __future__ import annotations

import pandas as pd
import pytest

from iiot.silver.telemetry import load_bronze, standardise


def bronze(rows: list[dict]) -> pd.DataFrame:
    """Bronze-style telemetry: every value is text, plus lineage columns."""
    base = {
        "datetime": "2015-01-01 06:00:00",
        "machineID": "1",
        "volt": "170.5",
        "rotate": "450.0",
        "pressure": "100.0",
        "vibration": "40.0",
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["_batch_id"] = "batch-1"
    df["_source_line"] = range(2, len(df) + 2)
    return df


def test_types_and_names():
    df, _ = standardise(bronze([{}]))
    assert list(df.columns) == [
        "timestamp",
        "machine_id",
        "volt",
        "rotate",
        "pressure",
        "vibration",
        "_batch_id",
        "_source_line",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert str(df["machine_id"].dtype) == "Int64"
    for col in ("volt", "rotate", "pressure", "vibration"):
        assert df[col].dtype == "float64"
    row = df.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2015-01-01 06:00:00")
    assert row["machine_id"] == 1
    assert row["volt"] == pytest.approx(170.5)


def test_values_converted_exactly():
    df, _ = standardise(bronze([{"volt": "176.217853015625", "machineID": " 42 "}]))
    assert df.loc[0, "volt"] == 176.217853015625
    assert df.loc[0, "machine_id"] == 42  # surrounding spaces ignored


def test_empty_and_unparseable_counted_separately():
    df, stats = standardise(
        bronze([{"volt": ""}, {"volt": "abc"}, {"volt": "-999.0"}, {"rotate": "  "}])
    )
    assert stats.empty["volt"] == 1
    assert stats.unparseable["volt"] == 1
    assert stats.empty["rotate"] == 1  # whitespace-only counts as empty
    assert df["volt"].isna().sum() == 2
    assert df.loc[2, "volt"] == -999.0  # out-of-range values are kept here; cleaning removes them


def test_rows_without_valid_key_are_dropped_and_counted():
    df, stats = standardise(
        bronze(
            [
                {},
                {"datetime": "yesterday"},
                {"datetime": ""},
                {"machineID": "1.5"},
                {"machineID": "-3"},
                {"machineID": "M7"},
            ]
        )
    )
    assert stats.rows_in == 6
    assert stats.rows_out == len(df) == 1
    assert stats.rows_dropped_no_key == 5
    assert stats.unparseable["timestamp"] == 1
    assert stats.empty["timestamp"] == 1
    assert stats.unparseable["machine_id"] == 3


def test_us_style_timestamps_are_parsed():
    df, _ = standardise(bronze([{"datetime": "1/3/2015 7:00:00 AM"}]))
    assert df.loc[0, "timestamp"] == pd.Timestamp("2015-01-03 07:00:00")


def test_lineage_columns_kept():
    df, _ = standardise(bronze([{}, {"datetime": "bad"}, {}]))
    assert df["_source_line"].tolist() == [2, 4]  # traceable to the original CSV lines
    assert (df["_batch_id"] == "batch-1").all()


def test_missing_bronze_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        standardise(bronze([{}]).drop(columns="pressure"))


def test_load_bronze_without_table_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot bronze ingest"):
        load_bronze(tmp_path)
