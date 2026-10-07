"""Tests for iiot.silver.tables (event and master tables)."""

from __future__ import annotations

import pandas as pd
import pytest

from iiot.silver.tables import (
    build_component_costs,
    build_events,
    build_machines,
    build_tables,
    event_specs,
)

SPECS = event_specs()


def text_frame(rows: list[dict]) -> pd.DataFrame:
    """Bronze-style table: all values text, plus lineage columns."""
    df = pd.DataFrame(rows).astype("string")
    df["_batch_id"] = "batch-1"
    df["_source_line"] = range(2, len(df) + 2)
    return df


MACHINES = text_frame(
    [
        {"machineID": "1", "model": "model3", "age": "18"},
        {"machineID": "2", "model": "model4", "age": "7"},
        {"machineID": "3", "model": "model1", "age": "0"},
    ]
)
LOCATION = text_frame(
    [
        {
            "machineID": m,
            "plant_id": "PUNE",
            "plant_name": "Pune Plant",
            "city": "Pune",
            "line_id": "PUNE-L1",
        }
        for m in ("1", "2", "3")
    ]
)


# --- machines --------------------------------------------------------------------


def test_machines_typed_and_joined_with_location():
    df, stats = build_machines(MACHINES, LOCATION)
    assert list(df.columns[:7]) == [
        "machine_id",
        "model",
        "age",
        "plant_id",
        "plant_name",
        "city",
        "line_id",
    ]
    assert df["machine_id"].tolist() == [1, 2, 3]
    assert str(df["machine_id"].dtype) == str(df["age"].dtype) == "Int64"
    assert df.loc[0, "line_id"] == "PUNE-L1"
    assert stats.rows_out == 3 and sum(stats.dropped.values()) == 0


def test_invalid_machines_dropped_by_reason():
    bad = pd.concat(
        [
            MACHINES,
            text_frame(
                [
                    {"machineID": "x", "model": "model1", "age": "1"},
                    {"machineID": "5", "model": "model9", "age": "1"},
                    {"machineID": "6", "model": "model1", "age": "-2"},
                    {"machineID": "1", "model": "model1", "age": "3"},
                ]
            ),
        ],
        ignore_index=True,
    )
    df, stats = build_machines(bad, LOCATION)
    assert stats.dropped == {"bad_machine_id": 1, "unknown_value": 1, "bad_age": 1, "duplicate": 1}
    assert df["machine_id"].tolist() == [1, 2, 3]


def test_machine_without_location_is_kept_and_flagged():
    df, stats = build_machines(MACHINES, LOCATION.iloc[:2])
    assert len(df) == 3
    assert stats.flagged == {"no_location": 1}
    assert pd.isna(df.loc[2, "line_id"])


# --- events ----------------------------------------------------------------------


def error_rows(*rows):
    return text_frame(
        [dict(zip(["datetime", "machineID", "errorID"], r, strict=True)) for r in rows]
    )


def test_events_typed_and_renamed():
    df, stats = build_events(
        error_rows(("2015-01-03 07:00:00", "1", "error1")), SPECS["errors"], {1, 2, 3}, "errors"
    )
    assert list(df.columns) == ["timestamp", "machine_id", "error_id", "_batch_id", "_source_line"]
    assert df.loc[0, "timestamp"] == pd.Timestamp("2015-01-03 07:00:00")
    assert df.loc[0, "machine_id"] == 1
    assert stats.rows_in == stats.rows_out == 1


def test_invalid_events_dropped_by_reason():
    df, stats = build_events(
        error_rows(
            ("2015-01-03 07:00:00", "1", "error1"),
            ("not a date", "1", "error1"),
            ("2015-01-03 08:00:00", "99", "error1"),
            ("2015-01-03 09:00:00", "2", "error42"),
            ("2015-01-03 07:00:00", "1", "error1"),  # duplicate of the first row
            ("2015-01-03 07:00:00", "1", "error2"),  # same time, different error: kept
        ),
        SPECS["errors"],
        {1, 2, 3},
        "errors",
    )
    assert stats.dropped == {
        "bad_timestamp": 1,
        "unknown_machine": 1,
        "unknown_value": 1,
        "duplicate": 1,
    }
    assert df["error_id"].tolist() == ["error1", "error2"]


def test_each_bad_row_counted_once():
    """A row with several problems is counted only under its first failed check."""
    _, stats = build_events(error_rows(("bad", "99", "error42")), SPECS["errors"], {1}, "errors")
    assert sum(stats.dropped.values()) == 1 and stats.dropped["bad_timestamp"] == 1


@pytest.mark.parametrize("name", ["maintenance", "failures"])
def test_component_events_use_configured_components(name):
    source = SPECS[name].value_source
    bronze = text_frame(
        [
            {"datetime": "2015-01-05 06:00:00", "machineID": "1", source: "comp4"},
            {"datetime": "2015-01-05 06:00:00", "machineID": "1", source: "comp9"},
        ]
    )
    df, stats = build_events(bronze, SPECS[name], {1}, name)
    assert df["component"].tolist() == ["comp4"]
    assert stats.dropped["unknown_value"] == 1


# --- component costs -------------------------------------------------------------


COSTS = text_frame(
    [
        {
            "component": "comp1",
            "repair_cost": "40000",
            "unplanned_repair_cost": "60000.0",
            "unplanned_downtime_hours": "8",
            "planned_downtime_hours": "2",
            "downtime_cost_per_hour": "50000",
            "unplanned_failure_cost": "460000.0",
            "planned_maintenance_cost": "140000",
            "saving_if_prevented": "320000.0",
            "currency": "INR",
        }
    ]
)


def test_component_costs_are_numeric():
    df, stats = build_component_costs(COSTS)
    assert df.loc[0, "saving_if_prevented"] == 320000.0
    assert pd.api.types.is_float_dtype(df["repair_cost"])
    assert df.loc[0, "currency"] == "INR"
    assert stats.rows_out == 1


def test_bad_cost_rows_dropped():
    bad = pd.concat(
        [COSTS, COSTS.assign(component="comp2", repair_cost="abc"), COSTS], ignore_index=True
    )
    df, stats = build_component_costs(bad)
    assert stats.dropped == {"bad_value": 1, "unknown_value": 0, "duplicate": 1}
    assert df["component"].tolist() == ["comp1"]


# --- end to end ------------------------------------------------------------------


def test_build_tables_writes_all_tables(tmp_path):
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    bronze.mkdir()
    MACHINES.to_parquet(bronze / "machines.parquet")
    LOCATION.to_parquet(bronze / "machine_location.parquet")
    error_rows(("2015-01-03 07:00:00", "1", "error1")).to_parquet(bronze / "errors.parquet")
    text_frame([{"datetime": "2015-01-05 06:00:00", "machineID": "2", "comp": "comp2"}]).to_parquet(
        bronze / "maintenance.parquet"
    )
    text_frame(
        [{"datetime": "2015-01-05 06:00:00", "machineID": "2", "failure": "comp2"}]
    ).to_parquet(bronze / "failures.parquet")
    COSTS.to_parquet(bronze / "component_costs.parquet")

    results = build_tables(bronze, silver)

    expected = ["machines", "errors", "maintenance", "failures", "component_costs"]
    assert list(results) == expected
    for table in expected:
        assert len(pd.read_parquet(silver / f"{table}.parquet")) == 1 or table == "machines"
    assert pd.read_parquet(silver / "failures.parquet").loc[0, "component"] == "comp2"


def test_build_tables_without_bronze_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot bronze ingest"):
        build_tables(tmp_path / "bronze", tmp_path / "silver")
