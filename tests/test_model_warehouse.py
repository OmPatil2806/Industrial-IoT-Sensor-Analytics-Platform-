"""Tests for iiot.model.warehouse (DuckDB star schema: dimension tables)."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from iiot.model import warehouse
from iiot.model.warehouse import WarehouseLockedError, build_warehouse, connect, sql_scripts


def ts(values: list[str]) -> pd.Series:
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
    pd.DataFrame({"timestamp": ts(["2015-01-01 06:00", "2015-01-10 06:00"])}).to_parquet(
        d / "telemetry.parquet"
    )
    pd.DataFrame({"timestamp": ts(["2015-01-03 07:00"])}).to_parquet(d / "errors.parquet")
    pd.DataFrame({"timestamp": ts(["2014-12-30 06:00"])}).to_parquet(d / "maintenance.parquet")
    pd.DataFrame({"timestamp": ts(["2015-01-05 06:00"])}).to_parquet(d / "failures.parquet")
    return d


@pytest.fixture
def built(silver, tmp_path):
    path = tmp_path / "warehouse" / "iiot.duckdb"
    counts = build_warehouse(silver, path)
    return path, counts


def test_all_dimension_tables_are_built(built):
    _, counts = built
    assert counts == {
        "dim_component": 2,
        "dim_date": 12,  # 2014-12-30 (maintenance) to 2015-01-10 (telemetry)
        "dim_error_type": 5,
        "dim_machine": 2,
        "dim_sensor": 4,
    }


def test_dimension_contents(built):
    path, _ = built
    con = connect(path)
    assert con.sql("SELECT line_id FROM dim_machine WHERE machine_id = 2").fetchone() == (
        "CHENNAI-L1",
    )
    assert con.sql(
        "SELECT saving_if_prevented FROM dim_component WHERE component = 'comp2'"
    ).fetchone() == (380000.0,)
    assert con.sql("SELECT valid_max FROM dim_sensor WHERE sensor = 'vibration'").fetchone() == (
        100.0,
    )
    assert con.sql(
        "SELECT error_number FROM dim_error_type WHERE error_id = 'error4'"
    ).fetchone() == (4,)
    con.close()


def test_date_dimension_attributes(built):
    path, _ = built
    con = connect(path)
    row = con.sql(
        "SELECT date_key, year, quarter, month, year_month, day_of_week, day_name, is_weekend "
        "FROM dim_date WHERE date = DATE '2015-01-03'"
    ).fetchone()
    con.close()
    assert row == (20150103, 2015, 1, 1, "2015-01", 6, "Saturday", True)


def test_primary_keys_reject_duplicates(built):
    path, _ = built
    con = duckdb.connect(str(path))
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO dim_machine VALUES (1, 'model1', 1, 'P', 'P', 'C', 'P-L1')")
    con.close()


def test_rebuild_replaces_the_file_and_leaves_no_temp_files(silver, built):
    path, _ = built
    build_warehouse(silver, path)
    assert sorted(p.name for p in path.parent.iterdir()) == ["iiot.duckdb"]


def test_read_only_connection_cannot_write(built):
    path, _ = built
    con = connect(path)
    with pytest.raises(duckdb.Error):
        con.execute("DELETE FROM dim_machine")
    con.close()


def test_locked_warehouse_gives_a_clear_error_and_keeps_the_old_file(silver, built, monkeypatch):
    path, _ = built
    before = path.read_bytes()

    def locked(src, dst):
        raise PermissionError("file in use")

    monkeypatch.setattr(warehouse.os, "replace", locked)
    with pytest.raises(WarehouseLockedError, match="Close or disconnect"):
        build_warehouse(silver, path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["iiot.duckdb"]


def test_failing_script_is_named(silver, tmp_path):
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "01_ok.sql").write_text("CREATE TABLE ok AS SELECT 1 AS x;", encoding="utf-8")
    (sql_dir / "02_broken.sql").write_text("SELECT * FROM no_such_table;", encoding="utf-8")
    with pytest.raises(RuntimeError, match="02_broken.sql"):
        build_warehouse(silver, tmp_path / "w.duckdb", sql_dir=sql_dir)
    assert not (tmp_path / "w.duckdb").exists()


def test_missing_silver_table_raises(silver, tmp_path):
    (silver / "machines.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="iiot silver build"):
        build_warehouse(silver, tmp_path / "w.duckdb")


def test_connect_without_warehouse_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot model build"):
        connect(tmp_path / "missing.duckdb")


def test_scripts_run_in_file_name_order():
    names = [p.name for p in sql_scripts()]
    assert names == sorted(names)
    assert names[:5] == [
        "01_dim_machine.sql",
        "02_dim_component.sql",
        "03_dim_sensor.sql",
        "04_dim_error_type.sql",
        "05_dim_date.sql",
    ]
