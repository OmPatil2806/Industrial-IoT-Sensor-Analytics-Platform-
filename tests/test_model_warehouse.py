"""Tests for iiot.model.warehouse (DuckDB star schema: dimension and fact tables)."""

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
    quality = ["ok", "filled", "missing"]
    telemetry = pd.DataFrame(
        {
            "machine_id": pd.array([1, 2], dtype="Int64"),
            "timestamp": ts(["2015-01-01 06:00", "2015-01-10 06:00"]),
            "volt": [170.0, None],
            "rotate": [450.0, 440.0],
            "pressure": [100.0, 101.0],
            "vibration": [40.0, 41.0],
        }
    )
    for sensor in ("volt", "rotate", "pressure", "vibration"):
        telemetry[f"{sensor}_quality"] = pd.Categorical(["ok", "ok"], quality)
    telemetry["volt_quality"] = pd.Categorical(["filled", "missing"], quality)
    telemetry.to_parquet(d / "telemetry.parquet")
    events(["2015-01-03 07:00"], [2], "error_id", ["error3"]).to_parquet(d / "errors.parquet")
    events(
        ["2014-12-30 06:00", "2015-01-05 06:00"], [1, 1], "component", ["comp2", "comp1"]
    ).to_parquet(d / "maintenance.parquet")
    events(
        ["2015-01-05 06:00", "2015-01-05 06:00"], [1, 1], "component", ["comp1", "comp2"]
    ).to_parquet(d / "failures.parquet")

    g = tmp_path / "gold"
    g.mkdir()
    pd.DataFrame(
        {
            "machine_id": pd.array([1, 2], dtype="Int64"),
            "model": ["model3", "model4"],
            "plant_id": ["PUNE", "CHENNAI"],
            "line_id": ["PUNE-L1", "CHENNAI-L1"],
            "month": ["2015-01", "2015-01"],
            "period_hours": [738.0, 738.0],
            "failures": [1, 0],
            "components_failed": [2, 0],
            "planned_maintenances": [0, 0],
            "unplanned_downtime_h": [10.0, 0.0],
            "planned_downtime_h": [0.0, 0.0],
            "downtime_h": [10.0, 0.0],
            "failure_cost": [1050000.0, 0.0],
            "maintenance_cost": [0.0, 0.0],
            "total_cost": [1050000.0, 0.0],
            "availability": [1 - 10 / 738, 1.0],
            "mtbf_h": [728.0, None],
            "mttr_h": [10.0, None],
        }
    ).to_parquet(g / "kpi_machine_monthly.parquet")
    return d


def events(times, machines, value_col, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": pd.array(machines, dtype="Int64"),
            "timestamp": ts(times),
            value_col: values,
        }
    )


def build(silver, path, **kwargs):
    return build_warehouse(silver, path, gold_dir=silver.parent / "gold", **kwargs)


@pytest.fixture
def built(silver, tmp_path):
    path = tmp_path / "warehouse" / "iiot.duckdb"
    counts = build(silver, path)
    return path, counts


def test_all_tables_are_built(built):
    _, counts = built
    assert counts == {
        "dim_component": 2,
        "dim_date": 12,  # 2014-12-30 (maintenance) to 2015-01-10 (telemetry)
        "dim_error_type": 5,
        "dim_machine": 2,
        "dim_sensor": 4,
        "fact_error": 1,
        "fact_failure": 2,
        "fact_machine_month": 2,
        "fact_maintenance": 2,
        "fact_sensor_reading": 8,  # 2 machine-hours x 4 sensors
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
    build(silver, path)
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
        build(silver, path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["iiot.duckdb"]


def test_failing_script_is_named(silver, tmp_path):
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "01_ok.sql").write_text("CREATE TABLE ok AS SELECT 1 AS x;", encoding="utf-8")
    (sql_dir / "02_broken.sql").write_text("SELECT * FROM no_such_table;", encoding="utf-8")
    with pytest.raises(RuntimeError, match="02_broken.sql"):
        build(silver, tmp_path / "w.duckdb", sql_dir=sql_dir)
    assert not (tmp_path / "w.duckdb").exists()


def test_missing_silver_table_raises(silver, tmp_path):
    (silver / "machines.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="iiot silver build"):
        build(silver, tmp_path / "w.duckdb")


def test_connect_without_warehouse_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot model build"):
        connect(tmp_path / "missing.duckdb")


def test_scripts_run_in_file_name_order():
    names = [p.name for p in sql_scripts()]
    assert names == sorted(names)
    assert names == [
        "01_dim_machine.sql",
        "02_dim_component.sql",
        "03_dim_sensor.sql",
        "04_dim_error_type.sql",
        "05_dim_date.sql",
        "06_fact_sensor_reading.sql",
        "07_fact_failure.sql",
        "08_fact_maintenance.sql",
        "09_fact_error.sql",
        "10_fact_machine_month.sql",
    ]


# --- fact tables -----------------------------------------------------------------


def test_sensor_readings_are_long_and_keep_missing_values(built):
    path, _ = built
    con = connect(path)
    rows = con.sql(
        "SELECT machine_id, date_key, sensor, value, quality FROM fact_sensor_reading "
        "WHERE sensor = 'volt' ORDER BY machine_id"
    ).fetchall()
    con.close()
    assert rows == [(1, 20150101, "volt", 170.0, "filled"), (2, 20150110, "volt", None, "missing")]


def test_failures_carry_costs_and_event_size(built):
    path, _ = built
    con = connect(path)
    rows = con.sql(
        "SELECT component, components_in_event, downtime_hours, failure_cost "
        "FROM fact_failure ORDER BY component"
    ).fetchall()
    con.close()
    assert rows == [("comp1", 2, 8.0, 460000.0), ("comp2", 2, 10.0, 590000.0)]


def test_maintenance_separates_failure_repairs_from_planned(built):
    path, _ = built
    con = connect(path)
    rows = con.sql(
        "SELECT date_key, component, maintenance_type, downtime_hours, maintenance_cost "
        "FROM fact_maintenance ORDER BY timestamp"
    ).fetchall()
    con.close()
    assert rows == [
        (20141230, "comp2", "planned", 3.0, 210000.0),
        (20150105, "comp1", "failure_repair", 0.0, 0.0),  # already costed in fact_failure
    ]


def test_errors_and_monthly_kpis(built):
    path, _ = built
    con = connect(path)
    assert con.sql("SELECT machine_id, date_key, error_id FROM fact_error").fetchall() == [
        (2, 20150103, "error3")
    ]
    row = con.sql(
        "SELECT month_date_key, year_month, failures, total_cost "
        "FROM fact_machine_month WHERE machine_id = 1"
    ).fetchone()
    con.close()
    assert row == (20150101, "2015-01", 1, 1050000.0)


def test_star_join_answers_a_business_question(built):
    path, _ = built
    con = connect(path)
    rows = con.sql(
        """
        SELECT m.line_id, d.year_month, sum(f.failure_cost) AS cost
        FROM fact_failure f
        JOIN dim_machine m USING (machine_id)
        JOIN dim_date d USING (date_key)
        GROUP BY 1, 2
        """
    ).fetchall()
    con.close()
    assert rows == [("PUNE-L1", "2015-01", 1050000.0)]


@pytest.mark.parametrize(
    ("table", "values"),
    [
        ("fact_failure", "99, 20150101, TIMESTAMP '2015-01-01', 'comp1', 1, 8, 1"),
        ("fact_error", "1, 20150101, TIMESTAMP '2015-01-01', 'error42'"),
        ("fact_maintenance", "1, 20990101, TIMESTAMP '2099-01-01', 'comp1', 'planned', 2, 1"),
        ("fact_maintenance", "1, 20150101, TIMESTAMP '2015-01-01', 'comp1', 'guess', 2, 1"),
    ],
    ids=["unknown machine", "unknown error", "unknown date", "bad maintenance type"],
)
def test_fact_constraints_reject_bad_rows(built, table, values):
    path, _ = built
    con = duckdb.connect(str(path))
    with pytest.raises(duckdb.ConstraintException):
        con.execute(f"INSERT INTO {table} VALUES ({values})")
    con.close()


def test_missing_gold_kpis_raise(silver, tmp_path):
    (silver.parent / "gold" / "kpi_machine_monthly.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="iiot gold build"):
        build(silver, tmp_path / "w.duckdb")
