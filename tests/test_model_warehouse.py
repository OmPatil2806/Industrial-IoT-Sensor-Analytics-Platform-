"""Tests for iiot.model.warehouse (DuckDB star schema: dimension and fact tables).

The `silver` and `built` fixtures (a tiny Silver + Gold layer and its warehouse) are
in conftest.py, shared with test_model_views.py.
"""

from __future__ import annotations

import duckdb
import pytest

from iiot.model import warehouse
from iiot.model.warehouse import WarehouseLockedError, build_warehouse, connect, sql_scripts


def build(silver, path, **kwargs):
    return build_warehouse(silver, path, gold_dir=silver.parent / "gold", **kwargs)


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
        "11_v_fleet_monthly.sql",
        "12_v_machine_health.sql",
        "13_v_component_reliability.sql",
        "14_v_line_performance.sql",
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
