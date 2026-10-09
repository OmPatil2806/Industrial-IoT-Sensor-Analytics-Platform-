"""Check the warehouse and write data/warehouse/model_report.json.

`build_model()` builds the warehouse and checks it; `check_model()` checks an
existing one. The checks:

    unique_keys        no duplicate or empty business key in any table, including
                       fact_sensor_reading, which has no declared primary key
    references         every key in every fact table exists in its dimension
                       (the foreign keys of fact_sensor_reading are not declared,
                       so this is the only thing protecting it)
    date_keys          every fact row's date_key is the date of its timestamp, and
                       dim_date has no missing day
    silver_reconciliation  the facts hold exactly the Silver data: row counts,
                       non-empty readings, sum of values and quality flags per sensor
    gold_reconciliation    fact_machine_month equals the Gold KPIs, and the KPI
                       failure / planned-maintenance events equal the events
                       counted from the fact tables in the analysis period
    views              every expected view exists, runs, returns rows, and its
                       totals equal fact_machine_month

`passed` is true only if every check passes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from iiot.config import get_settings
from iiot.model.warehouse import (
    build_warehouse,
    connect,
    require_file,
    sql_literal,
    table_counts,
    view_names,
    warehouse_path,
)
from iiot.utils.logger import get_logger

logger = get_logger("iiot.model.report")

REPORT_FILE_NAME = "model_report.json"
FLOAT_TOLERANCE = 1e-9  # relative; catches even a 1-rupee change in the cost totals

KEYS = {
    "dim_machine": ["machine_id"],
    "dim_component": ["component"],
    "dim_sensor": ["sensor"],
    "dim_error_type": ["error_id"],
    "dim_date": ["date_key"],
    "fact_sensor_reading": ["machine_id", "timestamp", "sensor"],
    "fact_failure": ["machine_id", "timestamp", "component"],
    "fact_maintenance": ["machine_id", "timestamp", "component"],
    "fact_error": ["machine_id", "timestamp", "error_id"],
    "fact_machine_month": ["machine_id", "year_month"],
}
# (fact table, column) -> (dimension table, column)
REFERENCES = {
    ("fact_sensor_reading", "machine_id"): ("dim_machine", "machine_id"),
    ("fact_sensor_reading", "date_key"): ("dim_date", "date_key"),
    ("fact_sensor_reading", "sensor"): ("dim_sensor", "sensor"),
    ("fact_failure", "machine_id"): ("dim_machine", "machine_id"),
    ("fact_failure", "date_key"): ("dim_date", "date_key"),
    ("fact_failure", "component"): ("dim_component", "component"),
    ("fact_maintenance", "machine_id"): ("dim_machine", "machine_id"),
    ("fact_maintenance", "date_key"): ("dim_date", "date_key"),
    ("fact_maintenance", "component"): ("dim_component", "component"),
    ("fact_error", "machine_id"): ("dim_machine", "machine_id"),
    ("fact_error", "date_key"): ("dim_date", "date_key"),
    ("fact_error", "error_id"): ("dim_error_type", "error_id"),
    ("fact_machine_month", "machine_id"): ("dim_machine", "machine_id"),
    ("fact_machine_month", "month_date_key"): ("dim_date", "date_key"),
}
EVENT_TABLES = {
    "fact_failure": "failures",
    "fact_maintenance": "maintenance",
    "fact_error": "errors",
}
EXPECTED_VIEWS = [
    "v_component_reliability",
    "v_fleet_monthly",
    "v_line_performance",
    "v_machine_health",
]
VIEWS_WITH_TOTALS = ["v_fleet_monthly", "v_machine_health", "v_line_performance"]
KPI_COLUMNS = ["failures", "components_failed", "planned_maintenances", "downtime_h", "total_cost"]


def _one(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()


def _parquet(folder: Path, table: str, layer: str = "silver") -> str:
    return f"read_parquet({sql_literal(require_file(folder / f'{table}.parquet', layer))})"


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= FLOAT_TOLERANCE * max(1.0, abs(a), abs(b))


def check_unique_keys(con: duckdb.DuckDBPyConnection) -> dict:
    problems = {}
    for table, cols in KEYS.items():
        key = ", ".join(cols)
        not_null = " AND ".join(f"{c} IS NOT NULL" for c in cols)
        rows, distinct, complete = _one(
            con,
            f"SELECT count(*), count(DISTINCT ({key})), count(*) FILTER (WHERE {not_null}) "
            f"FROM {table}",
        )
        duplicates, empty = rows - distinct, rows - complete
        if duplicates or empty:
            problems[table] = {"duplicate_keys": duplicates, "empty_keys": empty}
    return {"passed": not problems, "tables_checked": len(KEYS), "problems": problems}


def check_references(con: duckdb.DuckDBPyConnection) -> dict:
    orphans = {}
    for (fact, col), (dim, dim_col) in REFERENCES.items():
        (n,) = _one(
            con,
            f"SELECT count(*) FROM {fact} AS f "
            f"WHERE NOT EXISTS (SELECT 1 FROM {dim} AS d WHERE d.{dim_col} = f.{col})",
        )
        if n:
            orphans[f"{fact}.{col} -> {dim}"] = n
    return {"passed": not orphans, "references_checked": len(REFERENCES), "orphans": orphans}


def check_date_keys(con: duckdb.DuckDBPyConnection) -> dict:
    mismatched = {}
    for table in ("fact_sensor_reading", *EVENT_TABLES):
        (n,) = _one(
            con,
            f"SELECT count(*) FROM {table} "
            "WHERE date_key <> CAST(strftime(timestamp, '%Y%m%d') AS INTEGER)",
        )
        if n:
            mismatched[table] = n
    (n,) = _one(
        con,
        "SELECT count(*) FROM fact_machine_month "
        "WHERE month_date_key <> CAST(replace(year_month, '-', '') || '01' AS INTEGER)",
    )
    if n:
        mismatched["fact_machine_month"] = n
    days, first, last = _one(con, "SELECT count(*), min(date), max(date) FROM dim_date")
    missing_days = (last - first).days + 1 - days
    return {
        "passed": not mismatched and missing_days == 0,
        "mismatched_date_keys": mismatched,
        "calendar": f"{first} to {last}",
        "missing_days": missing_days,
    }


def check_silver_reconciliation(con: duckdb.DuckDBPyConnection, silver_dir: Path) -> dict:
    def silver(table: str) -> str:
        return _parquet(silver_dir, table)

    differences = {}
    sensors = [
        r[0] for r in con.execute("SELECT sensor FROM dim_sensor ORDER BY sensor").fetchall()
    ]
    (telemetry_rows,) = _one(con, f"SELECT count(*) FROM {silver('telemetry')}")
    for sensor in sensors:
        expected = _one(
            con,
            f"SELECT count(*), count({sensor}), sum({sensor}), "
            f"count(*) FILTER (WHERE {sensor}_quality = 'ok'), "
            f"count(*) FILTER (WHERE {sensor}_quality = 'filled'), "
            f"count(*) FILTER (WHERE {sensor}_quality = 'missing') "
            f"FROM {silver('telemetry')}",
        )
        actual = _one(
            con,
            "SELECT count(*), count(value), sum(value), "
            "count(*) FILTER (WHERE quality = 'ok'), "
            "count(*) FILTER (WHERE quality = 'filled'), "
            "count(*) FILTER (WHERE quality = 'missing') "
            f"FROM fact_sensor_reading WHERE sensor = '{sensor}'",
        )
        names = ["rows", "values", "sum", "ok", "filled", "missing"]
        for name, e, a in zip(names, expected, actual, strict=True):
            if not _close(e, a):
                differences[f"fact_sensor_reading.{sensor}.{name}"] = {"silver": e, "warehouse": a}

    for fact, table in EVENT_TABLES.items():
        (e,) = _one(con, f"SELECT count(*) FROM {silver(table)}")
        (a,) = _one(con, f"SELECT count(*) FROM {fact}")
        if e != a:
            differences[f"{fact}.rows"] = {"silver": e, "warehouse": a}
    (e,) = _one(con, f"SELECT count(*) FROM {silver('machines')}")
    (a,) = _one(con, "SELECT count(*) FROM dim_machine")
    if e != a:
        differences["dim_machine.rows"] = {"silver": e, "warehouse": a}

    return {
        "passed": not differences,
        "telemetry_hours": telemetry_rows,
        "sensor_readings": telemetry_rows * len(sensors),
        "differences": differences,
    }


def check_gold_reconciliation(con: duckdb.DuckDBPyConnection, gold_dir: Path) -> dict:
    kpis = _parquet(gold_dir, "kpi_machine_monthly", "gold")
    differences = {}
    sums = ", ".join(f"sum({c})" for c in KPI_COLUMNS)
    expected = _one(con, f"SELECT count(*), {sums} FROM {kpis}")
    actual = _one(con, f"SELECT count(*), {sums} FROM fact_machine_month")
    for name, e, a in zip(["rows", *KPI_COLUMNS], expected, actual, strict=True):
        if not _close(e, a):
            differences[f"fact_machine_month.{name}"] = {"gold": e, "warehouse": a}

    # Events counted from the fact tables, in the same period as the Gold KPIs.
    period = (
        "(SELECT min(timestamp) AS period_start, date_trunc('day', max(timestamp)) AS period_end "
        "FROM fact_sensor_reading) AS p"
    )
    in_period = "timestamp >= p.period_start AND timestamp < p.period_end"
    (failure_events,) = _one(
        con,
        f"SELECT count(DISTINCT (machine_id, timestamp)) FROM fact_failure, {period} "
        f"WHERE {in_period}",
    )
    (planned_events,) = _one(
        con,
        f"SELECT count(DISTINCT (machine_id, timestamp)) FROM fact_maintenance, {period} "
        f"WHERE maintenance_type = 'planned' AND {in_period}",
    )
    kpi_failures, kpi_planned = _one(
        con, "SELECT sum(failures), sum(planned_maintenances) FROM fact_machine_month"
    )
    for name, events, kpi in (
        ("failure_events", failure_events, kpi_failures),
        ("planned_maintenance_events", planned_events, kpi_planned),
    ):
        if events != kpi:
            differences[name] = {"fact_tables": events, "fact_machine_month": kpi}

    return {
        "passed": not differences,
        "failure_events": failure_events,
        "planned_maintenance_events": planned_events,
        "differences": differences,
    }


def check_views(con: duckdb.DuckDBPyConnection) -> dict:
    present = view_names(con)
    problems = {name: "missing" for name in EXPECTED_VIEWS if name not in present}
    rows = {}
    for name in present:
        try:
            (rows[name],) = _one(con, f"SELECT count(*) FROM {name}")
        except duckdb.Error as e:
            problems[name] = f"fails: {e}"
            continue
        if rows[name] == 0:
            problems[name] = "returns no rows"
    expected = _one(con, "SELECT sum(failures), sum(total_cost) FROM fact_machine_month")
    for name in VIEWS_WITH_TOTALS:
        if name in problems or name not in present:
            continue
        actual = _one(con, f"SELECT sum(failures), sum(total_cost) FROM {name}")
        if not all(_close(e, a) for e, a in zip(expected, actual, strict=True)):
            problems[name] = f"totals {actual} differ from fact_machine_month {expected}"
    return {"passed": not problems, "rows": rows, "problems": problems}


def check_model(
    path: Path | None = None,
    silver_dir: Path | None = None,
    gold_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run every check on an existing warehouse, write and return the report."""
    settings = get_settings()
    path = Path(path or warehouse_path())
    silver_dir = Path(silver_dir or settings.paths.silver)
    gold_dir = Path(gold_dir or settings.paths.gold)
    out_dir = Path(out_dir or path.parent)

    with connect(path) as con:
        checks = {
            "unique_keys": check_unique_keys(con),
            "references": check_references(con),
            "date_keys": check_date_keys(con),
            "silver_reconciliation": check_silver_reconciliation(con, silver_dir),
            "gold_reconciliation": check_gold_reconciliation(con, gold_dir),
            "views": check_views(con),
        }
        tables = table_counts(con)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "warehouse": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "passed": all(c["passed"] for c in checks.values()),
        "tables": tables,
        "checks": checks,
    }
    out = out_dir / REPORT_FILE_NAME
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log_summary(report, out)
    return report


def build_model(
    silver_dir: Path | None = None,
    gold_dir: Path | None = None,
    path: Path | None = None,
) -> dict:
    """Build the warehouse, check it, write and return the report."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    gold_dir = Path(gold_dir or settings.paths.gold)
    path = Path(path or warehouse_path())
    build_warehouse(silver_dir, path, gold_dir=gold_dir)
    return check_model(path, silver_dir, gold_dir)


def show_report(warehouse_dir: Path | None = None) -> dict:
    """Log the summary of the latest model_report.json and return the report."""
    path = Path(warehouse_dir or get_settings().paths.warehouse) / REPORT_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot model build` first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Report generated at %s", report["generated_at"])
    _log_summary(report, path)
    return report


def _log_summary(report: dict, out: Path) -> None:
    for name, check in report["checks"].items():
        (logger.info if check["passed"] else logger.error)(
            "%-22s %s", name, "PASS" if check["passed"] else "FAIL"
        )
    (logger.info if report["passed"] else logger.error)(
        "Model report: %s -> %s", "PASSED" if report["passed"] else "FAILED", out
    )
