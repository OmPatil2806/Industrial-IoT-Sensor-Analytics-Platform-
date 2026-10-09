"""Build the DuckDB star-schema warehouse: data/warehouse/iiot.duckdb.

The tables are defined in plain SQL files in `iiot/model/sql/`, run in file-name
order. Before running them, the builder exposes the inputs to SQL as `src_*`
views, so the SQL never needs file paths:

    src_machines          Silver machines (model, age, plant, line)
    src_component_costs   Silver component costs
    src_telemetry         Silver cleaned telemetry (with quality flags)
    src_failures, src_maintenance, src_errors   Silver events
    src_kpi_machine_monthly                     Gold KPIs per machine and month
    src_sensors           sensor names and valid ranges from config
    src_error_types       allowed error IDs from the raw data contract
    src_dates             every calendar day that appears in the data

Scripts 01-05 create the dimension tables, 06-10 the fact tables and 11+ the
analytical views (v_*), which are saved queries over the tables and cost no space.

The database is built in a temporary file and then swapped in, so a failed build
never leaves a half-built warehouse. If another program (e.g. DBeaver) has the
warehouse open, the swap fails with a clear message to close it.

Usage:
    from iiot.model.warehouse import build_warehouse, connect
    build_warehouse()
    con = connect(read_only=True)
    con.sql("SELECT * FROM dim_machine LIMIT 5").show()
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from iiot.config import get_settings
from iiot.ingestion.validate_raw import CONTRACTS
from iiot.utils.logger import get_logger

logger = get_logger("iiot.model.warehouse")

SQL_DIR = Path(__file__).parent / "sql"
WAREHOUSE_FILE_NAME = "iiot.duckdb"
SILVER_SOURCES = {
    "src_machines": "machines",
    "src_component_costs": "component_costs",
    "src_telemetry": "telemetry",
    "src_failures": "failures",
    "src_maintenance": "maintenance",
    "src_errors": "errors",
}
GOLD_SOURCES = {"src_kpi_machine_monthly": "kpi_machine_monthly"}
DATE_SOURCES = ("telemetry", "errors", "maintenance", "failures")


class WarehouseLockedError(RuntimeError):
    """The warehouse file is open in another program and cannot be replaced."""


def warehouse_path() -> Path:
    return get_settings().paths.warehouse / WAREHOUSE_FILE_NAME


def connect(path: Path | None = None, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the warehouse. Read-only by default, which never blocks a rebuild for long."""
    path = Path(path or warehouse_path())
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot model build` first.")
    return duckdb.connect(str(path), read_only=read_only)


def sql_literal(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def require_file(path: Path, layer: str = "silver") -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot {layer} build` first.")
    return path


def register_sources(con: duckdb.DuckDBPyConnection, silver_dir: Path, gold_dir: Path) -> None:
    """Expose every input to the SQL scripts as a `src_*` view."""
    settings = get_settings()
    for sources, folder, layer in (
        (SILVER_SOURCES, silver_dir, "silver"),
        (GOLD_SOURCES, gold_dir, "gold"),
    ):
        for view, table in sources.items():
            path = require_file(folder / f"{table}.parquet", layer)
            con.execute(
                f"CREATE TEMP VIEW {view} AS SELECT * FROM read_parquet({sql_literal(path)})"
            )

    sensors = pd.DataFrame(
        [
            {"sensor": name, "valid_min": limit.min, "valid_max": limit.max}
            for name, limit in settings.sensors.items()
        ]
    )
    con.register("src_sensors", sensors)
    errors = sorted(CONTRACTS["errors"].categories["errorID"])
    con.register("src_error_types", pd.DataFrame({"error_id": errors}))

    bounds = []
    for table in DATE_SOURCES:
        path = sql_literal(require_file(silver_dir / f"{table}.parquet"))
        bounds.append(
            f"SELECT min(timestamp) AS lo, max(timestamp) AS hi FROM read_parquet({path})"
        )
    lo, hi = con.execute(f"SELECT min(lo), max(hi) FROM ({' UNION ALL '.join(bounds)})").fetchone()
    dates = pd.date_range(pd.Timestamp(lo).normalize(), pd.Timestamp(hi).normalize(), freq="D")
    con.register("src_dates", pd.DataFrame({"date": dates}))


def sql_scripts(sql_dir: Path = SQL_DIR) -> list[Path]:
    return sorted(sql_dir.glob("*.sql"))


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    names = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE NOT temporary ORDER BY table_name"
        ).fetchall()
    ]
    return {n: con.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0] for n in names}


def view_names(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "SELECT view_name FROM duckdb_views() WHERE NOT internal AND NOT temporary "
            "ORDER BY view_name"
        ).fetchall()
    ]


def build_warehouse(
    silver_dir: Path | None = None,
    path: Path | None = None,
    sql_dir: Path = SQL_DIR,
    gold_dir: Path | None = None,
) -> dict[str, int]:
    """Run every SQL script into a fresh warehouse file. Returns rows per table."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    gold_dir = Path(gold_dir or settings.paths.gold)
    path = Path(path or warehouse_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.building")
    for leftover in (tmp, tmp.with_name(tmp.name + ".wal")):
        leftover.unlink(missing_ok=True)

    try:
        with duckdb.connect(str(tmp)) as con:
            register_sources(con, silver_dir, gold_dir)
            for script in sql_scripts(sql_dir):
                try:
                    con.execute(script.read_text(encoding="utf-8"))
                except duckdb.Error as e:
                    raise RuntimeError(f"{script.name} failed: {e}") from e
                logger.debug("Ran %s", script.name)
            counts = table_counts(con)
            views = view_names(con)
        try:
            os.replace(tmp, path)
        except PermissionError as e:
            raise WarehouseLockedError(
                f"Cannot replace {path}: it is open in another program "
                "(e.g. DBeaver or a notebook). Close or disconnect it and run the build again."
            ) from e
    finally:
        tmp.unlink(missing_ok=True)

    logger.info("Warehouse built: %d tables, %d views -> %s", len(counts), len(views), path)
    for name, rows in counts.items():
        logger.info("  %-24s %10s rows", name, f"{rows:,}")
    for name in views:
        logger.info("  %-24s %10s", name, "view")
    return counts
