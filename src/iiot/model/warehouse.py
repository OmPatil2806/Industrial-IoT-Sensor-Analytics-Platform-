"""Build the DuckDB star-schema warehouse: data/warehouse/iiot.duckdb.

The tables are defined in plain SQL files in `iiot/model/sql/`, run in file-name
order. Before running them, the builder exposes the inputs to SQL as `src_*`
views, so the SQL never needs file paths:

    src_machines          Silver machines (model, age, plant, line)
    src_component_costs   Silver component costs
    src_sensors           sensor names and valid ranges from config
    src_error_types       allowed error IDs from the raw data contract
    src_dates             every calendar day that appears in the data

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
SILVER_SOURCES = {"src_machines": "machines", "src_component_costs": "component_costs"}
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


def _sql_literal(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot silver build` first.")
    return path


def register_sources(con: duckdb.DuckDBPyConnection, silver_dir: Path) -> None:
    """Expose every input to the SQL scripts as a `src_*` view."""
    settings = get_settings()
    for view, table in SILVER_SOURCES.items():
        path = _require(silver_dir / f"{table}.parquet")
        con.execute(f"CREATE TEMP VIEW {view} AS SELECT * FROM read_parquet({_sql_literal(path)})")

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
        path = _sql_literal(_require(silver_dir / f"{table}.parquet"))
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


def build_warehouse(
    silver_dir: Path | None = None,
    path: Path | None = None,
    sql_dir: Path = SQL_DIR,
) -> dict[str, int]:
    """Run every SQL script into a fresh warehouse file. Returns rows per table."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    path = Path(path or warehouse_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.building")
    for leftover in (tmp, tmp.with_name(tmp.name + ".wal")):
        leftover.unlink(missing_ok=True)

    try:
        with duckdb.connect(str(tmp)) as con:
            register_sources(con, silver_dir)
            for script in sql_scripts(sql_dir):
                try:
                    con.execute(script.read_text(encoding="utf-8"))
                except duckdb.Error as e:
                    raise RuntimeError(f"{script.name} failed: {e}") from e
                logger.debug("Ran %s", script.name)
            counts = table_counts(con)
        try:
            os.replace(tmp, path)
        except PermissionError as e:
            raise WarehouseLockedError(
                f"Cannot replace {path}: it is open in another program "
                "(e.g. DBeaver or a notebook). Close or disconnect it and run the build again."
            ) from e
    finally:
        tmp.unlink(missing_ok=True)

    logger.info("Warehouse built: %d tables -> %s", len(counts), path)
    for name, rows in counts.items():
        logger.info("  %-20s %10s rows", name, f"{rows:,}")
    return counts
