"""Verify the Bronze layer with SQL (DuckDB) and write bronze_report.json.

For every registered table the report checks, using DuckDB queries on the
Parquet files:
    exists          the Parquet file is present
    readable        DuckDB can query it (not corrupted)
    rows_match_log  row count equals the count recorded at ingestion
    columns_match   columns equal the source columns + lineage metadata
    single_batch    all rows come from one ingestion batch
It also reports, without failing:
    stale           the raw source file changed since it was ingested
                    (re-run the ingestion to pick up the change)
    empty_cells     empty values per source column (Bronze keeps them as "")
    compression     CSV size / Parquet size

Usage (Python):
    from iiot.bronze.report import build_report
    build_report().passed
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from iiot.bronze.ingest import METADATA_COLUMNS
from iiot.bronze.pipeline import bronze_tables, file_sha256, read_log
from iiot.config import get_settings
from iiot.utils.logger import get_logger

logger = get_logger("iiot.bronze.report")

REPORT_FILE_NAME = "bronze_report.json"
CHECKS = ("exists", "readable", "rows_match_log", "columns_match", "single_batch")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@dataclass
class TableReport:
    table: str
    checks: dict[str, bool] = field(default_factory=dict)
    rows: int | None = None
    columns: list[str] = field(default_factory=list)
    empty_cells: dict[str, int] = field(default_factory=dict)
    batch_id: str | None = None
    ingested_at: str | None = None
    csv_bytes: int | None = None
    parquet_bytes: int | None = None
    compression_ratio: float | None = None
    stale: bool | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return all(self.checks.get(c, False) for c in CHECKS)


@dataclass
class BronzeReport:
    generated_at: str
    tables: list[TableReport]

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.tables)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "passed": self.passed,
            "tables": {t.table: {**asdict(t), "passed": t.passed} for t in self.tables},
        }


def _check_table(
    con: duckdb.DuckDBPyConnection, table: str, bronze_dir: Path, raw_dir: Path, log: dict
) -> TableReport:
    report = TableReport(table=table)
    path = bronze_dir / f"{table}.parquet"
    entry = log["tables"].get(table)

    report.checks["exists"] = path.exists()
    if not report.checks["exists"]:
        report.error = f"{path.name} not found - run the Bronze ingestion"
        return report

    try:
        rows, batches, batch_id, ingested_at = con.execute(
            "SELECT count(*), count(DISTINCT _batch_id), min(_batch_id), "
            "CAST(min(_ingested_at) AS VARCHAR) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
        columns = [
            r[0]
            for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
        ]
        source_columns = [c for c in columns if c not in METADATA_COLUMNS]
        if source_columns:
            sums = ", ".join(f"count(*) FILTER (WHERE {_quote(c)} = '')" for c in source_columns)
            empties = con.execute(f"SELECT {sums} FROM read_parquet(?)", [str(path)]).fetchone()
            report.empty_cells = {c: int(n) for c, n in zip(source_columns, empties, strict=True)}
    except duckdb.Error as e:
        report.checks["readable"] = False
        report.error = f"DuckDB could not read {path.name}: {e}"
        return report

    report.checks["readable"] = True
    report.rows, report.columns = int(rows), columns
    report.batch_id, report.ingested_at = batch_id, ingested_at
    report.checks["single_batch"] = batches == 1
    report.parquet_bytes = path.stat().st_size

    if entry is None:
        report.checks["rows_match_log"] = report.checks["columns_match"] = False
        report.error = "table not recorded in the ingestion log"
        return report

    report.checks["rows_match_log"] = report.rows == entry["rows"]
    report.checks["columns_match"] = columns == entry["columns"] + list(METADATA_COLUMNS)
    report.csv_bytes = entry["csv_bytes"]
    report.compression_ratio = round(entry["csv_bytes"] / report.parquet_bytes, 2)
    source = raw_dir / entry["source_file"]
    report.stale = not source.exists() or file_sha256(source) != entry["source_sha256"]
    return report


def build_report(
    bronze_dir: Path | None = None,
    raw_dir: Path | None = None,
    tables: list[str] | None = None,
    write: bool = True,
) -> BronzeReport:
    """Check every Bronze table, log a summary, optionally write bronze_report.json."""
    settings = get_settings()
    bronze_dir = Path(bronze_dir or settings.paths.bronze)
    raw_dir = Path(raw_dir or settings.paths.raw)
    tables = tables or list(bronze_tables(settings))
    log = read_log(bronze_dir)

    with duckdb.connect() as con:
        results = [_check_table(con, t, bronze_dir, raw_dir, log) for t in tables]
    report = BronzeReport(generated_at=datetime.now(UTC).isoformat(), tables=results)

    for t in results:
        if t.passed:
            empties = sum(t.empty_cells.values())
            logger.info(
                "PASS %-18s %10s rows  %5.1fx compression  %s empty cells%s",
                t.table,
                f"{t.rows:,}",
                t.compression_ratio,
                f"{empties:,}",
                "  (STALE: source changed, re-ingest)" if t.stale else "",
            )
        else:
            failed = [c for c in CHECKS if not t.checks.get(c, False)]
            logger.error("FAIL %-18s failed: %s  %s", t.table, ", ".join(failed), t.error or "")
    summary = f"{sum(t.passed for t in results)}/{len(results)} tables passed"
    (logger.info if report.passed else logger.error)("Bronze report: %s", summary)

    if write:
        bronze_dir.mkdir(parents=True, exist_ok=True)
        out = bronze_dir / REPORT_FILE_NAME
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info("Report written to %s", out)
    return report
