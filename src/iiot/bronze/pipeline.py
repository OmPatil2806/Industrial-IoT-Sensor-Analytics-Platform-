"""Load every raw source into Bronze as one batch, skipping unchanged files.

Usage (Python):
    from iiot.bronze.pipeline import ingest_all
    ingest_all()             # load new or changed files only
    ingest_all(force=True)   # reload everything

Idempotency: the SHA-256 hash of each source file is recorded in
data/bronze/_ingestion_log.json. A file whose hash matches the last load (and
whose Parquet table still exists) is skipped, so re-running never duplicates
or needlessly rewrites data.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from iiot.bronze.ingest import IngestResult, ingest_file, new_batch_id
from iiot.config import Settings, get_settings
from iiot.ingestion.inject_issues import DIRTY_FILE_NAME
from iiot.ingestion.reference_data import COMPONENT_COSTS_FILE, MACHINE_LOCATION_FILE
from iiot.utils.logger import get_logger

logger = get_logger("iiot.bronze.pipeline")

LOG_FILE_NAME = "_ingestion_log.json"
MAX_HISTORY = 50


def bronze_tables(settings: Settings | None = None) -> dict[str, str]:
    """Registry: Bronze table name -> raw source file name (in data/raw/).

    Telemetry comes from the dirty copy: in this project it plays the role of the
    real sensor feed. The clean original stays in raw/ as ground truth.
    """
    files = (settings or get_settings()).dataset.files
    return {
        "telemetry": DIRTY_FILE_NAME,
        "errors": files["errors"],
        "maintenance": files["maintenance"],
        "failures": files["failures"],
        "machines": files["machines"],
        "machine_location": MACHINE_LOCATION_FILE,
        "component_costs": COMPONENT_COSTS_FILE,
    }


@dataclass
class BatchResult:
    batch_id: str
    started_at: datetime
    loaded: list[IngestResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_log(bronze_dir: Path) -> dict:
    path = bronze_dir / LOG_FILE_NAME
    if not path.exists():
        return {"tables": {}, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_log(bronze_dir: Path, log: dict) -> None:
    path = bronze_dir / LOG_FILE_NAME
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(log, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def ingest_all(
    force: bool = False,
    raw_dir: Path | None = None,
    bronze_dir: Path | None = None,
    tables: dict[str, str] | None = None,
) -> BatchResult:
    """Load all registered sources into Bronze; return what was loaded and skipped."""
    settings = get_settings()
    raw_dir = Path(raw_dir or settings.paths.raw)
    bronze_dir = Path(bronze_dir or settings.paths.bronze)
    tables = tables or bronze_tables(settings)
    bronze_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast, before loading anything, if any source is missing.
    missing = [name for name in tables.values() if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing source files in {raw_dir}: {', '.join(missing)}. "
            "Run `iiot data prepare` first."
        )

    started_at = datetime.now(UTC)
    batch = BatchResult(batch_id=new_batch_id(started_at), started_at=started_at)
    log = read_log(bronze_dir)
    logger.info("Bronze batch %s: %d tables (force=%s)", batch.batch_id, len(tables), force)

    for table, file_name in tables.items():
        source = raw_dir / file_name
        sha = file_sha256(source)
        previous = log["tables"].get(table)
        unchanged = (
            previous is not None
            and previous["source_file"] == file_name
            and previous["source_sha256"] == sha
            and (bronze_dir / f"{table}.parquet").exists()
        )
        if unchanged and not force:
            logger.info("%-18s unchanged since batch %s, skipped", table, previous["batch_id"])
            batch.skipped.append(table)
            continue

        result = ingest_file(source, table, bronze_dir, batch.batch_id, started_at)
        batch.loaded.append(result)
        log["tables"][table] = {
            "source_file": file_name,
            "source_sha256": sha,
            "rows": result.rows,
            "columns": list(result.columns),
            "csv_bytes": result.csv_bytes,
            "parquet_bytes": result.parquet_bytes,
            "batch_id": batch.batch_id,
            "ingested_at": started_at.isoformat(),
        }
        _write_log(bronze_dir, log)  # after every table, so a later failure keeps progress

    log["history"] = (
        log["history"]
        + [
            {
                "batch_id": batch.batch_id,
                "started_at": started_at.isoformat(),
                "force": force,
                "loaded": [r.table for r in batch.loaded],
                "skipped": batch.skipped,
            }
        ]
    )[-MAX_HISTORY:]
    _write_log(bronze_dir, log)

    logger.info(
        "Bronze batch %s done: %d loaded, %d skipped (%s rows loaded)",
        batch.batch_id,
        len(batch.loaded),
        len(batch.skipped),
        f"{sum(r.rows for r in batch.loaded):,}",
    )
    return batch
