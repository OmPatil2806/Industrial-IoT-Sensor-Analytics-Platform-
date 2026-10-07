"""Load one raw CSV file into the Bronze layer.

Bronze rules:
    - Every source column is stored as TEXT, exactly as written in the CSV.
      Nothing is parsed, converted, cleaned or dropped (empty cells stay "").
      Typing and cleaning happen in the Silver layer, where problems are counted.
    - Lineage metadata is added to every row:
        _source_file   name of the CSV file
        _source_line   line number in that file (the header is line 1)
        _ingested_at   UTC time of the load
        _batch_id      identifier shared by all files loaded in the same run
    - Row counts are reconciled: the number of data lines in the CSV must equal
      the number of rows in the Parquet file, otherwise the load fails.
    - The Parquet file is written to a temporary name first and then renamed,
      so a failed load never leaves a half-written table behind.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from iiot.config import get_settings
from iiot.utils.logger import get_logger

logger = get_logger("iiot.bronze.ingest")

METADATA_COLUMNS = ("_source_file", "_source_line", "_ingested_at", "_batch_id")
COMPRESSION = "zstd"


class BronzeError(RuntimeError):
    """Raised when a file cannot be loaded into Bronze correctly."""


@dataclass(frozen=True)
class IngestResult:
    table: str
    source: Path
    output: Path
    rows: int
    columns: tuple[str, ...]
    csv_bytes: int
    parquet_bytes: int
    batch_id: str
    ingested_at: datetime


def new_batch_id(now: datetime | None = None) -> str:
    """Readable, unique batch id, e.g. 20261007T101502Z-3f9a1c."""
    now = now or datetime.now(UTC)
    return f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def count_data_lines(path: Path) -> int:
    """Count data rows in a CSV independently of pandas: all lines minus the header."""
    lines = 0
    last = b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":  # final line without a trailing newline
        lines += 1
    return max(lines - 1, 0)


def ingest_file(
    source: Path,
    table: str,
    output_dir: Path | None = None,
    batch_id: str | None = None,
    ingested_at: datetime | None = None,
) -> IngestResult:
    """Load `source` (CSV) into `<output_dir>/<table>.parquet` and return a summary."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    output_dir = Path(output_dir or get_settings().paths.bronze)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = batch_id or new_batch_id()
    ingested_at = ingested_at or datetime.now(UTC)

    # Text only, no NA conversion: values are kept exactly as received.
    df = pd.read_csv(source, dtype=str, keep_default_na=False, na_filter=False)
    if any(c.startswith("_") for c in df.columns):
        raise BronzeError(f"{source.name}: source column names must not start with '_'")
    source_columns = tuple(df.columns)

    df["_source_file"] = source.name
    df["_source_line"] = pd.RangeIndex(2, len(df) + 2)
    df["_ingested_at"] = pd.Timestamp(ingested_at)
    df["_batch_id"] = batch_id

    schema = pa.schema(
        [pa.field(c, pa.string()) for c in source_columns]
        + [
            pa.field("_source_file", pa.string()),
            pa.field("_source_line", pa.int64()),
            pa.field("_ingested_at", pa.timestamp("us", tz="UTC")),
            pa.field("_batch_id", pa.string()),
        ]
    )
    arrow_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    output = output_dir / f"{table}.parquet"
    tmp = output.with_name(f".{output.name}.tmp")
    try:
        pq.write_table(arrow_table, tmp, compression=COMPRESSION)
        csv_rows = count_data_lines(source)
        parquet_rows = pq.ParquetFile(tmp).metadata.num_rows
        if not csv_rows == len(df) == parquet_rows:
            raise BronzeError(
                f"{source.name}: row count mismatch (CSV lines {csv_rows:,}, "
                f"read {len(df):,}, Parquet {parquet_rows:,})"
            )
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)

    result = IngestResult(
        table=table,
        source=source,
        output=output,
        rows=parquet_rows,
        columns=source_columns,
        csv_bytes=source.stat().st_size,
        parquet_bytes=output.stat().st_size,
        batch_id=batch_id,
        ingested_at=ingested_at,
    )
    logger.info(
        "%-18s %10s rows  %8.1f MB CSV -> %6.1f MB Parquet  (reconciled)",
        table,
        f"{result.rows:,}",
        result.csv_bytes / 1_048_576,
        result.parquet_bytes / 1_048_576,
    )
    return result
