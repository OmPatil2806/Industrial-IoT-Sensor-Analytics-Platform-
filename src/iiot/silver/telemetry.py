"""Silver telemetry, step 1: typing and standardisation.

Bronze stores every value as text exactly as received. This step converts the
telemetry to proper types and consistent names, and counts every value that
could not be converted, so nothing is lost silently:

    Bronze (text)   Silver                 Type
    datetime    ->  timestamp              datetime64[ns]
    machineID   ->  machine_id             Int64 (nullable integer)
    volt ...    ->  volt, rotate,          float64
                    pressure, vibration
    _batch_id, _source_line                kept for lineage

Per column, two kinds of bad values are counted separately:
    empty        the cell was empty in the source ("")
    unparseable  the cell had text that is not a valid value (e.g. "abc", "1.5" as an ID)

Rows without a valid timestamp or machine_id cannot be placed in the time
series and are dropped (and counted). Rows with bad sensor values are kept with
the bad values set to null; later cleaning steps handle those.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import get_settings
from iiot.ingestion.validate_raw import parse_datetimes
from iiot.silver.cleaning import CleaningStats, clean
from iiot.utils.logger import get_logger

logger = get_logger("iiot.silver.telemetry")

RENAMES = {"datetime": "timestamp", "machineID": "machine_id"}
KEY_COLUMNS = ("timestamp", "machine_id")
LINEAGE_COLUMNS = ("_batch_id", "_source_line")


@dataclass
class TypingStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped_no_key: int = 0
    empty: dict[str, int] = field(default_factory=dict)
    unparseable: dict[str, int] = field(default_factory=dict)


def sensor_columns() -> list[str]:
    return list(get_settings().sensors)


def _to_float(text: pd.Series) -> pd.Series:
    return pd.to_numeric(text.replace("", np.nan), errors="coerce").astype("float64")


def _to_machine_id(text: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(text.replace("", np.nan), errors="coerce")
    valid = numbers.notna() & (numbers == numbers.round()) & (numbers > 0)
    return numbers.where(valid).astype("Int64")


def standardise(bronze: pd.DataFrame) -> tuple[pd.DataFrame, TypingStats]:
    """Convert Bronze telemetry (all text) to typed, renamed Silver columns."""
    sensors = sensor_columns()
    missing = [c for c in [*RENAMES, *sensors] if c not in bronze.columns]
    if missing:
        raise ValueError(f"Bronze telemetry is missing columns: {missing}")

    stats = TypingStats(rows_in=len(bronze))
    out = pd.DataFrame(index=bronze.index)
    converters = {
        "datetime": parse_datetimes,
        "machineID": _to_machine_id,
        **dict.fromkeys(sensors, _to_float),
    }
    for source, convert in converters.items():
        text = bronze[source].astype("string").fillna("").str.strip()
        is_empty = text == ""
        values = convert(text)
        name = RENAMES.get(source, source)
        out[name] = values
        stats.empty[name] = int(is_empty.sum())
        stats.unparseable[name] = int((values.isna() & ~is_empty).sum())

    for col in LINEAGE_COLUMNS:
        if col in bronze.columns:
            out[col] = bronze[col].to_numpy()

    has_key = out["timestamp"].notna() & out["machine_id"].notna()
    stats.rows_dropped_no_key = int((~has_key).sum())
    out = out[has_key].reset_index(drop=True)
    stats.rows_out = len(out)
    return out, stats


def load_bronze(bronze_dir: Path | None = None) -> pd.DataFrame:
    path = Path(bronze_dir or get_settings().paths.bronze) / "telemetry.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot bronze ingest` first.")
    return pd.read_parquet(path)


def build(
    bronze_dir: Path | None = None, silver_dir: Path | None = None
) -> tuple[pd.DataFrame, TypingStats, CleaningStats]:
    """Bronze telemetry -> typed -> cleaned -> data/silver/telemetry.parquet."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    typed, typing_stats = standardise(load_bronze(bronze_dir))
    log_typing_stats(typing_stats)
    cleaned, cleaning_stats = clean(typed, settings.sensors, settings.silver)
    log_cleaning_stats(cleaning_stats)

    silver_dir.mkdir(parents=True, exist_ok=True)
    out = silver_dir / "telemetry.parquet"
    tmp = out.with_name(f".{out.name}.tmp")
    try:
        cleaned.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("Wrote %s (%s rows)", out, f"{len(cleaned):,}")
    return cleaned, typing_stats, cleaning_stats


def log_cleaning_stats(stats: CleaningStats) -> None:
    logger.info(
        "Cleaned telemetry: %s rows in -> %s rows out",
        f"{stats.rows_in:,}",
        f"{stats.rows_out:,}",
    )
    logger.info(
        "  duplicates removed %s (conflicting %s), off-hour rows dropped %s, grid rows added %s",
        f"{stats.duplicates_removed:,}",
        f"{stats.conflicting_duplicates:,}",
        f"{stats.rows_off_hour_dropped:,}",
        f"{stats.grid_rows_added:,}",
    )
    for s in stats.out_of_range_nulled:
        logger.info(
            "  %-10s out-of-range %6s  stuck runs %4s (%5s values)  filled %6s  missing %6s",
            s,
            f"{stats.out_of_range_nulled[s]:,}",
            f"{stats.stuck_runs[s]:,}",
            f"{stats.stuck_values_nulled[s]:,}",
            f"{stats.filled[s]:,}",
            f"{stats.missing_after[s]:,}",
        )


def log_typing_stats(stats: TypingStats) -> None:
    logger.info(
        "Typed telemetry: %s rows in -> %s rows out (%s dropped: no valid timestamp/machine_id)",
        f"{stats.rows_in:,}",
        f"{stats.rows_out:,}",
        f"{stats.rows_dropped_no_key:,}",
    )
    for col in stats.empty:
        logger.info(
            "  %-12s empty %8s   unparseable %6s",
            col,
            f"{stats.empty[col]:,}",
            f"{stats.unparseable[col]:,}",
        )
