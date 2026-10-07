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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import get_settings
from iiot.ingestion.validate_raw import parse_datetimes
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
