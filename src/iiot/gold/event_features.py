"""Gold event and machine features, on the same rows as the sensor features.

For every (machine_id, timestamp) row of data/gold/sensor_features.parquet:

    error<N>_count_<W>h            number of errors of each type (error1..error5)
                                   in the last W hours (W = longest gold window,
                                   24 h by default): events with  t - W < time <= t
    errors_total_<W>h              all error types together
    hours_since_<comp>_replaced    hours since the component (comp1..comp4) was
                                   last replaced (maintenance at or before t);
                                   empty if it was never replaced
    model, age, plant_id, line_id  machine attributes from Silver

Like the sensor features, these only use events at or before the row's time,
so they cannot leak information about a future failure. Event times that are
not on a whole hour are rounded UP, so an event is never visible before it
actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import get_settings
from iiot.ingestion.validate_raw import CONTRACTS
from iiot.utils.io import write_parquet_atomic
from iiot.utils.logger import get_logger

logger = get_logger("iiot.gold.event_features")

KEYS = ["machine_id", "timestamp"]
OUTPUT_TABLE = "event_features"
MACHINE_COLUMNS = ["model", "age", "plant_id", "line_id"]


@dataclass
class EventFeatureStats:
    rows: int = 0
    features: int = 0
    errors_used: int = 0
    replacements_used: int = 0
    never_replaced: dict[str, int] = field(default_factory=dict)


def error_types() -> list[str]:
    return sorted(CONTRACTS["errors"].categories["errorID"])


def count_recent_events(spine: pd.DataFrame, events: pd.DataFrame, window_hours: int) -> np.ndarray:
    """For each spine row, count events of the same machine with t - window < time <= t."""
    counts = np.zeros(len(spine), dtype=np.int32)
    window = np.timedelta64(window_hours, "h")
    times_by_machine = {
        m: np.sort(g["timestamp"].to_numpy()) for m, g in events.groupby("machine_id")
    }
    for machine, rows in spine.groupby("machine_id").indices.items():
        times = times_by_machine.get(machine)
        if times is None:
            continue
        t = spine["timestamp"].to_numpy()[rows]
        counts[rows] = np.searchsorted(times, t, side="right") - np.searchsorted(
            times, t - window, side="right"
        )
    return counts


def hours_since_last(spine: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """Hours since the latest event of the same machine at or before each spine time."""
    # Same time precision on both sides: merge_asof refuses to match e.g. [us] with [ns].
    left = spine[KEYS].reset_index()
    left["timestamp"] = left["timestamp"].astype("datetime64[ns]")
    left = left.sort_values("timestamp")
    right = events[KEYS].rename(columns={"timestamp": "last_time"})
    right["last_time"] = right["last_time"].astype("datetime64[ns]")
    right = right.sort_values("last_time")
    merged = (
        pd.merge_asof(
            left,
            right,
            left_on="timestamp",
            right_on="last_time",
            by="machine_id",
            direction="backward",
            allow_exact_matches=True,
        )
        .set_index("index")
        .sort_index()
    )
    return ((merged["timestamp"] - merged["last_time"]) / pd.Timedelta(hours=1)).to_numpy()


def build_event_features(
    spine: pd.DataFrame,
    errors: pd.DataFrame,
    maintenance: pd.DataFrame,
    machines: pd.DataFrame,
    window_hours: int,
    components: list[str],
) -> tuple[pd.DataFrame, EventFeatureStats]:
    """Event and machine features for every (machine_id, timestamp) row of `spine`."""
    spine = spine[KEYS].reset_index(drop=True)
    stats = EventFeatureStats(rows=len(spine))
    errors = errors.assign(timestamp=errors["timestamp"].dt.ceil("h"))
    maintenance = maintenance.assign(timestamp=maintenance["timestamp"].dt.ceil("h"))
    out = spine.copy()

    for error_id in error_types():
        out[f"{error_id}_count_{window_hours}h"] = count_recent_events(
            spine, errors[errors["error_id"] == error_id], window_hours
        )
    count_cols = [c for c in out.columns if c.endswith(f"_count_{window_hours}h")]
    out[f"errors_total_{window_hours}h"] = out[count_cols].sum(axis=1).astype(np.int32)

    for comp in components:
        hours = hours_since_last(spine, maintenance[maintenance["component"] == comp])
        out[f"hours_since_{comp}_replaced"] = hours.astype(np.float32)
        stats.never_replaced[comp] = int(np.isnan(hours).sum())

    attributes = machines[["machine_id", *MACHINE_COLUMNS]]
    out = out.merge(attributes, on="machine_id", how="left", validate="many_to_one")
    for col in ("model", "plant_id", "line_id"):
        out[col] = out[col].astype("category")

    stats.features = out.shape[1] - len(KEYS)
    stats.errors_used = len(errors)
    stats.replacements_used = len(maintenance)
    return out, stats


def build(
    silver_dir: Path | None = None, gold_dir: Path | None = None
) -> tuple[pd.DataFrame, EventFeatureStats]:
    """Silver events + Gold sensor-feature rows -> data/gold/event_features.parquet."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    gold_dir = Path(gold_dir or settings.paths.gold)
    spine_path = gold_dir / "sensor_features.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(f"{spine_path} not found. Build the sensor features first.")
    tables = {}
    for name in ("errors", "maintenance", "machines"):
        path = silver_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `iiot silver build` first.")
        tables[name] = pd.read_parquet(path)

    features, stats = build_event_features(
        pd.read_parquet(spine_path, columns=KEYS),
        tables["errors"],
        tables["maintenance"],
        tables["machines"],
        window_hours=max(settings.gold.window_hours),
        components=list(settings.business.components),
    )
    out = write_parquet_atomic(features, gold_dir / f"{OUTPUT_TABLE}.parquet")
    logger.info(
        "Event features: %s rows, %d features (from %s errors, %s replacements) -> %s",
        f"{stats.rows:,}",
        stats.features,
        f"{stats.errors_used:,}",
        f"{stats.replacements_used:,}",
        out,
    )
    never = {c: n for c, n in stats.never_replaced.items() if n}
    if never:
        logger.info("  rows with no earlier replacement (left empty): %s", never)
    return features, stats
