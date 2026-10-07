"""Gold sensor features: rolling statistics that look BACKWARDS only.

For every machine, one feature row every `gold.feature_step_hours` hours. Each
row at time t only uses readings from t and the hours before it, never after,
so the features can never leak information about a future failure.

Per sensor (volt, rotate, pressure, vibration) and per window w in
`gold.window_hours` (default 3 h and 24 h):
    <sensor>_mean_<w>h    average of the readings in the last w hours
    <sensor>_std_<w>h     spread of the readings in the last w hours
Plus, per sensor:
    <sensor>_trend        mean over the shortest window minus mean over the
                          longest window (positive = rising recently)
    <sensor>_filled_share_<W>h   share of readings in the longest window that
    <sensor>_missing_share_<W>h  Silver filled / could not fill (data quality)

A statistic needs at least half of its window (and 2 readings) to be present,
otherwise it is left empty. Rows in a machine's first (longest window - 1)
hours are dropped as warm-up, because their longest window is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import Gold, get_settings
from iiot.utils.io import write_parquet_atomic
from iiot.utils.logger import get_logger

logger = get_logger("iiot.gold.sensor_features")

KEYS = ["machine_id", "timestamp"]
OUTPUT_TABLE = "sensor_features"


@dataclass
class SensorFeatureStats:
    rows_in: int = 0
    warmup_rows_dropped: int = 0
    rows_out: int = 0
    features: int = 0


def feature_names(sensors: list[str], windows: tuple[int, ...]) -> list[str]:
    longest = max(windows)
    names = []
    for s in sensors:
        for w in windows:
            names += [f"{s}_mean_{w}h", f"{s}_std_{w}h"]
        names += [f"{s}_trend", f"{s}_filled_share_{longest}h", f"{s}_missing_share_{longest}h"]
    return names


def _rolling(grouped, window: int, how: str) -> pd.Series:
    """Backward-looking rolling statistic per machine (window includes the current hour)."""
    min_periods = max(2, window // 2)
    rolled = getattr(grouped.rolling(window, min_periods=min_periods), how)()
    return rolled.reset_index(level=0, drop=True)


def build_sensor_features(
    silver: pd.DataFrame, sensors: list[str], gold: Gold
) -> tuple[pd.DataFrame, SensorFeatureStats]:
    """Compute rolling features on the hourly Silver telemetry, then keep every Nth hour."""
    stats = SensorFeatureStats(rows_in=len(silver))
    df = silver.sort_values(KEYS).reset_index(drop=True)
    machine = df["machine_id"]
    windows = tuple(gold.window_hours)
    shortest, longest = min(windows), max(windows)

    out = df[KEYS].copy()
    for s in sensors:
        values = df[s].groupby(machine)
        for w in windows:
            out[f"{s}_mean_{w}h"] = _rolling(values, w, "mean")
            out[f"{s}_std_{w}h"] = _rolling(values, w, "std")
        out[f"{s}_trend"] = out[f"{s}_mean_{shortest}h"] - out[f"{s}_mean_{longest}h"]
        quality = df[f"{s}_quality"].astype(str)
        for flag in ("filled", "missing"):
            share = (quality == flag).astype(float).groupby(machine)
            out[f"{s}_{flag}_share_{longest}h"] = (
                share.rolling(longest, min_periods=1).mean().reset_index(level=0, drop=True)
            )

    # Warm-up: the longest window is incomplete for each machine's first hours.
    hours_since_start = df["timestamp"] - df.groupby("machine_id")["timestamp"].transform("min")
    warm = hours_since_start < pd.Timedelta(hours=longest - 1)
    stats.warmup_rows_dropped = int(warm.sum())

    # Keep one row every `feature_step_hours`, aligned to the earliest timestamp overall.
    elapsed = (df["timestamp"] - df["timestamp"].min()) // pd.Timedelta(hours=1)
    on_step = (elapsed % gold.feature_step_hours) == 0
    out = out[~warm & on_step].reset_index(drop=True)

    names = feature_names(sensors, windows)
    out = out[KEYS + names]
    out[names] = out[names].astype(np.float32)  # half the size, ample precision for features
    stats.rows_out, stats.features = len(out), len(names)
    return out, stats


def build(
    silver_dir: Path | None = None, gold_dir: Path | None = None
) -> tuple[pd.DataFrame, SensorFeatureStats]:
    """Silver telemetry -> data/gold/sensor_features.parquet."""
    settings = get_settings()
    path = Path(silver_dir or settings.paths.silver) / "telemetry.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot silver build` first.")
    sensors = list(settings.sensors)
    features, stats = build_sensor_features(pd.read_parquet(path), sensors, settings.gold)
    out = write_parquet_atomic(
        features, Path(gold_dir or settings.paths.gold) / f"{OUTPUT_TABLE}.parquet"
    )
    logger.info(
        "Sensor features: %s hourly rows -> %s rows (every %dh, %s warm-up rows dropped), "
        "%d features -> %s",
        f"{stats.rows_in:,}",
        f"{stats.rows_out:,}",
        settings.gold.feature_step_hours,
        f"{stats.warmup_rows_dropped:,}",
        stats.features,
        out,
    )
    return features, stats
