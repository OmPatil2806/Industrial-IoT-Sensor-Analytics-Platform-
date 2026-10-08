"""Gold labels (what the models must predict) and the ML dataset.

Labels look FORWARD from each feature row at time t, over the prediction
horizon H (`ml.prediction_horizon_hours`, 24 h), using failures with
t < failure time <= t + H:

    fails_within_<H>h         1 if any component fails in the horizon, else 0
    <comp>_fails_within_<H>h  1 if that component fails in the horizon (one per
                              component: two components can fail together)
    failed_component          component(s) of the NEXT failure in the horizon,
                              e.g. "comp2" or "comp2+comp4"; "none" if no failure
    hours_to_failure          hours until the next failure (remaining useful
                              life); empty if no later failure is recorded

Rows whose horizon goes past the end of the data are dropped: we cannot know
whether they fail. (Features look backward, labels look forward, so the two
never overlap.)

ML dataset = sensor features + event features + labels, with a `split` column:
    train   t + H < ml.train_end_date   (the label window ends before the split)
    test    t >= ml.train_end_date
Rows in between are dropped as a gap: their labels would look into the test
period, which would leak test information into training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import get_settings
from iiot.utils.io import write_parquet_atomic
from iiot.utils.logger import get_logger

logger = get_logger("iiot.gold.labels")

KEYS = ["machine_id", "timestamp"]
NO_FAILURE = "none"


@dataclass
class LabelStats:
    rows_in: int = 0
    unobservable_dropped: int = 0
    rows_out: int = 0
    positives: int = 0
    failed_component_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetStats:
    rows: int = 0
    gap_dropped: int = 0
    split_rows: dict[str, int] = field(default_factory=dict)
    split_positive_rate: dict[str, float] = field(default_factory=dict)
    features: int = 0


def _ns(values) -> np.ndarray:
    return pd.to_datetime(pd.Series(values)).to_numpy(dtype="datetime64[ns]")


def build_labels(
    spine: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_hours: int,
    data_end: pd.Timestamp,
    components: list[str],
) -> tuple[pd.DataFrame, LabelStats]:
    """Forward-looking labels for every (machine_id, timestamp) row of `spine`."""
    stats = LabelStats(rows_in=len(spine))
    horizon = np.timedelta64(horizon_hours, "h")
    target = f"fails_within_{horizon_hours}h"

    observable = _ns(spine["timestamp"]) + horizon <= np.datetime64(pd.Timestamp(data_end), "ns")
    stats.unobservable_dropped = int((~observable).sum())
    out = spine.loc[observable, KEYS].reset_index(drop=True)
    t_all = _ns(out["timestamp"])

    hours = np.full(len(out), np.nan)
    next_components = np.full(len(out), NO_FAILURE, dtype=object)
    per_component = {c: np.zeros(len(out), dtype=np.int8) for c in components}

    events = failures.assign(timestamp=_ns(failures["timestamp"]))
    for machine, rows in out.groupby("machine_id").indices.items():
        mine = events[events["machine_id"] == machine]
        if mine.empty:
            continue
        t = t_all[rows]
        grouped = mine.groupby("timestamp")["component"].apply(lambda c: "+".join(sorted(c)))
        event_times = grouped.index.to_numpy(dtype="datetime64[ns]")
        nxt = np.searchsorted(event_times, t, side="right")  # first failure strictly after t
        has_next = nxt < len(event_times)
        hours[rows[has_next]] = (event_times[nxt[has_next]] - t[has_next]) / np.timedelta64(1, "h")
        labels = grouped.to_numpy()
        in_horizon = has_next.copy()
        in_horizon[has_next] = event_times[nxt[has_next]] <= t[has_next] + horizon
        next_components[rows[in_horizon]] = labels[nxt[in_horizon]]
        for comp in components:
            times = np.sort(mine.loc[mine["component"] == comp, "timestamp"].to_numpy())
            count = np.searchsorted(times, t + horizon, side="right") - np.searchsorted(
                times, t, side="right"
            )
            per_component[comp][rows] = (count > 0).astype(np.int8)

    out["hours_to_failure"] = hours.astype(np.float32)
    out[target] = (out["hours_to_failure"] <= horizon_hours).astype(np.int8)
    for comp in components:
        out[f"{comp}_fails_within_{horizon_hours}h"] = per_component[comp]
    out["failed_component"] = pd.Categorical(next_components)

    stats.rows_out = len(out)
    stats.positives = int(out[target].sum())
    stats.failed_component_counts = {
        k: int(v) for k, v in out["failed_component"].value_counts().items()
    }
    return out, stats


def build_ml_dataset(
    sensor: pd.DataFrame,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon_hours: int,
) -> tuple[pd.DataFrame, DatasetStats]:
    """Join features and labels; mark train/test and drop the gap between them."""
    stats = DatasetStats()
    df = sensor.merge(events, on=KEYS, how="inner", validate="one_to_one").merge(
        labels, on=KEYS, how="inner", validate="one_to_one"
    )
    t = _ns(df["timestamp"])
    split_at = np.datetime64(pd.Timestamp(train_end), "ns")
    horizon = np.timedelta64(horizon_hours, "h")
    is_train = t + horizon < split_at
    is_test = t >= split_at
    stats.gap_dropped = int((~is_train & ~is_test).sum())
    df["split"] = np.where(is_train, "train", "test")
    df = df[is_train | is_test].reset_index(drop=True)
    df["split"] = df["split"].astype("category")

    target = f"fails_within_{horizon_hours}h"
    stats.rows = len(df)
    for split in ("train", "test"):
        part = df[df["split"] == split]
        stats.split_rows[split] = len(part)
        stats.split_positive_rate[split] = (
            round(float(part[target].mean()), 4) if len(part) else 0.0
        )
    label_columns = set(labels.columns) - set(KEYS)
    stats.features = len([c for c in df.columns if c not in label_columns | set(KEYS) | {"split"}])
    return df, stats


def build(
    silver_dir: Path | None = None, gold_dir: Path | None = None
) -> tuple[pd.DataFrame, LabelStats, DatasetStats]:
    """Gold features + Silver failures -> labels.parquet and ml_dataset.parquet."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    gold_dir = Path(gold_dir or settings.paths.gold)
    paths = {
        "sensor": gold_dir / "sensor_features.parquet",
        "events": gold_dir / "event_features.parquet",
        "failures": silver_dir / "failures.parquet",
        "telemetry": silver_dir / "telemetry.parquet",
    }
    for name, path in paths.items():
        if not path.exists():
            step = "`iiot silver build`" if path.parent == silver_dir else f"the {name} features"
            raise FileNotFoundError(f"{path} not found. Build {step} first.")

    horizon = settings.ml.prediction_horizon_hours
    sensor = pd.read_parquet(paths["sensor"])
    data_end = pd.read_parquet(paths["telemetry"], columns=["timestamp"])["timestamp"].max()
    labels, label_stats = build_labels(
        sensor[KEYS],
        pd.read_parquet(paths["failures"]),
        horizon,
        data_end,
        list(settings.business.components),
    )
    write_parquet_atomic(labels, gold_dir / "labels.parquet")
    dataset, dataset_stats = build_ml_dataset(
        sensor,
        pd.read_parquet(paths["events"]),
        labels,
        pd.Timestamp(settings.ml.train_end_date),
        horizon,
    )
    out = write_parquet_atomic(dataset, gold_dir / "ml_dataset.parquet")

    logger.info(
        "Labels: %s rows (%s dropped: horizon past the end of the data), "
        "%s positive (%.2f%%), next failure: %s",
        f"{label_stats.rows_out:,}",
        f"{label_stats.unobservable_dropped:,}",
        f"{label_stats.positives:,}",
        100 * label_stats.positives / max(label_stats.rows_out, 1),
        label_stats.failed_component_counts,
    )
    logger.info(
        "ML dataset: %s rows, %d features, train %s (%.2f%% positive), test %s (%.2f%% positive), "
        "%s gap rows dropped -> %s",
        f"{dataset_stats.rows:,}",
        dataset_stats.features,
        f"{dataset_stats.split_rows['train']:,}",
        100 * dataset_stats.split_positive_rate["train"],
        f"{dataset_stats.split_rows['test']:,}",
        100 * dataset_stats.split_positive_rate["test"],
        f"{dataset_stats.gap_dropped:,}",
        out,
    )
    return dataset, label_stats, dataset_stats
