"""Inject realistic data-quality problems into a COPY of the raw telemetry.

The Azure PdM telemetry is unusually clean. Real sensor feeds are not, so this
step writes a dirty copy that gives the Silver layer real cleaning work:

    1. Stuck sensor   - a sensor repeats one value for several hours
    2. Missing values - readings dropped (empty cells)
    3. Spikes         - impossible readings outside the validation limits
    4. Duplicate rows - messages re-sent by the gateway

Rates and the random seed come from `data_quality_injection` in
config/settings.yaml, so the output is identical on every run. The original
file is never modified. A manifest records exactly what was injected, so the
Silver layer can later be checked against it.

Usage:
    python -m iiot.ingestion.inject_issues
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import DataQualityInjection, SensorLimit, get_settings
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ingestion.inject_issues")

DIRTY_FILE_NAME = "PdM_telemetry_dirty.csv"
MANIFEST_FILE_NAME = "injection_manifest.json"
N_EXAMPLES = 5
MAX_STUCK_ATTEMPTS = 100


def spike_values(limit: SensorLimit) -> tuple[float, float]:
    """Low/high sentinel values that are always outside a sensor's valid range."""
    return min(-999.0, limit.min - 1.0), max(9999.0, limit.max * 10.0)


def _examples(df: pd.DataFrame, mask: np.ndarray, sensor: str, rng) -> list[dict]:
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        return []
    picked = np.sort(rng.choice(rows, size=min(N_EXAMPLES, len(rows)), replace=False))
    return [
        {
            "datetime": str(df.at[i, "datetime"]),
            "machineID": int(df.at[i, "machineID"]),
            "sensor": sensor,
            "value": None if pd.isna(df.at[i, sensor]) else float(df.at[i, sensor]),
        }
        for i in picked
    ]


def _inject_stuck(
    df: pd.DataFrame, sensors: list[str], params: DataQualityInjection, rng
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Freeze a sensor at one value for a run of consecutive hourly readings."""
    n = len(df)
    stuck = {s: np.zeros(n, dtype=bool) for s in sensors}
    machine = df["machineID"].to_numpy()
    hours = pd.to_datetime(df["datetime"]).to_numpy().astype("datetime64[h]").astype(np.int64)
    episodes: list[dict] = []

    for _ in range(params.stuck_episodes):
        for _attempt in range(MAX_STUCK_ATTEMPTS):
            length = int(rng.integers(params.stuck_min_hours, params.stuck_max_hours + 1))
            start = int(rng.integers(0, n - length + 1))
            end = start + length  # exclusive
            sensor = sensors[int(rng.integers(len(sensors)))]
            same_machine = (machine[start:end] == machine[start]).all()
            consecutive = (np.diff(hours[start:end]) == 1).all()
            # Keep a one-row gap around episodes so neighbouring runs never merge.
            lo, hi = max(start - 1, 0), min(end + 1, n)
            if same_machine and consecutive and not stuck[sensor][lo:hi].any():
                break
        else:
            logger.warning("Could not place stuck episode after %d attempts", MAX_STUCK_ATTEMPTS)
            continue

        value = df.at[start, sensor]
        df.loc[start : end - 1, sensor] = value
        stuck[sensor][start:end] = True
        episodes.append(
            {
                "machineID": int(machine[start]),
                "sensor": sensor,
                "start": str(df.at[start, "datetime"]),
                "end": str(df.at[end - 1, "datetime"]),
                "hours": length,
                "value": float(value),
            }
        )
    episodes.sort(key=lambda e: (e["machineID"], e["start"], e["sensor"]))
    return stuck, episodes


def inject(
    clean: pd.DataFrame,
    params: DataQualityInjection,
    limits: dict[str, SensorLimit],
) -> tuple[pd.DataFrame, dict]:
    """Return (dirty_copy, manifest). The input DataFrame is not modified."""
    rng = np.random.default_rng(params.seed)
    df = clean.reset_index(drop=True).copy()
    sensors = list(limits)
    n = len(df)
    manifest: dict = {
        "seed": params.seed,
        "rates": {
            "missing_rate": params.missing_rate,
            "spike_rate": params.spike_rate,
            "duplicate_rate": params.duplicate_rate,
        },
        "rows_in": n,
    }

    # 1. Stuck sensors first, on clean values.
    stuck, episodes = _inject_stuck(df, sensors, params, rng)
    manifest["stuck_sensor"] = {
        "episodes": len(episodes),
        "cells_per_sensor": {s: int(stuck[s].sum()) for s in sensors},
        "episode_list": episodes,
    }

    # 2. Missing values and 3. spikes, never on stuck cells and never overlapping.
    missing_counts, spike_counts = {}, {}
    examples: dict[str, list] = {"missing_values": [], "spikes": []}
    for sensor in sensors:
        free = ~stuck[sensor]
        missing = free & (rng.random(n) < params.missing_rate)
        spike = free & ~missing & (rng.random(n) < params.spike_rate)

        df.loc[missing, sensor] = np.nan
        low, high = spike_values(limits[sensor])
        df.loc[spike, sensor] = rng.choice([low, high], size=int(spike.sum()))

        missing_counts[sensor] = int(missing.sum())
        spike_counts[sensor] = int(spike.sum())
        examples["missing_values"] += _examples(df, missing, sensor, rng)
        examples["spikes"] += _examples(df, spike, sensor, rng)
    manifest["missing_values"] = missing_counts
    manifest["spikes"] = spike_counts

    # 4. Duplicates last: exact copies placed right after the original row.
    duplicated = rng.random(n) < params.duplicate_rate
    df = df.iloc[np.repeat(np.arange(n), 1 + duplicated.astype(int))].reset_index(drop=True)
    manifest["duplicate_rows"] = int(duplicated.sum())
    manifest["rows_out"] = len(df)
    manifest["examples"] = examples
    return df, manifest


def run(raw_dir: Path | None = None) -> dict:
    """Read the raw telemetry, write the dirty copy and manifest, return the manifest."""
    settings = get_settings()
    raw_dir = raw_dir or settings.paths.raw
    source = raw_dir / settings.dataset.files["telemetry"]
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Run `python -m iiot.ingestion.download` first."
        )

    logger.info("Reading %s", source.name)
    clean = pd.read_csv(source, dtype={"datetime": str})
    dirty, manifest = inject(clean, settings.data_quality_injection, settings.sensors)
    manifest = {"source_file": source.name, "output_file": DIRTY_FILE_NAME, **manifest}

    dirty.to_csv(raw_dir / DIRTY_FILE_NAME, index=False)
    (raw_dir / MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Rows: %s in -> %s out", f"{manifest['rows_in']:,}", f"{manifest['rows_out']:,}")
    logger.info(
        "Stuck sensor:   %d episodes, %s cells",
        manifest["stuck_sensor"]["episodes"],
        f"{sum(manifest['stuck_sensor']['cells_per_sensor'].values()):,}",
    )
    logger.info("Missing values: %s cells", f"{sum(manifest['missing_values'].values()):,}")
    logger.info("Spikes:         %s cells", f"{sum(manifest['spikes'].values()):,}")
    logger.info("Duplicate rows: %s", f"{manifest['duplicate_rows']:,}")
    logger.info("Wrote %s and %s", DIRTY_FILE_NAME, MANIFEST_FILE_NAME)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inject data-quality issues into telemetry.")
    parser.parse_args(argv)
    try:
        run()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
