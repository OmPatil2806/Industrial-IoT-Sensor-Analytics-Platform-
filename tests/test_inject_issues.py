"""Tests for iiot.ingestion.inject_issues on small synthetic telemetry."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from iiot.config import get_settings
from iiot.ingestion.inject_issues import (
    DIRTY_FILE_NAME,
    MANIFEST_FILE_NAME,
    inject,
    run,
    spike_values,
)

SETTINGS = get_settings()
LIMITS = SETTINGS.sensors
SENSORS = list(LIMITS)
PARAMS = replace(
    SETTINGS.data_quality_injection,
    missing_rate=0.02,
    spike_rate=0.01,
    duplicate_rate=0.01,
    stuck_episodes=20,
)


def make_clean(n_machines: int = 5, hours: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2015-01-01 06:00", periods=hours, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    n = n_machines * hours
    return pd.DataFrame(
        {
            "datetime": np.tile(times, n_machines),
            "machineID": np.repeat(np.arange(1, n_machines + 1), hours),
            "volt": rng.normal(170, 15, n),
            "rotate": rng.normal(446, 52, n),
            "pressure": rng.normal(100, 11, n),
            "vibration": rng.normal(40, 5, n),
        }
    )


@pytest.fixture(scope="module")
def result():
    clean = make_clean()
    dirty, manifest = inject(clean, PARAMS, LIMITS)
    return clean, dirty, manifest


def test_row_counts(result):
    clean, dirty, manifest = result
    assert manifest["rows_in"] == len(clean)
    assert manifest["rows_out"] == len(dirty) == len(clean) + manifest["duplicate_rows"]


def test_manifest_counts_match_output(result):
    _, dirty, manifest = result
    assert int(dirty.duplicated().sum()) == manifest["duplicate_rows"]
    unique = dirty.drop_duplicates()
    for s in SENSORS:
        assert int(unique[s].isna().sum()) == manifest["missing_values"][s]
        outside = ~unique[s].between(LIMITS[s].min, LIMITS[s].max) & unique[s].notna()
        assert int(outside.sum()) == manifest["spikes"][s]


def test_rates_are_close_to_config():
    clean = make_clean(n_machines=20, hours=1000)
    _, manifest = inject(clean, PARAMS, LIMITS)
    cells = len(clean)
    for s in SENSORS:
        assert manifest["missing_values"][s] / cells == pytest.approx(PARAMS.missing_rate, rel=0.2)
        assert manifest["spikes"][s] / cells == pytest.approx(PARAMS.spike_rate, rel=0.3)
    assert manifest["duplicate_rows"] / len(clean) == pytest.approx(PARAMS.duplicate_rate, rel=0.3)


def test_spikes_are_outside_validation_limits():
    for s, limit in LIMITS.items():
        low, high = spike_values(limit)
        assert low < limit.min and high > limit.max, s


def test_input_is_not_modified():
    clean = make_clean()
    before = clean.copy()
    inject(clean, PARAMS, LIMITS)
    pd.testing.assert_frame_equal(clean, before)


def test_same_seed_is_reproducible_and_new_seed_differs():
    clean = make_clean()
    a, manifest_a = inject(clean, PARAMS, LIMITS)
    b, manifest_b = inject(clean, PARAMS, LIMITS)
    c, _ = inject(clean, replace(PARAMS, seed=PARAMS.seed + 1), LIMITS)
    pd.testing.assert_frame_equal(a, b)
    assert manifest_a == manifest_b
    assert not a.equals(c)


def test_stuck_episodes_are_valid(result):
    clean, _, manifest = result
    episodes = manifest["stuck_sensor"]["episode_list"]
    assert manifest["stuck_sensor"]["episodes"] == len(episodes) == PARAMS.stuck_episodes
    dirty_unique = inject(clean, replace(PARAMS, duplicate_rate=0.0), LIMITS)[0]
    for e in episodes:
        assert PARAMS.stuck_min_hours <= e["hours"] <= PARAMS.stuck_max_hours
        rows = dirty_unique[
            (dirty_unique["machineID"] == e["machineID"])
            & dirty_unique["datetime"].between(e["start"], e["end"])
        ]
        # One machine, consecutive hours, every value frozen (no missing values or spikes).
        assert len(rows) == e["hours"]
        assert (rows[e["sensor"]] == e["value"]).all()


def test_duplicates_sit_right_after_their_original(result):
    _, dirty, _ = result
    dup_positions = np.flatnonzero(dirty.duplicated().to_numpy())
    for pos in dup_positions:
        assert dirty.iloc[pos].equals(dirty.iloc[pos - 1])


def test_zero_rates_return_an_identical_copy():
    clean = make_clean()
    params = replace(PARAMS, missing_rate=0.0, spike_rate=0.0, duplicate_rate=0.0, stuck_episodes=0)
    dirty, manifest = inject(clean, params, LIMITS)
    pd.testing.assert_frame_equal(dirty, clean)
    assert manifest["duplicate_rows"] == 0


def test_run_writes_files_and_keeps_original(tmp_path):
    source = tmp_path / SETTINGS.dataset.files["telemetry"]
    make_clean().to_csv(source, index=False)
    original_bytes = source.read_bytes()

    manifest = run(raw_dir=tmp_path)

    assert source.read_bytes() == original_bytes
    assert (tmp_path / DIRTY_FILE_NAME).exists()
    saved = json.loads((tmp_path / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    assert saved == manifest
    assert saved["source_file"] == source.name


def test_run_without_source_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        run(raw_dir=tmp_path)
