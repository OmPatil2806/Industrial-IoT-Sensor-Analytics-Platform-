"""Tests for iiot.gold.sensor_features (backward-looking rolling features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iiot.config import Gold
from iiot.gold.sensor_features import build, build_sensor_features, feature_names

GOLD = Gold(feature_step_hours=3, window_hours=(3, 24))
SENSORS = ["volt"]


def silver(values, machine=1, quality=None, start="2015-01-01 06:00") -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "machine_id": pd.array([machine] * n, dtype="Int64"),
            "timestamp": pd.date_range(start, periods=n, freq="h"),
            "volt": np.asarray(values, dtype=float),
            "volt_quality": pd.Categorical(quality or ["ok"] * n, ["ok", "filled", "missing"]),
        }
    )


def features_at(df: pd.DataFrame, hour_index: int, start="2015-01-01 06:00") -> pd.Series:
    ts = pd.Timestamp(start) + pd.Timedelta(hours=hour_index)
    row = df[df["timestamp"] == ts]
    assert len(row) == 1, f"no feature row at {ts}"
    return row.iloc[0]


def test_feature_names():
    assert feature_names(["volt"], (3, 24)) == [
        "volt_mean_3h",
        "volt_std_3h",
        "volt_mean_24h",
        "volt_std_24h",
        "volt_trend",
        "volt_filled_share_24h",
        "volt_missing_share_24h",
    ]


def test_rolling_values_use_current_and_previous_hours():
    values = np.arange(48, dtype=float)  # 0, 1, 2, ... one per hour
    out, _ = build_sensor_features(silver(values), SENSORS, GOLD)
    row = features_at(out, 30)
    assert row["volt_mean_3h"] == pytest.approx((28 + 29 + 30) / 3)
    assert row["volt_mean_24h"] == pytest.approx(np.mean(np.arange(7, 31)))
    assert row["volt_std_3h"] == pytest.approx(1.0)
    assert row["volt_trend"] == pytest.approx(29.0 - 18.5)


def test_features_never_use_future_readings():
    """Changing readings after time t must not change any feature at time t."""
    values = np.random.default_rng(0).normal(170, 15, 72)
    before, _ = build_sensor_features(silver(values), SENSORS, GOLD)
    changed = values.copy()
    changed[31:] += 1000.0  # change everything after hour 30
    after, _ = build_sensor_features(silver(changed), SENSORS, GOLD)
    names = feature_names(SENSORS, GOLD.window_hours)
    pd.testing.assert_series_equal(features_at(before, 30)[names], features_at(after, 30)[names])
    assert features_at(after, 33)["volt_mean_3h"] != features_at(before, 33)["volt_mean_3h"]


def test_one_row_every_step_after_warm_up():
    out, stats = build_sensor_features(silver(np.arange(48.0)), SENSORS, GOLD)
    hours = (
        (out["timestamp"] - pd.Timestamp("2015-01-01 06:00")) // pd.Timedelta(hours=1)
    ).tolist()
    assert hours == [24, 27, 30, 33, 36, 39, 42, 45]  # warm-up (first 23 h) dropped, every 3 h
    assert stats.warmup_rows_dropped == 23
    assert stats.rows_out == len(out) and stats.features == 7


def test_machines_do_not_share_windows():
    df = pd.concat(
        [silver(np.full(30, 100.0) + np.arange(30) * 0.01), silver(np.full(30, 900.0), machine=2)],
        ignore_index=True,
    )
    out, _ = build_sensor_features(df, SENSORS, GOLD)
    first = out[out["machine_id"] == 1].iloc[0]
    assert first["volt_mean_24h"] < 101  # machine 2's 900s never leak in


def test_quality_shares_count_filled_and_missing():
    quality = ["ok"] * 18 + ["filled"] * 4 + ["missing"] * 2 + ["ok"] * 6
    values = [1.0] * 22 + [np.nan] * 2 + [1.0] * 6
    out, _ = build_sensor_features(silver(values, quality=quality), SENSORS, GOLD)
    row = features_at(out, 24)  # window = hours 1..24
    assert row["volt_filled_share_24h"] == pytest.approx(4 / 24)
    assert row["volt_missing_share_24h"] == pytest.approx(2 / 24)


def test_statistics_need_half_the_window():
    values = [np.nan] * 22 + [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    out, _ = build_sensor_features(silver(values), SENSORS, GOLD)
    row = features_at(out, 24)  # only 3 valid readings in the 24 h window
    assert np.isnan(row["volt_mean_24h"])
    assert row["volt_mean_3h"] == pytest.approx(6.0)  # hours 22-24 = 5, 6, 7: enough for 3 h


def test_features_are_float32():
    out, _ = build_sensor_features(silver(np.arange(48.0)), SENSORS, GOLD)
    assert (out[feature_names(SENSORS, GOLD.window_hours)].dtypes == np.float32).all()


def test_build_without_silver_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot silver build"):
        build(tmp_path / "silver", tmp_path / "gold")
