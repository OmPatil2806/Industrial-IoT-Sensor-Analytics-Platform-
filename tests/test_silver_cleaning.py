"""Tests for iiot.silver.cleaning (each rule, then the full sequence)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iiot.config import SensorLimit, Silver
from iiot.silver.cleaning import (
    clean,
    deduplicate,
    fill_short_gaps,
    null_out_of_range,
    null_stuck_runs,
    to_hourly_grid,
)

LIMITS = {"volt": SensorLimit(0.0, 300.0)}
RULES = Silver(stuck_min_run=3, max_fill_hours=3, fill_window_hours=24)


def series(values, machine=1, start="2015-01-01 06:00"):
    """Typed telemetry for one machine with hourly timestamps and the given volt values."""
    n = len(values)
    return pd.DataFrame(
        {
            "machine_id": pd.array([machine] * n, dtype="Int64"),
            "timestamp": pd.date_range(start, periods=n, freq="h"),
            "volt": pd.array(values, dtype="float64"),
            "_source_line": range(2, n + 2),
        }
    )


# --- 1. deduplicate ---------------------------------------------------------------


def test_deduplicate_keeps_first_copy():
    df = series([1.0, 2.0])
    dupes = pd.concat([df, df.iloc[[1]].assign(_source_line=99)], ignore_index=True)
    out, removed, conflicting = deduplicate(dupes, ["volt"])
    assert removed == 1 and conflicting == 0
    assert out["_source_line"].tolist() == [2, 3]


def test_deduplicate_counts_conflicting_values():
    df = series([1.0, 2.0])
    conflict = df.iloc[[1]].assign(volt=5.0, _source_line=99)
    out, removed, conflicting = deduplicate(pd.concat([df, conflict]), ["volt"])
    assert removed == 1 and conflicting == 1
    assert out["volt"].tolist() == [1.0, 2.0]  # first (source-order) value kept


def test_deduplicate_treats_matching_nulls_as_equal():
    df = series([np.nan, 2.0])
    out, removed, conflicting = deduplicate(pd.concat([df, df.iloc[[0]]]), ["volt"])
    assert (removed, conflicting) == (1, 0)


# --- 2. range check ---------------------------------------------------------------


def test_out_of_range_values_become_null():
    df, counts = null_out_of_range(series([-999.0, 0.0, 150.0, 300.0, 9999.0, np.nan]), LIMITS)
    assert counts == {"volt": 2}
    assert df["volt"].isna().tolist() == [True, False, False, False, True, True]


# --- 3. stuck sensors ------------------------------------------------------------


def test_stuck_run_keeps_first_value_and_nulls_repeats():
    df, runs, nulled = null_stuck_runs(series([1.0, 5.0, 5.0, 5.0, 5.0, 2.0]), ["volt"], 3)
    assert runs == {"volt": 1} and nulled == {"volt": 3}
    assert df["volt"].tolist()[:2] == [1.0, 5.0]
    assert df["volt"].isna().tolist() == [False, False, True, True, True, False]


def test_short_repeat_is_not_stuck():
    df, runs, nulled = null_stuck_runs(series([1.0, 5.0, 5.0, 2.0]), ["volt"], 3)
    assert runs == {"volt": 0} and nulled == {"volt": 0}


def test_stuck_run_needs_consecutive_hours_and_same_machine():
    gap = series([5.0, 5.0, 5.0])
    gap.loc[2, "timestamp"] += pd.Timedelta(hours=1)  # a missing hour breaks the run
    assert null_stuck_runs(gap, ["volt"], 3)[1] == {"volt": 0}

    two_machines = pd.concat([series([5.0, 5.0]), series([5.0], machine=2)], ignore_index=True)
    assert null_stuck_runs(two_machines, ["volt"], 3)[1] == {"volt": 0}


def test_nulls_break_stuck_runs():
    df, runs, _ = null_stuck_runs(series([5.0, 5.0, np.nan, 5.0, 5.0]), ["volt"], 3)
    assert runs == {"volt": 0}


# --- 4. hourly grid --------------------------------------------------------------


def test_missing_hours_are_added():
    df = series([1.0, 2.0, 3.0, 4.0]).drop(index=[1, 2])
    out, added, off_hour = to_hourly_grid(df)
    assert (added, off_hour) == (2, 0)
    assert out["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert out["volt"].isna().tolist() == [False, True, True, False]


def test_off_hour_rows_are_dropped():
    df = series([1.0, 2.0])
    df.loc[1, "timestamp"] += pd.Timedelta(minutes=30)
    out, _, off_hour = to_hourly_grid(df)
    assert off_hour == 1 and len(out) == 1


def test_grid_spans_each_machine_separately():
    df = pd.concat([series([1.0, 2.0]), series([3.0], machine=2, start="2015-01-05")])
    out, added, _ = to_hourly_grid(df)
    assert added == 0 and len(out) == 3


# --- 5. fill short gaps ----------------------------------------------------------


def test_short_gap_filled_with_window_mean():
    values = [10.0] * 12 + [np.nan] * 2 + [20.0] * 12
    out, counts = fill_short_gaps(series(values), ["volt"], max_hours=3, window_hours=24)
    assert counts == {"volt": 2}
    # Centred 25-slot window around row 12: rows 0-11 (12 x 10), 2 gaps, rows 14-24 (11 x 20).
    assert out["volt"].iloc[12] == pytest.approx((12 * 10 + 11 * 20) / 23)
    assert out["volt_quality"].tolist()[11:15] == ["ok", "filled", "filled", "ok"]


def test_long_gap_stays_missing():
    values = [10.0] * 12 + [np.nan] * 4 + [20.0] * 12
    out, counts = fill_short_gaps(series(values), ["volt"], max_hours=3, window_hours=24)
    assert counts == {"volt": 0}
    assert (out["volt_quality"].iloc[12:16] == "missing").all()


def test_gaps_at_the_edges_are_not_filled():
    values = [np.nan] + [10.0] * 20 + [np.nan]
    out, counts = fill_short_gaps(series(values), ["volt"], max_hours=3, window_hours=24)
    assert counts == {"volt": 0}
    assert out["volt_quality"].iloc[0] == out["volt_quality"].iloc[-1] == "missing"


def test_window_never_uses_another_machine():
    df = pd.concat(
        [series([10.0] * 10 + [np.nan] + [10.0] * 10), series([1000.0] * 21, machine=2)],
        ignore_index=True,
    )
    out, _ = fill_short_gaps(df, ["volt"], max_hours=3, window_hours=24)
    assert out["volt"].iloc[10] == pytest.approx(10.0)


def test_too_few_valid_neighbours_leaves_gap_missing():
    values = [10.0, np.nan, 10.0]  # only 2 valid readings, window 24 needs 12
    out, counts = fill_short_gaps(series(values), ["volt"], max_hours=3, window_hours=24)
    assert counts == {"volt": 0}


# --- full sequence ---------------------------------------------------------------


def varied(n: int, start: float = 100.0) -> list[float]:
    """Distinct readings, like real sensor data (identical repeats would look stuck)."""
    return [start + 0.1 * i for i in range(n)]


def test_clean_runs_all_rules_in_order():
    values = varied(10) + [9999.0] + varied(5, 200.0) + [50.0] * 4 + varied(10, 250.0)
    df = series(values)
    df = pd.concat([df, df.iloc[[3]].assign(_source_line=999)], ignore_index=True)  # duplicate
    out, stats = clean(df, LIMITS, RULES)

    assert stats.rows_in == 31 and stats.rows_out == 30
    assert stats.duplicates_removed == 1
    assert stats.out_of_range_nulled == {"volt": 1}
    assert stats.stuck_runs == {"volt": 1} and stats.stuck_values_nulled == {"volt": 3}
    # The spike (1 h) is filled; the stuck repeats (3 h) are filled too (gap <= 3 h).
    assert stats.filled == {"volt": 4} and stats.missing_after == {"volt": 0}
    assert out["volt"].iloc[16] == 50.0  # first value of the stuck run is kept


def test_clean_does_not_modify_input():
    df = series([1.0, 9999.0, 3.0])
    before = df.copy()
    clean(df, LIMITS, RULES)
    pd.testing.assert_frame_equal(df, before)
