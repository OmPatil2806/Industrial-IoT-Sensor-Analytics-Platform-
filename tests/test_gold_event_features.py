"""Tests for iiot.gold.event_features (error, maintenance and machine features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iiot.gold.event_features import (
    build,
    build_event_features,
    count_recent_events,
    hours_since_last,
)

T0 = pd.Timestamp("2015-01-10 00:00")
COMPONENTS = ["comp1", "comp2"]


def ts(hours: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=hours)


def spine(machine_hours: dict[int, list[float]]) -> pd.DataFrame:
    rows = [(m, ts(h)) for m, hours in machine_hours.items() for h in hours]
    return pd.DataFrame(
        {
            "machine_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "timestamp": [r[1] for r in rows],
        }
    )


def events(rows: list[tuple[int, float, str]], value_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": pd.array([r[0] for r in rows], dtype="Int64"),
            # typed even when empty, like a real Silver table
            "timestamp": pd.to_datetime([ts(r[1]) for r in rows]).astype("datetime64[ns]"),
            value_col: [r[2] for r in rows],
        }
    )


MACHINES = pd.DataFrame(
    {
        "machine_id": pd.array([1, 2], dtype="Int64"),
        "model": ["model3", "model4"],
        "age": pd.array([18, 7], dtype="Int64"),
        "plant_id": ["PUNE", "CHENNAI"],
        "plant_name": ["Pune Plant", "Chennai Plant"],
        "city": ["Pune", "Chennai"],
        "line_id": ["PUNE-L1", "CHENNAI-L2"],
    }
)


# --- error counts ----------------------------------------------------------------


def test_window_excludes_start_and_includes_end():
    s = spine({1: [24]})
    e = events(
        [(1, 0, "error1"), (1, 0.5, "error1"), (1, 24, "error1"), (1, 25, "error1")], "error_id"
    )
    # window is (0, 24]: the event at 0 is out, 0.5 and 24 are in, 25 is in the future
    assert count_recent_events(s, e, 24).tolist() == [2]


def test_counts_are_per_machine():
    s = spine({1: [24], 2: [24]})
    e = events([(1, 10, "error1"), (1, 11, "error1"), (2, 12, "error1")], "error_id")
    assert count_recent_events(s, e, 24).tolist() == [2, 1]


def test_error_features_per_type_and_total():
    s = spine({1: [24]})
    e = events([(1, 5, "error1"), (1, 6, "error3"), (1, 7, "error3")], "error_id")
    out, stats = build_event_features(s, e, events([], "component"), MACHINES, 24, COMPONENTS)
    row = out.iloc[0]
    assert row["error1_count_24h"] == 1
    assert row["error3_count_24h"] == 2
    assert row["error2_count_24h"] == row["error5_count_24h"] == 0
    assert row["errors_total_24h"] == 3


def test_future_events_never_counted():
    """Adding events after time t must not change any feature at time t."""
    s = spine({1: [24]})
    past = events([(1, 10, "error1"), (1, 5, "comp1")], "error_id")
    maint = events([(1, 5, "comp1")], "component")
    before, _ = build_event_features(s, past.iloc[:1], maint, MACHINES, 24, COMPONENTS)
    future_errors = pd.concat([past.iloc[:1], events([(1, 30, "error1")], "error_id")])
    future_maint = pd.concat([maint, events([(1, 30, "comp1")], "component")])
    after, _ = build_event_features(s, future_errors, future_maint, MACHINES, 24, COMPONENTS)
    pd.testing.assert_frame_equal(before, after)


def test_off_hour_events_are_rounded_up():
    s = spine({1: [10, 11]})
    e = events([(1, 10.5, "error2")], "error_id")
    out, _ = build_event_features(s, e, events([], "component"), MACHINES, 24, COMPONENTS)
    assert out["error2_count_24h"].tolist() == [0, 1]  # not visible at 10:00, visible at 11:00


# --- hours since replacement ------------------------------------------------------


def test_hours_since_last_replacement():
    s = spine({1: [0, 10, 30]})
    m = events([(1, -50, "comp1"), (1, 10, "comp1")], "component")
    assert hours_since_last(s, m).tolist() == [50.0, 0.0, 20.0]


def test_never_replaced_is_empty_and_counted():
    s = spine({1: [0], 2: [0]})
    m = events([(1, -5, "comp1")], "component")
    out, stats = build_event_features(s, events([], "error_id"), m, MACHINES, 24, COMPONENTS)
    assert out["hours_since_comp1_replaced"].tolist()[0] == 5.0
    assert np.isnan(out["hours_since_comp1_replaced"].tolist()[1])
    assert stats.never_replaced == {"comp1": 1, "comp2": 2}


def test_replacements_are_per_component():
    s = spine({1: [0]})
    m = events([(1, -5, "comp1"), (1, -100, "comp2")], "component")
    out, _ = build_event_features(s, events([], "error_id"), m, MACHINES, 24, COMPONENTS)
    assert out.loc[0, "hours_since_comp1_replaced"] == 5.0
    assert out.loc[0, "hours_since_comp2_replaced"] == 100.0


# --- machine attributes and shape -----------------------------------------------


def test_machine_attributes_joined_and_order_kept():
    s = spine({2: [0], 1: [0, 3]})
    out, stats = build_event_features(
        s, events([], "error_id"), events([], "component"), MACHINES, 24, COMPONENTS
    )
    assert out["machine_id"].tolist() == [2, 1, 1]  # same row order as the spine
    assert out["model"].tolist() == ["model4", "model3", "model3"]
    assert out["line_id"].tolist()[0] == "CHENNAI-L2"
    assert stats.rows == 3
    assert stats.features == 5 + 1 + len(COMPONENTS) + 4  # error types, total, components, machine


def test_build_without_sensor_features_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="sensor features"):
        build(tmp_path / "silver", tmp_path / "gold")
