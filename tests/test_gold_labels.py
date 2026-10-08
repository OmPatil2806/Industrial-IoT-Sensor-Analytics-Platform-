"""Tests for iiot.gold.labels (forward-looking labels and the ML dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iiot.gold.labels import build, build_labels, build_ml_dataset

T0 = pd.Timestamp("2015-03-01 00:00")
COMPONENTS = ["comp1", "comp2"]
END = T0 + pd.Timedelta(days=30)


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


def failures(rows: list[tuple[int, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "timestamp": pd.to_datetime([ts(r[1]) for r in rows]).astype("datetime64[ns]"),
            "component": [r[2] for r in rows],
        }
    )


def labels(machine_hours, fails, end=END):
    return build_labels(spine(machine_hours), failures(fails), 24, end, COMPONENTS)


def test_failure_window_is_after_t_up_to_horizon():
    out, _ = labels({1: [0, 6, 24, 30, 31]}, [(1, 30, "comp1")])
    # failure at hour 30: rows at 6..29 see it; the row AT 30 does not (it is not in the future)
    assert out["fails_within_24h"].tolist() == [0, 1, 1, 0, 0]
    assert out["hours_to_failure"].tolist()[:3] == [30.0, 24.0, 6.0]
    assert out["failed_component"].tolist() == ["none", "comp1", "comp1", "none", "none"]


def test_simultaneous_failures_are_kept():
    out, _ = labels({1: [0]}, [(1, 5, "comp2"), (1, 5, "comp1")])
    row = out.iloc[0]
    assert row["failed_component"] == "comp1+comp2"
    assert row["comp1_fails_within_24h"] == row["comp2_fails_within_24h"] == 1


def test_component_labels_see_every_failure_in_the_horizon():
    out, _ = labels({1: [0]}, [(1, 5, "comp1"), (1, 10, "comp2")])
    row = out.iloc[0]
    assert row["failed_component"] == "comp1"  # the NEXT failure
    assert row["comp1_fails_within_24h"] == 1 and row["comp2_fails_within_24h"] == 1


def test_no_later_failure_leaves_rul_empty():
    out, _ = labels({1: [100]}, [(1, 50, "comp1")])
    assert np.isnan(out["hours_to_failure"].iloc[0])
    assert out["fails_within_24h"].iloc[0] == 0


def test_failures_are_per_machine():
    out, _ = labels({1: [0], 2: [0]}, [(2, 5, "comp1")])
    assert out["fails_within_24h"].tolist() == [0, 1]


def test_rows_whose_horizon_passes_the_data_end_are_dropped():
    out, stats = labels({1: [0, 10, 20]}, [], end=ts(40))
    assert out["timestamp"].tolist() == [ts(0), ts(10)]  # 20 + 24 = 44 > 40
    assert stats.unobservable_dropped == 1 and stats.rows_out == 2


def test_label_stats():
    _, stats = labels({1: [0, 6, 50]}, [(1, 10, "comp1")])
    assert stats.positives == 2
    assert stats.failed_component_counts == {"comp1": 2, "none": 1}


# --- ML dataset ------------------------------------------------------------------


def test_ml_dataset_split_with_gap():
    split = ts(100)
    sp = spine({1: [0, 70, 76, 80, 99, 100, 130]})
    lab, _ = labels({1: [0, 70, 76, 80, 99, 100, 130]}, [])
    sensor = sp.assign(volt_mean_3h=1.0)
    events = sp.assign(errors_total_24h=0)
    ds, stats = build_ml_dataset(sensor, events, lab, split, 24)
    # train: t + 24 < 100 -> t < 76; test: t >= 100; 76..99 are gap
    assert ds["split"].astype(str).tolist() == ["train", "train", "test", "test"]
    assert stats.gap_dropped == 3
    assert stats.split_rows == {"train": 2, "test": 2}
    assert stats.features == 2  # volt_mean_3h, errors_total_24h


def test_ml_dataset_train_labels_never_reach_the_test_period():
    sp = spine({1: list(range(0, 200, 3))})
    lab, _ = labels({1: list(range(0, 200, 3))}, [(1, 101, "comp1")])
    ds, _ = build_ml_dataset(sp.assign(f=1.0), sp.assign(g=1), lab, ts(100), 24)
    train = ds[ds["split"] == "train"]
    assert (train["timestamp"] + pd.Timedelta(hours=24) < ts(100)).all()
    assert train["fails_within_24h"].sum() == 0  # the failure at 101 is test-period only


def test_ml_dataset_requires_matching_rows():
    sp = spine({1: [0]})
    lab, _ = labels({1: [0]}, [])
    with pytest.raises(pd.errors.MergeError):
        build_ml_dataset(pd.concat([sp, sp]).assign(f=1.0), sp.assign(g=1), lab, ts(100), 24)


def test_build_without_inputs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(tmp_path / "silver", tmp_path / "gold")
