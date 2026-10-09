"""Tests for iiot.ml.data: features and the fit / validation / development / test split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iiot.ml.data import (
    MLDataError,
    feature_columns,
    fingerprint,
    label_columns,
    load_splits,
    split,
)

LABELS = label_columns(24, ["comp1", "comp2", "comp3", "comp4"])
VALIDATION_START = pd.Timestamp("2015-08-01")
TRAIN_END = pd.Timestamp("2015-10-01")
HORIZON = pd.Timedelta(hours=24)


def ml_dataset() -> pd.DataFrame:
    """Two machines, a row every 3 h from June to October, split like Gold does."""
    times = pd.date_range("2015-06-01", "2015-10-31", freq="3h")
    df = pd.DataFrame(
        {
            "machine_id": np.repeat([1, 2], len(times)),
            "timestamp": np.tile(times, 2),
        }
    )
    n = len(df)
    rng = np.random.default_rng(0)
    df["volt_mean_3h"] = rng.normal(170, 5, n)
    df["error5_count_24h"] = rng.integers(0, 2, n)
    df["model"] = pd.Categorical(np.where(df["machine_id"] == 1, "model3", "model4"))
    df["age"] = np.where(df["machine_id"] == 1, 18, 7)
    df["plant_id"] = pd.Categorical(np.where(df["machine_id"] == 1, "PUNE", "CHENNAI"))
    df["line_id"] = pd.Categorical(np.where(df["machine_id"] == 1, "PUNE-L1", "CHENNAI-L1"))
    df["fails_within_24h"] = (np.arange(n) % 50 == 0).astype(int)
    for comp in ("comp1", "comp2", "comp3", "comp4"):
        df[f"{comp}_fails_within_24h"] = 0
    df["comp1_fails_within_24h"] = df["fails_within_24h"]
    df["failed_component"] = np.where(df["fails_within_24h"] == 1, "comp1", "none")
    df["hours_to_failure"] = np.where(df["fails_within_24h"] == 1, 12.0, np.nan)
    t = df["timestamp"]
    is_train, is_test = t + HORIZON < TRAIN_END, t >= TRAIN_END
    df["split"] = pd.Categorical(np.where(is_train, "train", "test"))
    return df[is_train | is_test].reset_index(drop=True)


@pytest.fixture
def splits():
    return split(ml_dataset(), VALIDATION_START, 24, LABELS, ("plant_id", "line_id"))


def test_features_exclude_keys_labels_split_and_configured_columns(splits):
    assert splits.features == ["volt_mean_3h", "error5_count_24h", "model", "age"]
    assert splits.target == "fails_within_24h"
    assert list(splits.X("fit").columns) == splits.features


def test_parts_are_in_time_order_and_never_overlap(splits):
    fit, val, dev, test = (splits.data[p] for p in ("fit", "validation", "development", "test"))
    assert (fit["timestamp"] + HORIZON < VALIDATION_START).all()  # fit labels end before validation
    assert (val["timestamp"] >= VALIDATION_START).all()
    assert (val["timestamp"] + HORIZON < TRAIN_END).all()  # validation labels end before test
    assert (test["timestamp"] >= TRAIN_END).all()
    keys = ["machine_id", "timestamp"]
    for a, b in ((fit, val), (fit, test), (val, test), (dev, test)):
        assert a.merge(b, on=keys).empty


def test_development_is_all_train_rows_including_the_gap(splits):
    dev, fit, val = splits.data["development"], splits.data["fit"], splits.data["validation"]
    gap_rows = 2 * 8  # 2 machines x 8 rows of 3 h whose label window crosses validation_start
    assert len(dev) == len(fit) + len(val) + gap_rows


def test_summary_counts_positives(splits):
    summary = splits.summary()
    assert summary["test"]["positives"] == int(splits.y("test").sum())
    assert summary["fit"]["from"].startswith("2015-06-01")
    assert summary["test"]["from"].startswith("2015-10-01")


def test_label_like_feature_is_rejected():
    df = ml_dataset().assign(vibration_before_failure=1.0)
    with pytest.raises(MLDataError, match="vibration_before_failure"):
        feature_columns(df, LABELS, ())


def test_missing_label_column_is_rejected():
    with pytest.raises(MLDataError, match="hours_to_failure"):
        feature_columns(ml_dataset().drop(columns="hours_to_failure"), LABELS, ())


def test_validation_start_after_the_training_data_is_rejected():
    with pytest.raises(MLDataError, match="validation"):
        split(ml_dataset(), TRAIN_END, 24, LABELS)


def test_dataset_without_test_rows_is_rejected():
    df = ml_dataset()
    with pytest.raises(MLDataError, match="train and test"):
        split(df[df["split"] == "train"], VALIDATION_START, 24, LABELS)


def test_fingerprint_is_stable_and_detects_changes():
    df = ml_dataset()
    assert fingerprint(df) == fingerprint(ml_dataset())
    changed = df.copy()
    changed.loc[5, "volt_mean_3h"] += 0.001
    assert fingerprint(changed) != fingerprint(df)


def test_load_splits_reads_gold_and_uses_config(tmp_path):
    ml_dataset().to_parquet(tmp_path / "ml_dataset.parquet")
    splits = load_splits(tmp_path)
    assert "plant_id" not in splits.features and "line_id" not in splits.features
    assert splits.data["validation"]["timestamp"].min() == VALIDATION_START


def test_load_splits_without_gold_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot gold build"):
        load_splits(tmp_path)
