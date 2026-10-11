"""Tests for iiot.ml.anomaly (small synthetic data, MLflow in a temporary folder)."""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from iiot.ml import anomaly
from iiot.ml.anomaly import (
    MODEL_CARD_FILE_NAME,
    MODEL_NAME,
    NORMAL,
    anomaly_score,
    evaluate_anomalies,
    healthy,
    rate_key,
    sensor_features,
    time_bucket,
    train_anomaly_model,
)
from iiot.ml.failure_model import CHAMPION_ALIAS
from tests.test_ml_failure_model import make_splits


def test_sensor_features_leave_out_errors_wear_machine_and_data_quality():
    features = [
        "volt_mean_3h",
        "rotate_std_24h",
        "vibration_trend",
        "pressure_missing_share_24h",
        "error5_count_24h",
        "hours_since_comp1_replaced",
        "model",
        "age",
    ]
    assert sensor_features(features) == ["volt_mean_3h", "rotate_std_24h", "vibration_trend"]


def test_healthy_rows_are_a_week_or_more_from_any_failure():
    df = pd.DataFrame(
        {
            "fails_within_24h": [1, 0, 0, 0],
            "hours_to_failure": [12.0, 100.0, 200.0, np.nan],
        }
    )
    assert list(healthy(df, "fails_within_24h").index) == [2, 3]


def test_time_buckets_and_metric_names():
    hours = pd.Series([3.0, 24.0, 30.0, 60.0, 100.0, 500.0, np.nan])
    assert list(time_bucket(hours)) == [
        "0-24 h",
        "0-24 h",
        "24-48 h",
        "48-72 h",
        "72-168 h",
        NORMAL,
        NORMAL,
    ]
    assert rate_key("24-48 h") == "flag_rate_24_48h"
    assert rate_key(NORMAL) == "flag_rate_normal"


def test_evaluation_worked_out_by_hand():
    t0 = pd.Timestamp("2015-10-01")
    df = pd.DataFrame(
        {
            "machine_id": [1] * 6,
            "timestamp": [t0 + pd.Timedelta(hours=3 * i) for i in range(6)],
            # rows 0-1: 30 h and 27 h before a failure (24-48 h); rows 2-3: 0-24 h before it;
            # rows 4-5: normal operation
            "hours_to_failure": [30.0, 27.0, 24.0, 21.0, np.nan, np.nan],
            "fails_within_24h": [0, 0, 1, 1, 0, 0],
        }
    )
    scores = np.array([0.9, 0.1, 0.9, 0.9, 0.9, 0.1])
    m = evaluate_anomalies(df, scores, 0.5, "fails_within_24h")
    assert m["flags"] == 4
    assert m["flag_rate_0_24h"] == 1.0
    assert m["flag_rate_24_48h"] == 0.5
    assert m["flag_rate_normal"] == 0.5
    assert m["flag_rate_48h"] == 0.75  # 3 of the 4 rows in the 48 h before the failure
    assert m["lift_48h"] == 1.5
    assert (m["failure_events"], m["event_recall_48h"]) == (1, 1.0)


@pytest.fixture
def mlruns(tmp_path):
    yield tmp_path / "mlruns"
    mlflow.end_run()


def test_model_flags_the_rise_before_failures_and_becomes_champion(mlruns, tmp_path):
    splits = make_splits(learnable=True)  # voltage rises in the 24 h before a failure
    card = train_anomaly_model(splits, mlruns_dir=mlruns, models_dir=tmp_path / "m")
    assert card["promoted"] and card["alias"] == CHAMPION_ALIAS
    assert card["test"]["flag_rate_0_24h"] > 0.8
    assert card["test"]["lift_48h"] >= anomaly.MIN_LIFT
    assert card["inputs"] == ["volt_mean_3h"]  # error5 is not a sensor input

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    flags = anomaly_score(model, splits.data["test"][card["inputs"]]) >= card["threshold"]
    assert int(flags.sum()) == card["test"]["flags"]

    client = MlflowClient()
    assert {a.path for a in client.list_artifacts(card["run_id"], "charts")} == {
        "charts/flag_rate_by_time_to_failure_test.png"
    }
    saved = json.loads((tmp_path / "m" / MODEL_CARD_FILE_NAME).read_text(encoding="utf-8"))
    assert saved["threshold"] == card["threshold"]


def test_threshold_flags_about_one_percent_of_healthy_validation_rows(mlruns, tmp_path):
    splits = make_splits(learnable=True)
    card = train_anomaly_model(splits, mlruns_dir=mlruns, models_dir=tmp_path / "m")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    rows = healthy(splits.data["validation"], splits.target)
    share = (anomaly_score(model, rows[card["inputs"]]) >= card["threshold"]).mean()
    assert share == pytest.approx(anomaly.FLAG_RATE, abs=0.01)


def test_model_below_the_minimum_lift_is_not_promoted(mlruns, tmp_path, monkeypatch):
    monkeypatch.setattr(anomaly, "MIN_LIFT", 1e9)
    card = train_anomaly_model(make_splits(True), mlruns_dir=mlruns, models_dir=tmp_path / "m")
    assert not card["promoted"] and card["alias"] is None
    with pytest.raises(MlflowException):
        MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
