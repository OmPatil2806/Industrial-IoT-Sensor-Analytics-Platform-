"""Tests for iiot.ml.explain (small synthetic data, MLflow in a temporary folder)."""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from iiot.ml.explain import (
    NO_REASON,
    NormalProfile,
    describe,
    explain_champion,
    explain_rows,
    feature_group,
    group_importance,
    reason_inputs_by_component,
)
from iiot.ml.failure_model import make_model, train_failure_model
from tests.test_ml_failure_model import COSTS, GRID, make_splits


@pytest.fixture(scope="module")
def splits():
    return make_splits(learnable=True)  # voltage rises before a failure; error5 is noise


@pytest.fixture(scope="module")
def model(splits):
    return make_model(GRID[0], seed=42).fit(splits.X("development"), splits.y("development"))


@pytest.mark.parametrize(
    ("feature", "group"),
    [
        ("volt_mean_24h", "voltage"),
        ("vibration_trend", "vibration"),
        ("error3_count_24h", "errors"),
        ("errors_total_24h", "errors"),
        ("hours_since_comp2_replaced", "component wear"),
        ("age", "machine"),
    ],
)
def test_feature_groups(feature, group):
    assert feature_group(feature) == group


@pytest.mark.parametrize(
    ("feature", "value", "normal", "text"),
    [
        ("vibration_mean_24h", 48.0, 40.0, "vibration 24 h mean 48.0 vs normal 40.0 (+20%)"),
        ("rotate_std_3h", 30.0, 15.0, "rotation 3 h variability 30.0 vs normal 15.0 (+100%)"),
        ("pressure_trend", 4.2, 0.1, "pressure trend (3 h vs 24 h mean) +4.2 (normal +0.1)"),
        ("error5_count_24h", 2, 0, "error5 raised 2 times in the last 24 h (normal 0)"),
        ("errors_total_24h", 3, 1, "errors (all types) raised 3 times in the last 24 h (normal 1)"),
        ("hours_since_comp2_replaced", 1250, 410, "comp2 last replaced 1,250 h ago (normal 410 h)"),
    ],
)
def test_describe(feature, value, normal, text):
    assert describe(feature, value, normal) == text


def test_normal_profile_uses_healthy_rows_per_machine():
    df = pd.DataFrame(
        {
            "machine_id": [1, 1, 1, 2, 2],
            "volt_mean_3h": [170.0, 172.0, 999.0, 160.0, 160.0],
            "fails_within_24h": [0, 0, 1, 0, 0],
        }
    )
    profile = NormalProfile.from_rows(df, ["volt_mean_3h"], "fails_within_24h")
    assert profile.median.loc[1, "volt_mean_3h"] == 171.0  # the failure row is ignored
    assert profile.median.loc[2, "volt_mean_3h"] == 160.0
    assert profile.std.loc[2, "volt_mean_3h"] > 0  # constant machine: fleet spread instead


def test_alert_reason_names_the_input_that_raised_the_risk(splits, model):
    test = splits.data["test"]
    profile = NormalProfile.from_rows(splits.data["development"], splits.features, splits.target)
    risk = model.predict_proba(splits.X("test"))[:, 1]
    rows = test[(test[splits.target] == 1) & (risk >= 0.5)].head(3)  # alerts before failures
    assert len(rows) == 3
    reasons = explain_rows(model, rows[splits.features], rows["machine_id"], profile)
    assert (reasons["reason_1_group"] == "voltage").all()
    assert reasons["reason_1"].str.startswith("voltage 3 h mean").all()
    assert (reasons["reason_1_input"] == "voltage").all()
    assert (reasons["risk_drop_1"] >= 0.05).all()


def test_normal_row_has_no_reason(splits, model):
    test = splits.data["test"]
    profile = NormalProfile.from_rows(splits.data["development"], splits.features, splits.target)
    row = test[test[splits.target] == 0].head(1)
    reasons = explain_rows(model, row[splits.features], row["machine_id"], profile)
    assert reasons.loc[0, "reason_1"] == NO_REASON
    assert reasons.loc[0, "reason_1_group"] is None


def test_group_importance_finds_the_signal(splits, model):
    table = group_importance(model, splits.X("test"), splits.y("test"), repeats=2)
    assert table.iloc[0]["group"] == "voltage"
    assert table.set_index("group").loc["voltage", "importance"] > 0.3
    assert abs(table.set_index("group").loc["errors", "importance"]) < 0.1  # error5 is noise


def test_reason_inputs_count_each_alert_once():
    alerts = pd.DataFrame(
        {
            "failed_component": ["comp1", "comp1", "comp2"],
            "reason_1_input": ["voltage", "error1", "rotation"],
            "reason_2_input": ["error1", "error1", None],
        }
    )
    table = reason_inputs_by_component(alerts)
    assert table.loc["comp1", "error1"] == 1.0  # both comp1 alerts, the second counted once
    assert table.loc["comp1", "voltage"] == 0.5
    assert table.loc["comp2", "rotation"] == 1.0


def test_explain_champion_logs_to_the_champion_run(splits, tmp_path):
    mlruns = tmp_path / "mlruns"
    card = train_failure_model(
        splits, COSTS, mlruns_dir=mlruns, models_dir=tmp_path / "models", grid=GRID[:1]
    )
    summary = explain_champion(splits, mlruns_dir=mlruns)
    mlflow.end_run()
    assert summary["run_id"] == card["run_id"]
    assert summary["group_importance"][0]["group"] == "voltage"
    assert summary["alerts_explained"] == card["test"]["alerts"]

    client = MlflowClient()
    files = {a.path for a in client.list_artifacts(card["run_id"], "explain")}
    assert files == {
        "explain/importance_by_group_test.png",
        "explain/importance_by_feature_test.png",
        "explain/importance_by_group_test.csv",
        "explain/importance_by_feature_test.csv",
        "explain/alert_reasons_test.csv",
        "explain/reason_inputs_by_component_test.csv",
    }
    assert client.get_run(card["run_id"]).data.tags["explained"] == "true"
    assert summary["reason_inputs_by_component"]["comp1"]["voltage"] > 0.5
