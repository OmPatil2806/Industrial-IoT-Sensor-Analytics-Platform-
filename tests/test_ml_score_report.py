"""Tests for iiot.ml.score and iiot.ml.report: score with all three champions, then check.

The three models are trained once on small synthetic data where each failing
component has its own sensor signature (MLflow in a temporary folder).
"""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest

from iiot.ml.anomaly import train_anomaly_model
from iiot.ml.component_model import train_component_model
from iiot.ml.evaluate import Costs
from iiot.ml.failure_model import train_failure_model
from iiot.ml.report import build_ml_report
from iiot.ml.score import PREDICTIONS_FILE_NAME, score_all
from tests.test_ml_component_model import make_splits

COSTS = Costs(
    savings={c: 1000.0 for c in ("comp1", "comp2", "comp3", "comp4")}, false_alarm_cost=200
)
GRID = [{"learning_rate": 0.1, "max_leaf_nodes": 7}]
SENSOR = ["volt_mean_3h", "rotate_mean_3h", "pressure_mean_3h", "vibration_mean_3h"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("ml")
    mlruns, models, gold = root / "mlruns", root / "models", root / "gold"
    gold.mkdir()
    splits = make_splits()
    # Component savings must exist for the failure model's evaluation.
    monkeypatch_module.setattr("iiot.ml.report.load_costs", lambda: COSTS)
    train_failure_model(splits, COSTS, mlruns_dir=mlruns, models_dir=models, grid=GRID)
    train_component_model(splits, mlruns_dir=mlruns, models_dir=models)
    train_anomaly_model(splits, mlruns_dir=mlruns, models_dir=models)
    mlflow.end_run()

    rows = pd.concat([splits.data["development"], splits.data["test"]], ignore_index=True)
    keys = ["machine_id", "timestamp"]
    rows[keys + SENSOR].to_parquet(gold / "sensor_features.parquet")
    events = [
        c for c in rows.columns if c in ("error5_count_24h", "model", "age", "plant_id", "line_id")
    ]
    rows[keys + events].to_parquet(gold / "event_features.parquet")
    predictions = score_all(splits, gold, mlruns, rebuild_warehouse=False)
    return {
        "splits": splits,
        "gold": gold,
        "mlruns": mlruns,
        "models": models,
        "predictions": predictions,
    }


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def report(setup) -> dict:
    return build_ml_report(setup["splits"], setup["gold"], setup["mlruns"], setup["models"])


def test_every_feature_row_is_scored_by_all_three_models(setup):
    p = setup["predictions"]
    assert len(p) == len(setup["splits"].data["development"]) + len(setup["splits"].data["test"])
    assert p["failure_risk"].between(0, 1).all()
    assert set(p["predicted_component"]) <= {"comp1", "comp2", "comp3", "comp4"}
    assert p.loc[p["alert"], "reason_1"].notna().all()
    assert p.loc[~p["alert"], "reason_1"].isna().all()
    assert p["out_of_sample"].equals(p["timestamp"] >= pd.Timestamp("2015-10-01"))
    assert (setup["gold"] / PREDICTIONS_FILE_NAME).exists()


def test_report_passes_on_fresh_predictions(setup):
    result = report(setup)
    assert result["passed"], {k: c for k, c in result["checks"].items() if not c["passed"]}
    assert result["checks"]["reproducible"]["max_difference"] < 1e-9
    assert (setup["models"] / "ml_report.json").exists()


@pytest.mark.parametrize(
    ("corrupt", "check"),
    [
        (lambda p: p.assign(alert=~p["alert"]), "thresholds_applied"),
        (lambda p: p.iloc[1:], "predictions_complete"),
        (lambda p: p.assign(failure_model_version="99"), "predictions_complete"),
        (lambda p: p.assign(failure_risk=p["failure_risk"] * 0.5), "reproducible"),
    ],
    ids=["flipped alerts", "missing row", "old model version", "changed scores"],
)
def test_report_catches_bad_predictions(setup, corrupt, check):
    path = setup["gold"] / PREDICTIONS_FILE_NAME
    good = pd.read_parquet(path)
    try:
        corrupt(good).to_parquet(path)
        result = report(setup)
        assert not result["checks"][check]["passed"]
        assert not result["passed"]
    finally:
        good.to_parquet(path)


def test_report_without_predictions_raises(setup, tmp_path):
    with pytest.raises(FileNotFoundError, match="Score the models"):
        build_ml_report(setup["splits"], tmp_path, setup["mlruns"], setup["models"])
