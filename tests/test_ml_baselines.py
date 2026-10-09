"""Tests for iiot.ml.baselines (MLflow runs go to a temporary folder)."""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest

from iiot.ml.baselines import BASELINES, ErrorRule, NeverFail, run_baselines
from iiot.ml.data import split
from iiot.ml.evaluate import Costs
from tests.test_ml_data import LABELS, VALIDATION_START, ml_dataset

COSTS = Costs(
    savings={c: 1000.0 for c in ("comp1", "comp2", "comp3", "comp4")}, false_alarm_cost=10
)


@pytest.fixture
def splits():
    return split(ml_dataset(), VALIDATION_START, 24, LABELS, ("plant_id", "line_id"))


def test_baseline_scores():
    X = pd.DataFrame({"error5_count_24h": [0, 2, 1]})
    assert list(NeverFail().score(X)) == [0, 0, 0]
    alerts = ErrorRule().score(X) >= ErrorRule().threshold
    assert list(alerts) == [False, True, True]


def test_run_baselines_evaluates_and_logs_each_baseline(splits, tmp_path):
    results = run_baselines(splits, COSTS, mlruns_dir=tmp_path / "mlruns")
    assert set(results) == {b.name for b in BASELINES}
    never = results["never-fail"]
    assert never["validation"]["alerts"] == 0 and never["test"]["net_saving"] == 0
    assert results["error5-rule"]["validation"]["alerts"] > 0

    runs = mlflow.search_runs(filter_string="tags.model_type = 'baseline'")
    assert sorted(runs["tags.mlflow.runName"]) == ["error5-rule", "never-fail"]
    row = runs[runs["tags.mlflow.runName"] == "error5-rule"].iloc[0]
    assert row["params.rule"] == ErrorRule().description
    assert row["metrics.val_net_saving"] == results["error5-rule"]["validation"]["net_saving"]
    assert "metrics.test_pr_auc" in runs.columns
