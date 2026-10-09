"""Tests for iiot.ml.failure_model (small synthetic data, MLflow in a temporary folder)."""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from iiot.ml.data import split
from iiot.ml.evaluate import Costs
from iiot.ml.failure_model import (
    CHAMPION_ALIAS,
    MODEL_CARD_FILE_NAME,
    MODEL_NAME,
    model_signature,
    shuffled_label_check,
    train_failure_model,
)
from tests.test_ml_data import LABELS, VALIDATION_START, ml_dataset

# A false alarm costs a fifth of a caught failure's saving: with ~2% positive rows,
# alerting at random then loses money (as with the real costs).
COSTS = Costs(
    savings={c: 1000.0 for c in ("comp1", "comp2", "comp3", "comp4")}, false_alarm_cost=200
)
GRID = [{"learning_rate": 0.1, "max_leaf_nodes": 7}, {"learning_rate": 0.2, "max_leaf_nodes": 7}]


def make_splits(learnable: bool):
    df = ml_dataset()
    rng = np.random.default_rng(3)
    if learnable:  # voltage rises before a failure; error5 says nothing
        df["volt_mean_3h"] = 170 + 25 * df["fails_within_24h"] + rng.normal(0, 3, len(df))
    df["error5_count_24h"] = rng.integers(0, 2, len(df))
    return split(df, VALIDATION_START, 24, LABELS, ("plant_id", "line_id"))


@pytest.fixture
def train(tmp_path):
    def _train(learnable: bool = True) -> dict:
        return train_failure_model(
            make_splits(learnable),
            COSTS,
            mlruns_dir=tmp_path / "mlruns",
            models_dir=tmp_path / "models",
            grid=GRID,
        )

    yield _train
    mlflow.end_run()


def test_model_that_beats_the_baselines_becomes_champion(train, tmp_path):
    card = train()
    assert card["beats_baselines"] and card["alias"] == CHAMPION_ALIAS
    assert card["test"]["events_caught"] > card["test_baselines"]["error5-rule"]["events_caught"]
    assert card["test"]["net_saving"] > card["test_baselines"]["error5-rule"]["net_saving"]
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert str(version.version) == str(card["version"])

    saved = json.loads((tmp_path / "models" / MODEL_CARD_FILE_NAME).read_text(encoding="utf-8"))
    assert saved["threshold"] == card["threshold"]
    assert saved["features"] == make_splits(True).features


def test_run_logs_candidates_charts_card_and_a_reloadable_model(train):
    card = train()
    children = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{card['run_id']}'")
    assert len(children) == len(GRID)
    assert "metrics.val_net_saving" in children.columns

    client = MlflowClient()
    paths = {a.path for a in client.list_artifacts(card["run_id"], "charts")}
    assert paths == {
        "charts/pr_curve_test.png",
        "charts/net_saving_vs_threshold_validation.png",
    }
    run = client.get_run(card["run_id"])
    assert run.data.tags["beats_baselines"] == "True"
    assert run.data.params["threshold_rule"] == "highest net saving on validation"
    assert "test_error5-rule_net_saving" in run.data.metrics

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    splits = make_splits(True)
    scores = model.predict_proba(splits.X("test"))[:, 1]
    alerts = scores >= card["threshold"]
    assert int(alerts.sum()) == card["test"]["alerts"]


def test_model_without_signal_is_registered_but_not_champion(train):
    card = train(learnable=False)
    assert not card["beats_baselines"] and card["alias"] is None
    with pytest.raises(MlflowException):
        MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)


def test_shuffled_labels_score_near_chance():
    result = shuffled_label_check(make_splits(True), GRID[0], seed=42)
    assert result["passed"]
    assert result["shuffled_label_val_pr_auc"] < 3 * result["chance_pr_auc"]


def test_signature_describes_categories_as_text_and_numbers_as_floats():
    X = pd.DataFrame({"model": pd.Categorical(["model1"]), "error1_count_24h": [2], "age": [5]})
    signature = model_signature(X, np.array([0.5]))
    types = {col.name: col.type.name for col in signature.inputs.inputs}
    assert types == {"model": "string", "error1_count_24h": "double", "age": "double"}
