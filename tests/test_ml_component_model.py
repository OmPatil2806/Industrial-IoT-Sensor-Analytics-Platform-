"""Tests for iiot.ml.component_model (small synthetic data, MLflow in a temporary folder)."""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from iiot.ml.component_model import (
    MODEL_CARD_FILE_NAME,
    MODEL_NAME,
    component_labels,
    confusion,
    evaluate_diagnosis,
    frequency_baseline,
    train_component_model,
)
from iiot.ml.data import split
from iiot.ml.failure_model import CHAMPION_ALIAS
from tests.test_ml_data import LABELS, VALIDATION_START, ml_dataset

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
COMP_LABELS = [f"{c}_fails_within_24h" for c in COMPONENTS]


def make_splits():
    """Each failing component has its own sensor signature, as in the real data."""
    df = ml_dataset()
    rng = np.random.default_rng(5)
    positive = df["fails_within_24h"].to_numpy() == 1
    which = np.full(len(df), -1)
    which[positive] = np.arange(positive.sum()) % 4
    for k, label in enumerate(COMP_LABELS):
        df[label] = (which == k).astype(int)
    df["failed_component"] = np.where(positive, np.array(COMPONENTS)[which], "none")
    noise = rng.normal(0, 2, (len(df), 4))
    df["volt_mean_3h"] = 170 + 25 * df["comp1_fails_within_24h"] + noise[:, 0]
    df["rotate_mean_3h"] = 450 - 60 * df["comp2_fails_within_24h"] + noise[:, 1]
    df["pressure_mean_3h"] = 100 + 20 * df["comp3_fails_within_24h"] + noise[:, 2]
    df["vibration_mean_3h"] = 40 + 15 * df["comp4_fails_within_24h"] + noise[:, 3]
    return split(df, VALIDATION_START, 24, LABELS, ("plant_id", "line_id"))


def rows(labels: list[list[int]]) -> pd.DataFrame:
    """Rows with the given per-component labels (target = any component fails)."""
    df = pd.DataFrame(labels, columns=COMP_LABELS)
    df["fails_within_24h"] = df[COMP_LABELS].max(axis=1)
    return df


def test_component_labels():
    assert component_labels(make_splits()) == COMP_LABELS


def test_diagnosis_metrics_worked_out_by_hand():
    df = rows([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    risk = pd.DataFrame(
        [
            [0.9, 0.1, 0.0, 0.0],  # right, exact set right
            [0.0, 0.6, 0.7, 0.0],  # top (comp3) failed; set {comp2, comp3} exactly right
            [0.0, 0.8, 0.0, 0.3],  # wrong: comp2 ranked first, comp4 failed
            [0.0, 0.0, 0.0, 0.0],  # no failure ahead: not judged
        ],
        columns=COMPONENTS,
    )
    m = evaluate_diagnosis(df, risk, COMP_LABELS, "fails_within_24h")
    assert m["rows_before_failure"] == 3
    assert m["top1_accuracy"] == pytest.approx(2 / 3)
    assert m["exact_set_accuracy"] == pytest.approx(2 / 3)
    assert m["pr_auc_comp1"] == 1.0


def test_confusion_counts_single_component_failures_only():
    df = rows([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    risk = pd.DataFrame(
        [[0.9, 0, 0, 0], [0, 0.9, 0, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 0.9]], columns=COMPONENTS
    )
    table = confusion(df, risk, COMP_LABELS, "fails_within_24h")
    assert table.loc["comp1", "comp1"] == 1 and table.loc["comp1", "comp2"] == 1
    assert table.loc["comp4", "comp4"] == 1
    assert int(table.to_numpy().sum()) == 3  # the 2-component failure is left out
    assert list(table.index) == list(table.columns) == COMPONENTS


def test_frequency_baseline_uses_fit_shares():
    splits = make_splits()
    baseline = frequency_baseline(splits, COMP_LABELS, "test")
    fit = splits.data["fit"]
    expected = fit.loc[fit["fails_within_24h"] == 1, COMP_LABELS].mean().to_numpy()
    assert len(baseline) == len(splits.data["test"])
    assert np.allclose(baseline.iloc[0].to_numpy(), expected)


def test_training_registers_a_champion_that_names_the_right_component(tmp_path):
    splits = make_splits()
    card = train_component_model(splits, mlruns_dir=tmp_path / "mlruns", models_dir=tmp_path / "m")
    mlflow.end_run()
    assert card["beats_baseline"] and card["alias"] == CHAMPION_ALIAS
    assert card["test"]["top1_accuracy"] >= 0.9 > card["test_baseline"]["top1_accuracy"]

    client = MlflowClient()
    assert str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version) == str(
        card["version"]
    )
    artifacts = {a.path for a in client.list_artifacts(card["run_id"], "charts")}
    assert artifacts == {"charts/confusion_test.png"}
    saved = json.loads((tmp_path / "m" / MODEL_CARD_FILE_NAME).read_text(encoding="utf-8"))
    assert saved["components"] == COMPONENTS

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    probabilities = model.predict_proba(splits.X("test"))
    assert len(probabilities) == 4  # one classifier per component
