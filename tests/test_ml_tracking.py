"""Tests for iiot.ml.tracking: local MLflow tracking (always in a temporary folder)."""

from __future__ import annotations

import json

import mlflow
import pandas as pd
import pytest

from iiot.ml import tracking
from iiot.ml.data import split
from tests.test_ml_data import LABELS, VALIDATION_START, ml_dataset


@pytest.fixture
def mlruns(tmp_path):
    yield tmp_path / "mlruns"
    mlflow.end_run()  # never leave a run open for the next test


def test_tracking_uri_points_at_a_sqlite_file(tmp_path):
    uri = tracking.tracking_uri(tmp_path / "mlruns")
    assert uri.startswith("sqlite:///") and uri.endswith("/mlruns/mlflow.db")


def test_run_is_tagged_and_records_the_data(mlruns):
    splits = split(ml_dataset(), VALIDATION_START, 24, LABELS, ("plant_id",))
    with tracking.start_run("test-run", model_type="failure_24h", mlruns_dir=mlruns) as run:
        tracking.log_splits(splits)
        mlflow.log_metric("pr_auc", 0.5)

    logged = mlflow.get_run(run.info.run_id)
    assert logged.data.tags["model_type"] == "failure_24h"
    assert logged.data.tags["phase"] == tracking.PHASE
    assert logged.data.tags["git_commit"]
    assert logged.data.params["target"] == "fails_within_24h"
    assert logged.data.params["fit_rows"] == str(len(splits.data["fit"]))
    assert len(logged.data.params["data_fingerprint"]) == 16
    assert logged.data.metrics["pr_auc"] == 0.5
    assert (mlruns / tracking.DB_FILE_NAME).exists()

    saved = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="data")
    with open(f"{saved}/splits.json", encoding="utf-8") as f:
        record = json.load(f)
    assert record["features"] == splits.features
    assert set(record["splits"]) == {"fit", "validation", "development", "test"}


def test_experiment_is_created_once_and_reused(mlruns):
    first = tracking.setup(mlruns, "test-experiment")
    second = tracking.setup(mlruns, "test-experiment")
    assert first == second
    assert len(mlflow.search_experiments(filter_string="name = 'test-experiment'")) == 1


def test_runs_with_the_same_data_have_the_same_fingerprint(mlruns):
    splits = split(ml_dataset(), VALIDATION_START, 24, LABELS)
    ids = []
    for name in ("a", "b"):
        with tracking.start_run(name, model_type="test", mlruns_dir=mlruns) as run:
            tracking.log_splits(splits)
        ids.append(run.info.run_id)
    a, b = (mlflow.get_run(i).data.params["data_fingerprint"] for i in ids)
    assert a == b


def test_git_commit_outside_a_repository_is_unknown(tmp_path):
    assert tracking.git_commit(tmp_path) == "unknown"


def test_git_commit_in_the_project():
    commit = tracking.git_commit()
    assert commit != "unknown"
    assert pd.Series([commit]).str.match(r"^[0-9a-f]{7,}(-dirty)?$").all()
