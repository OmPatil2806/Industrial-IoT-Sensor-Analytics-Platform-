"""MLflow experiment tracking, stored locally: no server and no account needed.

Every training run is recorded in a SQLite database, mlruns/mlflow.db, and its
files (models, charts, model cards) in mlruns/artifacts/. To browse the runs,
run this from the project folder and open http://localhost:5000:

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

Each run is tagged with the project phase, the model type and the git commit of
the code that trained it (with "-dirty" if there were uncommitted changes), and
`log_splits` records exactly which data it was trained and evaluated on.

Usage:
    from iiot.ml import tracking
    with tracking.start_run("failure-model", model_type="failure_24h"):
        tracking.log_splits(splits)
        mlflow.log_metric("pr_auc", 0.8)
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pandas as pd

from iiot.config import PROJECT_ROOT, get_settings
from iiot.ml.data import Splits, fingerprint
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.tracking")

DB_FILE_NAME = "mlflow.db"
ARTIFACTS_DIR_NAME = "artifacts"
PHASE = "7-machine-learning"


def tracking_uri(mlruns_dir: Path) -> str:
    """SQLite URI of the tracking database (works on Windows and Linux)."""
    return f"sqlite:///{(Path(mlruns_dir).resolve() / DB_FILE_NAME).as_posix()}"


def setup(mlruns_dir: Path | None = None, experiment: str | None = None) -> str:
    """Point MLflow at the local database and select the experiment. Returns its ID."""
    settings = get_settings()
    mlruns_dir = Path(mlruns_dir or settings.paths.mlruns)
    experiment = experiment or settings.ml.mlflow_experiment
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(mlruns_dir))
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        artifacts = (mlruns_dir.resolve() / ARTIFACTS_DIR_NAME).as_uri()
        experiment_id = mlflow.create_experiment(experiment, artifact_location=artifacts)
        logger.info("Created MLflow experiment '%s' in %s", experiment, mlruns_dir)
    else:
        experiment_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def git_commit(repo: Path = PROJECT_ROOT) -> str:
    """Current commit hash, with "-dirty" if the code has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if changes else commit


@contextmanager
def start_run(
    run_name: str,
    model_type: str,
    mlruns_dir: Path | None = None,
    experiment: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start a tagged MLflow run in the project experiment."""
    setup(mlruns_dir, experiment)
    tags = {"phase": PHASE, "model_type": model_type, "git_commit": git_commit()}
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        logger.info("MLflow run '%s' started (id %s)", run_name, run.info.run_id)
        yield run


def log_splits(splits: Splits) -> None:
    """Record which data the run used: fingerprint, sizes, positive rates and features."""
    summary = splits.summary()
    data = pd.concat([splits.data["development"], splits.data["test"]], ignore_index=True)
    params = {
        "data_fingerprint": fingerprint(data),
        "target": splits.target,
        "n_features": len(splits.features),
    }
    for part, info in summary.items():
        params[f"{part}_rows"] = info["rows"]
        params[f"{part}_period"] = f"{info['from'][:10]} to {info['to'][:10]}"
    mlflow.log_params(params)
    mlflow.log_metrics(
        {f"{part}_positive_rate": info["positive_rate"] for part, info in summary.items()}
    )
    mlflow.log_dict({"features": splits.features, "splits": summary}, "data/splits.json")
