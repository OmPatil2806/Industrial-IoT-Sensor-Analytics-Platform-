"""Load the Gold ML dataset and split it for model training.

Gold already splits by time into `train` (label window ends before
ml.train_end_date) and `test` (from ml.train_end_date). Here the Gold train rows
are split once more, the same way, so models can be tuned without touching test:

    fit          t + H < validation_start   used to fit the models
    validation   validation_start <= t      used to tune and choose the alert threshold
    development  fit + gap + validation     used to refit the final model
    test         t >= train_end             used ONCE, for the final evaluation

Rows between fit and validation (their label window crosses validation_start)
are dropped from fit and validation, for the same reason Gold drops them.

Features are every column except the keys, the labels, the split column and
ml.exclude_features. As a guard against leakage, any feature whose name looks
like a label ("fail") is rejected.

Usage:
    from iiot.ml.data import load_splits
    splits = load_splits()
    X, y = splits.X("fit"), splits.y("fit")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from iiot.config import get_settings
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.data")

DATASET_FILE_NAME = "ml_dataset.parquet"
KEYS = ["machine_id", "timestamp"]
PARTS = ("fit", "validation", "development", "test")


class MLDataError(ValueError):
    """The ML dataset is missing columns or would leak labels into the features."""


def label_columns(horizon_hours: int, components: list[str]) -> list[str]:
    """The label columns Gold writes for a given horizon and component list."""
    return [
        f"fails_within_{horizon_hours}h",
        *[f"{c}_fails_within_{horizon_hours}h" for c in components],
        "failed_component",
        "hours_to_failure",
    ]


def feature_columns(df: pd.DataFrame, labels: list[str], exclude: tuple[str, ...]) -> list[str]:
    """Every model input column, in dataset order."""
    missing = [c for c in [*KEYS, *labels, "split"] if c not in df.columns]
    if missing:
        raise MLDataError(f"ML dataset is missing columns: {missing}")
    dropped = set(KEYS) | set(labels) | {"split"} | set(exclude)
    features = [c for c in df.columns if c not in dropped]
    suspicious = [c for c in features if "fail" in c.lower()]
    if suspicious:
        raise MLDataError(f"Columns look like labels and must not be features: {suspicious}")
    return features


def fingerprint(df: pd.DataFrame) -> str:
    """A short hash of the data, logged with every model so it can be traced to its data."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class Splits:
    data: dict[str, pd.DataFrame]
    features: list[str]
    target: str
    labels: list[str]

    def X(self, part: str) -> pd.DataFrame:  # model inputs; "X" as in scikit-learn
        return self.data[part][self.features]

    def y(self, part: str, target: str | None = None) -> pd.Series:
        return self.data[part][target or self.target]

    def summary(self) -> dict:
        """Rows, positives, positive rate and time range per part (logged to MLflow)."""
        out = {}
        for part, df in self.data.items():
            y = df[self.target]
            out[part] = {
                "rows": len(df),
                "positives": int(y.sum()),
                "positive_rate": round(float(y.mean()), 5) if len(df) else 0.0,
                "from": str(df["timestamp"].min()),
                "to": str(df["timestamp"].max()),
            }
        return out


def split(
    df: pd.DataFrame,
    validation_start: pd.Timestamp,
    horizon_hours: int,
    labels: list[str],
    exclude: tuple[str, ...] = (),
) -> Splits:
    """Split the Gold ML dataset into fit / validation / development / test."""
    features = feature_columns(df, labels, exclude)
    is_train = df["split"].astype(str) == "train"
    is_test = df["split"].astype(str) == "test"
    t = df["timestamp"]
    horizon = pd.Timedelta(hours=horizon_hours)
    development = df[is_train]
    test = df[is_test]
    if len(development) == 0 or len(test) == 0:
        raise MLDataError("The ML dataset needs both train and test rows")
    if development["timestamp"].max() >= test["timestamp"].min():
        raise MLDataError("Train rows must all come before the test rows")
    data = {
        "fit": df[is_train & (t + horizon < validation_start)],
        "validation": df[is_train & (t >= validation_start)],
        "development": development,
        "test": test,
    }
    data = {part: d.reset_index(drop=True) for part, d in data.items()}
    for part in ("fit", "validation"):
        if len(data[part]) == 0:
            raise MLDataError(f"No {part} rows: check ml.validation_start_date")
    return Splits(data=data, features=features, target=labels[0], labels=labels)


def load_dataset(gold_dir: Path | None = None) -> pd.DataFrame:
    path = Path(gold_dir or get_settings().paths.gold) / DATASET_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot gold build` first.")
    return pd.read_parquet(path)


def load_splits(gold_dir: Path | None = None) -> Splits:
    """Load the Gold ML dataset and split it using the settings in config."""
    settings = get_settings()
    ml = settings.ml
    labels = label_columns(ml.prediction_horizon_hours, list(settings.business.components))
    splits = split(
        load_dataset(gold_dir),
        pd.Timestamp(ml.validation_start_date),
        ml.prediction_horizon_hours,
        labels,
        ml.exclude_features,
    )
    for part, info in splits.summary().items():
        logger.info(
            "%-12s %8s rows, %5.2f%% positive, %s to %s",
            part,
            f"{info['rows']:,}",
            100 * info["positive_rate"],
            info["from"][:10],
            info["to"][:10],
        )
    logger.info("%d features (excluded: %s)", len(splits.features), ", ".join(ml.exclude_features))
    return splits
