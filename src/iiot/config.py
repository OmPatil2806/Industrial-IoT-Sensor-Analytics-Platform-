"""Load and validate config/settings.yaml.

Usage:
    from iiot.config import get_settings

    settings = get_settings()
    settings.paths.bronze                 # absolute Path to data/bronze
    settings.sensors["vibration"].max     # 100.0
    settings.ml.prediction_horizon_hours  # 24

Set the IIOT_CONFIG environment variable to use a different settings file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class Paths:
    raw: Path
    bronze: Path
    silver: Path
    gold: Path
    models: Path
    reports: Path
    logs: Path

    def ensure_exist(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Dataset:
    kaggle_slug: str
    files: dict[str, str]


@dataclass(frozen=True)
class SensorLimit:
    min: float
    max: float


@dataclass(frozen=True)
class DataQualityInjection:
    seed: int
    missing_rate: float
    spike_rate: float
    duplicate_rate: float
    stuck_episodes: int
    stuck_min_hours: int
    stuck_max_hours: int


@dataclass(frozen=True)
class Business:
    currency: str
    downtime_cost_per_hour: float
    repair_hours: dict[str, float]
    component_repair_cost: dict[str, float]


@dataclass(frozen=True)
class ML:
    prediction_horizon_hours: int
    train_end_date: date
    random_seed: int


@dataclass(frozen=True)
class Logging:
    level: str
    file_name: str


@dataclass(frozen=True)
class Settings:
    paths: Paths
    dataset: Dataset
    sensors: dict[str, SensorLimit]
    data_quality_injection: DataQualityInjection
    business: Business
    ml: ML
    logging: Logging


def _validate(settings: Settings) -> None:
    for name, limit in settings.sensors.items():
        if limit.min >= limit.max:
            raise ValueError(f"Sensor '{name}': min ({limit.min}) must be below max ({limit.max})")
    dq = settings.data_quality_injection
    for name in ("missing_rate", "spike_rate", "duplicate_rate"):
        if not 0.0 <= getattr(dq, name) <= 1.0:
            raise ValueError(f"data_quality_injection.{name} must be between 0 and 1")
    if dq.stuck_episodes < 0:
        raise ValueError("data_quality_injection.stuck_episodes must not be negative")
    if not 2 <= dq.stuck_min_hours <= dq.stuck_max_hours:
        raise ValueError("data_quality_injection: need 2 <= stuck_min_hours <= stuck_max_hours")
    if settings.ml.prediction_horizon_hours <= 0:
        raise ValueError("ml.prediction_horizon_hours must be positive")
    if settings.logging.level.upper() not in (
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ):
        raise ValueError(f"logging.level '{settings.logging.level}' is not a valid log level")


def load_settings(path: Path | str | None = None) -> Settings:
    """Read a settings YAML file and return a validated Settings object."""
    path = Path(path or os.environ.get("IIOT_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ml = raw["ml"]
    settings = Settings(
        paths=Paths(**{k: PROJECT_ROOT / v for k, v in raw["paths"].items()}),
        dataset=Dataset(**raw["dataset"]),
        sensors={k: SensorLimit(**v) for k, v in raw["sensors"].items()},
        data_quality_injection=DataQualityInjection(**raw["data_quality_injection"]),
        business=Business(**raw["business"]),
        ml=ML(
            prediction_horizon_hours=ml["prediction_horizon_hours"],
            train_end_date=date.fromisoformat(str(ml["train_end_date"])),
            random_seed=ml["random_seed"],
        ),
        logging=Logging(**raw["logging"]),
    )
    _validate(settings)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings loaded once and shared across the application."""
    return load_settings()
