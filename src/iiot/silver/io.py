"""Shared Silver helpers: reading Bronze tables and writing Silver tables safely."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from iiot.config import get_settings
from iiot.utils.io import write_parquet_atomic


def read_bronze(table: str, bronze_dir: Path | None = None) -> pd.DataFrame:
    path = Path(bronze_dir or get_settings().paths.bronze) / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot bronze ingest` first.")
    return pd.read_parquet(path)


def write_silver(df: pd.DataFrame, table: str, silver_dir: Path | None = None) -> Path:
    """Write a Silver table atomically (a failure never leaves a partial table)."""
    silver_dir = Path(silver_dir or get_settings().paths.silver)
    return write_parquet_atomic(df, silver_dir / f"{table}.parquet")
