"""Shared Silver helpers: reading Bronze tables and writing Silver tables safely."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from iiot.config import get_settings


def read_bronze(table: str, bronze_dir: Path | None = None) -> pd.DataFrame:
    path = Path(bronze_dir or get_settings().paths.bronze) / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot bronze ingest` first.")
    return pd.read_parquet(path)


def write_silver(df: pd.DataFrame, table: str, silver_dir: Path | None = None) -> Path:
    """Write to a temporary file, then rename, so a failure never leaves a partial table."""
    silver_dir = Path(silver_dir or get_settings().paths.silver)
    silver_dir.mkdir(parents=True, exist_ok=True)
    out = silver_dir / f"{table}.parquet"
    tmp = out.with_name(f".{out.name}.tmp")
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out
