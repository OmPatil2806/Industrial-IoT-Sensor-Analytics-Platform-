"""File helpers shared by the data layers."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> Path:
    """Write to a temporary file, then rename, so a failure never leaves a partial table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
