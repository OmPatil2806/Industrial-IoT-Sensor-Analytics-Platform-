"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from iiot.config import DEFAULT_CONFIG_PATH


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., Path]:
    """Write a copy of the real settings.yaml with selected values overridden.

    Example:
        path = write_config({"sensors": {"volt": {"min": 10, "max": 5}}})
    """

    def _write(overrides: dict | None = None) -> Path:
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        for section, values in (overrides or {}).items():
            if isinstance(values, dict):
                raw[section].update(values)
            else:
                raw[section] = values
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    return _write
