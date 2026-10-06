"""Tests for iiot.utils.logger."""

from __future__ import annotations

import logging

import pytest

from iiot.config import get_settings
from iiot.utils.logger import _HANDLER_TAG, ROOT_LOGGER_NAME, get_logger


def _reset_logging() -> None:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    get_settings.cache_clear()


@pytest.fixture
def log_dir(tmp_path, write_config, monkeypatch):
    """Point logging at a temporary folder so tests never touch the real logs/."""
    logs = tmp_path / "logs"
    monkeypatch.setenv("IIOT_CONFIG", str(write_config({"paths": {"logs": str(logs)}})))
    _reset_logging()
    yield logs
    _reset_logging()


def test_messages_are_written_to_log_file(log_dir):
    get_logger("iiot.test").info("Loaded %s rows", 42)
    content = (log_dir / "pipeline.log").read_text(encoding="utf-8")
    assert "| INFO     | iiot.test | Loaded 42 rows" in content


def _own_handlers() -> list[logging.Handler]:
    return [
        h for h in logging.getLogger(ROOT_LOGGER_NAME).handlers if getattr(h, _HANDLER_TAG, False)
    ]


def test_handlers_are_added_only_once(log_dir):
    for _ in range(3):
        get_logger("iiot.test")
    assert len(_own_handlers()) == 2


def test_configures_even_if_another_tool_added_a_handler(log_dir):
    """A foreign handler (e.g. from pytest or a library) must not block our setup."""
    foreign = logging.NullHandler()
    logging.getLogger(ROOT_LOGGER_NAME).addHandler(foreign)
    get_logger("iiot.test").info("still logged")
    assert len(_own_handlers()) == 2
    assert "still logged" in (log_dir / "pipeline.log").read_text(encoding="utf-8")


def test_names_are_placed_under_iiot_namespace(log_dir):
    assert get_logger("bronze").name == "iiot.bronze"
    assert get_logger("iiot.silver").name == "iiot.silver"
    assert get_logger("bronze") is get_logger("iiot.bronze")


def test_debug_messages_hidden_at_info_level(log_dir):
    logger = get_logger("iiot.test")
    logger.debug("hidden detail")
    logger.info("visible")
    content = (log_dir / "pipeline.log").read_text(encoding="utf-8")
    assert "visible" in content
    assert "hidden detail" not in content
