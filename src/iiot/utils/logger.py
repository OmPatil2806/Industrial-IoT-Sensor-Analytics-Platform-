"""Project-wide logging.

Usage:
    from iiot.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Loaded %s: %d rows", file_name, len(df))

Messages go to the console and to logs/pipeline.log (rotated at 5 MB).
The level and file name come from the `logging` section of config/settings.yaml.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from iiot.config import get_settings

ROOT_LOGGER_NAME = "iiot"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3


def _configure_root_logger() -> logging.Logger:
    """Attach console and file handlers to the 'iiot' logger exactly once."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if root.handlers:
        return root

    settings = get_settings()
    root.setLevel(settings.logging.level.upper())
    root.propagate = False
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    settings.paths.logs.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.paths.logs / settings.logging.file_name,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the 'iiot' namespace, e.g. get_logger(__name__)."""
    _configure_root_logger()
    if name != ROOT_LOGGER_NAME and not name.startswith(ROOT_LOGGER_NAME + "."):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
