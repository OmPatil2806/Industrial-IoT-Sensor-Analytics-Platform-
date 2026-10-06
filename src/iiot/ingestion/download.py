"""Download the Azure Predictive Maintenance dataset from Kaggle into data/raw/.

Usage:
    python -m iiot.ingestion.download            # skip if files already exist
    python -m iiot.ingestion.download --force    # always re-download

Kaggle authentication (any one of these):
    1. kaggle auth login                       (browser login, recommended)
    2. KAGGLE_API_TOKEN environment variable   (token from kaggle.com/settings/api)
    3. ~/.kaggle/access_token file             (same token, saved to a file)
    4. ~/.kaggle/kaggle.json                   (legacy username + key file)
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable
from pathlib import Path

from iiot.config import get_settings
from iiot.utils.logger import get_logger

# Explicit name so logs read "iiot.ingestion.download" even when run with `python -m`.
logger = get_logger("iiot.ingestion.download")

# A downloader receives (kaggle_slug, destination_dir) and saves the unzipped files there.
Downloader = Callable[[str, Path], None]


class DownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded or is incomplete."""


def manual_download_help(slug: str, raw_dir: Path) -> str:
    return (
        "Automatic download failed. Either set up Kaggle authentication:\n"
        "  - run:  kaggle auth login\n"
        "  - or create a token at https://www.kaggle.com/settings/api and set it in the\n"
        "    KAGGLE_API_TOKEN environment variable\n"
        "or download the dataset manually:\n"
        f"  1. Open https://www.kaggle.com/datasets/{slug}\n"
        "  2. Click 'Download' and extract the ZIP file\n"
        f"  3. Copy the 5 PdM_*.csv files into: {raw_dir}"
    )


def kaggle_downloader(slug: str, dest: Path) -> None:
    """Download and unzip a Kaggle dataset using the official Kaggle API."""
    try:
        # Imported here: the kaggle package tries to authenticate on import.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    except SystemExit as e:
        # The Kaggle library calls exit(1) when no credentials are found.
        raise DownloadError("Kaggle authentication failed: no credentials found.") from e
    except Exception as e:  # network errors, 403/404 from the API, etc.
        raise DownloadError(f"Kaggle download failed: {e}") from e


def _move_nested_files(raw_dir: Path, file_names: list[str]) -> None:
    """Move expected files to raw_dir if the ZIP extracted them into a subfolder."""
    for name in file_names:
        target = raw_dir / name
        if target.exists():
            continue
        found = next((p for p in raw_dir.rglob(name) if p.is_file()), None)
        if found is not None:
            shutil.move(str(found), target)
            logger.debug("Moved %s -> %s", found, target)


def download_dataset(
    force: bool = False,
    raw_dir: Path | None = None,
    downloader: Downloader = kaggle_downloader,
) -> list[Path]:
    """Ensure all source CSV files are present in raw_dir and return their paths."""
    settings = get_settings()
    raw_dir = raw_dir or settings.paths.raw
    slug = settings.dataset.kaggle_slug
    file_names = list(settings.dataset.files.values())
    paths = [raw_dir / name for name in file_names]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not force and all(p.exists() for p in paths):
        logger.info("All %d dataset files already in %s, skipping download", len(paths), raw_dir)
        return paths

    logger.info("Downloading Kaggle dataset '%s' into %s", slug, raw_dir)
    try:
        downloader(slug, raw_dir)
    except DownloadError as e:
        raise DownloadError(f"{e}\n\n{manual_download_help(slug, raw_dir)}") from e

    _move_nested_files(raw_dir, file_names)
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise DownloadError(
            f"Download finished but files are missing: {', '.join(missing)}\n\n"
            + manual_download_help(slug, raw_dir)
        )

    for p in paths:
        logger.info("  %-22s %8.1f MB", p.name, p.stat().st_size / 1_048_576)
    logger.info("Dataset download complete: %d files", len(paths))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Azure PdM dataset from Kaggle.")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args(argv)
    try:
        download_dataset(force=args.force)
    except DownloadError as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
