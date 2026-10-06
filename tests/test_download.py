"""Tests for iiot.ingestion.download (no internet or Kaggle account needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iiot.config import get_settings
from iiot.ingestion.download import DownloadError, download_dataset, main

FILE_NAMES = list(get_settings().dataset.files.values())


def fake_downloader(subfolder: str = "", skip: tuple[str, ...] = ()):
    """Return a downloader that writes small CSVs instead of calling Kaggle."""
    calls: list[str] = []

    def _download(slug: str, dest: Path) -> None:
        calls.append(slug)
        out = dest / subfolder
        out.mkdir(parents=True, exist_ok=True)
        for name in FILE_NAMES:
            if name not in skip:
                (out / name).write_text("datetime,machineID\n2015-01-01 06:00:00,1\n")

    _download.calls = calls
    return _download


def test_downloads_all_files(tmp_path):
    downloader = fake_downloader()
    paths = download_dataset(raw_dir=tmp_path, downloader=downloader)
    assert [p.name for p in paths] == FILE_NAMES
    assert all(p.exists() for p in paths)
    assert downloader.calls == [get_settings().dataset.kaggle_slug]


def test_skips_download_when_files_exist(tmp_path):
    download_dataset(raw_dir=tmp_path, downloader=fake_downloader())
    second = fake_downloader()
    download_dataset(raw_dir=tmp_path, downloader=second)
    assert second.calls == []


def test_force_redownloads(tmp_path):
    download_dataset(raw_dir=tmp_path, downloader=fake_downloader())
    second = fake_downloader()
    download_dataset(raw_dir=tmp_path, downloader=second, force=True)
    assert len(second.calls) == 1


def test_files_extracted_into_subfolder_are_moved_up(tmp_path):
    paths = download_dataset(raw_dir=tmp_path, downloader=fake_downloader(subfolder="archive"))
    assert all(p.parent == tmp_path and p.exists() for p in paths)


def test_missing_file_after_download_raises(tmp_path):
    downloader = fake_downloader(skip=("PdM_errors.csv",))
    with pytest.raises(DownloadError, match="PdM_errors.csv"):
        download_dataset(raw_dir=tmp_path, downloader=downloader)


def test_failed_download_includes_manual_instructions(tmp_path):
    def failing(slug: str, dest: Path) -> None:
        raise DownloadError("Kaggle authentication failed: no credentials found.")

    with pytest.raises(DownloadError) as exc:
        download_dataset(raw_dir=tmp_path, downloader=failing)
    message = str(exc.value)
    assert "authentication failed" in message
    assert "kaggle auth login" in message
    assert "kaggle.com/datasets/" in message


def test_main_returns_error_code_on_failure(monkeypatch):
    def failing(*args, **kwargs):
        raise DownloadError("boom")

    logged: list[str] = []
    monkeypatch.setattr("iiot.ingestion.download.download_dataset", failing)
    monkeypatch.setattr(
        "iiot.ingestion.download.logger.error", lambda msg, *args: logged.append(msg % args)
    )
    assert main([]) == 1
    assert logged == ["boom"]
