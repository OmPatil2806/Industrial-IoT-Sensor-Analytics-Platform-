"""Command-line interface: `iiot <group> <command>`.

Examples:
    iiot data prepare            # download -> validate -> inject issues -> reference data
    iiot data download --force   # re-download the Kaggle dataset
    iiot data validate           # run the raw data contract only

Also available as `python -m iiot ...`.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from iiot.ingestion import download, inject_issues, reference_data, validate_raw
from iiot.utils.logger import get_logger

logger = get_logger("iiot.cli")


class StepFailed(RuntimeError):
    """A pipeline step reported failure; later steps must not run."""


def _download(args: argparse.Namespace) -> None:
    download.download_dataset(force=getattr(args, "force", False))


def _validate(args: argparse.Namespace) -> None:
    if not validate_raw.validate_raw().passed:
        raise StepFailed("Raw data contract failed - see the FAIL lines above.")


def _inject(args: argparse.Namespace) -> None:
    inject_issues.run()


def _reference(args: argparse.Namespace) -> None:
    reference_data.run()


DATA_STEPS: dict[str, tuple[Callable[[argparse.Namespace], None], str]] = {
    "download": (_download, "download the Azure PdM dataset from Kaggle"),
    "validate": (_validate, "check the raw files against the data contract"),
    "inject": (_inject, "write the telemetry copy with injected data-quality issues"),
    "reference": (_reference, "generate plant/line and component cost reference data"),
}


def _run_steps(names: list[str], args: argparse.Namespace) -> int:
    total_start = time.perf_counter()
    for i, name in enumerate(names, 1):
        func, description = DATA_STEPS[name]
        logger.info("[%d/%d] %s: %s", i, len(names), name, description)
        start = time.perf_counter()
        try:
            func(args)
        except (StepFailed, download.DownloadError, FileNotFoundError) as e:
            logger.error("Step '%s' failed: %s", name, e)
            if i < len(names):
                logger.error("Stopped. Remaining steps not run: %s", ", ".join(names[i:]))
            return 1
        logger.info("[%d/%d] %s done in %.1fs", i, len(names), name, time.perf_counter() - start)
    if len(names) > 1:
        logger.info("All %d steps done in %.1fs", len(names), time.perf_counter() - total_start)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iiot", description="Industrial IoT Sensor Analytics Platform"
    )
    groups = parser.add_subparsers(dest="group", required=True)

    data = groups.add_parser("data", help="acquire and prepare the raw data (Phase 2)")
    commands = data.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="run all data preparation steps in order")
    prepare.add_argument(
        "--force-download", dest="force", action="store_true", help="re-download the dataset"
    )
    for name, (_, description) in DATA_STEPS.items():
        sub = commands.add_parser(name, help=description)
        if name == "download":
            sub.add_argument("--force", action="store_true", help="re-download even if present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.group == "data":
        names = list(DATA_STEPS) if args.command == "prepare" else [args.command]
        return _run_steps(names, args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
