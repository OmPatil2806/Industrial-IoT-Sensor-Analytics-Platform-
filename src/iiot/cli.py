"""Command-line interface: `iiot <group> <command>`.

Examples:
    iiot run                     # full pipeline: data -> Bronze -> Silver -> Gold -> warehouse
    iiot data prepare            # download -> validate -> inject issues -> reference data
    iiot data download --force   # re-download the Kaggle dataset
    iiot data validate           # run the raw data contract only
    iiot bronze ingest           # load raw files into Bronze (unchanged files skipped)
    iiot bronze report           # verify Bronze tables with DuckDB
    iiot silver build            # clean data into Silver and prove the cleaning worked
    iiot silver report           # show the latest Silver quality report
    iiot gold build              # features, labels, ML dataset, KPIs + quality checks
    iiot gold report             # show the latest Gold report
    iiot model build             # build the DuckDB star-schema warehouse + model checks
    iiot model report            # show the latest model report

Also available as `python -m iiot ...`.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from iiot.bronze import pipeline as bronze_pipeline
from iiot.bronze import report as bronze_report
from iiot.bronze.ingest import BronzeError
from iiot.gold import report as gold_report
from iiot.ingestion import download, inject_issues, reference_data, validate_raw
from iiot.model import report as model_report
from iiot.model.warehouse import WarehouseLockedError
from iiot.silver import report as silver_report
from iiot.utils.logger import get_logger

logger = get_logger("iiot.cli")


class StepFailed(RuntimeError):
    """A pipeline step reported failure; later steps must not run."""


def _download(args: argparse.Namespace) -> None:
    download.download_dataset(force=getattr(args, "force_download", False))


def _validate(args: argparse.Namespace) -> None:
    if not validate_raw.validate_raw().passed:
        raise StepFailed("Raw data contract failed - see the FAIL lines above.")


def _inject(args: argparse.Namespace) -> None:
    inject_issues.run()


def _reference(args: argparse.Namespace) -> None:
    reference_data.run()


def _bronze_ingest(args: argparse.Namespace) -> None:
    bronze_pipeline.ingest_all(force=getattr(args, "force_bronze", False))


def _bronze_report(args: argparse.Namespace) -> None:
    if not bronze_report.build_report().passed:
        raise StepFailed("Bronze verification failed - see the FAIL lines above.")


def _silver_build(args: argparse.Namespace) -> None:
    if not silver_report.build_silver()["passed"]:
        raise StepFailed("Silver quality checks failed - see the MISMATCH/NO lines above.")


def _silver_report(args: argparse.Namespace) -> None:
    if not silver_report.show_report()["passed"]:
        raise StepFailed("The latest Silver quality report did not pass.")


def _gold_build(args: argparse.Namespace) -> None:
    if not gold_report.build_gold()["passed"]:
        raise StepFailed("Gold checks failed - see the FAIL lines above.")


def _gold_report(args: argparse.Namespace) -> None:
    if not gold_report.show_report()["passed"]:
        raise StepFailed("The latest Gold report did not pass.")


def _model_build(args: argparse.Namespace) -> None:
    if not model_report.build_model()["passed"]:
        raise StepFailed("Model checks failed - see the FAIL lines above.")


def _model_report(args: argparse.Namespace) -> None:
    if not model_report.show_report()["passed"]:
        raise StepFailed("The latest model report did not pass.")


Step = tuple[Callable[[argparse.Namespace], None], str]

# Every pipeline step, keyed by name. Groups and `iiot run` pick steps from here.
STEPS: dict[str, Step] = {
    "download": (_download, "download the Azure PdM dataset from Kaggle"),
    "validate": (_validate, "check the raw files against the data contract"),
    "inject": (_inject, "write the telemetry copy with injected data-quality issues"),
    "reference": (_reference, "generate plant/line and component cost reference data"),
    "bronze-ingest": (_bronze_ingest, "load raw files into Bronze Parquet tables"),
    "bronze-report": (_bronze_report, "verify Bronze tables with DuckDB"),
    "silver-build": (_silver_build, "clean Bronze into Silver tables and prove the cleaning"),
    "silver-report": (_silver_report, "show the latest Silver quality report"),
    "gold-build": (_gold_build, "build features, labels, ML dataset and KPIs, and check them"),
    "gold-report": (_gold_report, "show the latest Gold report"),
    "model-build": (_model_build, "build the star-schema warehouse and check it"),
    "model-report": (_model_report, "show the latest model report"),
}
DATA_STEPS = ["download", "validate", "inject", "reference"]
BRONZE_STEPS = ["bronze-ingest", "bronze-report"]
FULL_PIPELINE = DATA_STEPS + BRONZE_STEPS + ["silver-build", "gold-build", "model-build"]


def _run_steps(names: list[str], args: argparse.Namespace) -> int:
    total_start = time.perf_counter()
    for i, name in enumerate(names, 1):
        func, description = STEPS[name]
        logger.info("[%d/%d] %s: %s", i, len(names), name, description)
        start = time.perf_counter()
        try:
            func(args)
        except (
            StepFailed,
            download.DownloadError,
            BronzeError,
            FileNotFoundError,
            WarehouseLockedError,
        ) as e:
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

    run = groups.add_parser(
        "run", help="run the full pipeline (data -> Bronze -> Silver -> Gold -> warehouse)"
    )
    run.add_argument("--force-download", action="store_true", help="re-download the dataset")
    run.add_argument("--force-bronze", action="store_true", help="reload every Bronze table")

    data = groups.add_parser("data", help="acquire and prepare the raw data (Phase 2)")
    data_cmds = data.add_subparsers(dest="command", required=True)
    prepare = data_cmds.add_parser("prepare", help="run all data preparation steps in order")
    prepare.add_argument("--force-download", action="store_true", help="re-download the dataset")
    for name in DATA_STEPS:
        sub = data_cmds.add_parser(name, help=STEPS[name][1])
        if name == "download":
            sub.add_argument(
                "--force",
                dest="force_download",
                action="store_true",
                help="re-download even if present",
            )

    bronze = groups.add_parser("bronze", help="load and verify the Bronze layer (Phase 3)")
    bronze_cmds = bronze.add_subparsers(dest="command", required=True)
    ingest = bronze_cmds.add_parser("ingest", help=STEPS["bronze-ingest"][1])
    ingest.add_argument(
        "--force", dest="force_bronze", action="store_true", help="reload even unchanged files"
    )
    bronze_cmds.add_parser("report", help=STEPS["bronze-report"][1])

    silver = groups.add_parser("silver", help="clean data into the Silver layer (Phase 4)")
    silver_cmds = silver.add_subparsers(dest="command", required=True)
    silver_cmds.add_parser("build", help=STEPS["silver-build"][1])
    silver_cmds.add_parser("report", help=STEPS["silver-report"][1])

    gold = groups.add_parser("gold", help="build ML features, labels and KPIs (Phase 5)")
    gold_cmds = gold.add_subparsers(dest="command", required=True)
    gold_cmds.add_parser("build", help=STEPS["gold-build"][1])
    gold_cmds.add_parser("report", help=STEPS["gold-report"][1])

    model = groups.add_parser("model", help="build and check the DuckDB warehouse (Phase 6)")
    model_cmds = model.add_subparsers(dest="command", required=True)
    model_cmds.add_parser("build", help=STEPS["model-build"][1])
    model_cmds.add_parser("report", help=STEPS["model-report"][1])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.group == "run":
        names = FULL_PIPELINE
    elif args.group == "data":
        names = DATA_STEPS if args.command == "prepare" else [args.command]
    else:  # bronze, silver, gold or model
        names = [f"{args.group}-{args.command}"]
    return _run_steps(names, args)


if __name__ == "__main__":
    raise SystemExit(main())
