"""Build the whole Silver layer and prove the cleaning works.

`build_silver()` builds every Silver table and writes
data/silver/silver_quality_report.json with:

    telemetry.typing      empty / unparseable values per column, rows dropped
    telemetry.cleaning    what each cleaning rule changed
    telemetry.quality     ok / filled / missing counts per sensor
    tables                rows in/out, dropped and flagged rows per table
    manifest_check        PROOF 1: Silver found exactly what was injected
                          (duplicates, spikes per sensor, stuck runs and values)
    ground_truth_check    PROOF 2: compared with the clean original telemetry,
                          every `ok` value is identical, and the error of the
                          filled values (with linear interpolation for comparison)
    passed                both proofs passed (a proof is skipped, not failed,
                          when its input file does not exist)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from iiot.config import get_settings
from iiot.ingestion.inject_issues import MANIFEST_FILE_NAME
from iiot.silver import tables, telemetry
from iiot.silver.cleaning import KEYS, CleaningStats
from iiot.utils.logger import get_logger

logger = get_logger("iiot.silver.report")

REPORT_FILE_NAME = "silver_quality_report.json"


def check_against_manifest(stats: CleaningStats, manifest: dict) -> dict:
    """Compare what the cleaning found with what the injection step recorded."""
    stuck = manifest["stuck_sensor"]
    episodes_per_sensor: dict[str, int] = {}
    for episode in stuck["episode_list"]:
        episodes_per_sensor[episode["sensor"]] = episodes_per_sensor.get(episode["sensor"], 0) + 1
    # In each stuck run the first value is real, so the repeats nulled = cells - episodes.
    expected_stuck_values = {
        s: cells - episodes_per_sensor.get(s, 0) for s, cells in stuck["cells_per_sensor"].items()
    }
    checks = {
        "duplicates": {
            "injected": manifest["duplicate_rows"],
            "found": stats.duplicates_removed,
        },
        "spikes": {"injected": manifest["spikes"], "found": stats.out_of_range_nulled},
        "stuck_runs": {
            "injected": stuck["episodes"],
            "found": sum(stats.stuck_runs.values()),
        },
        "stuck_values": {"injected": expected_stuck_values, "found": stats.stuck_values_nulled},
    }
    for check in checks.values():
        check["match"] = check["injected"] == check["found"]
    return {
        "status": "checked",
        "passed": all(c["match"] for c in checks.values()),
        "checks": checks,
    }


def load_ground_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path)
    truth = truth.rename(columns={"datetime": "timestamp", "machineID": "machine_id"})
    truth["timestamp"] = pd.to_datetime(truth["timestamp"]).astype("datetime64[ns]")
    truth["machine_id"] = truth["machine_id"].astype("Int64")
    return truth


def check_against_ground_truth(
    silver: pd.DataFrame, truth: pd.DataFrame, sensors: list[str]
) -> dict:
    """Compare cleaned telemetry with the clean original it was derived from."""
    silver = silver.assign(timestamp=silver["timestamp"].astype("datetime64[ns]"))
    joined = silver.merge(truth[KEYS + sensors], on=KEYS, how="inner", suffixes=("", "_true"))
    result: dict = {"status": "checked", "rows_compared": len(joined), "sensors": {}}
    for s in sensors:
        quality = joined[f"{s}_quality"]
        ok, filled = quality == "ok", quality == "filled"
        mismatches = int((joined.loc[ok, s] != joined.loc[ok, f"{s}_true"]).sum())

        # Linear interpolation on the same gaps, for comparison with the window mean.
        before_fill = joined[s].where(~filled)
        linear = before_fill.groupby(joined["machine_id"]).transform(
            lambda x: x.interpolate(limit_area="inside")
        )
        error = (joined.loc[filled, s] - joined.loc[filled, f"{s}_true"]).abs()
        linear_error = (linear[filled] - joined.loc[filled, f"{s}_true"]).abs()
        result["sensors"][s] = {
            "ok_values": int(ok.sum()),
            "ok_mismatches": mismatches,
            "filled_values": int(filled.sum()),
            "filled_mae": round(float(error.mean()), 4) if filled.any() else None,
            "linear_interpolation_mae": round(float(linear_error.mean()), 4)
            if filled.any()
            else None,
            "sensor_std": round(float(joined[f"{s}_true"].std()), 4),
        }
    result["passed"] = all(v["ok_mismatches"] == 0 for v in result["sensors"].values())
    return result


def build_silver(
    bronze_dir: Path | None = None,
    silver_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> dict:
    """Build all Silver tables, run both proofs, write and return the report."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    raw_dir = Path(raw_dir or settings.paths.raw)
    sensors = list(settings.sensors)

    cleaned, typing_stats, cleaning_stats = telemetry.build(bronze_dir, silver_dir)
    table_stats = tables.build_tables(bronze_dir, silver_dir)

    manifest_path = raw_dir / MANIFEST_FILE_NAME
    manifest_check = (
        check_against_manifest(
            cleaning_stats, json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if manifest_path.exists()
        else {"status": "skipped", "reason": f"{manifest_path.name} not found"}
    )
    truth_path = raw_dir / settings.dataset.files["telemetry"]
    truth_check = (
        check_against_ground_truth(cleaned, load_ground_truth(truth_path), sensors)
        if truth_path.exists()
        else {"status": "skipped", "reason": f"{truth_path.name} not found"}
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": manifest_check.get("passed", True) and truth_check.get("passed", True),
        "telemetry": {
            "typing": asdict(typing_stats),
            "cleaning": asdict(cleaning_stats),
            "quality": {
                s: {k: int(v) for k, v in cleaned[f"{s}_quality"].value_counts().items()}
                for s in sensors
            },
        },
        "tables": {name: asdict(stats) for name, stats in table_stats.items()},
        "manifest_check": manifest_check,
        "ground_truth_check": truth_check,
    }
    silver_dir.mkdir(parents=True, exist_ok=True)
    out = silver_dir / REPORT_FILE_NAME
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log_summary(report, out)
    return report


def show_report(silver_dir: Path | None = None) -> dict:
    """Log the summary of the latest silver_quality_report.json and return the report."""
    path = Path(silver_dir or get_settings().paths.silver) / REPORT_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot silver build` first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Report generated at %s", report["generated_at"])
    _log_summary(report, path)
    return report


def _log_summary(report: dict, out: Path) -> None:
    manifest = report["manifest_check"]
    if manifest["status"] == "checked":
        for name, check in manifest["checks"].items():
            found = check["found"]
            total = sum(found.values()) if isinstance(found, dict) else found
            logger.info(
                "Manifest %-13s %s  (%s found)",
                name,
                "MATCH" if check["match"] else "MISMATCH",
                f"{total:,}",
            )
    else:
        logger.warning("Manifest check skipped: %s", manifest["reason"])

    truth = report["ground_truth_check"]
    if truth["status"] == "checked":
        for s, v in truth["sensors"].items():
            logger.info(
                "Truth %-10s ok values identical: %s   filled MAE %s (linear would be %s)",
                s,
                "yes" if v["ok_mismatches"] == 0 else f"NO ({v['ok_mismatches']} differ)",
                v["filled_mae"],
                v["linear_interpolation_mae"],
            )
    else:
        logger.warning("Ground-truth check skipped: %s", truth["reason"])

    (logger.info if report["passed"] else logger.error)(
        "Silver quality report: %s -> %s", "PASSED" if report["passed"] else "FAILED", out
    )
