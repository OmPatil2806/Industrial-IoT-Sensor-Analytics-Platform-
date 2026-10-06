"""Generate business reference data that the source dataset does not contain.

Outputs (in data/raw/):
    ref_machine_location.csv   machineID -> plant, city, production line
    ref_component_costs.csv    per component: repair cost, repair hours, and the
                               total cost of an unplanned failure vs. planned maintenance

Plant layout and costs come from `plant_layout` and `business` in
config/settings.yaml. Costs are illustrative assumptions, not real plant figures.

Usage:
    python -m iiot.ingestion.reference_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import Business, PlantLayout, get_settings
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ingestion.reference_data")

MACHINE_LOCATION_FILE = "ref_machine_location.csv"
COMPONENT_COSTS_FILE = "ref_component_costs.csv"


def build_machine_location(machine_ids: list[int], layout: PlantLayout) -> pd.DataFrame:
    """Assign machines to production lines: shuffled with a fixed seed, split evenly."""
    lines = [
        {
            "plant_id": plant.plant_id,
            "plant_name": plant.name,
            "city": plant.city,
            "line_id": f"{plant.plant_id}-L{n}",
        }
        for plant in layout.plants
        for n in range(1, plant.lines + 1)
    ]
    rng = np.random.default_rng(layout.seed)
    shuffled = rng.permutation(sorted(machine_ids))
    rows = [
        {"machineID": int(machine_id), **line}
        for line, group in zip(lines, np.array_split(shuffled, len(lines)), strict=True)
        for machine_id in group
    ]
    return pd.DataFrame(rows).sort_values("machineID").reset_index(drop=True)


def build_component_costs(business: Business) -> pd.DataFrame:
    """Cost of each component failing unexpectedly vs. being replaced on schedule."""
    unplanned_h = business.repair_hours["unplanned_failure"]
    planned_h = business.repair_hours["planned_maintenance"]
    df = pd.DataFrame(
        {
            "component": list(business.component_repair_cost),
            "repair_cost": list(business.component_repair_cost.values()),
        }
    )
    df["unplanned_downtime_hours"] = unplanned_h
    df["planned_downtime_hours"] = planned_h
    df["downtime_cost_per_hour"] = business.downtime_cost_per_hour
    df["unplanned_failure_cost"] = df["repair_cost"] + unplanned_h * df["downtime_cost_per_hour"]
    df["planned_maintenance_cost"] = df["repair_cost"] + planned_h * df["downtime_cost_per_hour"]
    df["saving_if_prevented"] = df["unplanned_failure_cost"] - df["planned_maintenance_cost"]
    df["currency"] = business.currency
    return df


def run(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Build both reference tables from the machines file and config, and save them."""
    settings = get_settings()
    raw_dir = raw_dir or settings.paths.raw
    machines_file = raw_dir / settings.dataset.files["machines"]
    if not machines_file.exists():
        raise FileNotFoundError(
            f"{machines_file} not found. Run `python -m iiot.ingestion.download` first."
        )

    machine_ids = pd.read_csv(machines_file)["machineID"].tolist()
    location = build_machine_location(machine_ids, settings.plant_layout)
    costs = build_component_costs(settings.business)

    location.to_csv(raw_dir / MACHINE_LOCATION_FILE, index=False)
    costs.to_csv(raw_dir / COMPONENT_COSTS_FILE, index=False)

    per_line = location.groupby(["plant_id", "line_id"]).size()
    logger.info(
        "Assigned %d machines to %d plants / %d lines (%d-%d machines per line)",
        len(location),
        location["plant_id"].nunique(),
        len(per_line),
        per_line.min(),
        per_line.max(),
    )
    for row in costs.itertuples():
        logger.info(
            "  %s: failure %s %s vs planned %s %s (saving %s %s)",
            row.component,
            row.currency,
            f"{row.unplanned_failure_cost:,.0f}",
            row.currency,
            f"{row.planned_maintenance_cost:,.0f}",
            row.currency,
            f"{row.saving_if_prevented:,.0f}",
        )
    logger.info("Wrote %s and %s", MACHINE_LOCATION_FILE, COMPONENT_COSTS_FILE)
    return {"machine_location": location, "component_costs": costs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate business reference data.")
    parser.parse_args(argv)
    try:
        run()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
