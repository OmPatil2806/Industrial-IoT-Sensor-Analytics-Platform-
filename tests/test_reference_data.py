"""Tests for iiot.ingestion.reference_data."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from iiot.config import Plant, get_settings
from iiot.ingestion.reference_data import (
    COMPONENT_COSTS_FILE,
    MACHINE_LOCATION_FILE,
    build_component_costs,
    build_machine_location,
    run,
)

SETTINGS = get_settings()
LAYOUT = SETTINGS.plant_layout
MACHINES = list(range(1, 101))


def test_every_machine_assigned_exactly_once():
    loc = build_machine_location(MACHINES, LAYOUT)
    assert sorted(loc["machineID"]) == MACHINES
    assert loc["machineID"].is_unique


def test_machines_split_evenly_across_lines():
    loc = build_machine_location(MACHINES, LAYOUT)
    per_line = loc.groupby("line_id").size()
    assert len(per_line) == LAYOUT.total_lines
    assert per_line.max() - per_line.min() <= 1


def test_lines_belong_to_their_plant():
    loc = build_machine_location(MACHINES, LAYOUT)
    assert (loc["line_id"].str.split("-").str[0] == loc["plant_id"]).all()
    plants = {p.plant_id: p for p in LAYOUT.plants}
    for row in loc.itertuples():
        assert row.city == plants[row.plant_id].city


def test_assignment_is_reproducible_and_seed_dependent():
    a = build_machine_location(MACHINES, LAYOUT)
    b = build_machine_location(MACHINES, LAYOUT)
    c = build_machine_location(MACHINES, replace(LAYOUT, seed=LAYOUT.seed + 1))
    pd.testing.assert_frame_equal(a, b)
    assert not a["line_id"].equals(c["line_id"])


def test_uneven_split_differs_by_at_most_one():
    layout = replace(LAYOUT, plants=(Plant("A", "Plant A", "X", 3),))
    per_line = build_machine_location(list(range(1, 11)), layout).groupby("line_id").size()
    assert sorted(per_line) == [3, 3, 4]


def test_component_costs():
    business = SETTINGS.business
    costs = build_component_costs(business).set_index("component")
    assert list(costs.index) == list(business.components)
    rate = business.downtime_cost_per_hour
    for comp, c in business.components.items():
        row = costs.loc[comp]
        expected_unplanned = (
            c.repair_cost * business.emergency_repair_premium + c.unplanned_downtime_hours * rate
        )
        expected_planned = c.repair_cost + c.planned_downtime_hours * rate
        assert row["unplanned_failure_cost"] == pytest.approx(expected_unplanned)
        assert row["planned_maintenance_cost"] == pytest.approx(expected_planned)
        assert row["saving_if_prevented"] == pytest.approx(expected_unplanned - expected_planned)
    assert (costs["currency"] == business.currency).all()


def test_savings_differ_between_components():
    """Per-component downtime and the emergency premium make some failures costlier."""
    savings = build_component_costs(SETTINGS.business)["saving_if_prevented"]
    assert savings.nunique() == len(savings)
    assert (savings > 0).all()


def test_emergency_premium_raises_unplanned_cost():
    base = replace(SETTINGS.business, emergency_repair_premium=1.0)
    premium = replace(SETTINGS.business, emergency_repair_premium=2.0)
    low = build_component_costs(base)["unplanned_failure_cost"]
    high = build_component_costs(premium)["unplanned_failure_cost"]
    assert (high > low).all()


def test_run_writes_both_files(tmp_path):
    pd.DataFrame({"machineID": [1, 2, 3, 4, 5], "model": "model1", "age": 5}).to_csv(
        tmp_path / SETTINGS.dataset.files["machines"], index=False
    )
    run(raw_dir=tmp_path)
    loc = pd.read_csv(tmp_path / MACHINE_LOCATION_FILE)
    costs = pd.read_csv(tmp_path / COMPONENT_COSTS_FILE)
    assert sorted(loc["machineID"]) == [1, 2, 3, 4, 5]
    assert set(costs["component"]) == {"comp1", "comp2", "comp3", "comp4"}


def test_run_without_machines_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        run(raw_dir=tmp_path)
