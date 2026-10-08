"""Tests for the analytical views (v_*) and docs/example_queries.sql.

Uses the tiny warehouse from conftest.py: two machines on two lines; the analysis
period runs from 2015-01-01 06:00 to 2015-01-10 00:00. Machine 1 has one failure
event at 2015-01-05 06:00 (comp1 + comp2, 10 h downtime, cost 1,050,000); machine 2
has one error. The only planned maintenance (2014-12-30) is before the period.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from iiot.model.warehouse import connect, view_names

EXAMPLE_QUERIES = Path(__file__).parents[1] / "docs" / "example_queries.sql"
VIEWS = ["v_component_reliability", "v_fleet_monthly", "v_line_performance", "v_machine_health"]


@pytest.fixture
def con(built):
    path, _ = built
    with connect(path) as c:
        yield c


def rows(con, sql: str) -> list[dict]:
    return con.sql(sql).df().to_dict("records")


def test_all_views_are_created(con):
    assert view_names(con) == VIEWS


def test_fleet_monthly_recomputes_ratios_from_sums(con):
    [month] = rows(con, "SELECT * FROM v_fleet_monthly")
    assert month["year_month"] == "2015-01"
    assert (month["machines"], month["failures"], month["components_failed"]) == (2, 1, 2)
    assert month["availability"] == pytest.approx(1 - 10 / (2 * 738))
    assert month["mtbf_h"] == pytest.approx(2 * 738 - 10)
    assert month["mttr_h"] == 10
    assert month["total_cost"] == 1_050_000


def test_machine_health(con):
    m1, m2 = rows(con, "SELECT * FROM v_machine_health ORDER BY machine_id")
    assert (m1["failures"], m1["cost_rank"], m1["errors"]) == (1, 1, 0)
    assert m1["days_since_last_failure"] == pytest.approx(4.75)  # 01-05 06:00 -> 01-10 00:00
    assert m1["filled_share"] == pytest.approx(0.25)  # 1 of 4 readings filled
    assert m1["missing_share"] == 0
    assert (m2["failures"], m2["cost_rank"], m2["errors"], m2["errors_last_30d"]) == (0, 2, 1, 1)
    assert m2["missing_share"] == pytest.approx(0.25)
    assert m2["mtbf_h"] != m2["mtbf_h"]  # NaN: no failures, no MTBF
    assert str(m2["last_failure_at"]) == "NaT"


def test_component_reliability_counts_only_the_analysis_period(con):
    comp1, comp2 = rows(con, "SELECT * FROM v_component_reliability")
    assert (comp1["failures"], comp1["in_multi_component_events"]) == (1, 1)
    assert comp1["failure_cost"] == 460_000
    assert comp1["saving_if_all_prevented"] == 320_000
    # comp2's planned replacement on 2014-12-30 is before the period: not counted.
    assert (comp2["failures"], comp2["planned_replacements"], comp2["unplanned_share"]) == (1, 0, 1)
    assert comp1["mean_days_between_failures"] != comp1["mean_days_between_failures"]  # NaN


def test_mean_days_between_failures(built):
    path, _ = built
    with duckdb.connect(str(path)) as c:
        c.execute(
            "INSERT INTO fact_failure "
            "VALUES (1, 20150108, TIMESTAMP '2015-01-08 06:00', 'comp1', 1, 8, 460000)"
        )
        gap = c.sql(
            "SELECT mean_days_between_failures FROM v_component_reliability "
            "WHERE component = 'comp1'"
        ).fetchone()[0]
    assert gap == 3.0


def test_line_performance_ranks_lines(con):
    chennai, pune = rows(con, "SELECT * FROM v_line_performance")
    assert (chennai["line_id"], chennai["availability"], chennai["availability_rank"]) == (
        "CHENNAI-L1",
        1.0,
        1,
    )
    assert pune["availability"] == pytest.approx(1 - 10 / 738)
    assert (pune["availability_rank"], pune["failures_per_machine"]) == (2, 1.0)
    assert pune["cost_per_machine"] == 1_050_000


def test_views_reconcile_with_the_fact_table(con):
    expected = con.sql("SELECT sum(failures), sum(total_cost) FROM fact_machine_month").fetchone()
    for view in ("v_fleet_monthly", "v_machine_health", "v_line_performance"):
        assert con.sql(f"SELECT sum(failures), sum(total_cost) FROM {view}").fetchone() == expected


def example_queries() -> list[str]:
    with duckdb.connect() as c:
        return [s.query for s in c.extract_statements(EXAMPLE_QUERIES.read_text(encoding="utf-8"))]


def test_example_queries_file_has_ten_questions():
    assert len(example_queries()) == 10


@pytest.mark.parametrize("number", range(1, 11), ids=lambda n: f"Q{n}")
def test_example_query_runs(con, number):
    result = con.sql(example_queries()[number - 1]).fetchall()
    assert isinstance(result, list)
