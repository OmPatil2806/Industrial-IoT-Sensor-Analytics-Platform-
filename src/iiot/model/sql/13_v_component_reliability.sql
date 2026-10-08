-- v_component_reliability: one row per component, inside the analysis period.
--   failures                    component failures (a 2-component event counts for both)
--   in_multi_component_events   of those, failures that happened together with another
--   component_downtime_h        sum of this component's repair hours (an event's real
--                               downtime is the longest repair, see v_fleet_monthly)
--   mean_days_between_failures  average gap between two failures of this component on
--                               the same machine
--   planned_replacements        planned replacement records (one per component; the
--                               Gold KPIs count planned EVENTS, which can hold several)
--   unplanned_share             failures / (failures + planned replacements)
--   saving_if_all_prevented     failures x (unplanned failure cost - planned cost):
--                               the prize if every failure had been caught in time
CREATE VIEW v_component_reliability AS
WITH period AS (
    SELECT min(timestamp) AS period_start, date_trunc('day', max(timestamp)) AS period_end
    FROM fact_sensor_reading
),
failures AS (
    SELECT f.*
    FROM fact_failure AS f, period AS p
    WHERE f.timestamp >= p.period_start AND f.timestamp < p.period_end
),
failure_totals AS (
    SELECT component,
           count(*)                                         AS failures,
           count(DISTINCT machine_id)                       AS machines_affected,
           count(*) FILTER (WHERE components_in_event > 1)  AS in_multi_component_events,
           sum(downtime_hours)                              AS component_downtime_h,
           sum(failure_cost)                                AS failure_cost
    FROM failures
    GROUP BY component
),
gaps AS (
    SELECT component, avg(date_diff('hour', previous_at, timestamp)) / 24.0 AS mean_days
    FROM (
        SELECT component, timestamp,
               lag(timestamp) OVER (PARTITION BY machine_id, component ORDER BY timestamp)
                   AS previous_at
        FROM failures
    )
    WHERE previous_at IS NOT NULL
    GROUP BY component
),
planned AS (
    SELECT mt.component, count(*) AS planned_replacements, sum(mt.maintenance_cost) AS planned_cost
    FROM fact_maintenance AS mt, period AS p
    WHERE mt.maintenance_type = 'planned'
      AND mt.timestamp >= p.period_start AND mt.timestamp < p.period_end
    GROUP BY mt.component
)
SELECT
    c.component,
    coalesce(f.failures, 0)                                         AS failures,
    coalesce(f.machines_affected, 0)                                AS machines_affected,
    coalesce(f.in_multi_component_events, 0)                        AS in_multi_component_events,
    coalesce(f.component_downtime_h, 0)                             AS component_downtime_h,
    coalesce(f.failure_cost, 0)                                     AS failure_cost,
    g.mean_days                                                     AS mean_days_between_failures,
    coalesce(pl.planned_replacements, 0)                            AS planned_replacements,
    coalesce(pl.planned_cost, 0)                                    AS planned_cost,
    coalesce(f.failures, 0)
        / nullif(coalesce(f.failures, 0) + coalesce(pl.planned_replacements, 0), 0)
                                                                    AS unplanned_share,
    c.unplanned_failure_cost,
    c.planned_maintenance_cost,
    coalesce(f.failures, 0) * c.saving_if_prevented                 AS saving_if_all_prevented
FROM dim_component AS c
LEFT JOIN failure_totals AS f USING (component)
LEFT JOIN gaps AS g USING (component)
LEFT JOIN planned AS pl USING (component)
ORDER BY c.component;
