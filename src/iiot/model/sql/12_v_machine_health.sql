-- v_machine_health: one row per machine, a health card for maintenance planners.
-- Reliability and cost come from fact_machine_month (the analysis period); events are
-- counted inside the same period: from the first telemetry hour up to the last
-- midnight of the telemetry, as in the Gold KPIs.
--   errors_last_30d          errors in the last 30 days of the period
--   days_since_last_failure  days from the last failure to the end of the period
--   filled_share / missing_share  share of sensor readings the Silver layer had to
--                                 fill or could not fill (data quality per machine)
--   cost_rank                1 = the machine with the highest total cost
CREATE VIEW v_machine_health AS
WITH period AS (
    SELECT min(timestamp) AS period_start, date_trunc('day', max(timestamp)) AS period_end
    FROM fact_sensor_reading
),
kpi AS (
    SELECT machine_id,
           sum(period_hours)                          AS period_hours,
           CAST(sum(failures) AS INTEGER)             AS failures,
           CAST(sum(components_failed) AS INTEGER)    AS components_failed,
           CAST(sum(planned_maintenances) AS INTEGER) AS planned_maintenances,
           sum(unplanned_downtime_h)                  AS unplanned_downtime_h,
           sum(downtime_h)                            AS downtime_h,
           sum(failure_cost)                          AS failure_cost,
           sum(maintenance_cost)                      AS maintenance_cost,
           sum(total_cost)                            AS total_cost
    FROM fact_machine_month
    GROUP BY machine_id
),
errors AS (
    SELECT e.machine_id,
           count(*) AS errors,
           count(*) FILTER (WHERE e.timestamp >= p.period_end - INTERVAL 30 DAY) AS errors_last_30d
    FROM fact_error AS e, period AS p
    WHERE e.timestamp >= p.period_start AND e.timestamp < p.period_end
    GROUP BY e.machine_id
),
last_failure AS (
    SELECT f.machine_id, max(f.timestamp) AS last_failure_at
    FROM fact_failure AS f, period AS p
    WHERE f.timestamp >= p.period_start AND f.timestamp < p.period_end
    GROUP BY f.machine_id
),
quality AS (
    SELECT machine_id,
           avg(CAST(quality = 'filled' AS INTEGER))  AS filled_share,
           avg(CAST(quality = 'missing' AS INTEGER)) AS missing_share
    FROM fact_sensor_reading
    GROUP BY machine_id
)
SELECT
    m.machine_id, m.model, m.age, m.plant_id, m.line_id,
    coalesce(k.failures, 0)                                               AS failures,
    coalesce(k.components_failed, 0)                                      AS components_failed,
    coalesce(k.planned_maintenances, 0)                                   AS planned_maintenances,
    coalesce(k.downtime_h, 0)                                             AS downtime_h,
    1 - k.downtime_h / k.period_hours                                     AS availability,
    (k.period_hours - k.downtime_h) / nullif(k.failures, 0)               AS mtbf_h,
    k.unplanned_downtime_h / nullif(k.failures, 0)                        AS mttr_h,
    coalesce(k.failure_cost, 0)                                           AS failure_cost,
    coalesce(k.maintenance_cost, 0)                                       AS maintenance_cost,
    coalesce(k.total_cost, 0)                                             AS total_cost,
    coalesce(e.errors, 0)                                                 AS errors,
    coalesce(e.errors_last_30d, 0)                                        AS errors_last_30d,
    lf.last_failure_at,
    date_diff('hour', lf.last_failure_at, p.period_end) / 24.0            AS days_since_last_failure,
    q.filled_share,
    q.missing_share,
    rank() OVER (ORDER BY coalesce(k.total_cost, 0) DESC)                 AS cost_rank
FROM dim_machine AS m
CROSS JOIN period AS p
LEFT JOIN kpi AS k USING (machine_id)
LEFT JOIN errors AS e USING (machine_id)
LEFT JOIN last_failure AS lf USING (machine_id)
LEFT JOIN quality AS q USING (machine_id)
ORDER BY m.machine_id;
