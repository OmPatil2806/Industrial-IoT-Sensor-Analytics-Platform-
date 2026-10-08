-- v_fleet_monthly: the whole fleet per calendar month (the headline trend).
-- Source: fact_machine_month. Ratios are recomputed from the summed hours and counts,
-- never averaged, exactly like the Gold KPIs.
--   availability  1 - downtime / machine-hours
--   mtbf_h        operating machine-hours per failure event
--   mttr_h        unplanned downtime hours per failure event
CREATE VIEW v_fleet_monthly AS
SELECT
    year_month,
    count(*)                                        AS machines,
    CAST(sum(failures) AS INTEGER)                  AS failures,
    CAST(sum(components_failed) AS INTEGER)         AS components_failed,
    CAST(sum(planned_maintenances) AS INTEGER)      AS planned_maintenances,
    sum(unplanned_downtime_h)                       AS unplanned_downtime_h,
    sum(planned_downtime_h)                         AS planned_downtime_h,
    sum(downtime_h)                                 AS downtime_h,
    1 - sum(downtime_h) / sum(period_hours)         AS availability,
    (sum(period_hours) - sum(downtime_h)) / nullif(sum(failures), 0) AS mtbf_h,
    sum(unplanned_downtime_h) / nullif(sum(failures), 0)             AS mttr_h,
    sum(failure_cost)                               AS failure_cost,
    sum(maintenance_cost)                           AS maintenance_cost,
    sum(total_cost)                                 AS total_cost
FROM fact_machine_month
GROUP BY year_month
ORDER BY year_month;
