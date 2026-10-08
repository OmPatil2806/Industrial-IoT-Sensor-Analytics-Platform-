-- v_line_performance: one row per production line over the whole analysis period,
-- for comparing lines and plants. Source: fact_machine_month + dim_machine.
-- Ratios are recomputed from summed hours and counts. availability_rank 1 = best line.
CREATE VIEW v_line_performance AS
SELECT
    m.plant_id,
    any_value(m.plant_name)                          AS plant_name,
    m.line_id,
    count(DISTINCT m.machine_id)                     AS machines,
    CAST(sum(k.failures) AS INTEGER)                 AS failures,
    CAST(sum(k.components_failed) AS INTEGER)        AS components_failed,
    CAST(sum(k.planned_maintenances) AS INTEGER)     AS planned_maintenances,
    sum(k.unplanned_downtime_h)                      AS unplanned_downtime_h,
    sum(k.planned_downtime_h)                        AS planned_downtime_h,
    sum(k.downtime_h)                                AS downtime_h,
    1 - sum(k.downtime_h) / sum(k.period_hours)      AS availability,
    (sum(k.period_hours) - sum(k.downtime_h)) / nullif(sum(k.failures), 0) AS mtbf_h,
    sum(k.unplanned_downtime_h) / nullif(sum(k.failures), 0)              AS mttr_h,
    sum(k.failure_cost)                              AS failure_cost,
    sum(k.maintenance_cost)                          AS maintenance_cost,
    sum(k.total_cost)                                AS total_cost,
    sum(k.total_cost) / count(DISTINCT m.machine_id) AS cost_per_machine,
    sum(k.failures) / count(DISTINCT m.machine_id)   AS failures_per_machine,
    rank() OVER (ORDER BY 1 - sum(k.downtime_h) / sum(k.period_hours) DESC) AS availability_rank
FROM fact_machine_month AS k
JOIN dim_machine AS m USING (machine_id)
GROUP BY m.plant_id, m.line_id
ORDER BY m.plant_id, m.line_id;
