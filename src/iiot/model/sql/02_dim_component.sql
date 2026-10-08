-- dim_component: one row per replaceable component, with its repair and downtime costs.
-- Source: Silver `component_costs` (illustrative cost assumptions from config).
CREATE TABLE dim_component (
    component                 VARCHAR PRIMARY KEY,
    repair_cost               DOUBLE,
    unplanned_repair_cost     DOUBLE,
    unplanned_downtime_hours  DOUBLE,
    planned_downtime_hours    DOUBLE,
    downtime_cost_per_hour    DOUBLE,
    unplanned_failure_cost    DOUBLE,
    planned_maintenance_cost  DOUBLE,
    saving_if_prevented       DOUBLE,
    currency                  VARCHAR
);

INSERT INTO dim_component
SELECT component, repair_cost, unplanned_repair_cost, unplanned_downtime_hours,
       planned_downtime_hours, downtime_cost_per_hour, unplanned_failure_cost,
       planned_maintenance_cost, saving_if_prevented, currency
FROM src_component_costs
ORDER BY component;
