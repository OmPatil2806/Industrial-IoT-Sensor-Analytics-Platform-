-- fact_failure: one row per component failure.
-- Source: Silver `failures`; costs and downtime from dim_component.
-- Components that fail together on one machine at the same time form one failure
-- event (components_in_event > 1). Event-level downtime (the longest repair, since
-- repairs run in parallel) is computed in the views and the Gold KPIs.
CREATE TABLE fact_failure (
    machine_id           INTEGER   NOT NULL REFERENCES dim_machine (machine_id),
    date_key             INTEGER   NOT NULL REFERENCES dim_date (date_key),
    timestamp            TIMESTAMP NOT NULL,
    component            VARCHAR   NOT NULL REFERENCES dim_component (component),
    components_in_event  INTEGER   NOT NULL,
    downtime_hours       DOUBLE    NOT NULL,   -- this component's unplanned repair time
    failure_cost         DOUBLE    NOT NULL,   -- this component's unplanned failure cost
    PRIMARY KEY (machine_id, timestamp, component)
);

INSERT INTO fact_failure
SELECT f.machine_id,
       CAST(strftime(f.timestamp, '%Y%m%d') AS INTEGER),
       f.timestamp,
       f.component,
       count(*) OVER (PARTITION BY f.machine_id, f.timestamp),
       c.unplanned_downtime_hours,
       c.unplanned_failure_cost
FROM src_failures AS f
JOIN dim_component AS c USING (component)
ORDER BY f.machine_id, f.timestamp, f.component;
