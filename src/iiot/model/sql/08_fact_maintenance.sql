-- fact_maintenance: one row per component replacement (full history, from 2014).
-- Source: Silver `maintenance`; costs and downtime from dim_component.
--   maintenance_type = 'failure_repair'  the replacement that repairs a failure (same
--                                        machine, time and component as a failure).
--                                        Its downtime and cost are already in
--                                        fact_failure, so they are 0 here.
--   maintenance_type = 'planned'         a scheduled replacement, with planned downtime
--                                        and planned maintenance cost.
CREATE TABLE fact_maintenance (
    machine_id        INTEGER   NOT NULL REFERENCES dim_machine (machine_id),
    date_key          INTEGER   NOT NULL REFERENCES dim_date (date_key),
    timestamp         TIMESTAMP NOT NULL,
    component         VARCHAR   NOT NULL REFERENCES dim_component (component),
    maintenance_type  VARCHAR   NOT NULL CHECK (maintenance_type IN ('planned', 'failure_repair')),
    downtime_hours    DOUBLE    NOT NULL,
    maintenance_cost  DOUBLE    NOT NULL,
    PRIMARY KEY (machine_id, timestamp, component)
);

INSERT INTO fact_maintenance
SELECT m.machine_id,
       CAST(strftime(m.timestamp, '%Y%m%d') AS INTEGER),
       m.timestamp,
       m.component,
       CASE WHEN f.machine_id IS NULL THEN 'planned' ELSE 'failure_repair' END,
       CASE WHEN f.machine_id IS NULL THEN c.planned_downtime_hours ELSE 0 END,
       CASE WHEN f.machine_id IS NULL THEN c.planned_maintenance_cost ELSE 0 END
FROM src_maintenance AS m
JOIN dim_component AS c USING (component)
LEFT JOIN src_failures AS f
       ON f.machine_id = m.machine_id AND f.timestamp = m.timestamp AND f.component = m.component
ORDER BY m.machine_id, m.timestamp, m.component;
