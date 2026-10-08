-- fact_machine_month: reliability and cost KPIs per machine and calendar month.
-- Source: Gold `kpi_machine_monthly` (see iiot.gold.kpis for the business rules).
-- month_date_key points to the first day of the month in dim_date.
CREATE TABLE fact_machine_month (
    machine_id            INTEGER NOT NULL REFERENCES dim_machine (machine_id),
    month_date_key        INTEGER NOT NULL REFERENCES dim_date (date_key),
    year_month            VARCHAR NOT NULL,
    period_hours          DOUBLE  NOT NULL,
    failures              INTEGER NOT NULL,
    components_failed     INTEGER NOT NULL,
    planned_maintenances  INTEGER NOT NULL,
    unplanned_downtime_h  DOUBLE  NOT NULL,
    planned_downtime_h    DOUBLE  NOT NULL,
    downtime_h            DOUBLE  NOT NULL,
    availability          DOUBLE,
    mtbf_h                DOUBLE,
    mttr_h                DOUBLE,
    failure_cost          DOUBLE  NOT NULL,
    maintenance_cost      DOUBLE  NOT NULL,
    total_cost            DOUBLE  NOT NULL,
    PRIMARY KEY (machine_id, year_month)
);

INSERT INTO fact_machine_month
SELECT machine_id,
       CAST(replace(month, '-', '') || '01' AS INTEGER),
       month,
       period_hours, failures, components_failed, planned_maintenances,
       unplanned_downtime_h, planned_downtime_h, downtime_h,
       availability, mtbf_h, mttr_h,
       failure_cost, maintenance_cost, total_cost
FROM src_kpi_machine_monthly
ORDER BY machine_id, month;
