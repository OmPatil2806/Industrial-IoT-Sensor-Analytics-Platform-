-- dim_error_type: one row per error code the machines can raise.
-- Source: the allowed error IDs in the raw data contract.
CREATE TABLE dim_error_type (
    error_id      VARCHAR PRIMARY KEY,
    error_number  INTEGER NOT NULL
);

INSERT INTO dim_error_type
SELECT error_id, CAST(regexp_extract(error_id, '[0-9]+') AS INTEGER)
FROM src_error_types
ORDER BY error_id;
