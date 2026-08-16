CREATE TABLE trace_configuration (
    configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    record_size_bytes INTEGER NOT NULL,
    ring_capacity INTEGER NOT NULL,
    producer_count INTEGER NOT NULL,
    collector_batch_size INTEGER NOT NULL,
    collector_strategy TEXT NOT NULL,
    timestamp_method TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (record_size_bytes IN (32, 64)),
    CHECK (ring_capacity > 0),
    CHECK (producer_count > 0),
    CHECK (collector_batch_size > 0)
);

CREATE TABLE trace_benchmarks (
    trace_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    configuration_id INTEGER REFERENCES trace_configuration(configuration_id),
    workload TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    baseline_duration_ns INTEGER NOT NULL,
    traced_duration_ns INTEGER NOT NULL,
    events_generated INTEGER NOT NULL,
    events_written INTEGER NOT NULL,
    events_dropped INTEGER NOT NULL,
    events_per_second REAL NOT NULL,
    producer_cpu_ns INTEGER,
    collector_cpu_ns INTEGER,
    observer_overhead_raw REAL NOT NULL,
    ring_high_water_mark INTEGER,
    trace_bytes INTEGER,
    disk_write_bytes_per_second REAL,
    try_push_p50_ns REAL,
    try_push_p95_ns REAL,
    try_push_p99_ns REAL,
    raw_samples_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (events_generated = events_written + events_dropped)
);

CREATE INDEX trace_benchmarks_session_idx ON trace_benchmarks(session_id);

INSERT INTO schema_migrations(version, name)
VALUES (3, 'Versioned binary Flight Recorder and observer-effect laboratory');

PRAGMA user_version = 3;
