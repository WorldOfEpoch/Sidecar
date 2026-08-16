CREATE TABLE host_memory_benchmarks (
    host_memory_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    mode TEXT NOT NULL,
    requested_bytes INTEGER NOT NULL,
    measured_repetitions INTEGER NOT NULL,
    status TEXT NOT NULL,
    safety_status TEXT NOT NULL,
    timer_method TEXT NOT NULL,
    timer_bracket_overhead_ns INTEGER NOT NULL,
    timer_resolution_ns INTEGER NOT NULL,
    cuda_allocation_flags TEXT,
    cuda_registration_flags TEXT,
    statistics_json TEXT NOT NULL,
    amortization_json TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE host_memory_samples (
    host_memory_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    host_memory_benchmark_id INTEGER NOT NULL
        REFERENCES host_memory_benchmarks(host_memory_benchmark_id) ON DELETE CASCADE,
    repetition INTEGER NOT NULL,
    cold_setup INTEGER NOT NULL,
    status TEXT NOT NULL,
    allocation_raw_ns INTEGER,
    allocation_corrected_ns INTEGER,
    first_touch_raw_ns INTEGER,
    first_touch_corrected_ns INTEGER,
    warm_touch_raw_ns INTEGER,
    warm_touch_corrected_ns INTEGER,
    registration_raw_ns INTEGER,
    registration_corrected_ns INTEGER,
    unregistration_raw_ns INTEGER,
    unregistration_corrected_ns INTEGER,
    cleanup_raw_ns INTEGER,
    cleanup_corrected_ns INTEGER,
    before_snapshot_json TEXT NOT NULL,
    after_setup_snapshot_json TEXT NOT NULL,
    after_cleanup_snapshot_json TEXT NOT NULL,
    available_recovery_delta_bytes INTEGER,
    available_recovery_percent REAL,
    noisy INTEGER NOT NULL,
    cuda_error INTEGER,
    windows_error INTEGER,
    message TEXT,
    CHECK (cold_setup IN (0, 1)),
    CHECK (noisy IN (0, 1))
);

CREATE TABLE memory_pressure_events (
    memory_pressure_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    requested_bytes INTEGER NOT NULL,
    classification TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE persistent_arena_tests (
    persistent_arena_test_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    backend TEXT NOT NULL,
    capacity_bytes INTEGER NOT NULL,
    reuse_count INTEGER NOT NULL,
    setup_raw_ns INTEGER,
    cleanup_raw_ns INTEGER,
    reuse_statistics_json TEXT NOT NULL,
    raw_reuse_samples_json TEXT NOT NULL,
    contents_verified INTEGER NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (contents_verified IN (0, 1))
);

CREATE INDEX host_memory_benchmarks_session_idx ON host_memory_benchmarks(session_id);
CREATE INDEX host_memory_samples_benchmark_idx
    ON host_memory_samples(host_memory_benchmark_id);

INSERT INTO schema_migrations(version, name)
VALUES (4, 'Host memory lifecycle laboratory');

PRAGMA user_version = 4;
