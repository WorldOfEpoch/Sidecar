CREATE TABLE storage_datasets (
    storage_dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    storage_id INTEGER NOT NULL REFERENCES storage_devices(storage_id),
    machine_hash TEXT NOT NULL REFERENCES hardware_profiles(machine_hash),
    format_version INTEGER NOT NULL,
    generator TEXT NOT NULL,
    seed INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    identity_hash TEXT NOT NULL,
    file_path TEXT NOT NULL,
    volume_name TEXT NOT NULL,
    physical_disk_number INTEGER NOT NULL,
    verified INTEGER NOT NULL,
    full_verification INTEGER NOT NULL,
    verified_bytes INTEGER NOT NULL,
    verified_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (verified IN (0, 1)),
    CHECK (full_verification IN (0, 1)),
    UNIQUE(machine_hash, identity_hash, file_path)
);

CREATE TABLE storage_health_snapshots (
    storage_health_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    storage_id INTEGER NOT NULL REFERENCES storage_devices(storage_id),
    session_id INTEGER REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    critical_warning INTEGER,
    temperature_c REAL,
    available_spare_percent INTEGER,
    available_spare_threshold_percent INTEGER,
    percentage_used INTEGER,
    data_units_read_low64 INTEGER,
    data_units_written_low64 INTEGER,
    host_read_commands_low64 INTEGER,
    host_write_commands_low64 INTEGER,
    controller_busy_minutes_low64 INTEGER,
    power_cycles_low64 INTEGER,
    power_on_hours_low64 INTEGER,
    unsafe_shutdowns_low64 INTEGER,
    media_data_errors_low64 INTEGER,
    error_log_entries_low64 INTEGER,
    raw_evidence_json TEXT NOT NULL,
    message TEXT,
    captured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE storage_benchmark_configurations (
    storage_benchmark_configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    storage_dataset_id INTEGER NOT NULL REFERENCES storage_datasets(storage_dataset_id),
    configuration_hash TEXT NOT NULL,
    backend TEXT NOT NULL,
    access_pattern TEXT NOT NULL,
    destination_kind TEXT NOT NULL,
    block_bytes INTEGER NOT NULL,
    queue_depth INTEGER NOT NULL,
    request_count INTEGER NOT NULL,
    outstanding_byte_cap INTEGER NOT NULL,
    timeout_ms INTEGER NOT NULL,
    ordering_seed INTEGER NOT NULL,
    window_bytes INTEGER NOT NULL,
    phase TEXT NOT NULL,
    cache_classification TEXT NOT NULL,
    status TEXT NOT NULL,
    skip_reason TEXT,
    UNIQUE(session_id, configuration_hash)
);

CREATE TABLE storage_benchmarks (
    storage_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    storage_benchmark_configuration_id INTEGER NOT NULL
        REFERENCES storage_benchmark_configurations(storage_benchmark_configuration_id)
        ON DELETE CASCADE,
    health_before_id INTEGER REFERENCES storage_health_snapshots(storage_health_snapshot_id),
    health_after_id INTEGER REFERENCES storage_health_snapshots(storage_health_snapshot_id),
    status TEXT NOT NULL,
    verified INTEGER NOT NULL,
    completed_bytes INTEGER NOT NULL,
    wall_time_ns INTEGER NOT NULL,
    bytes_per_second REAL NOT NULL,
    iops REAL NOT NULL,
    statistics_json TEXT NOT NULL,
    cpu_json TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (verified IN (0, 1))
);

CREATE TABLE storage_samples (
    storage_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    storage_benchmark_id INTEGER NOT NULL REFERENCES storage_benchmarks(storage_benchmark_id)
        ON DELETE CASCADE,
    request_index INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    file_offset INTEGER NOT NULL,
    requested_bytes INTEGER NOT NULL,
    completed_bytes INTEGER NOT NULL,
    submission_reference_ns INTEGER NOT NULL,
    submission_cost_ns INTEGER NOT NULL,
    completion_latency_ns INTEGER NOT NULL,
    completion_processing_ns INTEGER NOT NULL,
    status TEXT NOT NULL,
    native_error INTEGER,
    UNIQUE(storage_benchmark_id, request_index)
);

CREATE TABLE storage_deadline_profiles (
    storage_deadline_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    storage_benchmark_id INTEGER NOT NULL REFERENCES storage_benchmarks(storage_benchmark_id)
        ON DELETE CASCADE,
    deadline_ns INTEGER NOT NULL,
    hits INTEGER NOT NULL,
    misses INTEGER NOT NULL,
    success_rate REAL NOT NULL,
    compliant_bytes_per_second REAL NOT NULL,
    UNIQUE(storage_benchmark_id, deadline_ns)
);

CREATE TABLE storage_backend_profiles (
    storage_backend_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    backend TEXT NOT NULL,
    access_pattern TEXT NOT NULL,
    destination_kind TEXT NOT NULL,
    peak_bytes_per_second REAL NOT NULL,
    peak_block_bytes INTEGER NOT NULL,
    peak_queue_depth INTEGER NOT NULL,
    knee_80_queue_depth INTEGER,
    knee_90_queue_depth INTEGER,
    knee_95_queue_depth INTEGER,
    safe_envelopes_json TEXT NOT NULL,
    supply_profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, backend, access_pattern, destination_kind)
);

CREATE INDEX storage_datasets_storage_idx ON storage_datasets(storage_id);
CREATE INDEX storage_health_storage_idx ON storage_health_snapshots(storage_id);
CREATE INDEX storage_health_session_idx ON storage_health_snapshots(session_id);
CREATE INDEX storage_configurations_session_idx ON storage_benchmark_configurations(session_id);
CREATE INDEX storage_samples_benchmark_idx ON storage_samples(storage_benchmark_id);
CREATE INDEX storage_deadlines_benchmark_idx ON storage_deadline_profiles(storage_benchmark_id);
CREATE INDEX storage_profiles_session_idx ON storage_backend_profiles(session_id);

INSERT INTO schema_migrations(version, name)
VALUES (7, 'NVMe storage physics laboratory');

PRAGMA user_version = 7;
