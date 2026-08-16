CREATE TABLE cuda_transfer_configurations (
    cuda_transfer_configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    configuration_hash TEXT NOT NULL,
    experiment TEXT NOT NULL,
    direction TEXT NOT NULL,
    host_memory_class TEXT NOT NULL,
    api_mode TEXT NOT NULL,
    transfer_bytes INTEGER NOT NULL,
    stream_mode TEXT NOT NULL,
    batch_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    warmup_count INTEGER NOT NULL,
    measured_repetitions INTEGER NOT NULL,
    device_index INTEGER NOT NULL,
    host_buffer_bytes INTEGER NOT NULL,
    device_buffer_bytes INTEGER NOT NULL,
    validation_mode TEXT NOT NULL,
    timer_mode TEXT NOT NULL,
    host_timer_bracket_overhead_ns INTEGER NOT NULL,
    host_timer_resolution_ns INTEGER NOT NULL,
    cuda_event_mode TEXT NOT NULL,
    ordering_seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(session_id, configuration_hash)
);

CREATE TABLE cuda_transfer_benchmarks (
    cuda_transfer_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cuda_transfer_configuration_id INTEGER NOT NULL REFERENCES cuda_transfer_configurations(cuda_transfer_configuration_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    verified INTEGER NOT NULL,
    noisy INTEGER NOT NULL,
    async_behavior TEXT NOT NULL,
    statistics_json TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (verified IN (0, 1)),
    CHECK (noisy IN (0, 1))
);

CREATE TABLE cuda_transfer_samples (
    cuda_transfer_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cuda_transfer_benchmark_id INTEGER NOT NULL REFERENCES cuda_transfer_benchmarks(cuda_transfer_benchmark_id) ON DELETE CASCADE,
    repetition INTEGER NOT NULL,
    payload_bytes INTEGER NOT NULL,
    host_api_raw_ns INTEGER NOT NULL,
    device_duration_ns INTEGER NOT NULL,
    end_to_end_raw_ns INTEGER NOT NULL,
    status TEXT NOT NULL,
    cuda_error INTEGER,
    message TEXT
);

CREATE TABLE cuda_bidirectional_benchmarks (
    cuda_bidirectional_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    host_memory_class TEXT NOT NULL,
    transfer_bytes INTEGER NOT NULL,
    measured_repetitions INTEGER NOT NULL,
    isolated_h2d_median_ns REAL NOT NULL,
    isolated_d2h_median_ns REAL NOT NULL,
    concurrent_h2d_median_ns REAL NOT NULL,
    concurrent_d2h_median_ns REAL NOT NULL,
    makespan_median_ns REAL NOT NULL,
    aggregate_bytes_per_second REAL NOT NULL,
    concurrency_benefit REAL NOT NULL,
    verified INTEGER NOT NULL,
    status TEXT NOT NULL,
    raw_samples_json TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (verified IN (0, 1))
);

CREATE TABLE cuda_link_state_observations (
    cuda_link_state_observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    temperature_c INTEGER,
    graphics_clock_mhz INTEGER,
    memory_clock_mhz INTEGER,
    power_watts REAL,
    power_limit_watts REAL,
    pcie_generation INTEGER,
    pcie_width INTEGER,
    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE cuda_transfer_profiles (
    cuda_transfer_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    direction TEXT NOT NULL,
    host_memory_class TEXT NOT NULL,
    api_mode TEXT NOT NULL,
    peak_bytes_per_second REAL NOT NULL,
    peak_size_bytes INTEGER NOT NULL,
    knee_80_bytes INTEGER NOT NULL,
    knee_90_bytes INTEGER NOT NULL,
    knee_95_bytes INTEGER NOT NULL,
    candidate_sizes_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX cuda_transfer_configurations_session_idx ON cuda_transfer_configurations(session_id);
CREATE INDEX cuda_transfer_samples_benchmark_idx ON cuda_transfer_samples(cuda_transfer_benchmark_id);
CREATE INDEX cuda_bidirectional_session_idx ON cuda_bidirectional_benchmarks(session_id);
CREATE INDEX cuda_link_state_session_idx ON cuda_link_state_observations(session_id);

INSERT INTO schema_migrations(version, name)
VALUES (5, 'CUDA host-device transfer physics laboratory');

PRAGMA user_version = 5;
