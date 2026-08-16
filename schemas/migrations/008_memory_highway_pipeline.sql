CREATE TABLE host_copy_configurations (
    host_copy_configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    configuration_hash TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    worker_count INTEGER NOT NULL,
    warmups INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    concurrent_compute TEXT,
    compute_window_us REAL NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(session_id, configuration_hash)
);

CREATE TABLE host_copy_benchmarks (
    host_copy_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    host_copy_configuration_id INTEGER NOT NULL
        REFERENCES host_copy_configurations(host_copy_configuration_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    statistics_json TEXT NOT NULL,
    affinity TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE host_copy_samples (
    host_copy_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    host_copy_benchmark_id INTEGER NOT NULL
        REFERENCES host_copy_benchmarks(host_copy_benchmark_id) ON DELETE CASCADE,
    sample_index INTEGER NOT NULL,
    wall_ns INTEGER NOT NULL,
    process_cpu_ns INTEGER NOT NULL,
    compute_ns INTEGER NOT NULL,
    bytes_per_second REAL NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    UNIQUE(host_copy_benchmark_id, sample_index)
);

CREATE TABLE pipeline_configurations (
    pipeline_configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    storage_dataset_id INTEGER NOT NULL REFERENCES storage_datasets(storage_dataset_id),
    configuration_hash TEXT NOT NULL,
    pipeline_type TEXT NOT NULL,
    aggregate_bytes INTEGER NOT NULL,
    chunk_bytes INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    buffer_depth INTEGER NOT NULL,
    compute_type TEXT NOT NULL,
    compute_window_us REAL NOT NULL,
    deadline_ns INTEGER NOT NULL,
    measured_repetitions INTEGER NOT NULL,
    ordering_seed INTEGER NOT NULL,
    timeout_ms INTEGER NOT NULL,
    phase TEXT NOT NULL,
    contention_test TEXT NOT NULL,
    host_copy_workers INTEGER NOT NULL,
    device_index INTEGER NOT NULL,
    arena_allocation_id TEXT NOT NULL,
    refinement_reason TEXT,
    status TEXT NOT NULL,
    UNIQUE(session_id, configuration_hash)
);

CREATE TABLE pipeline_benchmarks (
    pipeline_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_configuration_id INTEGER NOT NULL
        REFERENCES pipeline_configurations(pipeline_configuration_id) ON DELETE CASCADE,
    health_before_id INTEGER REFERENCES storage_health_snapshots(storage_health_snapshot_id),
    health_after_id INTEGER REFERENCES storage_health_snapshots(storage_health_snapshot_id),
    status TEXT NOT NULL,
    statistics_json TEXT NOT NULL,
    gpu_before_json TEXT NOT NULL,
    gpu_after_json TEXT NOT NULL,
    host_before_json TEXT NOT NULL,
    host_after_json TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE pipeline_samples (
    pipeline_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_benchmark_id INTEGER NOT NULL
        REFERENCES pipeline_benchmarks(pipeline_benchmark_id) ON DELETE CASCADE,
    sample_index INTEGER NOT NULL,
    c0_reference_ns INTEGER NOT NULL,
    p0_reference_ns INTEGER NOT NULL,
    cc_ns INTEGER NOT NULL,
    pc_ns INTEGER NOT NULL,
    makespan_ns INTEGER NOT NULL,
    compute_path_added_ns REAL NOT NULL,
    compute_path_added_percent REAL NOT NULL,
    pipeline_slowdown REAL NOT NULL,
    compute_slowdown REAL NOT NULL,
    pipeline_overlap_raw REAL NOT NULL,
    pipeline_overlap_normalized REAL NOT NULL,
    deadline_ns INTEGER NOT NULL,
    ready_ahead_ns INTEGER NOT NULL,
    deadline_hit INTEGER NOT NULL,
    storage_ns INTEGER NOT NULL,
    host_copy_ns INTEGER NOT NULL,
    h2d_ns INTEGER NOT NULL,
    fill_latency_ns INTEGER NOT NULL,
    steady_state_interval_ns INTEGER NOT NULL,
    drain_latency_ns INTEGER NOT NULL,
    gpu_waiting_for_data_ns INTEGER NOT NULL,
    h2d_waiting_for_source_ns INTEGER NOT NULL,
    pinned_slot_wait_ns INTEGER NOT NULL,
    pageable_slot_wait_ns INTEGER NOT NULL,
    nvme_queue_starved_ns INTEGER NOT NULL,
    slot_ids_json TEXT NOT NULL,
    verified INTEGER NOT NULL,
    status TEXT NOT NULL,
    native_error INTEGER,
    message TEXT,
    CHECK(deadline_hit IN (0, 1)),
    CHECK(verified IN (0, 1)),
    UNIQUE(pipeline_benchmark_id, sample_index)
);

CREATE TABLE pipeline_stage_samples (
    pipeline_stage_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_benchmark_id INTEGER NOT NULL
        REFERENCES pipeline_benchmarks(pipeline_benchmark_id) ON DELETE CASCADE,
    sample_index INTEGER NOT NULL,
    block_id INTEGER NOT NULL,
    slot_id INTEGER NOT NULL,
    stage_type TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    file_offset INTEGER NOT NULL,
    host_submit_ns INTEGER NOT NULL,
    host_start_ns INTEGER NOT NULL,
    host_finish_ns INTEGER NOT NULL,
    device_duration_ns INTEGER NOT NULL,
    wait_ns INTEGER NOT NULL,
    status TEXT NOT NULL,
    native_error INTEGER,
    UNIQUE(pipeline_benchmark_id, sample_index, block_id, stage_type)
);

CREATE TABLE pipeline_deadline_profiles (
    pipeline_deadline_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_benchmark_id INTEGER NOT NULL
        REFERENCES pipeline_benchmarks(pipeline_benchmark_id) ON DELETE CASCADE,
    deadline_ns INTEGER NOT NULL,
    hits INTEGER NOT NULL,
    misses INTEGER NOT NULL,
    success_rate REAL NOT NULL,
    ready_ahead_statistics_json TEXT NOT NULL,
    lateness_statistics_json TEXT NOT NULL,
    p999_supported INTEGER NOT NULL,
    CHECK(p999_supported IN (0, 1)),
    UNIQUE(pipeline_benchmark_id, deadline_ns)
);

CREATE TABLE pipeline_slot_profiles (
    pipeline_slot_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_benchmark_id INTEGER NOT NULL
        REFERENCES pipeline_benchmarks(pipeline_benchmark_id) ON DELETE CASCADE,
    machine_hash TEXT NOT NULL REFERENCES hardware_profiles(machine_hash),
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    arena_allocation_id TEXT NOT NULL,
    slot_id INTEGER NOT NULL,
    arena_offset INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    transfer_count INTEGER NOT NULL,
    verification_failures INTEGER NOT NULL,
    deadline_hits INTEGER NOT NULL,
    deadline_misses INTEGER NOT NULL,
    host_copy_statistics_json TEXT NOT NULL,
    h2d_statistics_json TEXT NOT NULL,
    observation TEXT NOT NULL,
    UNIQUE(pipeline_benchmark_id, slot_id)
);

CREATE TABLE pipeline_health_observations (
    pipeline_health_observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_benchmark_id INTEGER NOT NULL
        REFERENCES pipeline_benchmarks(pipeline_benchmark_id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    nvme_health_snapshot_id INTEGER REFERENCES storage_health_snapshots(storage_health_snapshot_id),
    gpu_telemetry_json TEXT NOT NULL,
    host_memory_json TEXT NOT NULL,
    process_cpu_json TEXT NOT NULL,
    status TEXT NOT NULL,
    captured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX host_copy_config_session_idx ON host_copy_configurations(session_id);
CREATE INDEX host_copy_samples_benchmark_idx ON host_copy_samples(host_copy_benchmark_id);
CREATE INDEX pipeline_config_session_idx ON pipeline_configurations(session_id);
CREATE INDEX pipeline_samples_benchmark_idx ON pipeline_samples(pipeline_benchmark_id);
CREATE INDEX pipeline_stages_benchmark_idx ON pipeline_stage_samples(pipeline_benchmark_id);
CREATE INDEX pipeline_deadlines_benchmark_idx ON pipeline_deadline_profiles(pipeline_benchmark_id);
CREATE INDEX pipeline_slots_session_idx ON pipeline_slot_profiles(session_id);
CREATE INDEX pipeline_health_benchmark_idx ON pipeline_health_observations(pipeline_benchmark_id);

INSERT INTO schema_migrations(version, name)
VALUES (8, 'integrated memory-highway pipeline physics');

PRAGMA user_version = 8;
