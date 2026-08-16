PRAGMA application_id = 0x53444352;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hardware_profiles (
    machine_hash TEXT PRIMARY KEY,
    discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    host_name TEXT,
    os_name TEXT NOT NULL,
    os_version TEXT NOT NULL,
    os_build TEXT,
    cpu_model TEXT NOT NULL,
    physical_core_count INTEGER,
    logical_core_count INTEGER,
    numa_node_count INTEGER,
    installed_ram_bytes INTEGER,
    available_ram_bytes INTEGER,
    ram_speed_mt_s INTEGER,
    ram_channel_configuration TEXT,
    motherboard_manufacturer TEXT,
    motherboard_model TEXT,
    firmware_revision TEXT,
    discovery_json TEXT,
    CHECK (physical_core_count IS NULL OR physical_core_count > 0),
    CHECK (logical_core_count IS NULL OR logical_core_count > 0),
    CHECK (installed_ram_bytes IS NULL OR installed_ram_bytes >= 0)
);

CREATE TABLE IF NOT EXISTS gpu_devices (
    gpu_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_hash TEXT NOT NULL REFERENCES hardware_profiles(machine_hash) ON DELETE CASCADE,
    gpu_uuid TEXT NOT NULL,
    model TEXT NOT NULL,
    vram_bytes INTEGER,
    pci_domain INTEGER,
    pci_bus INTEGER,
    pci_device INTEGER,
    negotiated_pcie_generation INTEGER,
    maximum_pcie_generation INTEGER,
    negotiated_lane_width INTEGER,
    maximum_lane_width INTEGER,
    cuda_compute_capability_major INTEGER,
    cuda_compute_capability_minor INTEGER,
    async_engine_count INTEGER,
    device_attributes_json TEXT,
    UNIQUE (machine_hash, gpu_uuid)
);

CREATE TABLE IF NOT EXISTS storage_devices (
    storage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_hash TEXT NOT NULL REFERENCES hardware_profiles(machine_hash) ON DELETE CASCADE,
    persistent_id TEXT NOT NULL,
    model TEXT,
    firmware TEXT,
    capacity_bytes INTEGER,
    filesystem TEXT,
    bus_type TEXT,
    pci_path TEXT,
    pcie_generation INTEGER,
    lane_width INTEGER,
    numa_node INTEGER,
    topology_json TEXT,
    UNIQUE (machine_hash, persistent_id)
);

CREATE TABLE IF NOT EXISTS benchmark_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_hash TEXT NOT NULL REFERENCES hardware_profiles(machine_hash),
    sidecar_spec_version TEXT NOT NULL,
    sidecar_git_commit TEXT NOT NULL,
    llama_cpp_git_commit TEXT,
    cuda_runtime_version INTEGER,
    cuda_driver_version INTEGER,
    cuda_toolkit_version TEXT,
    nvidia_driver_version TEXT,
    os_version TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    trace_mode TEXT NOT NULL,
    trace_overhead_pct REAL,
    gpu_temperature_start_c REAL,
    gpu_power_limit_watts REAL,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    notes TEXT,
    CHECK (status IN ('RUNNING', 'COMPLETE', 'FAILED', 'ABORTED'))
);

CREATE TABLE IF NOT EXISTS software_versions (
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    component TEXT NOT NULL,
    version TEXT,
    git_commit TEXT,
    source TEXT,
    PRIMARY KEY (session_id, component)
);

CREATE TABLE IF NOT EXISTS models (
    model_hash TEXT PRIMARY KEY,
    hash_algorithm TEXT NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    format TEXT,
    architecture TEXT,
    quantization TEXT,
    parameter_estimate INTEGER,
    layer_count INTEGER,
    is_moe INTEGER,
    expert_count INTEGER,
    scanned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT,
    CHECK (size_bytes >= 0),
    CHECK (is_moe IS NULL OR is_moe IN (0, 1))
);

CREATE TABLE IF NOT EXISTS memory_benchmarks (
    memory_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    memory_tier TEXT NOT NULL,
    operation TEXT NOT NULL,
    allocation_bytes INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    warmup_runs INTEGER NOT NULL,
    mean_us REAL,
    median_us REAL,
    min_us REAL,
    max_us REAL,
    stddev_us REAL,
    p95_us REAL,
    p99_us REAL,
    raw_samples_json TEXT,
    error_code TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS storage_microbenchmarks (
    storage_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    storage_id INTEGER NOT NULL REFERENCES storage_devices(storage_id),
    io_mode TEXT NOT NULL,
    block_size_bytes INTEGER NOT NULL,
    queue_depth INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    throughput_bytes_per_second REAL,
    iops REAL,
    p50_us REAL,
    p90_us REAL,
    p95_us REAL,
    p99_us REAL,
    p999_us REAL,
    stddev_us REAL,
    worst_us REAL,
    cpu_utilization_pct REAL,
    deadline_us REAL,
    deadline_compliance_pct REAL,
    deadline_compliant_bandwidth_bytes_per_second REAL,
    raw_samples_json TEXT,
    error_code TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS transfer_benchmarks (
    transfer_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    gpu_id INTEGER NOT NULL REFERENCES gpu_devices(gpu_id),
    direction TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    destination_tier TEXT NOT NULL,
    transfer_size_bytes INTEGER NOT NULL,
    stream_count INTEGER NOT NULL DEFAULT 1,
    repetitions INTEGER NOT NULL,
    warmup_runs INTEGER NOT NULL,
    mean_us REAL,
    median_us REAL,
    min_us REAL,
    max_us REAL,
    stddev_us REAL,
    p95_us REAL,
    p99_us REAL,
    throughput_bytes_per_second REAL,
    raw_samples_json TEXT,
    error_code TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS contention_benchmarks (
    contention_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    gpu_id INTEGER REFERENCES gpu_devices(gpu_id),
    storage_id INTEGER REFERENCES storage_devices(storage_id),
    scenario TEXT NOT NULL,
    repetitions INTEGER NOT NULL,
    compute_slowdown_pct REAL,
    h2d_slowdown_pct REAL,
    d2h_slowdown_pct REAL,
    nvme_slowdown_pct REAL,
    gpu_clock_mean_mhz REAL,
    gpu_power_mean_watts REAL,
    gpu_temperature_mean_c REAL,
    cpu_utilization_pct REAL,
    raw_samples_json TEXT,
    error_code TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS transfer_compute_grid (
    grid_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    gpu_id INTEGER NOT NULL REFERENCES gpu_devices(gpu_id),
    compute_type TEXT NOT NULL,
    transfer_size_bytes INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    compute_solo_us REAL NOT NULL,
    transfer_solo_us REAL NOT NULL,
    compute_concurrent_us REAL NOT NULL,
    transfer_concurrent_us REAL NOT NULL,
    concurrent_makespan_us REAL NOT NULL,
    compute_slowdown_pct REAL NOT NULL,
    transfer_slowdown_pct REAL NOT NULL,
    overlap_efficiency_raw REAL NOT NULL,
    overlap_efficiency_normalized REAL NOT NULL,
    critical_path_delta_us REAL NOT NULL,
    critical_path_added_us REAL NOT NULL,
    critical_path_added_p95_us REAL,
    critical_path_added_p99_us REAL,
    fits_without_delay INTEGER NOT NULL,
    fit_tolerance_us REAL NOT NULL,
    raw_samples_json TEXT,
    CHECK (fits_without_delay IN (0, 1))
);

CREATE TABLE IF NOT EXISTS inference_runs (
    inference_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    model_hash TEXT NOT NULL REFERENCES models(model_hash),
    backend_mode TEXT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_token_count INTEGER NOT NULL,
    generation_token_count INTEGER NOT NULL,
    sampling_json TEXT NOT NULL,
    context_size INTEGER NOT NULL,
    model_load_ms REAL,
    time_to_first_token_ms REAL,
    prompt_tokens_per_second REAL,
    generation_tokens_per_second REAL,
    peak_vram_bytes INTEGER,
    peak_system_ram_bytes INTEGER,
    gpu_utilization_mean_pct REAL,
    cpu_utilization_mean_pct REAL,
    gpu_power_mean_watts REAL,
    gpu_temperature_mean_c REAL,
    raw_samples_json TEXT,
    error_code TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS trace_files (
    trace_file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    record_size_bytes INTEGER NOT NULL,
    trace_mode TEXT NOT NULL,
    trace_bytes INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    dropped_events INTEGER NOT NULL,
    ring_high_water_mark INTEGER,
    is_complete INTEGER NOT NULL,
    sha256 TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (record_size_bytes IN (32, 64)),
    CHECK (is_complete IN (0, 1))
);

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (1, 'M0 initial forensic schema');

PRAGMA user_version = 1;

