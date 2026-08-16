CREATE TABLE compute_workload_profiles (
    compute_workload_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    workload TEXT NOT NULL,
    target_compute_us REAL NOT NULL,
    calibration_statistics_json TEXT NOT NULL,
    calibration_samples INTEGER NOT NULL,
    alu_iterations INTEGER NOT NULL,
    memory_working_set_bytes INTEGER NOT NULL,
    memory_passes INTEGER NOT NULL,
    memory_block_size INTEGER NOT NULL,
    memory_elements_per_thread INTEGER NOT NULL,
    gemm_m INTEGER NOT NULL,
    gemm_n INTEGER NOT NULL,
    gemm_k INTEGER NOT NULL,
    gemm_repetitions INTEGER NOT NULL,
    gemm_a_type TEXT NOT NULL,
    gemm_b_type TEXT NOT NULL,
    gemm_c_type TEXT NOT NULL,
    gemm_compute_type TEXT NOT NULL,
    gemm_math_mode TEXT NOT NULL,
    gemm_algorithm TEXT NOT NULL,
    cublas_version INTEGER,
    validated INTEGER NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (validated IN (0, 1)),
    UNIQUE(session_id, workload, target_compute_us)
);

CREATE TABLE cuda_overlap_configurations (
    cuda_overlap_configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    compute_workload_profile_id INTEGER NOT NULL REFERENCES compute_workload_profiles(compute_workload_profile_id) ON DELETE CASCADE,
    configuration_hash TEXT NOT NULL,
    phase TEXT NOT NULL,
    direction TEXT NOT NULL,
    host_memory_class TEXT NOT NULL,
    transfer_bytes INTEGER NOT NULL,
    target_compute_us REAL NOT NULL,
    measured_repetitions INTEGER NOT NULL,
    ordering_seed INTEGER NOT NULL,
    gate_delay_ns INTEGER NOT NULL,
    gate_margin_ns INTEGER NOT NULL,
    device_index INTEGER NOT NULL,
    instrumentation_mode TEXT NOT NULL,
    refinement_reason TEXT NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(session_id, configuration_hash, phase)
);

CREATE TABLE cuda_overlap_benchmarks (
    cuda_overlap_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cuda_overlap_configuration_id INTEGER NOT NULL REFERENCES cuda_overlap_configurations(cuda_overlap_configuration_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    refined INTEGER NOT NULL,
    statistics_json TEXT NOT NULL,
    telemetry_before_json TEXT NOT NULL,
    telemetry_after_json TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (refined IN (0, 1))
);

CREATE TABLE cuda_overlap_samples (
    cuda_overlap_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cuda_overlap_benchmark_id INTEGER NOT NULL REFERENCES cuda_overlap_benchmarks(cuda_overlap_benchmark_id) ON DELETE CASCADE,
    repetition INTEGER NOT NULL,
    baseline_block_id INTEGER NOT NULL,
    c0_reference_ns INTEGER NOT NULL,
    t0_reference_ns INTEGER NOT NULL,
    cc_ns INTEGER NOT NULL,
    tc_ns INTEGER NOT NULL,
    makespan_device_primary_ns INTEGER NOT NULL,
    makespan_device_crosscheck_ns INTEGER NOT NULL,
    makespan_host_ns INTEGER NOT NULL,
    host_submission_ns INTEGER NOT NULL,
    gate_delay_ns INTEGER NOT NULL,
    gate_actual_ns INTEGER NOT NULL,
    gate_margin_ns INTEGER NOT NULL,
    critical_path_delta_ns REAL NOT NULL,
    compute_path_added_ns REAL NOT NULL,
    compute_path_added_percent REAL NOT NULL,
    compute_slowdown REAL NOT NULL,
    transfer_slowdown REAL NOT NULL,
    overlap_efficiency_raw REAL NOT NULL,
    overlap_efficiency_normalized REAL NOT NULL,
    hidden_fraction_raw REAL NOT NULL,
    hidden_fraction_normalized REAL NOT NULL,
    compute_retention REAL NOT NULL,
    gate_valid INTEGER NOT NULL,
    fit_1_percent INTEGER NOT NULL,
    fit_2_percent INTEGER NOT NULL,
    fit_5_percent INTEGER NOT NULL,
    instrumentation_status TEXT NOT NULL,
    status TEXT NOT NULL,
    native_error INTEGER,
    message TEXT,
    CHECK (gate_valid IN (0, 1)),
    CHECK (fit_1_percent IN (0, 1)),
    CHECK (fit_2_percent IN (0, 1)),
    CHECK (fit_5_percent IN (0, 1))
);

CREATE TABLE cuda_overlap_fit_profiles (
    cuda_overlap_fit_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    workload TEXT NOT NULL,
    direction TEXT NOT NULL,
    host_memory_class TEXT NOT NULL,
    target_compute_us REAL NOT NULL,
    tolerance_percent REAL NOT NULL,
    percentile_criterion TEXT NOT NULL DEFAULT 'P99',
    largest_measured_bytes INTEGER NOT NULL,
    p99_added_percent REAL NOT NULL,
    fit_rate REAL NOT NULL,
    confidence TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, workload, direction, host_memory_class,
           target_compute_us, tolerance_percent, percentile_criterion)
);

CREATE TABLE instrumentation_control_benchmarks (
    instrumentation_control_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    workload TEXT NOT NULL,
    direction TEXT NOT NULL,
    transfer_bytes INTEGER NOT NULL,
    target_compute_us REAL NOT NULL,
    instrumentation_mode TEXT NOT NULL,
    compute_statistics_json TEXT NOT NULL,
    transfer_statistics_json TEXT NOT NULL,
    host_statistics_json TEXT NOT NULL,
    compute_bias_percent REAL NOT NULL,
    transfer_bias_percent REAL NOT NULL,
    rejected INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (rejected IN (0, 1))
);

CREATE INDEX compute_workload_profiles_session_idx
    ON compute_workload_profiles(session_id);
CREATE INDEX cuda_overlap_configurations_session_idx
    ON cuda_overlap_configurations(session_id);
CREATE INDEX cuda_overlap_samples_benchmark_idx
    ON cuda_overlap_samples(cuda_overlap_benchmark_id);
CREATE INDEX cuda_overlap_fit_profiles_session_idx
    ON cuda_overlap_fit_profiles(session_id);
CREATE INDEX instrumentation_control_session_idx
    ON instrumentation_control_benchmarks(session_id);

INSERT INTO schema_migrations(version, name)
VALUES (6, 'CUDA compute and transfer overlap physics laboratory');

PRAGMA user_version = 6;
