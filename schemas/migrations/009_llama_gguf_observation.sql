CREATE TABLE llama_dependencies (
    llama_dependency_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dependency_name TEXT NOT NULL,
    upstream_url TEXT NOT NULL,
    upstream_commit TEXT NOT NULL,
    dirty INTEGER NOT NULL,
    build_configuration TEXT NOT NULL,
    cuda_enabled INTEGER NOT NULL,
    compiler TEXT NOT NULL,
    sidecar_git_commit TEXT NOT NULL,
    binary_identity_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(dirty IN (0, 1)),
    CHECK(cuda_enabled IN (0, 1)),
    UNIQUE(dependency_name, upstream_commit, build_configuration, sidecar_git_commit)
);

CREATE TABLE gguf_models (
    model_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_repository TEXT,
    source_revision TEXT,
    ollama_model_tag TEXT,
    ollama_digest TEXT,
    original_blob_path TEXT,
    alias_path TEXT,
    local_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    gguf_version INTEGER NOT NULL,
    architecture TEXT,
    quantization TEXT,
    parameter_count INTEGER,
    tensor_count INTEGER NOT NULL,
    layer_count INTEGER,
    expert_count INTEGER,
    metadata_json TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    verification_detail TEXT,
    llama_cpp_commit TEXT NOT NULL,
    inspected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(file_size_bytes >= 0),
    CHECK(tensor_count >= 0)
);

CREATE TABLE gguf_tensors (
    model_id TEXT NOT NULL REFERENCES gguf_models(model_id) ON DELETE CASCADE,
    tensor_id INTEGER NOT NULL,
    tensor_name TEXT NOT NULL,
    tensor_type TEXT NOT NULL,
    dimensions_json TEXT NOT NULL,
    tensor_bytes INTEGER NOT NULL,
    file_offset INTEGER NOT NULL,
    file_span_bytes INTEGER NOT NULL,
    alignment_bytes INTEGER NOT NULL,
    layer_index INTEGER,
    expert_index INTEGER,
    tensor_role TEXT NOT NULL,
    persistent_weight INTEGER NOT NULL,
    PRIMARY KEY(model_id, tensor_id),
    CHECK(persistent_weight IN (0, 1))
);

CREATE TABLE gguf_model_storage_provenance (
    gguf_model_storage_provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES gguf_models(model_id) ON DELETE CASCADE,
    requested_path TEXT NOT NULL,
    resolved_path TEXT NOT NULL,
    volume_path TEXT NOT NULL,
    volume_unique_id TEXT,
    physical_device TEXT NOT NULL,
    physical_disk_number INTEGER,
    device_model TEXT NOT NULL,
    bus_type TEXT NOT NULL,
    is_samsung_990_pro INTEGER NOT NULL,
    is_usb_external INTEGER NOT NULL,
    storage_role TEXT NOT NULL,
    copy_relationship TEXT,
    checked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(is_samsung_990_pro IN (0, 1)),
    CHECK(is_usb_external IN (0, 1)),
    UNIQUE(model_id, requested_path, storage_role)
);

CREATE TABLE inference_fixtures (
    fixture_id TEXT PRIMARY KEY,
    fixture_name TEXT NOT NULL,
    fixture_version INTEGER NOT NULL,
    text_sha256 TEXT NOT NULL,
    text_bytes INTEGER NOT NULL,
    token_count INTEGER,
    token_ids_json TEXT,
    description TEXT NOT NULL
);

CREATE TABLE inference_configurations (
    inference_configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES gguf_models(model_id),
    fixture_id TEXT REFERENCES inference_fixtures(fixture_id),
    configuration_hash TEXT NOT NULL,
    phase TEXT NOT NULL,
    observer_mode TEXT NOT NULL,
    gpu_layers INTEGER NOT NULL,
    context_size INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    generated_tokens INTEGER NOT NULL,
    warmups INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    sampling_json TEXT NOT NULL,
    backend_mode TEXT NOT NULL,
    load_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(session_id, configuration_hash)
);

CREATE TABLE inference_benchmarks (
    inference_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_configuration_id INTEGER NOT NULL REFERENCES inference_configurations(inference_configuration_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    model_load_ns INTEGER,
    prompt_statistics_json TEXT NOT NULL,
    decode_statistics_json TEXT NOT NULL,
    perf_context_json TEXT NOT NULL,
    telemetry_json TEXT NOT NULL,
    graph_split_detail TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE inference_samples (
    inference_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_benchmark_id INTEGER NOT NULL REFERENCES inference_benchmarks(inference_benchmark_id) ON DELETE CASCADE,
    sample_index INTEGER NOT NULL,
    prompt_ns INTEGER NOT NULL,
    decode_total_ns INTEGER NOT NULL,
    prompt_tokens_per_second REAL NOT NULL,
    generation_tokens_per_second REAL NOT NULL,
    graph_node_count INTEGER NOT NULL,
    graph_split_count INTEGER,
    observer_event_count INTEGER NOT NULL,
    dropped_event_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(inference_benchmark_id, sample_index)
);

CREATE TABLE inference_token_samples (
    inference_token_sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_benchmark_id INTEGER NOT NULL REFERENCES inference_benchmarks(inference_benchmark_id) ON DELETE CASCADE,
    sample_index INTEGER NOT NULL,
    phase TEXT NOT NULL,
    token_index INTEGER NOT NULL,
    input_token_id INTEGER NOT NULL,
    output_token_id INTEGER,
    context_depth INTEGER NOT NULL,
    decode_start_ns INTEGER NOT NULL,
    decode_end_ns INTEGER NOT NULL,
    decode_duration_ns INTEGER NOT NULL,
    observer_mode TEXT NOT NULL,
    graph_node_count INTEGER NOT NULL,
    layer_sequence_reference TEXT,
    unique_weight_tensor_count INTEGER NOT NULL,
    unique_weight_bytes INTEGER NOT NULL,
    projected_blocks_json TEXT NOT NULL,
    telemetry_reference TEXT,
    status TEXT NOT NULL,
    UNIQUE(inference_benchmark_id, sample_index, phase, token_index)
);

CREATE TABLE inference_observer_profiles (
    observer_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_benchmark_id INTEGER NOT NULL REFERENCES inference_benchmarks(inference_benchmark_id) ON DELETE CASCADE,
    observer_mode TEXT NOT NULL,
    ask_calls INTEGER NOT NULL,
    materialized_calls INTEGER NOT NULL,
    compact_events INTEGER NOT NULL,
    dropped_events INTEGER NOT NULL,
    ring_high_water_mark INTEGER,
    callback_contract TEXT NOT NULL,
    dictionary_json TEXT NOT NULL
);

CREATE TABLE inference_observer_events (
    observer_profile_id INTEGER NOT NULL REFERENCES inference_observer_profiles(observer_profile_id) ON DELETE CASCADE,
    event_index INTEGER NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    eval_index INTEGER NOT NULL,
    sequence_index INTEGER NOT NULL,
    tensor_id_0 INTEGER,
    tensor_id_1 INTEGER,
    op_id INTEGER NOT NULL,
    layer_index INTEGER,
    event_class INTEGER NOT NULL,
    flags INTEGER NOT NULL,
    PRIMARY KEY(observer_profile_id, event_index)
);

CREATE TABLE tensor_demand_events (
    tensor_demand_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_benchmark_id INTEGER NOT NULL REFERENCES inference_benchmarks(inference_benchmark_id) ON DELETE CASCADE,
    token_index INTEGER NOT NULL,
    sequence_index INTEGER NOT NULL,
    tensor_id INTEGER NOT NULL,
    first_event_index INTEGER NOT NULL,
    layer_index INTEGER,
    tensor_bytes INTEGER NOT NULL,
    file_offset INTEGER NOT NULL,
    causal_known_at_ns INTEGER,
    demand_ns INTEGER NOT NULL,
    UNIQUE(inference_benchmark_id, token_index, tensor_id)
);

CREATE TABLE tensor_block_projections (
    tensor_block_projection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_benchmark_id INTEGER REFERENCES inference_benchmarks(inference_benchmark_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES gguf_models(model_id),
    block_size_bytes INTEGER NOT NULL,
    scope TEXT NOT NULL,
    scope_index INTEGER,
    useful_tensor_bytes INTEGER NOT NULL,
    projected_block_count INTEGER NOT NULL,
    projected_bytes INTEGER NOT NULL,
    overfetch_bytes INTEGER NOT NULL,
    overfetch_ratio REAL NOT NULL,
    block_ids_json TEXT NOT NULL
);

CREATE TABLE shadow_pipeline_results (
    shadow_pipeline_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_benchmark_id INTEGER NOT NULL REFERENCES inference_benchmarks(inference_benchmark_id) ON DELETE CASCADE,
    token_index INTEGER NOT NULL,
    tensor_or_block_id INTEGER NOT NULL,
    block_size_bytes INTEGER NOT NULL,
    pipeline_type TEXT NOT NULL,
    lookahead_assumption TEXT NOT NULL,
    knowledge_mode TEXT NOT NULL,
    available_lead_time_ns INTEGER NOT NULL,
    pipeline_profile_reference TEXT NOT NULL,
    deadline_ns INTEGER NOT NULL,
    predicted_hit INTEGER NOT NULL,
    margin_ns INTEGER NOT NULL,
    extrapolated INTEGER NOT NULL,
    CHECK(predicted_hit IN (0, 1)),
    CHECK(extrapolated IN (0, 1))
);

CREATE TABLE observer_overhead_benchmarks (
    observer_overhead_benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(session_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES gguf_models(model_id),
    phase TEXT NOT NULL,
    observer_mode TEXT NOT NULL,
    paired_repetitions INTEGER NOT NULL,
    baseline_samples_json TEXT NOT NULL,
    observed_samples_json TEXT NOT NULL,
    paired_delta_statistics_json TEXT NOT NULL,
    overhead_percent REAL NOT NULL,
    confidence_interval_json TEXT NOT NULL,
    production_gate_percent REAL NOT NULL,
    gate_passed INTEGER NOT NULL,
    CHECK(gate_passed IN (0, 1))
);

CREATE TABLE moe_demand_profiles (
    moe_demand_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_benchmark_id INTEGER REFERENCES inference_benchmarks(inference_benchmark_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES gguf_models(model_id),
    status TEXT NOT NULL,
    layer_index INTEGER,
    expert_count INTEGER,
    selected_experts_observable INTEGER NOT NULL,
    expert_working_set_bytes INTEGER,
    reuse_json TEXT NOT NULL,
    causal_cue_json TEXT NOT NULL,
    CHECK(selected_experts_observable IN (0, 1))
);

CREATE INDEX gguf_tensors_model_layer_idx ON gguf_tensors(model_id, layer_index);
CREATE INDEX gguf_storage_model_idx ON gguf_model_storage_provenance(model_id);
CREATE INDEX inference_config_session_idx ON inference_configurations(session_id);
CREATE INDEX inference_samples_benchmark_idx ON inference_samples(inference_benchmark_id);
CREATE INDEX inference_tokens_benchmark_idx ON inference_token_samples(inference_benchmark_id);
CREATE INDEX observer_events_profile_idx ON inference_observer_events(observer_profile_id);
CREATE INDEX tensor_demand_benchmark_idx ON tensor_demand_events(inference_benchmark_id, token_index);
CREATE INDEX shadow_pipeline_benchmark_idx ON shadow_pipeline_results(inference_benchmark_id, token_index);

INSERT INTO schema_migrations(version, name)
VALUES (9, 'llama.cpp GGUF real-workload observation laboratory');

PRAGMA user_version = 9;
