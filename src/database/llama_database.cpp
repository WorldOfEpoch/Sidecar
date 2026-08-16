#include "sidecar/database/database.hpp"

#include <sqlite3.h>

#include <optional>
#include <string>

namespace sidecar::database {
namespace {

void Check(sqlite3* db, int code) {
    if (code != SQLITE_OK) {
        throw DatabaseError(code, sqlite3_errmsg(db));
    }
}

class Statement final {
public:
    Statement(sqlite3* db, const char* sql) : db_(db) {
        Check(db_, sqlite3_prepare_v2(db_, sql, -1, &statement_, nullptr));
    }
    ~Statement() { sqlite3_finalize(statement_); }
    sqlite3_stmt* get() const noexcept { return statement_; }
private:
    sqlite3* db_{};
    sqlite3_stmt* statement_{};
};

void BindText(sqlite3* db, sqlite3_stmt* statement, int index,
              const std::string& value) {
    Check(db, sqlite3_bind_text(statement, index, value.c_str(),
                                static_cast<int>(value.size()), SQLITE_TRANSIENT));
}

void BindOptionalText(sqlite3* db, sqlite3_stmt* statement, int index,
                      const std::optional<std::string>& value) {
    if (value) BindText(db, statement, index, *value);
    else Check(db, sqlite3_bind_null(statement, index));
}

void BindOptionalInt(sqlite3* db, sqlite3_stmt* statement, int index,
                     const std::optional<std::int64_t>& value) {
    if (value) Check(db, sqlite3_bind_int64(statement, index, *value));
    else Check(db, sqlite3_bind_null(statement, index));
}

void Done(sqlite3* db, sqlite3_stmt* statement) {
    const int code = sqlite3_step(statement);
    if (code != SQLITE_DONE) throw DatabaseError(code, sqlite3_errmsg(db));
}

std::int64_t Scalar(sqlite3* db, const char* sql) {
    Statement statement(db, sql);
    const int code = sqlite3_step(statement.get());
    if (code != SQLITE_ROW) throw DatabaseError(code, sqlite3_errmsg(db));
    return sqlite3_column_int64(statement.get(), 0);
}

}  // namespace

std::int64_t Database::UpsertLlamaDependency(const LlamaDependencyInput& input) {
    static constexpr const char* kSql = R"sql(
INSERT INTO llama_dependencies(
    dependency_name, upstream_url, upstream_commit, dirty, build_configuration,
    cuda_enabled, compiler, sidecar_git_commit, binary_identity_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(dependency_name, upstream_commit, build_configuration, sidecar_git_commit)
DO UPDATE SET upstream_url=excluded.upstream_url, dirty=excluded.dirty,
    cuda_enabled=excluded.cuda_enabled, compiler=excluded.compiler,
    binary_identity_json=excluded.binary_identity_json, recorded_at=CURRENT_TIMESTAMP
RETURNING llama_dependency_id;
)sql";
    Statement statement(handle_, kSql);
    auto* raw = statement.get();
    BindText(handle_, raw, 1, input.dependency_name);
    BindText(handle_, raw, 2, input.upstream_url);
    BindText(handle_, raw, 3, input.upstream_commit);
    Check(handle_, sqlite3_bind_int(raw, 4, input.dirty ? 1 : 0));
    BindText(handle_, raw, 5, input.build_configuration);
    Check(handle_, sqlite3_bind_int(raw, 6, input.cuda_enabled ? 1 : 0));
    BindText(handle_, raw, 7, input.compiler);
    BindText(handle_, raw, 8, input.sidecar_git_commit);
    BindText(handle_, raw, 9, input.binary_identity_json);
    const int code = sqlite3_step(raw);
    if (code != SQLITE_ROW) throw DatabaseError(code, sqlite3_errmsg(handle_));
    return sqlite3_column_int64(raw, 0);
}

void Database::UpsertGgufModel(const GgufModelInput& input) {
    static constexpr const char* kSql = R"sql(
INSERT INTO gguf_models(
    model_id, source, source_repository, source_revision, ollama_model_tag,
    ollama_digest, original_blob_path, alias_path, local_path, sha256,
    file_size_bytes, gguf_version, architecture, quantization, parameter_count,
    tensor_count, layer_count, expert_count, metadata_json, verification_status,
    verification_detail, llama_cpp_commit)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(model_id) DO UPDATE SET
    source=excluded.source, source_repository=excluded.source_repository,
    source_revision=excluded.source_revision, ollama_model_tag=excluded.ollama_model_tag,
    ollama_digest=excluded.ollama_digest, original_blob_path=excluded.original_blob_path,
    alias_path=excluded.alias_path, local_path=excluded.local_path, sha256=excluded.sha256,
    file_size_bytes=excluded.file_size_bytes, gguf_version=excluded.gguf_version,
    architecture=excluded.architecture, quantization=excluded.quantization,
    parameter_count=excluded.parameter_count, tensor_count=excluded.tensor_count,
    layer_count=excluded.layer_count, expert_count=excluded.expert_count,
    metadata_json=excluded.metadata_json, verification_status=excluded.verification_status,
    verification_detail=excluded.verification_detail, llama_cpp_commit=excluded.llama_cpp_commit,
    inspected_at=CURRENT_TIMESTAMP;
)sql";
    Statement statement(handle_, kSql);
    auto* raw = statement.get();
    BindText(handle_, raw, 1, input.model_id);
    BindText(handle_, raw, 2, input.source);
    BindOptionalText(handle_, raw, 3, input.source_repository);
    BindOptionalText(handle_, raw, 4, input.source_revision);
    BindOptionalText(handle_, raw, 5, input.ollama_model_tag);
    BindOptionalText(handle_, raw, 6, input.ollama_digest);
    BindOptionalText(handle_, raw, 7, input.original_blob_path);
    BindOptionalText(handle_, raw, 8, input.alias_path);
    BindText(handle_, raw, 9, input.local_path);
    BindText(handle_, raw, 10, input.sha256);
    Check(handle_, sqlite3_bind_int64(raw, 11, input.file_size_bytes));
    Check(handle_, sqlite3_bind_int64(raw, 12, input.gguf_version));
    BindOptionalText(handle_, raw, 13, input.architecture);
    BindOptionalText(handle_, raw, 14, input.quantization);
    BindOptionalInt(handle_, raw, 15, input.parameter_count);
    Check(handle_, sqlite3_bind_int64(raw, 16, input.tensor_count));
    BindOptionalInt(handle_, raw, 17, input.layer_count);
    BindOptionalInt(handle_, raw, 18, input.expert_count);
    BindText(handle_, raw, 19, input.metadata_json);
    BindText(handle_, raw, 20, input.verification_status);
    BindOptionalText(handle_, raw, 21, input.verification_detail);
    BindText(handle_, raw, 22, input.llama_cpp_commit);
    Done(handle_, raw);
}

void Database::ReplaceGgufTensors(const std::string& model_id,
                                  const std::vector<GgufTensorInput>& tensors) {
    Execute("BEGIN IMMEDIATE;");
    try {
        {
            Statement remove(handle_, "DELETE FROM gguf_tensors WHERE model_id=?;");
            BindText(handle_, remove.get(), 1, model_id);
            Done(handle_, remove.get());
        }
        static constexpr const char* kSql = R"sql(
INSERT INTO gguf_tensors(
    model_id, tensor_id, tensor_name, tensor_type, dimensions_json, tensor_bytes,
    file_offset, file_span_bytes, alignment_bytes, layer_index, expert_index,
    tensor_role, persistent_weight)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
        for (const auto& tensor : tensors) {
            Statement statement(handle_, kSql);
            auto* raw = statement.get();
            BindText(handle_, raw, 1, tensor.model_id);
            Check(handle_, sqlite3_bind_int64(raw, 2, tensor.tensor_id));
            BindText(handle_, raw, 3, tensor.tensor_name);
            BindText(handle_, raw, 4, tensor.tensor_type);
            BindText(handle_, raw, 5, tensor.dimensions_json);
            Check(handle_, sqlite3_bind_int64(raw, 6, tensor.tensor_bytes));
            Check(handle_, sqlite3_bind_int64(raw, 7, tensor.file_offset));
            Check(handle_, sqlite3_bind_int64(raw, 8, tensor.file_span_bytes));
            Check(handle_, sqlite3_bind_int64(raw, 9, tensor.alignment_bytes));
            BindOptionalInt(handle_, raw, 10, tensor.layer_index);
            BindOptionalInt(handle_, raw, 11, tensor.expert_index);
            BindText(handle_, raw, 12, tensor.tensor_role);
            Check(handle_, sqlite3_bind_int(raw, 13, tensor.persistent_weight ? 1 : 0));
            Done(handle_, raw);
        }
        Execute("COMMIT;");
    } catch (...) {
        try { Execute("ROLLBACK;"); } catch (...) {}
        throw;
    }
}

void Database::UpsertGgufModelStorage(const GgufModelStorageInput& input) {
    // WU9 migration 009 was still uncommitted when the storage-provenance gate
    // was added. This idempotent create preserves development databases that
    // had already exercised an earlier migration-009 draft; final databases
    // receive the identical definition from the migration itself.
    Execute(R"sql(
CREATE TABLE IF NOT EXISTS gguf_model_storage_provenance (
    gguf_model_storage_provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES gguf_models(model_id) ON DELETE CASCADE,
    requested_path TEXT NOT NULL, resolved_path TEXT NOT NULL,
    volume_path TEXT NOT NULL, volume_unique_id TEXT,
    physical_device TEXT NOT NULL, physical_disk_number INTEGER,
    device_model TEXT NOT NULL, bus_type TEXT NOT NULL,
    is_samsung_990_pro INTEGER NOT NULL, is_usb_external INTEGER NOT NULL,
    storage_role TEXT NOT NULL, copy_relationship TEXT,
    checked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(is_samsung_990_pro IN (0,1)), CHECK(is_usb_external IN (0,1)),
    UNIQUE(model_id, requested_path, storage_role));
CREATE INDEX IF NOT EXISTS gguf_storage_model_idx
    ON gguf_model_storage_provenance(model_id);
)sql");
    Statement statement(handle_, R"sql(
INSERT INTO gguf_model_storage_provenance(model_id, requested_path, resolved_path,
    volume_path, volume_unique_id, physical_device, physical_disk_number,
    device_model, bus_type, is_samsung_990_pro, is_usb_external, storage_role,
    copy_relationship)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(model_id, requested_path, storage_role) DO UPDATE SET
    resolved_path=excluded.resolved_path, volume_path=excluded.volume_path,
    volume_unique_id=excluded.volume_unique_id, physical_device=excluded.physical_device,
    physical_disk_number=excluded.physical_disk_number, device_model=excluded.device_model,
    bus_type=excluded.bus_type, is_samsung_990_pro=excluded.is_samsung_990_pro,
    is_usb_external=excluded.is_usb_external,
    copy_relationship=excluded.copy_relationship, checked_at=CURRENT_TIMESTAMP;
)sql");
    auto* raw=statement.get();
    BindText(handle_,raw,1,input.model_id); BindText(handle_,raw,2,input.requested_path);
    BindText(handle_,raw,3,input.resolved_path); BindText(handle_,raw,4,input.volume_path);
    BindOptionalText(handle_,raw,5,input.volume_unique_id); BindText(handle_,raw,6,input.physical_device);
    BindOptionalInt(handle_,raw,7,input.physical_disk_number); BindText(handle_,raw,8,input.device_model);
    BindText(handle_,raw,9,input.bus_type); Check(handle_,sqlite3_bind_int(raw,10,input.is_samsung_990_pro?1:0));
    Check(handle_,sqlite3_bind_int(raw,11,input.is_usb_external?1:0)); BindText(handle_,raw,12,input.storage_role);
    BindOptionalText(handle_,raw,13,input.copy_relationship); Done(handle_,raw);
}

void Database::UpsertInferenceFixture(const InferenceFixtureInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO inference_fixtures(fixture_id, fixture_name, fixture_version, text_sha256,
    text_bytes, token_count, token_ids_json, description)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(fixture_id) DO UPDATE SET fixture_name=excluded.fixture_name,
    fixture_version=excluded.fixture_version, text_sha256=excluded.text_sha256,
    text_bytes=excluded.text_bytes, token_count=excluded.token_count,
    token_ids_json=excluded.token_ids_json, description=excluded.description;
)sql");
    auto* raw=statement.get();
    BindText(handle_,raw,1,input.fixture_id); BindText(handle_,raw,2,input.fixture_name);
    Check(handle_,sqlite3_bind_int64(raw,3,input.fixture_version)); BindText(handle_,raw,4,input.text_sha256);
    Check(handle_,sqlite3_bind_int64(raw,5,input.text_bytes)); BindOptionalInt(handle_,raw,6,input.token_count);
    BindOptionalText(handle_,raw,7,input.token_ids_json); BindText(handle_,raw,8,input.description);
    Done(handle_,raw);
}

std::int64_t Database::InsertInferenceConfiguration(
    const InferenceConfigurationRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO inference_configurations(session_id, model_id, fixture_id, configuration_hash,
    phase, observer_mode, gpu_layers, context_size, prompt_tokens, generated_tokens,
    warmups, repetitions, seed, sampling_json, backend_mode, load_mode, status)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    Check(handle_,sqlite3_bind_int64(raw,1,input.session_id)); BindText(handle_,raw,2,input.model_id);
    BindOptionalText(handle_,raw,3,input.fixture_id); BindText(handle_,raw,4,input.configuration_hash);
    BindText(handle_,raw,5,input.phase); BindText(handle_,raw,6,input.observer_mode);
    Check(handle_,sqlite3_bind_int64(raw,7,input.gpu_layers)); Check(handle_,sqlite3_bind_int64(raw,8,input.context_size));
    Check(handle_,sqlite3_bind_int64(raw,9,input.prompt_tokens)); Check(handle_,sqlite3_bind_int64(raw,10,input.generated_tokens));
    Check(handle_,sqlite3_bind_int64(raw,11,input.warmups)); Check(handle_,sqlite3_bind_int64(raw,12,input.repetitions));
    Check(handle_,sqlite3_bind_int64(raw,13,input.seed)); BindText(handle_,raw,14,input.sampling_json);
    BindText(handle_,raw,15,input.backend_mode); BindText(handle_,raw,16,input.load_mode); BindText(handle_,raw,17,input.status);
    Done(handle_,raw); return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertInferenceBenchmark(const InferenceBenchmarkRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO inference_benchmarks(inference_configuration_id, status, model_load_ns,
    prompt_statistics_json, decode_statistics_json, perf_context_json, telemetry_json,
    graph_split_detail, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    Check(handle_,sqlite3_bind_int64(raw,1,input.configuration_id)); BindText(handle_,raw,2,input.status);
    BindOptionalInt(handle_,raw,3,input.model_load_ns); BindText(handle_,raw,4,input.prompt_statistics_json);
    BindText(handle_,raw,5,input.decode_statistics_json); BindText(handle_,raw,6,input.perf_context_json);
    BindText(handle_,raw,7,input.telemetry_json); BindText(handle_,raw,8,input.graph_split_detail);
    BindOptionalText(handle_,raw,9,input.message); Done(handle_,raw); return sqlite3_last_insert_rowid(handle_);
}

void Database::InsertInferenceSample(const InferenceSampleRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO inference_samples(inference_benchmark_id, sample_index, prompt_ns,
    decode_total_ns, prompt_tokens_per_second, generation_tokens_per_second,
    graph_node_count, graph_split_count, observer_event_count, dropped_event_count, status)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    Check(handle_,sqlite3_bind_int64(raw,1,input.benchmark_id)); Check(handle_,sqlite3_bind_int64(raw,2,input.sample_index));
    Check(handle_,sqlite3_bind_int64(raw,3,input.prompt_ns)); Check(handle_,sqlite3_bind_int64(raw,4,input.decode_total_ns));
    Check(handle_,sqlite3_bind_double(raw,5,input.prompt_tokens_per_second)); Check(handle_,sqlite3_bind_double(raw,6,input.generation_tokens_per_second));
    Check(handle_,sqlite3_bind_int64(raw,7,input.graph_node_count)); BindOptionalInt(handle_,raw,8,input.graph_split_count);
    Check(handle_,sqlite3_bind_int64(raw,9,input.observer_event_count)); Check(handle_,sqlite3_bind_int64(raw,10,input.dropped_event_count));
    BindText(handle_,raw,11,input.status); Done(handle_,raw);
}

void Database::InsertInferenceToken(const InferenceTokenRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO inference_token_samples(inference_benchmark_id, sample_index, phase, token_index,
    input_token_id, output_token_id, context_depth, decode_start_ns, decode_end_ns,
    decode_duration_ns, observer_mode, graph_node_count, layer_sequence_reference,
    unique_weight_tensor_count, unique_weight_bytes, projected_blocks_json,
    telemetry_reference, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    Check(handle_,sqlite3_bind_int64(raw,1,input.benchmark_id)); Check(handle_,sqlite3_bind_int64(raw,2,input.sample_index));
    BindText(handle_,raw,3,input.phase); Check(handle_,sqlite3_bind_int64(raw,4,input.token_index));
    Check(handle_,sqlite3_bind_int64(raw,5,input.input_token_id)); BindOptionalInt(handle_,raw,6,input.output_token_id);
    Check(handle_,sqlite3_bind_int64(raw,7,input.context_depth)); Check(handle_,sqlite3_bind_int64(raw,8,input.decode_start_ns));
    Check(handle_,sqlite3_bind_int64(raw,9,input.decode_end_ns)); Check(handle_,sqlite3_bind_int64(raw,10,input.decode_duration_ns));
    BindText(handle_,raw,11,input.observer_mode); Check(handle_,sqlite3_bind_int64(raw,12,input.graph_node_count));
    BindOptionalText(handle_,raw,13,input.layer_sequence_reference); Check(handle_,sqlite3_bind_int64(raw,14,input.unique_weight_tensor_count));
    Check(handle_,sqlite3_bind_int64(raw,15,input.unique_weight_bytes)); BindText(handle_,raw,16,input.projected_blocks_json);
    BindOptionalText(handle_,raw,17,input.telemetry_reference); BindText(handle_,raw,18,input.status); Done(handle_,raw);
}

std::int64_t Database::InsertObserverProfile(const ObserverProfileRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO inference_observer_profiles(inference_benchmark_id, observer_mode, ask_calls,
    materialized_calls, compact_events, dropped_events, ring_high_water_mark,
    callback_contract, dictionary_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    Check(handle_,sqlite3_bind_int64(raw,1,input.benchmark_id)); BindText(handle_,raw,2,input.observer_mode);
    Check(handle_,sqlite3_bind_int64(raw,3,input.ask_calls)); Check(handle_,sqlite3_bind_int64(raw,4,input.materialized_calls));
    Check(handle_,sqlite3_bind_int64(raw,5,input.compact_events)); Check(handle_,sqlite3_bind_int64(raw,6,input.dropped_events));
    BindOptionalInt(handle_,raw,7,input.ring_high_water_mark); BindText(handle_,raw,8,input.callback_contract);
    BindText(handle_,raw,9,input.dictionary_json); Done(handle_,raw); return sqlite3_last_insert_rowid(handle_);
}

void Database::InsertObserverEvent(const ObserverEventRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO inference_observer_events(observer_profile_id, event_index, timestamp_ns,
    eval_index, sequence_index, tensor_id_0, tensor_id_1, op_id, layer_index,
    event_class, flags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    Check(handle_,sqlite3_bind_int64(raw,1,input.observer_profile_id)); Check(handle_,sqlite3_bind_int64(raw,2,input.event_index));
    Check(handle_,sqlite3_bind_int64(raw,3,input.timestamp_ns)); Check(handle_,sqlite3_bind_int64(raw,4,input.eval_index));
    Check(handle_,sqlite3_bind_int64(raw,5,input.sequence_index)); BindOptionalInt(handle_,raw,6,input.tensor_id_0);
    BindOptionalInt(handle_,raw,7,input.tensor_id_1); Check(handle_,sqlite3_bind_int64(raw,8,input.op_id));
    BindOptionalInt(handle_,raw,9,input.layer_index); Check(handle_,sqlite3_bind_int64(raw,10,input.event_class));
    Check(handle_,sqlite3_bind_int64(raw,11,input.flags)); Done(handle_,raw);
}

void Database::UpsertTensorDemandEvent(const TensorDemandRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO tensor_demand_events(inference_benchmark_id, token_index, sequence_index,
    tensor_id, first_event_index, layer_index, tensor_bytes, file_offset,
    causal_known_at_ns, demand_ns)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(inference_benchmark_id, token_index, tensor_id) DO UPDATE SET
    sequence_index=MIN(sequence_index, excluded.sequence_index),
    first_event_index=MIN(first_event_index, excluded.first_event_index),
    layer_index=COALESCE(layer_index, excluded.layer_index),
    tensor_bytes=excluded.tensor_bytes, file_offset=excluded.file_offset,
    causal_known_at_ns=COALESCE(causal_known_at_ns, excluded.causal_known_at_ns),
    demand_ns=MIN(demand_ns, excluded.demand_ns);
)sql");
    auto* raw = statement.get();
    Check(handle_, sqlite3_bind_int64(raw, 1, input.benchmark_id));
    Check(handle_, sqlite3_bind_int64(raw, 2, input.token_index));
    Check(handle_, sqlite3_bind_int64(raw, 3, input.sequence_index));
    Check(handle_, sqlite3_bind_int64(raw, 4, input.tensor_id));
    Check(handle_, sqlite3_bind_int64(raw, 5, input.first_event_index));
    BindOptionalInt(handle_, raw, 6, input.layer_index);
    Check(handle_, sqlite3_bind_int64(raw, 7, input.tensor_bytes));
    Check(handle_, sqlite3_bind_int64(raw, 8, input.file_offset));
    BindOptionalInt(handle_, raw, 9, input.causal_known_at_ns);
    Check(handle_, sqlite3_bind_int64(raw, 10, input.demand_ns));
    Done(handle_, raw);
}

void Database::InsertTensorBlockProjection(const TensorBlockProjectionRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO tensor_block_projections(inference_benchmark_id, model_id, block_size_bytes,
    scope, scope_index, useful_tensor_bytes, projected_block_count, projected_bytes,
    overfetch_bytes, overfetch_ratio, block_ids_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    BindOptionalInt(handle_,raw,1,input.benchmark_id); BindText(handle_,raw,2,input.model_id);
    Check(handle_,sqlite3_bind_int64(raw,3,input.block_size_bytes)); BindText(handle_,raw,4,input.scope);
    BindOptionalInt(handle_,raw,5,input.scope_index); Check(handle_,sqlite3_bind_int64(raw,6,input.useful_tensor_bytes));
    Check(handle_,sqlite3_bind_int64(raw,7,input.projected_block_count)); Check(handle_,sqlite3_bind_int64(raw,8,input.projected_bytes));
    Check(handle_,sqlite3_bind_int64(raw,9,input.overfetch_bytes)); Check(handle_,sqlite3_bind_double(raw,10,input.overfetch_ratio));
    BindText(handle_,raw,11,input.block_ids_json); Done(handle_,raw);
}

void Database::InsertObserverOverhead(const ObserverOverheadRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO observer_overhead_benchmarks(session_id, model_id, phase, observer_mode,
    paired_repetitions, baseline_samples_json, observed_samples_json,
    paired_delta_statistics_json, overhead_percent, confidence_interval_json,
    production_gate_percent, gate_passed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    Check(handle_,sqlite3_bind_int64(raw,1,input.session_id)); BindText(handle_,raw,2,input.model_id);
    BindText(handle_,raw,3,input.phase); BindText(handle_,raw,4,input.observer_mode);
    Check(handle_,sqlite3_bind_int64(raw,5,input.paired_repetitions)); BindText(handle_,raw,6,input.baseline_samples_json);
    BindText(handle_,raw,7,input.observed_samples_json); BindText(handle_,raw,8,input.paired_delta_statistics_json);
    Check(handle_,sqlite3_bind_double(raw,9,input.overhead_percent)); BindText(handle_,raw,10,input.confidence_interval_json);
    Check(handle_,sqlite3_bind_double(raw,11,input.production_gate_percent)); Check(handle_,sqlite3_bind_int(raw,12,input.gate_passed?1:0));
    Done(handle_,raw);
}

void Database::InsertMoeDemandProfile(const MoeDemandProfileRecordInput& input) {
    if (!input.inference_benchmark_id) {
        Statement remove(handle_,
            "DELETE FROM moe_demand_profiles WHERE model_id=? AND inference_benchmark_id IS NULL;");
        BindText(handle_, remove.get(), 1, input.model_id);
        Done(handle_, remove.get());
    }
    Statement statement(handle_, R"sql(
INSERT INTO moe_demand_profiles(inference_benchmark_id, model_id, status, layer_index,
    expert_count, selected_experts_observable, expert_working_set_bytes,
    reuse_json, causal_cue_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw = statement.get();
    BindOptionalInt(handle_, raw, 1, input.inference_benchmark_id);
    BindText(handle_, raw, 2, input.model_id);
    BindText(handle_, raw, 3, input.status);
    BindOptionalInt(handle_, raw, 4, input.layer_index);
    BindOptionalInt(handle_, raw, 5, input.expert_count);
    Check(handle_, sqlite3_bind_int(raw, 6, input.selected_experts_observable ? 1 : 0));
    BindOptionalInt(handle_, raw, 7, input.expert_working_set_bytes);
    BindText(handle_, raw, 8, input.reuse_json);
    BindText(handle_, raw, 9, input.causal_cue_json);
    Done(handle_, raw);
}

std::optional<std::int64_t> Database::LatestInferenceBenchmarkId(
    const std::string& model_id) const {
    Statement statement(handle_, R"sql(
SELECT b.inference_benchmark_id FROM inference_benchmarks b
JOIN inference_configurations c USING(inference_configuration_id)
WHERE c.model_id=? ORDER BY b.inference_benchmark_id DESC LIMIT 1;
)sql");
    BindText(handle_,statement.get(),1,model_id);
    const int code=sqlite3_step(statement.get());
    if(code==SQLITE_DONE) return std::nullopt;
    if(code!=SQLITE_ROW) throw DatabaseError(code,sqlite3_errmsg(handle_));
    return sqlite3_column_int64(statement.get(),0);
}

void Database::InsertShadowPipelineResult(const ShadowPipelineRecordInput& input) {
    Statement statement(handle_, R"sql(
INSERT INTO shadow_pipeline_results(inference_benchmark_id, token_index,
    tensor_or_block_id, block_size_bytes, pipeline_type, lookahead_assumption,
    knowledge_mode, available_lead_time_ns, pipeline_profile_reference,
    deadline_ns, predicted_hit, margin_ns, extrapolated)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql");
    auto* raw=statement.get();
    Check(handle_,sqlite3_bind_int64(raw,1,input.inference_benchmark_id));
    Check(handle_,sqlite3_bind_int64(raw,2,input.token_index));
    Check(handle_,sqlite3_bind_int64(raw,3,input.tensor_or_block_id));
    Check(handle_,sqlite3_bind_int64(raw,4,input.block_size_bytes));
    BindText(handle_,raw,5,input.pipeline_type); BindText(handle_,raw,6,input.lookahead_assumption);
    BindText(handle_,raw,7,input.knowledge_mode); Check(handle_,sqlite3_bind_int64(raw,8,input.available_lead_time_ns));
    BindText(handle_,raw,9,input.pipeline_profile_reference); Check(handle_,sqlite3_bind_int64(raw,10,input.deadline_ns));
    Check(handle_,sqlite3_bind_int(raw,11,input.predicted_hit?1:0)); Check(handle_,sqlite3_bind_int64(raw,12,input.margin_ns));
    Check(handle_,sqlite3_bind_int(raw,13,input.extrapolated?1:0)); Done(handle_,raw);
}

LlamaObservationCounts Database::LlamaCounts() const {
    LlamaObservationCounts counts;
    counts.dependencies = Scalar(handle_, "SELECT COUNT(*) FROM llama_dependencies;");
    counts.models = Scalar(handle_, "SELECT COUNT(*) FROM gguf_models;");
    counts.tensors = Scalar(handle_, "SELECT COUNT(*) FROM gguf_tensors;");
    counts.storage_provenance = Scalar(handle_, "SELECT COUNT(*) FROM gguf_model_storage_provenance;");
    counts.configurations = Scalar(handle_, "SELECT COUNT(*) FROM inference_configurations;");
    counts.benchmarks = Scalar(handle_, "SELECT COUNT(*) FROM inference_benchmarks;");
    counts.token_samples = Scalar(handle_, "SELECT COUNT(*) FROM inference_token_samples;");
    counts.observer_events = Scalar(handle_, "SELECT COUNT(*) FROM inference_observer_events;");
    counts.tensor_demand_events = Scalar(handle_, "SELECT COUNT(*) FROM tensor_demand_events;");
    counts.block_projections = Scalar(handle_, "SELECT COUNT(*) FROM tensor_block_projections;");
    counts.shadow_results = Scalar(handle_, "SELECT COUNT(*) FROM shadow_pipeline_results;");
    counts.overhead_benchmarks = Scalar(handle_, "SELECT COUNT(*) FROM observer_overhead_benchmarks;");
    counts.moe_profiles = Scalar(handle_, "SELECT COUNT(*) FROM moe_demand_profiles;");
    return counts;
}

}  // namespace sidecar::database
