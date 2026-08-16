#include "sidecar/database/database.hpp"

#include <sqlite3.h>

namespace sidecar::database {
namespace {

class StorageStatement final {
public:
    StorageStatement(sqlite3* database, const char* sql) : database_(database) {
        const int result = sqlite3_prepare_v2(database_, sql, -1, &value_, nullptr);
        if (result != SQLITE_OK) throw DatabaseError(result, sqlite3_errmsg(database_));
    }
    ~StorageStatement() { if (value_) sqlite3_finalize(value_); }
    [[nodiscard]] sqlite3_stmt* get() const noexcept { return value_; }
private:
    sqlite3* database_{};
    sqlite3_stmt* value_{};
};

void Check(sqlite3* database, int result, int expected = SQLITE_OK) {
    if (result != expected) throw DatabaseError(result, sqlite3_errmsg(database));
}
void Text(sqlite3* database, sqlite3_stmt* statement, int index, const std::string& value) {
    Check(database, sqlite3_bind_text(statement, index, value.c_str(),
                                      static_cast<int>(value.size()), SQLITE_TRANSIENT));
}
void OptionalText(sqlite3* database, sqlite3_stmt* statement, int index,
                  const std::optional<std::string>& value) {
    if (value) Text(database, statement, index, *value);
    else Check(database, sqlite3_bind_null(statement, index));
}
void OptionalInt(sqlite3* database, sqlite3_stmt* statement, int index,
                 const std::optional<std::int64_t>& value) {
    if (value) Check(database, sqlite3_bind_int64(statement, index, *value));
    else Check(database, sqlite3_bind_null(statement, index));
}
void OptionalDouble(sqlite3* database, sqlite3_stmt* statement, int index,
                    const std::optional<double>& value) {
    if (value) Check(database, sqlite3_bind_double(statement, index, *value));
    else Check(database, sqlite3_bind_null(statement, index));
}

}  // namespace

std::int64_t Database::UpsertStorageDataset(const StorageDatasetInput& value) {
    static constexpr const char* kSql = R"sql(
INSERT INTO storage_datasets(
 storage_id,machine_hash,format_version,generator,seed,size_bytes,identity_hash,
 file_path,volume_name,physical_disk_number,verified,full_verification,verified_bytes,verified_at)
VALUES ((SELECT storage_id FROM storage_devices WHERE machine_hash=? AND persistent_id=?),
 ?,?,?,?,?,?,?,?,?,?,?,?,CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END)
ON CONFLICT(machine_hash,identity_hash,file_path) DO UPDATE SET
 storage_id=excluded.storage_id,updated_at=CURRENT_TIMESTAMP,
 verified=MAX(storage_datasets.verified,excluded.verified),
 full_verification=MAX(storage_datasets.full_verification,excluded.full_verification),
 verified_bytes=MAX(storage_datasets.verified_bytes,excluded.verified_bytes),
 verified_at=CASE WHEN excluded.verified_bytes>=storage_datasets.verified_bytes
                  THEN excluded.verified_at ELSE storage_datasets.verified_at END,
 volume_name=excluded.volume_name,
 physical_disk_number=excluded.physical_disk_number
RETURNING storage_dataset_id;)sql";
    StorageStatement statement(handle_, kSql);
    auto* raw = statement.get();
    Text(handle_, raw, 1, value.machine_hash); Text(handle_, raw, 2, value.persistent_id);
    Text(handle_, raw, 3, value.machine_hash);
    Check(handle_, sqlite3_bind_int64(raw, 4, value.format_version));
    Text(handle_, raw, 5, value.generator);
    Check(handle_, sqlite3_bind_int64(raw, 6, value.seed));
    Check(handle_, sqlite3_bind_int64(raw, 7, value.size_bytes));
    Text(handle_, raw, 8, value.identity_hash); Text(handle_, raw, 9, value.file_path);
    Text(handle_, raw, 10, value.volume_name);
    Check(handle_, sqlite3_bind_int64(raw, 11, value.physical_disk_number));
    Check(handle_, sqlite3_bind_int(raw, 12, value.verified));
    Check(handle_, sqlite3_bind_int(raw, 13, value.full_verification));
    Check(handle_, sqlite3_bind_int64(raw, 14, value.verified_bytes));
    Check(handle_, sqlite3_bind_int(raw, 15, value.verified));
    Check(handle_, sqlite3_step(raw), SQLITE_ROW);
    return sqlite3_column_int64(raw, 0);
}

std::int64_t Database::InsertStorageHealthSnapshot(
    const StorageHealthSnapshotInput& value) {
    static constexpr const char* kSql = R"sql(
INSERT INTO storage_health_snapshots(
 storage_id,session_id,phase,provider,status,critical_warning,temperature_c,
 available_spare_percent,available_spare_threshold_percent,percentage_used,
 data_units_read_low64,data_units_written_low64,host_read_commands_low64,
 host_write_commands_low64,controller_busy_minutes_low64,power_cycles_low64,
 power_on_hours_low64,unsafe_shutdowns_low64,media_data_errors_low64,
 error_log_entries_low64,raw_evidence_json,message)
VALUES ((SELECT storage_id FROM storage_devices WHERE machine_hash=? AND persistent_id=?),
 ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);)sql";
    StorageStatement statement(handle_, kSql);
    auto* raw = statement.get();
    Text(handle_, raw, 1, value.machine_hash); Text(handle_, raw, 2, value.persistent_id);
    OptionalInt(handle_, raw, 3, value.session_id);
    Text(handle_, raw, 4, value.phase); Text(handle_, raw, 5, value.provider);
    Text(handle_, raw, 6, value.status);
    OptionalInt(handle_, raw, 7, value.critical_warning);
    OptionalDouble(handle_, raw, 8, value.temperature_c);
    OptionalInt(handle_, raw, 9, value.available_spare_percent);
    OptionalInt(handle_, raw, 10, value.available_spare_threshold_percent);
    OptionalInt(handle_, raw, 11, value.percentage_used);
    OptionalInt(handle_, raw, 12, value.data_units_read_low64);
    OptionalInt(handle_, raw, 13, value.data_units_written_low64);
    OptionalInt(handle_, raw, 14, value.host_read_commands_low64);
    OptionalInt(handle_, raw, 15, value.host_write_commands_low64);
    OptionalInt(handle_, raw, 16, value.controller_busy_minutes_low64);
    OptionalInt(handle_, raw, 17, value.power_cycles_low64);
    OptionalInt(handle_, raw, 18, value.power_on_hours_low64);
    OptionalInt(handle_, raw, 19, value.unsafe_shutdowns_low64);
    OptionalInt(handle_, raw, 20, value.media_data_errors_low64);
    OptionalInt(handle_, raw, 21, value.error_log_entries_low64);
    Text(handle_, raw, 22, value.raw_evidence_json);
    OptionalText(handle_, raw, 23, value.message);
    Check(handle_, sqlite3_step(raw), SQLITE_DONE);
    return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertStorageConfiguration(const StorageConfigurationInput& value) {
    static constexpr const char* kSql = R"sql(
INSERT INTO storage_benchmark_configurations(
 session_id,storage_dataset_id,configuration_hash,backend,access_pattern,destination_kind,
 block_bytes,queue_depth,request_count,outstanding_byte_cap,timeout_ms,ordering_seed,
 window_bytes,phase,cache_classification,status,skip_reason)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);)sql";
    StorageStatement statement(handle_, kSql); auto* raw = statement.get();
    Check(handle_, sqlite3_bind_int64(raw, 1, value.session_id));
    Check(handle_, sqlite3_bind_int64(raw, 2, value.storage_dataset_id));
    Text(handle_, raw, 3, value.configuration_hash); Text(handle_, raw, 4, value.backend);
    Text(handle_, raw, 5, value.access_pattern); Text(handle_, raw, 6, value.destination_kind);
    Check(handle_, sqlite3_bind_int64(raw, 7, value.block_bytes));
    Check(handle_, sqlite3_bind_int64(raw, 8, value.queue_depth));
    Check(handle_, sqlite3_bind_int64(raw, 9, value.request_count));
    Check(handle_, sqlite3_bind_int64(raw, 10, value.outstanding_byte_cap));
    Check(handle_, sqlite3_bind_int64(raw, 11, value.timeout_ms));
    Check(handle_, sqlite3_bind_int64(raw, 12, value.ordering_seed));
    Check(handle_, sqlite3_bind_int64(raw, 13, value.window_bytes));
    Text(handle_, raw, 14, value.phase); Text(handle_, raw, 15, value.cache_classification);
    Text(handle_, raw, 16, value.status); OptionalText(handle_, raw, 17, value.skip_reason);
    Check(handle_, sqlite3_step(raw), SQLITE_DONE);
    return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertStorageBenchmark(const StorageBenchmarkInput& value) {
    static constexpr const char* kSql = R"sql(
INSERT INTO storage_benchmarks(
 storage_benchmark_configuration_id,health_before_id,health_after_id,status,verified,
 completed_bytes,wall_time_ns,bytes_per_second,iops,statistics_json,cpu_json,message)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?);)sql";
    StorageStatement statement(handle_, kSql); auto* raw = statement.get();
    Check(handle_, sqlite3_bind_int64(raw, 1, value.configuration_id));
    OptionalInt(handle_, raw, 2, value.health_before_id);
    OptionalInt(handle_, raw, 3, value.health_after_id);
    Text(handle_, raw, 4, value.status);
    Check(handle_, sqlite3_bind_int(raw, 5, value.verified));
    Check(handle_, sqlite3_bind_int64(raw, 6, value.completed_bytes));
    Check(handle_, sqlite3_bind_int64(raw, 7, value.wall_time_ns));
    Check(handle_, sqlite3_bind_double(raw, 8, value.bytes_per_second));
    Check(handle_, sqlite3_bind_double(raw, 9, value.iops));
    Text(handle_, raw, 10, value.statistics_json); Text(handle_, raw, 11, value.cpu_json);
    OptionalText(handle_, raw, 12, value.message);
    Check(handle_, sqlite3_step(raw), SQLITE_DONE);
    return sqlite3_last_insert_rowid(handle_);
}

void Database::InsertStorageSample(const StorageSampleInput& value) {
    static constexpr const char* kSql =
        "INSERT INTO storage_samples(storage_benchmark_id,request_index,batch_id,file_offset,"
        "requested_bytes,completed_bytes,submission_reference_ns,submission_cost_ns,"
        "completion_latency_ns,completion_processing_ns,status,native_error)"
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?);";
    StorageStatement statement(handle_, kSql); auto* raw = statement.get();
    Check(handle_, sqlite3_bind_int64(raw, 1, value.benchmark_id));
    Check(handle_, sqlite3_bind_int64(raw, 2, value.request_index));
    Check(handle_, sqlite3_bind_int64(raw, 3, value.batch_id));
    Check(handle_, sqlite3_bind_int64(raw, 4, value.file_offset));
    Check(handle_, sqlite3_bind_int64(raw, 5, value.requested_bytes));
    Check(handle_, sqlite3_bind_int64(raw, 6, value.completed_bytes));
    Check(handle_, sqlite3_bind_int64(raw, 7, value.submission_reference_ns));
    Check(handle_, sqlite3_bind_int64(raw, 8, value.submission_cost_ns));
    Check(handle_, sqlite3_bind_int64(raw, 9, value.completion_latency_ns));
    Check(handle_, sqlite3_bind_int64(raw, 10, value.completion_processing_ns));
    Text(handle_, raw, 11, value.status); OptionalInt(handle_, raw, 12, value.native_error);
    Check(handle_, sqlite3_step(raw), SQLITE_DONE);
}

void Database::InsertStorageDeadline(const StorageDeadlineInput& value) {
    StorageStatement statement(handle_,
        "INSERT INTO storage_deadline_profiles(storage_benchmark_id,deadline_ns,hits,misses,"
        "success_rate,compliant_bytes_per_second)VALUES(?,?,?,?,?,?);");
    auto* raw = statement.get();
    Check(handle_, sqlite3_bind_int64(raw, 1, value.benchmark_id));
    Check(handle_, sqlite3_bind_int64(raw, 2, value.deadline_ns));
    Check(handle_, sqlite3_bind_int64(raw, 3, value.hits));
    Check(handle_, sqlite3_bind_int64(raw, 4, value.misses));
    Check(handle_, sqlite3_bind_double(raw, 5, value.success_rate));
    Check(handle_, sqlite3_bind_double(raw, 6, value.compliant_bytes_per_second));
    Check(handle_, sqlite3_step(raw), SQLITE_DONE);
}

void Database::InsertStorageBackendProfile(const StorageBackendProfileInput& value) {
    static constexpr const char* kSql = R"sql(
INSERT INTO storage_backend_profiles(
 session_id,backend,access_pattern,destination_kind,peak_bytes_per_second,peak_block_bytes,
 peak_queue_depth,knee_80_queue_depth,knee_90_queue_depth,knee_95_queue_depth,
 safe_envelopes_json,supply_profile_json)VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(session_id,backend,access_pattern,destination_kind) DO UPDATE SET
 peak_bytes_per_second=excluded.peak_bytes_per_second,peak_block_bytes=excluded.peak_block_bytes,
 peak_queue_depth=excluded.peak_queue_depth,knee_80_queue_depth=excluded.knee_80_queue_depth,
 knee_90_queue_depth=excluded.knee_90_queue_depth,knee_95_queue_depth=excluded.knee_95_queue_depth,
 safe_envelopes_json=excluded.safe_envelopes_json,supply_profile_json=excluded.supply_profile_json;)sql";
    StorageStatement statement(handle_, kSql); auto* raw = statement.get();
    Check(handle_, sqlite3_bind_int64(raw, 1, value.session_id));
    Text(handle_, raw, 2, value.backend); Text(handle_, raw, 3, value.access_pattern);
    Text(handle_, raw, 4, value.destination_kind);
    Check(handle_, sqlite3_bind_double(raw, 5, value.peak_bytes_per_second));
    Check(handle_, sqlite3_bind_int64(raw, 6, value.peak_block_bytes));
    Check(handle_, sqlite3_bind_int64(raw, 7, value.peak_queue_depth));
    OptionalInt(handle_, raw, 8, value.knee_80_queue_depth);
    OptionalInt(handle_, raw, 9, value.knee_90_queue_depth);
    OptionalInt(handle_, raw, 10, value.knee_95_queue_depth);
    Text(handle_, raw, 11, value.safe_envelopes_json);
    Text(handle_, raw, 12, value.supply_profile_json);
    Check(handle_, sqlite3_step(raw), SQLITE_DONE);
}

StoragePhysicsCounts Database::StorageCounts() const {
    StoragePhysicsCounts value;
    if (!TableExists("storage_datasets")) return value;
    value.datasets = QueryInt64("SELECT COUNT(*) FROM storage_datasets;");
    value.health_snapshots = QueryInt64("SELECT COUNT(*) FROM storage_health_snapshots;");
    value.configurations = QueryInt64("SELECT COUNT(*) FROM storage_benchmark_configurations;");
    value.benchmarks = QueryInt64("SELECT COUNT(*) FROM storage_benchmarks;");
    value.samples = QueryInt64("SELECT COUNT(*) FROM storage_samples;");
    value.deadline_profiles = QueryInt64("SELECT COUNT(*) FROM storage_deadline_profiles;");
    value.backend_profiles = QueryInt64("SELECT COUNT(*) FROM storage_backend_profiles;");
    return value;
}

std::vector<StorageBenchmarkSummary> Database::LatestStorageBenchmarks() const {
    std::vector<StorageBenchmarkSummary> values;
    if (!TableExists("storage_benchmarks")) return values;
    static constexpr const char* kSql = R"sql(
SELECT c.session_id,c.backend,c.access_pattern,c.destination_kind,c.block_bytes,c.queue_depth,
 b.status,b.bytes_per_second,b.iops,
 json_extract(b.statistics_json,'$.completion_latency_ns.count'),
 json_extract(b.statistics_json,'$.completion_latency_ns.p50'),
 json_extract(b.statistics_json,'$.completion_latency_ns.p95'),
 json_extract(b.statistics_json,'$.completion_latency_ns.p99')
FROM storage_benchmarks b JOIN storage_benchmark_configurations c
 ON c.storage_benchmark_configuration_id=b.storage_benchmark_configuration_id
WHERE c.session_id=(SELECT COALESCE(MAX(session_id),0) FROM storage_benchmark_configurations)
ORDER BY c.backend,c.access_pattern,c.destination_kind,c.block_bytes,c.queue_depth;)sql";
    StorageStatement statement(handle_, kSql);
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        StorageBenchmarkSummary value;
        auto text = [&](int column) {
            const auto* raw = sqlite3_column_text(statement.get(), column);
            return raw ? std::string(reinterpret_cast<const char*>(raw)) : std::string();
        };
        value.session_id=sqlite3_column_int64(statement.get(),0);
        value.backend=text(1); value.access_pattern=text(2); value.destination_kind=text(3);
        value.block_bytes=sqlite3_column_int64(statement.get(),4);
        value.queue_depth=sqlite3_column_int64(statement.get(),5); value.status=text(6);
        value.bytes_per_second=sqlite3_column_double(statement.get(),7);
        value.iops=sqlite3_column_double(statement.get(),8);
        value.sample_count=sqlite3_column_int64(statement.get(),9);
        value.p50_ns=sqlite3_column_double(statement.get(),10);
        value.p95_ns=sqlite3_column_double(statement.get(),11);
        value.p99_ns=sqlite3_column_double(statement.get(),12);
        values.push_back(std::move(value));
    }
    return values;
}

}  // namespace sidecar::database
