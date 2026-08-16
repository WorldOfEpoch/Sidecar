#include "sidecar/database/database.hpp"

#include <sqlite3.h>

namespace sidecar::database {
namespace {
class Statement final {
public:
    Statement(sqlite3* database, const char* sql) : database_(database) {
        const auto result = sqlite3_prepare_v2(database, sql, -1, &value_, nullptr);
        if (result != SQLITE_OK) throw DatabaseError(result, sqlite3_errmsg(database));
    }
    ~Statement() { if (value_) sqlite3_finalize(value_); }
    sqlite3_stmt* get() const noexcept { return value_; }
private:
    sqlite3* database_{};
    sqlite3_stmt* value_{};
};
void Check(sqlite3* database, int result, int expected = SQLITE_OK) {
    if (result != expected) throw DatabaseError(result, sqlite3_errmsg(database));
}
void Text(sqlite3* db, sqlite3_stmt* s, int i, const std::string& v) {
    Check(db, sqlite3_bind_text(s, i, v.c_str(), static_cast<int>(v.size()), SQLITE_TRANSIENT));
}
void OptionalText(sqlite3* db, sqlite3_stmt* s, int i,
                  const std::optional<std::string>& v) {
    if (v) Text(db, s, i, *v); else Check(db, sqlite3_bind_null(s, i));
}
void OptionalInt(sqlite3* db, sqlite3_stmt* s, int i,
                 const std::optional<std::int64_t>& v) {
    if (v) Check(db, sqlite3_bind_int64(s, i, *v)); else Check(db, sqlite3_bind_null(s, i));
}
std::string ColumnText(sqlite3_stmt* s, int i) {
    const auto* value = sqlite3_column_text(s, i);
    return value ? reinterpret_cast<const char*>(value) : std::string{};
}
std::int64_t LastId(sqlite3* db) { return sqlite3_last_insert_rowid(db); }
}

std::int64_t Database::InsertHostCopyConfiguration(const HostCopyConfigurationInput& v) {
    Statement q(handle_, "INSERT INTO host_copy_configurations(session_id,configuration_hash,"
        "bytes,worker_count,warmups,repetitions,concurrent_compute,compute_window_us,phase,status)"
        " VALUES(?,?,?,?,?,?,?,?,?,?);");
    auto* s=q.get(); int i=1;
    Check(handle_,sqlite3_bind_int64(s,i++,v.session_id)); Text(handle_,s,i++,v.configuration_hash);
    Check(handle_,sqlite3_bind_int64(s,i++,v.bytes)); Check(handle_,sqlite3_bind_int64(s,i++,v.worker_count));
    Check(handle_,sqlite3_bind_int64(s,i++,v.warmups)); Check(handle_,sqlite3_bind_int64(s,i++,v.repetitions));
    OptionalText(handle_,s,i++,v.concurrent_compute); Check(handle_,sqlite3_bind_double(s,i++,v.compute_window_us));
    Text(handle_,s,i++,v.phase); Text(handle_,s,i++,v.status); Check(handle_,sqlite3_step(s),SQLITE_DONE);
    return LastId(handle_);
}

std::int64_t Database::InsertHostCopyBenchmark(const HostCopyBenchmarkInput& v) {
    Statement q(handle_, "INSERT INTO host_copy_benchmarks(host_copy_configuration_id,status,"
        "statistics_json,affinity,message) VALUES(?,?,?,?,?);"); auto* s=q.get();
    Check(handle_,sqlite3_bind_int64(s,1,v.configuration_id)); Text(handle_,s,2,v.status);
    Text(handle_,s,3,v.statistics_json); Text(handle_,s,4,v.affinity); OptionalText(handle_,s,5,v.message);
    Check(handle_,sqlite3_step(s),SQLITE_DONE); return LastId(handle_);
}

void Database::InsertHostCopySample(const HostCopySampleInput& v) {
    Statement q(handle_, "INSERT INTO host_copy_samples(host_copy_benchmark_id,sample_index,wall_ns,"
        "process_cpu_ns,compute_ns,bytes_per_second,status,message) VALUES(?,?,?,?,?,?,?,?);");
    auto* s=q.get(); int i=1;
    Check(handle_,sqlite3_bind_int64(s,i++,v.benchmark_id)); Check(handle_,sqlite3_bind_int64(s,i++,v.sample_index));
    Check(handle_,sqlite3_bind_int64(s,i++,v.wall_ns)); Check(handle_,sqlite3_bind_int64(s,i++,v.process_cpu_ns));
    Check(handle_,sqlite3_bind_int64(s,i++,v.compute_ns)); Check(handle_,sqlite3_bind_double(s,i++,v.bytes_per_second));
    Text(handle_,s,i++,v.status); OptionalText(handle_,s,i++,v.message); Check(handle_,sqlite3_step(s),SQLITE_DONE);
}

std::int64_t Database::InsertPipelineConfiguration(const PipelineConfigurationInput& v) {
    Statement q(handle_, "INSERT INTO pipeline_configurations(session_id,storage_dataset_id,"
        "configuration_hash,pipeline_type,aggregate_bytes,chunk_bytes,chunk_count,buffer_depth,"
        "compute_type,compute_window_us,deadline_ns,measured_repetitions,ordering_seed,timeout_ms,"
        "phase,contention_test,host_copy_workers,device_index,arena_allocation_id,refinement_reason,status)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"); auto* s=q.get(); int i=1;
    Check(handle_,sqlite3_bind_int64(s,i++,v.session_id)); Check(handle_,sqlite3_bind_int64(s,i++,v.storage_dataset_id));
    Text(handle_,s,i++,v.configuration_hash); Text(handle_,s,i++,v.pipeline_type);
    for(auto x:{v.aggregate_bytes,v.chunk_bytes,v.chunk_count,v.buffer_depth}) Check(handle_,sqlite3_bind_int64(s,i++,x));
    Text(handle_,s,i++,v.compute_type); Check(handle_,sqlite3_bind_double(s,i++,v.compute_window_us));
    for(auto x:{v.deadline_ns,v.measured_repetitions,v.ordering_seed,v.timeout_ms}) Check(handle_,sqlite3_bind_int64(s,i++,x));
    Text(handle_,s,i++,v.phase); Text(handle_,s,i++,v.contention_test);
    Check(handle_,sqlite3_bind_int64(s,i++,v.host_copy_workers)); Check(handle_,sqlite3_bind_int64(s,i++,v.device_index));
    Text(handle_,s,i++,v.arena_allocation_id); OptionalText(handle_,s,i++,v.refinement_reason);
    Text(handle_,s,i++,v.status); Check(handle_,sqlite3_step(s),SQLITE_DONE); return LastId(handle_);
}

std::int64_t Database::InsertPipelineBenchmark(const PipelineBenchmarkInput& v) {
    Statement q(handle_, "INSERT INTO pipeline_benchmarks(pipeline_configuration_id,health_before_id,"
        "health_after_id,status,statistics_json,gpu_before_json,gpu_after_json,host_before_json,"
        "host_after_json,message) VALUES(?,?,?,?,?,?,?,?,?,?);"); auto* s=q.get(); int i=1;
    Check(handle_,sqlite3_bind_int64(s,i++,v.configuration_id)); OptionalInt(handle_,s,i++,v.health_before_id);
    OptionalInt(handle_,s,i++,v.health_after_id); Text(handle_,s,i++,v.status); Text(handle_,s,i++,v.statistics_json);
    Text(handle_,s,i++,v.gpu_before_json); Text(handle_,s,i++,v.gpu_after_json);
    Text(handle_,s,i++,v.host_before_json); Text(handle_,s,i++,v.host_after_json);
    OptionalText(handle_,s,i++,v.message); Check(handle_,sqlite3_step(s),SQLITE_DONE); return LastId(handle_);
}

void Database::InsertPipelineSample(const PipelineSampleInput& v) {
    Statement q(handle_, "INSERT INTO pipeline_samples(pipeline_benchmark_id,sample_index,c0_reference_ns,"
        "p0_reference_ns,cc_ns,pc_ns,makespan_ns,compute_path_added_ns,compute_path_added_percent,"
        "pipeline_slowdown,compute_slowdown,pipeline_overlap_raw,pipeline_overlap_normalized,deadline_ns,"
        "ready_ahead_ns,deadline_hit,storage_ns,host_copy_ns,h2d_ns,fill_latency_ns,steady_state_interval_ns,"
        "drain_latency_ns,gpu_waiting_for_data_ns,h2d_waiting_for_source_ns,pinned_slot_wait_ns,"
        "pageable_slot_wait_ns,nvme_queue_starved_ns,slot_ids_json,verified,status,native_error,message)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"); auto* s=q.get(); int i=1;
    for(auto x:{v.benchmark_id,v.sample_index,v.c0_reference_ns,v.p0_reference_ns,v.cc_ns,v.pc_ns,v.makespan_ns})
        Check(handle_,sqlite3_bind_int64(s,i++,x));
    for(auto x:{v.compute_path_added_ns,v.compute_path_added_percent,v.pipeline_slowdown,v.compute_slowdown,
                v.pipeline_overlap_raw,v.pipeline_overlap_normalized}) Check(handle_,sqlite3_bind_double(s,i++,x));
    for(auto x:{v.deadline_ns,v.ready_ahead_ns,static_cast<std::int64_t>(v.deadline_hit),v.storage_ns,
                v.host_copy_ns,v.h2d_ns,v.fill_latency_ns,v.steady_state_interval_ns,v.drain_latency_ns,
                v.gpu_waiting_for_data_ns,v.h2d_waiting_for_source_ns,v.pinned_slot_wait_ns,
                v.pageable_slot_wait_ns,v.nvme_queue_starved_ns}) Check(handle_,sqlite3_bind_int64(s,i++,x));
    Text(handle_,s,i++,v.slot_ids_json); Check(handle_,sqlite3_bind_int64(s,i++,v.verified?1:0));
    Text(handle_,s,i++,v.status); OptionalInt(handle_,s,i++,v.native_error); OptionalText(handle_,s,i++,v.message);
    Check(handle_,sqlite3_step(s),SQLITE_DONE);
}

void Database::InsertPipelineStageSample(const PipelineStageSampleInput& v) {
    Statement q(handle_, "INSERT INTO pipeline_stage_samples(pipeline_benchmark_id,sample_index,block_id,"
        "slot_id,stage_type,bytes,file_offset,host_submit_ns,host_start_ns,host_finish_ns,device_duration_ns,"
        "wait_ns,status,native_error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?);"); auto* s=q.get(); int i=1;
    for(auto x:{v.benchmark_id,v.sample_index,v.block_id,v.slot_id}) Check(handle_,sqlite3_bind_int64(s,i++,x));
    Text(handle_,s,i++,v.stage_type);
    for(auto x:{v.bytes,v.file_offset,v.host_submit_ns,v.host_start_ns,v.host_finish_ns,v.device_duration_ns,v.wait_ns})
        Check(handle_,sqlite3_bind_int64(s,i++,x));
    Text(handle_,s,i++,v.status); OptionalInt(handle_,s,i++,v.native_error); Check(handle_,sqlite3_step(s),SQLITE_DONE);
}

void Database::InsertPipelineDeadlineProfile(const PipelineDeadlineProfileInput& v) {
    Statement q(handle_, "INSERT INTO pipeline_deadline_profiles(pipeline_benchmark_id,deadline_ns,hits,"
        "misses,success_rate,ready_ahead_statistics_json,lateness_statistics_json,p999_supported)"
        " VALUES(?,?,?,?,?,?,?,?);"); auto* s=q.get(); int i=1;
    for(auto x:{v.benchmark_id,v.deadline_ns,v.hits,v.misses}) Check(handle_,sqlite3_bind_int64(s,i++,x));
    Check(handle_,sqlite3_bind_double(s,i++,v.success_rate)); Text(handle_,s,i++,v.ready_ahead_statistics_json);
    Text(handle_,s,i++,v.lateness_statistics_json); Check(handle_,sqlite3_bind_int64(s,i++,v.p999_supported?1:0));
    Check(handle_,sqlite3_step(s),SQLITE_DONE);
}

void Database::InsertPipelineSlotProfile(const PipelineSlotProfileInput& v) {
    Statement q(handle_, "INSERT INTO pipeline_slot_profiles(pipeline_benchmark_id,machine_hash,session_id,"
        "arena_allocation_id,slot_id,arena_offset,bytes,transfer_count,verification_failures,deadline_hits,"
        "deadline_misses,host_copy_statistics_json,h2d_statistics_json,observation)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?);"); auto* s=q.get(); int i=1;
    Check(handle_,sqlite3_bind_int64(s,i++,v.benchmark_id)); Text(handle_,s,i++,v.machine_hash);
    Check(handle_,sqlite3_bind_int64(s,i++,v.session_id)); Text(handle_,s,i++,v.arena_allocation_id);
    for(auto x:{v.slot_id,v.arena_offset,v.bytes,v.transfer_count,v.verification_failures,v.deadline_hits,v.deadline_misses})
        Check(handle_,sqlite3_bind_int64(s,i++,x));
    Text(handle_,s,i++,v.host_copy_statistics_json); Text(handle_,s,i++,v.h2d_statistics_json);
    Text(handle_,s,i++,v.observation); Check(handle_,sqlite3_step(s),SQLITE_DONE);
}

void Database::InsertPipelineHealthObservation(const PipelineHealthObservationInput& v) {
    Statement q(handle_, "INSERT INTO pipeline_health_observations(pipeline_benchmark_id,phase,"
        "nvme_health_snapshot_id,gpu_telemetry_json,host_memory_json,process_cpu_json,status)"
        " VALUES(?,?,?,?,?,?,?);"); auto* s=q.get(); int i=1;
    Check(handle_,sqlite3_bind_int64(s,i++,v.benchmark_id)); Text(handle_,s,i++,v.phase);
    OptionalInt(handle_,s,i++,v.nvme_health_snapshot_id); Text(handle_,s,i++,v.gpu_telemetry_json);
    Text(handle_,s,i++,v.host_memory_json); Text(handle_,s,i++,v.process_cpu_json);
    Text(handle_,s,i++,v.status); Check(handle_,sqlite3_step(s),SQLITE_DONE);
}

PipelinePhysicsCounts Database::PipelineCounts() const {
    PipelinePhysicsCounts r;
    r.host_copy_configurations=QueryInt64("SELECT COUNT(*) FROM host_copy_configurations;");
    r.host_copy_benchmarks=QueryInt64("SELECT COUNT(*) FROM host_copy_benchmarks;");
    r.host_copy_samples=QueryInt64("SELECT COUNT(*) FROM host_copy_samples;");
    r.pipeline_configurations=QueryInt64("SELECT COUNT(*) FROM pipeline_configurations;");
    r.pipeline_benchmarks=QueryInt64("SELECT COUNT(*) FROM pipeline_benchmarks;");
    r.pipeline_samples=QueryInt64("SELECT COUNT(*) FROM pipeline_samples;");
    r.pipeline_stage_samples=QueryInt64("SELECT COUNT(*) FROM pipeline_stage_samples;");
    r.deadline_profiles=QueryInt64("SELECT COUNT(*) FROM pipeline_deadline_profiles;");
    r.slot_profiles=QueryInt64("SELECT COUNT(*) FROM pipeline_slot_profiles;");
    r.health_observations=QueryInt64("SELECT COUNT(*) FROM pipeline_health_observations;");
    return r;
}

std::vector<PipelineBenchmarkSummary> Database::LatestPipelineBenchmarks() const {
    Statement q(handle_, "SELECT c.session_id,c.pipeline_type,c.aggregate_bytes,c.chunk_bytes,c.buffer_depth,"
        "c.compute_type,c.compute_window_us,c.contention_test,c.phase,b.status,"
        "(SELECT COUNT(*) FROM pipeline_samples s WHERE s.pipeline_benchmark_id=b.pipeline_benchmark_id),"
        "COALESCE(json_extract(b.statistics_json,'$.end_to_end_ns.p50'),0),"
        "COALESCE(json_extract(b.statistics_json,'$.end_to_end_ns.p95'),0),"
        "COALESCE(json_extract(b.statistics_json,'$.end_to_end_ns.p99'),0),"
        "COALESCE((SELECT success_rate FROM pipeline_deadline_profiles d WHERE d.pipeline_benchmark_id="
        "b.pipeline_benchmark_id AND d.deadline_ns=64000000),0),"
        "COALESCE(json_extract(b.statistics_json,'$.compute_path_added_percent.p99'),0) "
        "FROM pipeline_benchmarks b JOIN pipeline_configurations c USING(pipeline_configuration_id) "
        "ORDER BY b.pipeline_benchmark_id DESC LIMIT 200;");
    std::vector<PipelineBenchmarkSummary> result;
    while (sqlite3_step(q.get()) == SQLITE_ROW) {
        PipelineBenchmarkSummary v;
        v.session_id=sqlite3_column_int64(q.get(),0); v.pipeline_type=ColumnText(q.get(),1);
        v.aggregate_bytes=sqlite3_column_int64(q.get(),2); v.chunk_bytes=sqlite3_column_int64(q.get(),3);
        v.buffer_depth=sqlite3_column_int64(q.get(),4); v.compute_type=ColumnText(q.get(),5);
        v.compute_window_us=sqlite3_column_double(q.get(),6); v.contention_test=ColumnText(q.get(),7);
        v.phase=ColumnText(q.get(),8); v.status=ColumnText(q.get(),9); v.sample_count=sqlite3_column_int64(q.get(),10);
        v.p50_ns=sqlite3_column_double(q.get(),11); v.p95_ns=sqlite3_column_double(q.get(),12);
        v.p99_ns=sqlite3_column_double(q.get(),13); v.deadline_success_64ms=sqlite3_column_double(q.get(),14);
        v.p99_compute_added_percent=sqlite3_column_double(q.get(),15); result.push_back(std::move(v));
    }
    return result;
}

}  // namespace sidecar::database
