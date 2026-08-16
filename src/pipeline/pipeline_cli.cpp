#include "sidecar/pipeline/cli.hpp"

#include "sidecar/core/sha256.hpp"
#include "sidecar/database/database.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/pipeline/physics.hpp"
#include "sidecar/version.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace sidecar::pipeline {
namespace {

struct Options {
    std::string command;
    bool json{false}, persist{true}, dry_run{false}, compute_specified{false},
        phase_specified{false};
    std::string refinement_reason;
    std::filesystem::path database{std::filesystem::path("data") / "sidecar.db"};
    std::optional<std::string> device;
    std::optional<std::filesystem::path> dataset;
    PipelineType path{PipelineType::NvmePinnedH2D};
    std::uint64_t aggregate{64ULL << 20U}, chunk{64ULL << 20U};
    std::uint32_t depth{2}, repetitions{10}, workers{1}, stream_seconds{30};
    cuda::ComputeWorkload compute{cuda::ComputeWorkload::SyntheticAlu};
    double compute_window_us{32'000};
    std::uint64_t deadline_ns{32'000'000}, timeout_ms{120'000};
    PipelinePhase phase{PipelinePhase::Coarse};
    std::optional<ContentionTest> test;
    int cuda_device{0};
};

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::uint64_t ParseSize(std::string value) {
    const auto lowered = Lower(value);
    std::uint64_t multiplier = 1;
    std::string number = lowered;
    for (const auto& [suffix, scale] :
         std::initializer_list<std::pair<std::string_view, std::uint64_t>>{
             {"gib",1ULL<<30U},{"gb",1'000'000'000ULL},{"g",1ULL<<30U},
             {"mib",1ULL<<20U},{"mb",1'000'000ULL},{"m",1ULL<<20U},
             {"kib",1ULL<<10U},{"kb",1'000ULL},{"k",1ULL<<10U}}) {
        if (lowered.ends_with(suffix)) {
            multiplier=scale; number.resize(number.size()-suffix.size()); break;
        }
    }
    std::size_t consumed=0; const auto base=std::stoull(number,&consumed);
    if(consumed!=number.size() || base>UINT64_MAX/multiplier)
        throw std::invalid_argument("invalid byte size: "+value);
    return base*multiplier;
}

double ParseDurationUs(std::string value) {
    value=Lower(value); double multiplier=1000.0;
    if(value.ends_with("us")){multiplier=1; value.resize(value.size()-2);}
    else if(value.ends_with("ms")){multiplier=1000; value.resize(value.size()-2);}
    else if(value.ends_with("s")){multiplier=1'000'000; value.resize(value.size()-1);}
    return std::stod(value)*multiplier;
}

PipelineType ParsePath(const std::string& value) {
    const auto v=Lower(value);
    if(v=="pageable"||v=="a"||v=="normal") return PipelineType::NvmePageablePinnedH2D;
    if(v=="pinned"||v=="b"||v=="urgent") return PipelineType::NvmePinnedH2D;
    throw std::invalid_argument("unknown pipeline path: "+value);
}
cuda::ComputeWorkload ParseCompute(const std::string& value) {
    const auto v=Lower(value);
    if(v=="alu"||v=="synthetic_alu") return cuda::ComputeWorkload::SyntheticAlu;
    if(v=="memory"||v=="memory_bound") return cuda::ComputeWorkload::MemoryBound;
    if(v=="gemm"||v=="fp16_gemm") return cuda::ComputeWorkload::Fp16Gemm;
    throw std::invalid_argument("unknown compute workload: "+value);
}
PipelinePhase ParsePhase(const std::string& value) {
    const auto v=Lower(value);
    if(v=="validate") return PipelinePhase::Validate;
    if(v=="host-copy") return PipelinePhase::HostCopy;
    if(v=="baseline") return PipelinePhase::Baseline;
    if(v=="coarse") return PipelinePhase::Coarse;
    if(v=="refine") return PipelinePhase::Refine;
    if(v=="repeat") return PipelinePhase::Repeat;
    if(v=="stream") return PipelinePhase::Stream;
    throw std::invalid_argument("unknown pipeline phase: "+value);
}
ContentionTest ParseTest(std::string value) {
    value=Lower(value); if(value.size()>1) value=value.substr(0,1);
    if(value.empty()||value[0]<'a'||value[0]>'j') throw std::invalid_argument("test must be A..J");
    return static_cast<ContentionTest>(value[0]-'a');
}

Options Parse(int argc,char** argv) {
    if(argc<3) throw std::invalid_argument("pipeline subcommand required");
    Options o; o.command=Lower(argv[2]);
    for(int i=3;i<argc;++i){
        const std::string a(argv[i]);
        auto next=[&](){if(++i>=argc) throw std::invalid_argument(a+" requires a value"); return std::string(argv[i]);};
        if(a=="--json") o.json=true;
        else if(a=="--no-persist") o.persist=false;
        else if(a=="--dry-run") o.dry_run=true;
        else if(a=="--database") o.database=next();
        else if(a=="--device") o.device=next();
        else if(a=="--dataset") o.dataset=std::filesystem::path(next());
        else if(a=="--path") o.path=ParsePath(next());
        else if(a=="--aggregate-size"||a=="--size") o.aggregate=ParseSize(next());
        else if(a=="--chunk-size") o.chunk=ParseSize(next());
        else if(a=="--depth") o.depth=static_cast<std::uint32_t>(std::stoul(next()));
        else if(a=="--compute"){o.compute=ParseCompute(next());o.compute_specified=true;}
        else if(a=="--compute-window") o.compute_window_us=ParseDurationUs(next());
        else if(a=="--deadline") o.deadline_ns=static_cast<std::uint64_t>(ParseDurationUs(next())*1000.0);
        else if(a=="--repetitions") o.repetitions=static_cast<std::uint32_t>(std::stoul(next()));
        else if(a=="--workers") o.workers=static_cast<std::uint32_t>(std::stoul(next()));
        else if(a=="--seconds") o.stream_seconds=static_cast<std::uint32_t>(std::stoul(next()));
        else if(a=="--timeout-ms") o.timeout_ms=std::stoull(next());
        else if(a=="--phase"){o.phase=ParsePhase(next());o.phase_specified=true;}
        else if(a=="--refinement-reason") o.refinement_reason=next();
        else if(a=="--test") o.test=ParseTest(next());
        else if(a=="--cuda-device") o.cuda_device=std::stoi(next());
        else throw std::invalid_argument("unknown pipeline option: "+a);
    }
    return o;
}

PipelineConfiguration Configuration(const Options& o) {
    PipelineConfiguration c;
    c.pipeline_type=o.path; c.aggregate_bytes=o.aggregate; c.chunk_bytes=o.chunk;
    c.buffer_depth=o.depth; c.compute_workload=o.compute; c.compute_window_us=o.compute_window_us;
    c.deadline_ns=o.deadline_ns; c.repetitions=o.repetitions; c.timeout_ms=o.timeout_ms;
    c.phase=o.phase; c.contention_test=o.test.value_or(
        o.path==PipelineType::NvmePageablePinnedH2D?ContentionTest::I:ContentionTest::J);
    c.host_copy_workers=o.workers; c.device_index=o.cuda_device;
    c.refinement_reason=o.refinement_reason;
    if(c.refinement_reason.empty() && c.phase==PipelinePhase::Refine)
        c.refinement_reason="manual evidence-selected boundary refinement";
    if(c.refinement_reason.empty() && c.phase==PipelinePhase::Repeat)
        c.refinement_reason="independent repeatability pass";
    return c;
}

hardware::DiscoveryReport Discover(database::Database* db) {
    auto p=hardware::CreateNativeDiscoveryProvider(); hardware::HardwareDiscoveryService s(*p);
    auto r=s.Discover(); if(db) hardware::PersistDiscovery(*db,r); return r;
}

std::optional<database::Database> OpenDatabase(const Options& o) {
    if(!o.persist) return std::nullopt;
    if(o.database.has_parent_path()) std::filesystem::create_directories(o.database.parent_path());
    auto db=database::Database::Open(o.database,database::OpenMode::CreateOrOpen); db.Initialize(); return db;
}

std::int64_t StartSession(database::Database& db,const hardware::DiscoveryReport& machine,
                          std::string notes) {
    const auto version=CurrentVersionInfo();
    database::BenchmarkSessionInput in;
    in.machine_hash=machine.identity.machine_hash; in.sidecar_spec_version=std::string(version.spec_version);
    in.sidecar_git_commit=std::string(version.git_commit); in.trace_mode="PIPELINE_PHYSICS_WU8";
    in.notes=std::move(notes); in.cuda_runtime_version=machine.snapshot.cuda_runtime_version;
    in.cuda_driver_version=machine.snapshot.cuda_driver_version;
    if(!machine.snapshot.nvidia_driver_version.empty()) in.nvidia_driver_version=machine.snapshot.nvidia_driver_version;
    in.os_version=machine.snapshot.operating_system.version; return db.StartBenchmarkSession(in);
}

database::StorageDatasetInput DatasetInput(const storage::StorageTarget& t,
                                           const storage::DatasetIdentity& d) {
    return {t.machine_hash,t.persistent_id,d.generator,d.identity_hash,d.path.generic_string(),t.volume_name,
        d.format_version,static_cast<std::int64_t>(d.seed),static_cast<std::int64_t>(d.size_bytes),
        t.physical_disk_number,d.verified,d.full_verification,static_cast<std::int64_t>(d.verified_bytes)};
}

template<class T> std::optional<std::int64_t> DbInt(const std::optional<T>& v) {
    return v?std::optional<std::int64_t>(static_cast<std::int64_t>(*v)):std::nullopt;
}
database::StorageHealthSnapshotInput HealthInput(const storage::StorageTarget& t,
    const storage::HealthSnapshot& h,std::int64_t session) {
    database::StorageHealthSnapshotInput v; v.machine_hash=t.machine_hash; v.persistent_id=t.persistent_id;
    v.session_id=session; v.phase=h.phase; v.provider=h.provider; v.status=h.status;
    v.critical_warning=DbInt(h.critical_warning); v.temperature_c=h.temperature_c;
    v.available_spare_percent=DbInt(h.available_spare_percent); v.available_spare_threshold_percent=DbInt(h.available_spare_threshold_percent);
    v.percentage_used=DbInt(h.percentage_used); v.data_units_read_low64=DbInt(h.data_units_read_low64);
    v.data_units_written_low64=DbInt(h.data_units_written_low64); v.host_read_commands_low64=DbInt(h.host_read_commands_low64);
    v.host_write_commands_low64=DbInt(h.host_write_commands_low64); v.controller_busy_minutes_low64=DbInt(h.controller_busy_minutes_low64);
    v.power_cycles_low64=DbInt(h.power_cycles_low64); v.power_on_hours_low64=DbInt(h.power_on_hours_low64);
    v.unsafe_shutdowns_low64=DbInt(h.unsafe_shutdowns_low64); v.media_data_errors_low64=DbInt(h.media_data_errors_low64);
    v.error_log_entries_low64=DbInt(h.error_log_entries_low64); v.raw_evidence_json=h.raw_evidence_json;
    if(!h.message.empty()) v.message=h.message; return v;
}

std::string DistributionJson(const trace::Distribution& d) {
    std::ostringstream o; o<<std::setprecision(17)<<"{\"count\":"<<d.count<<",\"max\":"<<d.maximum<<",\"mean\":"<<d.mean
        <<",\"median\":"<<d.median<<",\"min\":"<<d.minimum<<",\"p50\":"<<d.p50
        <<",\"p90\":"<<d.p90<<",\"p95\":"<<d.p95<<",\"p99\":"<<d.p99
        <<",\"p999\":";if(d.p999)o<<*d.p999;else o<<"null";
    o<<",\"stddev\":"<<d.stddev<<'}'; return o.str();
}
std::string TelemetryJson(const cuda::TransferTelemetry& t) {
    std::ostringstream o; o<<"{\"available\":"<<(t.available?"true":"false");
    auto u=[&](const char* n,const auto& v){o<<",\""<<n<<"\":";if(v)o<<*v;else o<<"null";};
    u("temperature_c",t.temperature_c);u("graphics_clock_mhz",t.graphics_clock_mhz);
    u("memory_clock_mhz",t.memory_clock_mhz);u("power_watts",t.power_watts);
    u("power_limit_watts",t.power_limit_watts);u("pcie_generation",t.pcie_generation);u("pcie_width",t.pcie_width);
    o<<'}'; return o.str();
}
std::string MemoryJson(const memory::MemorySnapshot& m) {
    std::ostringstream o;o<<'{';bool first=true;
    auto u=[&](const char* n,const auto& v){if(!first)o<<',';first=false;o<<'"'<<n<<"\":";if(v)o<<*v;else o<<"null";};
    u("available_physical_bytes",m.available_physical_bytes);u("visible_physical_bytes",m.visible_physical_bytes);
    u("process_working_set_bytes",m.process_working_set_bytes);u("process_private_bytes",m.process_private_bytes);
    u("memory_load_percent",m.memory_load_percent);o<<'}';return o.str();
}
std::string SlotsJson(const std::vector<std::uint32_t>& values){std::ostringstream o;o<<'[';for(std::size_t i=0;i<values.size();++i){if(i)o<<',';o<<values[i];}o<<']';return o.str();}

void PersistHostCopy(database::Database& db,std::int64_t session,const HostCopyResult& r) {
    std::ostringstream material; material<<r.configuration.bytes<<'|'<<r.configuration.worker_count<<'|'
        <<r.configuration.compute_window_us<<'|'<<(r.configuration.concurrent_compute?cuda::ToString(*r.configuration.concurrent_compute):"NONE")
        <<'|'<<ToString(r.configuration.phase);
    database::HostCopyConfigurationInput c; c.session_id=session;c.configuration_hash=core::Sha256Hex(material.str());
    c.bytes=r.configuration.bytes;c.worker_count=r.configuration.worker_count;c.warmups=r.configuration.warmups;
    c.repetitions=r.configuration.repetitions;if(r.configuration.concurrent_compute)c.concurrent_compute=cuda::ToString(*r.configuration.concurrent_compute);
    c.compute_window_us=r.configuration.compute_window_us;c.phase=ToString(r.configuration.phase);c.status=ToString(r.status);
    db.BeginWriteTransaction();try{
        const auto cid=db.InsertHostCopyConfiguration(c); database::HostCopyBenchmarkInput b;
        b.configuration_id=cid;b.status=ToString(r.status);b.statistics_json=HostCopyToJson(r);b.affinity=r.affinity;if(!r.message.empty())b.message=r.message;
        const auto bid=db.InsertHostCopyBenchmark(b);for(const auto&s:r.samples)db.InsertHostCopySample({bid,s.sample_index,
            static_cast<std::int64_t>(s.wall_ns),static_cast<std::int64_t>(s.process_cpu_ns),static_cast<std::int64_t>(s.compute_ns),
            s.bytes_per_second,ToString(s.status),s.message.empty()?std::nullopt:std::optional<std::string>(s.message)});
        db.CommitWriteTransaction();}catch(...){db.RollbackWriteTransaction();throw;}
}

void PersistPipeline(database::Database& db,std::int64_t session,std::int64_t dataset_id,
                     const std::string& machine_hash,const storage::StorageTarget& target,
                     const PipelineResult& r) {
    const auto before=db.InsertStorageHealthSnapshot(HealthInput(target,r.health_before,session));
    const auto after=db.InsertStorageHealthSnapshot(HealthInput(target,r.health_after,session));
    const auto& x=r.configuration; database::PipelineConfigurationInput c;
    c.session_id=session;c.storage_dataset_id=dataset_id;c.configuration_hash=PipelineConfigurationIdentity(x);
    c.pipeline_type=ToString(x.pipeline_type);c.aggregate_bytes=x.aggregate_bytes;c.chunk_bytes=x.chunk_bytes;
    c.chunk_count=x.aggregate_bytes/x.chunk_bytes;c.buffer_depth=x.buffer_depth;c.compute_type=cuda::ToString(x.compute_workload);
    c.compute_window_us=x.compute_window_us;c.deadline_ns=x.deadline_ns;c.measured_repetitions=r.samples.size();
    c.ordering_seed=x.ordering_seed;c.timeout_ms=x.timeout_ms;c.phase=ToString(x.phase);c.contention_test=ToString(x.contention_test);
    c.host_copy_workers=x.host_copy_workers;c.device_index=x.device_index;c.arena_allocation_id=r.arena_allocation_id;
    if(!x.refinement_reason.empty())c.refinement_reason=x.refinement_reason;c.status=ToString(r.status);
    db.BeginWriteTransaction();try{
        const auto cid=db.InsertPipelineConfiguration(c);database::PipelineBenchmarkInput b;
        b.configuration_id=cid;b.health_before_id=before;b.health_after_id=after;b.status=ToString(r.status);
        b.statistics_json=PipelineResultToJson(r);b.gpu_before_json=TelemetryJson(r.gpu_before);b.gpu_after_json=TelemetryJson(r.gpu_after);
        b.host_before_json=MemoryJson(r.host_before);b.host_after_json=MemoryJson(r.host_after);if(!r.message.empty())b.message=r.message;
        const auto bid=db.InsertPipelineBenchmark(b);
        for(const auto&s:r.samples){database::PipelineSampleInput v;v.benchmark_id=bid;v.sample_index=s.sample_index;
            v.c0_reference_ns=s.c0_reference_ns;v.p0_reference_ns=s.p0_reference_ns;v.cc_ns=s.cc_ns;v.pc_ns=s.pc_ns;v.makespan_ns=s.makespan_ns;
            v.compute_path_added_ns=s.metrics.compute_path_added_ns;v.compute_path_added_percent=s.metrics.compute_path_added_percent;
            v.pipeline_slowdown=s.metrics.pipeline_slowdown;v.compute_slowdown=s.metrics.compute_slowdown;
            v.pipeline_overlap_raw=s.metrics.pipeline_overlap_raw;v.pipeline_overlap_normalized=s.metrics.pipeline_overlap_normalized;
            v.deadline_ns=x.deadline_ns;v.ready_ahead_ns=s.ready_ahead_ns;v.deadline_hit=s.deadline_hit;v.storage_ns=s.storage_ns;
            v.host_copy_ns=s.host_copy_ns;v.h2d_ns=s.h2d_ns;v.fill_latency_ns=s.fill_latency_ns;v.steady_state_interval_ns=s.steady_state_interval_ns;
            v.drain_latency_ns=s.drain_latency_ns;v.gpu_waiting_for_data_ns=s.gpu_waiting_for_data_ns;v.h2d_waiting_for_source_ns=s.h2d_waiting_for_source_ns;
            v.pinned_slot_wait_ns=s.pinned_slot_wait_ns;v.pageable_slot_wait_ns=s.pageable_slot_wait_ns;v.nvme_queue_starved_ns=s.nvme_queue_starved_ns;
            v.slot_ids_json=SlotsJson(s.slot_ids);v.verified=s.verified;v.status=ToString(s.status);if(s.native_error)v.native_error=s.native_error;if(!s.message.empty())v.message=s.message;
            db.InsertPipelineSample(v);}
        for(const auto&s:r.stages){db.InsertPipelineStageSample({bid,s.sample_index,s.block_id,s.slot_id,s.stage_type,
            static_cast<std::int64_t>(s.bytes),static_cast<std::int64_t>(s.file_offset),static_cast<std::int64_t>(s.host_submit_ns),
            static_cast<std::int64_t>(s.host_start_ns),static_cast<std::int64_t>(s.host_finish_ns),static_cast<std::int64_t>(s.device_duration_ns),
            static_cast<std::int64_t>(s.wait_ns),ToString(s.status),s.native_error?std::optional<std::int64_t>(s.native_error):std::nullopt});}
        for(const auto&d:r.deadlines)db.InsertPipelineDeadlineProfile({bid,static_cast<std::int64_t>(d.deadline_ns),
            static_cast<std::int64_t>(d.hits),static_cast<std::int64_t>(d.misses),d.success_rate,DistributionJson(d.ready_ahead_ns),
            DistributionJson(d.lateness_ns),d.p999_success_claim.has_value()});
        for(const auto&s:r.slots)db.InsertPipelineSlotProfile({bid,session,machine_hash,s.arena_allocation_id,s.slot_id,
            static_cast<std::int64_t>(s.arena_offset),static_cast<std::int64_t>(s.bytes),static_cast<std::int64_t>(s.transfer_count),
            static_cast<std::int64_t>(s.verification_failures),static_cast<std::int64_t>(s.deadline_hits),static_cast<std::int64_t>(s.deadline_misses),
            DistributionJson(s.host_copy_ns),DistributionJson(s.h2d_ns),s.observation});
        db.InsertPipelineHealthObservation({bid,"BEFORE",before,TelemetryJson(r.gpu_before),MemoryJson(r.host_before),"{}",ToString(r.status)});
        db.InsertPipelineHealthObservation({bid,"AFTER",after,TelemetryJson(r.gpu_after),MemoryJson(r.host_after),"{}",ToString(r.status)});
        db.CommitWriteTransaction();}catch(...){db.RollbackWriteTransaction();throw;}
}

int Report(const Options& o) {
    if(!std::filesystem::exists(o.database)){std::cerr<<"database does not exist: "<<o.database.string()<<'\n';return 2;}
    auto db=database::Database::Open(o.database,database::OpenMode::ExistingReadWrite);db.Initialize();
    const auto c=db.PipelineCounts();const auto rows=db.LatestPipelineBenchmarks();
    if(o.json){std::cout<<"{\"counts\":{\"deadline_profiles\":"<<c.deadline_profiles<<",\"health_observations\":"<<c.health_observations
        <<",\"host_copy_benchmarks\":"<<c.host_copy_benchmarks<<",\"host_copy_samples\":"<<c.host_copy_samples
        <<",\"pipeline_benchmarks\":"<<c.pipeline_benchmarks<<",\"pipeline_samples\":"<<c.pipeline_samples
        <<",\"pipeline_stage_samples\":"<<c.pipeline_stage_samples<<",\"slot_profiles\":"<<c.slot_profiles<<"},\"latest\":[";
        for(std::size_t i=0;i<rows.size();++i){if(i)std::cout<<',';const auto&r=rows[i];std::cout<<"{\"aggregate_bytes\":"<<r.aggregate_bytes
            <<",\"buffer_depth\":"<<r.buffer_depth<<",\"chunk_bytes\":"<<r.chunk_bytes<<",\"compute\":\""<<r.compute_type
            <<"\",\"compute_window_us\":"<<r.compute_window_us<<",\"contention_test\":\""<<r.contention_test
            <<"\",\"deadline_success_64ms\":"<<r.deadline_success_64ms<<",\"p99_compute_added_percent\":"<<r.p99_compute_added_percent
            <<",\"p99_ns\":"<<r.p99_ns<<",\"path\":\""<<r.pipeline_type<<"\",\"phase\":\""<<r.phase
            <<"\",\"samples\":"<<r.sample_count<<",\"session_id\":"<<r.session_id<<",\"status\":\""<<r.status<<"\"}";}
        std::cout<<"]}\n";}else{std::cout<<"SIDECAR WU8 PIPELINE REPORT\n\nPipeline benchmarks: "<<c.pipeline_benchmarks
            <<"\nPipeline samples: "<<c.pipeline_samples<<"\nStage samples: "<<c.pipeline_stage_samples
            <<"\nHost-copy samples: "<<c.host_copy_samples<<"\nDeadline profiles: "<<c.deadline_profiles<<"\nSlot profiles: "<<c.slot_profiles<<'\n';}
    return 0;
}

void PrintHostCopy(const HostCopyResult& r) {
    std::cout<<"SIDECAR PAGEABLE -> PINNED HOST COPY\n\nBytes: "<<r.configuration.bytes
        <<"\nWorkers: "<<r.configuration.worker_count<<"\nStatus: "<<ToString(r.status)
        <<"\nP50/P95/P99: "<<r.wall_ns.p50/1e6<<" / "<<r.wall_ns.p95/1e6<<" / "<<r.wall_ns.p99/1e6
        <<" ms\nMedian bandwidth: "<<r.bytes_per_second.median/1e9<<" GB/s\nCPU P50: "<<r.process_cpu_ns.p50/1e6<<" ms\n";
}

} // namespace

int RunPipelineCli(int argc,char** argv) {
    const auto o=Parse(argc,argv);
    if(o.command=="report") return Report(o);
    auto storage_provider=storage::CreateNativeStorageProvider();
    auto target=storage_provider->ResolveTarget(o.device,o.dataset);
    auto dataset=storage_provider->InspectDataset(target);
    if(!dataset.exists || dataset.identity_hash!=storage::DatasetIdentityHash(storage::kDefaultDatasetBytes)){
        std::cerr<<"authoritative WU7 dataset identity is unavailable or mismatched\n";return 3;}
    auto provider=CreateNativePipelineProvider();
    if(o.command=="info"){
        auto& compute=cuda::NativeOverlapProvider();const auto device=compute.InspectDevice(o.cuda_device);
        if(o.json)std::cout<<"{\"cuda_supported\":"<<(provider->SupportsCuda()?"true":"false")<<",\"dataset_identity\":\""
            <<dataset.identity_hash<<"\",\"dataset_path\":\""<<dataset.path.generic_string()<<"\",\"device\":\""<<device.name
            <<"\",\"nvme\":\""<<target.model<<"\",\"pipelines\":[\"NVME_PAGEABLE_PINNED_H2D\",\"NVME_PINNED_H2D\"]}\n";
        else std::cout<<"SIDECAR MEMORY-HIGHWAY INFORMATION\n\nNVMe: "<<target.model<<"\nDataset: "<<dataset.path.string()
            <<"\nIdentity: "<<dataset.identity_hash<<"\nGPU: "<<device.name<<"\nCUDA: "<<(provider->SupportsCuda()?"available":"disabled")<<'\n';
        return 0;
    }
    if(o.command=="plan"){
        auto& compute=cuda::NativeOverlapProvider();const auto device=compute.InspectDevice(o.cuda_device);
        auto plan=BuildDefaultPlan(target,dataset,memory::NativeHostMemoryProvider().Snapshot(),device,o.repetitions);
        if(o.json)std::cout<<PipelinePlanToJson(plan)<<'\n';else std::cout<<"SIDECAR WU8 PIPELINE DRY RUN\n\nEntries: "<<plan.entries.size()
            <<"\nEstimated bytes read: "<<plan.estimated_total_read_bytes<<"\nPageable/pinned paths: yes/yes\nDepths: 1,2,3,4\nCompute: ALU, MEMORY_BOUND, FP16_GEMM\nWindows: 32 ms, 64 ms\nTelemetry: "<<plan.telemetry_policy<<"\nHealth: "<<plan.health_safety<<'\n';
        return 0;
    }
    std::optional<database::Database> db=OpenDatabase(o);std::optional<hardware::DiscoveryReport> machine;
    std::optional<std::int64_t> session;std::int64_t dataset_id=0;
    if(db){machine=Discover(&*db);target.machine_hash=machine->identity.machine_hash;dataset_id=db->UpsertStorageDataset(DatasetInput(target,dataset));
        session=StartSession(*db,*machine,"WU8 "+o.command+" pipeline physics");}
    if(o.command=="host-copy"){
        HostCopyConfiguration c;c.bytes=o.aggregate;c.worker_count=o.workers;c.repetitions=o.repetitions;
        c.phase=o.phase_specified?o.phase:PipelinePhase::HostCopy;
        if(o.compute_specified){c.concurrent_compute=o.compute;c.compute_window_us=o.compute_window_us;}
        auto r=provider->RunHostCopy(c);if(db)PersistHostCopy(*db,*session,r);
        if(db)db->CompleteBenchmarkSession(*session,r.status==PipelineStatus::Success?"COMPLETE":"FAILED");
        if(o.json)std::cout<<HostCopyToJson(r)<<'\n';else PrintHostCopy(r);
        return (r.status==PipelineStatus::Success||r.status==PipelineStatus::SkippedUnsupported)?0:4;
    }
    std::vector<PipelineConfiguration> configurations;
    if(o.command=="matrix"){
        for(int test=0;test<10;++test){auto c=Configuration(o);c.contention_test=static_cast<ContentionTest>(test);
            c.pipeline_type=(test==8)?PipelineType::NvmePageablePinnedH2D:(test==9?PipelineType::NvmePinnedH2D:o.path);
            c.phase=PipelinePhase::Coarse;configurations.push_back(c);}
    }else{
        auto c=Configuration(o);
        if(o.command=="baseline")c.phase=PipelinePhase::Baseline;
        else if(o.command=="stream"){c.phase=PipelinePhase::Stream;c.stream_seconds=o.stream_seconds;}
        else if(o.command=="validate"){c.phase=PipelinePhase::Validate;c.aggregate_bytes=4ULL<<20U;c.chunk_bytes=2ULL<<20U;c.buffer_depth=2;c.repetitions=1;}
        else if(o.command!="run")throw std::invalid_argument("unknown pipeline command: "+o.command);
        configurations.push_back(c);
    }
    std::vector<PipelineResult> results;bool success=true;
    for(const auto& c:configurations){auto r=provider->Run(target,c);success=success&&
        (r.status==PipelineStatus::Success||r.status==PipelineStatus::SkippedUnsupported);
        if(db)PersistPipeline(*db,*session,dataset_id,machine->identity.machine_hash,target,r);results.push_back(std::move(r));}
    if(db)db->CompleteBenchmarkSession(*session,success?"COMPLETE":"FAILED");
    if(o.json){std::cout<<'[';for(std::size_t i=0;i<results.size();++i){if(i)std::cout<<',';std::cout<<PipelineResultToJson(results[i]);}std::cout<<"]\n";}
    else for(const auto&r:results)std::cout<<FormatPipelineResult(r)<<'\n';
    return success?0:5;
}

} // namespace sidecar::pipeline
