#include "sidecar/database/database.hpp"
#include "sidecar/pipeline/physics.hpp"
#include "sidecar/storage/physics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
using namespace sidecar::pipeline;

void Expect(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::filesystem::path TempDb(){return std::filesystem::temp_directory_path()/
    ("sidecar-wu8-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".db");}

sidecar::storage::StorageTarget Target(){
    sidecar::storage::StorageTarget t;t.machine_hash="pipeline-machine";t.persistent_id="nvme-test";
    t.alignment.file_offset_alignment_bytes=4096;t.alignment.buffer_alignment_bytes=4096;return t;
}

void TestStateMachineAndOwnership(){
    SlotStateMachine slot(3);Expect(slot.state()==SlotState::Free&&slot.slotId()==3,"initial slot state");
    slot.Transition(SlotState::NvmeReading);Expect(slot.generation()==1,"generation did not advance");
    slot.Transition(SlotState::HostCopyPending);slot.Transition(SlotState::HostCopyActive);
    slot.Transition(SlotState::H2DReady);slot.Transition(SlotState::H2DActive);
    slot.Transition(SlotState::GpuReady);slot.Transition(SlotState::VerifyPending);slot.Transition(SlotState::Free);
    bool rejected=false;try{slot.Transition(SlotState::H2DActive);}catch(const std::logic_error&){rejected=true;}
    Expect(rejected,"slot ownership permitted FREE -> H2D_ACTIVE");
    slot.Transition(SlotState::NvmeReading);slot.ResetAfterCancellation();
    Expect(slot.state()==SlotState::Free,"cancellation did not release slot");
}

void TestRotationDeadlineAndMetrics(){
    std::vector<SlotStateMachine> slots;for(std::uint32_t i=0;i<3;++i)slots.emplace_back(i);
    for(std::uint32_t block=0;block<9;++block){auto& s=slots[block%slots.size()];
        s.Transition(SlotState::NvmeReading);s.Transition(SlotState::H2DReady);
        s.Transition(SlotState::H2DActive);s.Transition(SlotState::GpuReady);
        s.Transition(SlotState::VerifyPending);s.Transition(SlotState::Free);}
    for(const auto&s:slots)Expect(s.generation()==3,"round-robin rotation failed");
    Expect(CalculateReadyAhead(32'000'000,20'000'000)==12'000'000,"ready-ahead positive math");
    Expect(CalculateReadyAhead(16'000'000,20'000'000)==-4'000'000,"ready-ahead negative math");
    const auto m=CalculatePipelineMetrics(100,80,105,88,110);
    Expect(std::abs(m.compute_path_added_percent-10.0)<1e-9,"compute path metric");
    Expect(std::abs(m.pipeline_slowdown-0.1)<1e-9,"pipeline slowdown metric");
    Expect(std::abs(m.pipeline_overlap_raw-0.875)<1e-9,"pipeline overlap metric");
    Expect(!m.fit_5_percent,"fit band classification");
    Expect(CalculateCommonTimelineMakespan(64'000'000,25'000,41'000'000)==64'000'000,
           "common makespan included post-timing verification work");
    Expect(CalculateCommonTimelineMakespan(32'000'000,25'000,41'000'000)==41'025'000,
           "common makespan lost the pipeline branch origin");
}

void TestDeadlinesAndStatistics(){
    std::vector<PipelineSample> samples(1000);
    for(std::size_t i=0;i<samples.size();++i){samples[i].pc_ns=i==999?70'000'000:30'000'000;
        samples[i].status=PipelineStatus::Success;}
    const auto profiles=EvaluatePipelineDeadlines(samples,{32'000'000,64'000'000});
    Expect(profiles[0].hits==999&&profiles[0].misses==1,"deadline hit/miss counts");
    Expect(profiles[0].p999_success_claim.has_value(),"99.9 support threshold");
    Expect(profiles[1].lateness_ns.maximum==6'000'000,"lateness distribution");
    auto values=std::vector<double>(1000,1.0);values.back()=7.0;
    const auto distribution=sidecar::trace::CalculateDistribution(values);
    Expect(distribution.count==1000&&distribution.p999.has_value()&&
               *distribution.p999==1.0,
           "observed P99.9 requires and uses 1000 samples");
    HostCopyResult copy;copy.configuration.bytes=1024;copy.samples={{0,1000,200,0,1.024e9,PipelineStatus::Success,{}}};
    FinalizeHostCopy(copy);Expect(copy.wall_ns.p50==1000&&copy.bytes_per_second.p50==1.024e9,"host-copy metrics");
}

void TestPlanSafetyRefinementAndIdentity(){
    auto t=Target();sidecar::storage::DatasetIdentity d;d.identity_hash=sidecar::storage::DatasetIdentityHash(sidecar::storage::kDefaultDatasetBytes);
    sidecar::memory::MemorySnapshot host;host.available_physical_bytes=96ULL<<30U;
    sidecar::cuda::DeviceMemoryInfo device;device.cuda_supported=true;device.free_bytes=20ULL<<30U;
    const auto plan=BuildDefaultPlan(t,d,host,device,3);Expect(plan.entries.size()==240,"coarse plan cardinality");
    Expect(plan.estimated_total_read_bytes>0,"coarse plan read estimate");
    const auto id1=PipelineConfigurationIdentity(plan.entries[0].configuration);
    const auto id2=PipelineConfigurationIdentity(plan.entries[0].configuration);
    Expect(id1==id2&&id1.size()==64,"configuration identity stability");
    auto constrained=host;constrained.available_physical_bytes=64ULL<<20U;
    const auto unsafe=BuildDefaultPlan(t,d,constrained,device,1);
    Expect(std::any_of(unsafe.entries.begin(),unsafe.entries.end(),[](const auto&e){return e.disposition==PipelineStatus::SkippedSafetyLimit;}),"memory safety planning");
    PipelineResult cell;cell.status=PipelineStatus::Success;cell.configuration=plan.entries[0].configuration;
    cell.samples.resize(100);for(auto&s:cell.samples){s.status=PipelineStatus::Success;s.pc_ns=31'000'000;}
    cell.samples.back().pc_ns=33'000'000;FinalizePipeline(cell);
    Expect(!SelectRefinementConfigurations({cell}).empty(),"deadline boundary refinement");
    Expect(d.identity_hash=="da0f22a709d6d4f6d2424fac8335a2a65d0e9d358190b40b695a8ceba1774c97","WU7 dataset identity");
}

class MockProvider final:public IPipelineProvider{
public:PipelineStatus next{PipelineStatus::Success};bool SupportsCuda()const noexcept override{return true;}
    HostCopyResult RunHostCopy(const HostCopyConfiguration& c)override{HostCopyResult r;r.configuration=c;r.status=next;return r;}
    PipelineResult Run(const sidecar::storage::StorageTarget&,const PipelineConfiguration& c)override{
        PipelineResult r;r.configuration=c;r.status=next;PipelineSample s;s.status=next;
        s.verified=next==PipelineStatus::Success;r.samples.push_back(s);return r;}
};

void TestMockFailuresAndJson(){
    MockProvider p;for(const auto status:{PipelineStatus::StorageError,PipelineStatus::H2DError,
        PipelineStatus::HostCopyError,PipelineStatus::Cancelled,PipelineStatus::DataVerificationFailure}){
        p.next=status;Expect(p.Run(Target(),{}).status==status,"mock error propagation");}
    PipelineResult r;r.configuration.aggregate_bytes=64ULL<<20U;r.configuration.chunk_bytes=32ULL<<20U;
    r.status=PipelineStatus::Success;PipelineSample s;s.status=PipelineStatus::Success;s.verified=true;r.samples.push_back(s);
    FinalizePipeline(r);const auto first=PipelineResultToJson(r);const auto second=PipelineResultToJson(r);
    Expect(first==second&&first.find("\"ready_ahead_ns\"")!=std::string::npos&&
               first.find("\"p999\":null")!=std::string::npos&&
               first.find("\"count\":1")!=std::string::npos,
           "stable pipeline JSON with explicit statistical support");
}

void TestMigrationAndPersistence(){
    const auto path=TempDb();struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code e;std::filesystem::remove(p,e);}}cleanup{path};
    auto db=sidecar::database::Database::Open(path,sidecar::database::OpenMode::CreateOrOpen);db.Initialize();
    Expect(db.Status().schema_version==sidecar::database::kCurrentSchemaVersion&&
           db.Status().latest_migration==sidecar::database::kCurrentSchemaVersion,
           "current migration status");
    sidecar::database::HardwareProfileInput hw;hw.machine_hash="pipeline-machine";hw.host_name="test";
    hw.os_name="Windows";hw.os_version="11";hw.os_build="x";hw.cpu_model="cpu";db.UpsertHardwareProfile(hw);
    sidecar::database::StorageDeviceInput sd;sd.persistent_id="nvme-test";sd.model="NVMe";
    db.RefreshHardwareInventory(hw,{}, {sd},false,true);
    sidecar::database::BenchmarkSessionInput session_input;session_input.machine_hash="pipeline-machine";
    session_input.sidecar_spec_version="M0-FROZEN-1";session_input.sidecar_git_commit="commit";
    session_input.trace_mode="PIPELINE_PHYSICS_WU8";session_input.notes="test";
    const auto session=db.StartBenchmarkSession(session_input);
    sidecar::database::StorageDatasetInput ds;ds.machine_hash="pipeline-machine";ds.persistent_id="nvme-test";ds.generator="g";
    ds.identity_hash="dataset";ds.file_path="dataset.bin";ds.volume_name="v";ds.size_bytes=4096;ds.verified=true;
    const auto dataset=db.UpsertStorageDataset(ds);
    sidecar::database::HostCopyConfigurationInput hc;hc.session_id=session;hc.configuration_hash="hc";hc.bytes=4096;hc.worker_count=1;hc.repetitions=1;hc.phase="HOST_COPY";hc.status="SUCCESS";
    const auto hcid=db.InsertHostCopyConfiguration(hc);const auto hbid=db.InsertHostCopyBenchmark({hcid,"SUCCESS","{}","UNCONTROLLED",std::nullopt});
    db.InsertHostCopySample({hbid,0,100,100,0,1e9,"SUCCESS",std::nullopt});
    sidecar::database::PipelineConfigurationInput pc;pc.session_id=session;pc.storage_dataset_id=dataset;pc.configuration_hash="pc";
    pc.pipeline_type="NVME_PINNED_H2D";pc.aggregate_bytes=4096;pc.chunk_bytes=4096;pc.chunk_count=1;pc.buffer_depth=1;
    pc.compute_type="SYNTHETIC_ALU";pc.compute_window_us=1000;pc.deadline_ns=1'000'000;pc.measured_repetitions=1;
    pc.phase="VALIDATE";pc.contention_test="J_PIPELINE_B_COMPUTE";pc.host_copy_workers=1;pc.arena_allocation_id="arena";pc.status="SUCCESS";
    const auto pcid=db.InsertPipelineConfiguration(pc);const auto pbid=db.InsertPipelineBenchmark({pcid,std::nullopt,std::nullopt,"SUCCESS","{}","{}","{}","{}","{}",std::nullopt});
    sidecar::database::PipelineSampleInput ps;ps.benchmark_id=pbid;ps.deadline_ns=1'000'000;ps.slot_ids_json="[0]";ps.verified=true;ps.status="SUCCESS";db.InsertPipelineSample(ps);
    db.InsertPipelineStageSample({pbid,0,0,0,"NVME",4096,0,0,0,1,0,0,"SUCCESS",std::nullopt});
    db.InsertPipelineDeadlineProfile({pbid,1'000'000,1,0,1.0,"{}","{}",false});
    db.InsertPipelineSlotProfile({pbid,session,"pipeline-machine","arena",0,0,4096,1,0,1,0,"{}","{}","NO_REGION_VARIATION_OBSERVED"});
    db.InsertPipelineHealthObservation({pbid,"AFTER",std::nullopt,"{}","{}","{}","SUCCESS"});
    db.CompleteBenchmarkSession(session,"COMPLETE");const auto counts=db.PipelineCounts();
    Expect(counts.host_copy_samples==1&&counts.pipeline_samples==1&&counts.pipeline_stage_samples==1&&counts.slot_profiles==1,"WU8 persistence counts");
    Expect(db.LatestPipelineBenchmarks().size()==1,"WU8 report query");
    Expect(db.Status().foreign_key_violations==0,"WU8 foreign keys");
}
}

int main(){
    const std::pair<const char*,void(*)()> tests[]={{"state_machine_ownership",TestStateMachineAndOwnership},
        {"rotation_deadline_metrics",TestRotationDeadlineAndMetrics},{"deadline_statistics",TestDeadlinesAndStatistics},
        {"plan_safety_refinement_identity",TestPlanSafetyRefinementAndIdentity},{"mock_failures_json",TestMockFailuresAndJson},
        {"migration_persistence",TestMigrationAndPersistence}};
    int failures=0;for(const auto&[name,test]:tests)try{test();std::cout<<"[PASS] "<<name<<'\n';}catch(const std::exception&e){++failures;std::cerr<<"[FAIL] "<<name<<": "<<e.what()<<'\n';}
    return failures?1:0;
}
