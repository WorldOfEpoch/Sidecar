#include "sidecar/pipeline/physics.hpp"

#include "sidecar/core/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace sidecar::pipeline {
namespace {

std::string Escape(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(static_cast<unsigned char>(c)) << std::dec;
                } else {
                    out << c;
                }
        }
    }
    out << '"';
    return out.str();
}

std::string DistributionJson(const trace::Distribution& value) {
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"count\":" << value.count << ",\"max\":" << value.maximum
        << ",\"mean\":" << value.mean
        << ",\"median\":" << value.median << ",\"min\":" << value.minimum
        << ",\"p50\":" << value.p50 << ",\"p90\":" << value.p90
        << ",\"p95\":" << value.p95 << ",\"p99\":" << value.p99
        << ",\"p999\":";
    if (value.p999) out << *value.p999;
    else out << "null";
    out << ",\"stddev\":" << value.stddev << '}';
    return out.str();
}

std::vector<double> Values(const std::vector<PipelineSample>& samples,
                           double (*selector)(const PipelineSample&)) {
    std::vector<double> values;
    for (const auto& sample : samples)
        if (sample.status == PipelineStatus::Success) values.push_back(selector(sample));
    return values;
}

bool IsFinitePositive(double value) { return std::isfinite(value) && value > 0; }

}  // namespace

const char* ToString(PipelineType value) noexcept {
    switch (value) {
        case PipelineType::NvmePageablePinnedH2D: return "NVME_PAGEABLE_PINNED_H2D";
        case PipelineType::NvmePinnedH2D: return "NVME_PINNED_H2D";
    }
    return "NVME_PINNED_H2D";
}

const char* ToString(SlotState value) noexcept {
    switch (value) {
        case SlotState::Free: return "FREE";
        case SlotState::NvmeReading: return "NVME_READING";
        case SlotState::HostCopyPending: return "HOST_COPY_PENDING";
        case SlotState::HostCopyActive: return "HOST_COPY_ACTIVE";
        case SlotState::H2DReady: return "H2D_READY";
        case SlotState::H2DActive: return "H2D_ACTIVE";
        case SlotState::GpuReady: return "GPU_READY";
        case SlotState::VerifyPending: return "VERIFY_PENDING";
    }
    return "FREE";
}

const char* ToString(PipelinePhase value) noexcept {
    switch (value) {
        case PipelinePhase::Validate: return "VALIDATE";
        case PipelinePhase::HostCopy: return "HOST_COPY";
        case PipelinePhase::Baseline: return "BASELINE";
        case PipelinePhase::Coarse: return "COARSE";
        case PipelinePhase::Refine: return "REFINE";
        case PipelinePhase::Repeat: return "REPEAT";
        case PipelinePhase::Stream: return "STREAM";
    }
    return "COARSE";
}

const char* ToString(ContentionTest value) noexcept {
    switch (value) {
        case ContentionTest::A: return "A_COMPUTE_ONLY";
        case ContentionTest::B: return "B_NVME_PAGEABLE_ONLY";
        case ContentionTest::C: return "C_PAGEABLE_PINNED_ONLY";
        case ContentionTest::D: return "D_H2D_ONLY";
        case ContentionTest::E: return "E_NVME_PINNED_ONLY";
        case ContentionTest::F: return "F_NVME_PAGEABLE_COPY";
        case ContentionTest::G: return "G_COPY_H2D";
        case ContentionTest::H: return "H_NVME_H2D_PINNED";
        case ContentionTest::I: return "I_PIPELINE_A_COMPUTE";
        case ContentionTest::J: return "J_PIPELINE_B_COMPUTE";
    }
    return "J_PIPELINE_B_COMPUTE";
}

const char* ToString(PipelineStatus value) noexcept {
    switch (value) {
        case PipelineStatus::Success: return "SUCCESS";
        case PipelineStatus::SkippedUnsupported: return "SKIPPED_UNSUPPORTED";
        case PipelineStatus::SkippedSafetyLimit: return "SKIPPED_SAFETY_LIMIT";
        case PipelineStatus::InvalidConfiguration: return "INVALID_CONFIGURATION";
        case PipelineStatus::StorageError: return "STORAGE_ERROR";
        case PipelineStatus::HostCopyError: return "HOST_COPY_ERROR";
        case PipelineStatus::H2DError: return "H2D_ERROR";
        case PipelineStatus::ComputeError: return "COMPUTE_ERROR";
        case PipelineStatus::TimedOut: return "TIMED_OUT";
        case PipelineStatus::Cancelled: return "CANCELLED";
        case PipelineStatus::DataVerificationFailure: return "DATA_VERIFICATION_FAILURE";
        case PipelineStatus::HardwareHealthChange: return "HARDWARE_HEALTH_CHANGE";
        case PipelineStatus::InternalError: return "INTERNAL_ERROR";
    }
    return "INTERNAL_ERROR";
}

SlotStateMachine::SlotStateMachine(std::uint32_t slot_id) noexcept : slot_id_(slot_id) {}
std::uint32_t SlotStateMachine::slotId() const noexcept { return slot_id_; }
SlotState SlotStateMachine::state() const noexcept { return state_; }
std::uint64_t SlotStateMachine::generation() const noexcept { return generation_; }

bool SlotStateMachine::CanTransition(SlotState next) const noexcept {
    switch (state_) {
        case SlotState::Free:
            return next == SlotState::NvmeReading || next == SlotState::HostCopyPending ||
                   next == SlotState::H2DReady;
        case SlotState::NvmeReading:
            return next == SlotState::HostCopyPending || next == SlotState::H2DReady ||
                   next == SlotState::VerifyPending;
        case SlotState::HostCopyPending: return next == SlotState::HostCopyActive;
        case SlotState::HostCopyActive:
            return next == SlotState::H2DReady || next == SlotState::VerifyPending;
        case SlotState::H2DReady: return next == SlotState::H2DActive;
        case SlotState::H2DActive: return next == SlotState::GpuReady;
        case SlotState::GpuReady: return next == SlotState::VerifyPending;
        case SlotState::VerifyPending: return next == SlotState::Free;
    }
    return false;
}

void SlotStateMachine::Transition(SlotState next) {
    if (!CanTransition(next))
        throw std::logic_error(std::string("invalid pipeline slot transition ") +
                               ToString(state_) + " -> " + ToString(next));
    if (state_ == SlotState::Free) ++generation_;
    state_ = next;
}

void SlotStateMachine::ResetAfterCancellation() noexcept { state_ = SlotState::Free; }

PipelineMetrics CalculatePipelineMetrics(double c0_ns, double p0_ns, double cc_ns,
                                         double pc_ns, double makespan_ns) noexcept {
    PipelineMetrics result;
    result.compute_path_delta_ns = makespan_ns - c0_ns;
    result.compute_path_added_ns = std::max(0.0, result.compute_path_delta_ns);
    if (c0_ns > 0) {
        result.compute_path_added_percent = result.compute_path_added_ns / c0_ns * 100.0;
        result.compute_slowdown = (cc_ns - c0_ns) / c0_ns;
    }
    if (p0_ns > 0) result.pipeline_slowdown = (pc_ns - p0_ns) / p0_ns;
    const double denominator = std::min(c0_ns, p0_ns);
    if (denominator > 0)
        result.pipeline_overlap_raw = (c0_ns + p0_ns - makespan_ns) / denominator;
    result.pipeline_overlap_normalized = std::clamp(result.pipeline_overlap_raw, 0.0, 1.0);
    result.fit_1_percent = result.compute_path_added_percent <= 1.0;
    result.fit_2_percent = result.compute_path_added_percent <= 2.0;
    result.fit_5_percent = result.compute_path_added_percent <= 5.0;
    return result;
}

std::uint64_t CalculateCommonTimelineMakespan(
    std::uint64_t compute_branch_host_ns,
    std::uint64_t pipeline_start_offset_ns,
    std::uint64_t pipeline_timed_ns) noexcept {
    const auto pipeline_branch = pipeline_timed_ns >
            std::numeric_limits<std::uint64_t>::max() - pipeline_start_offset_ns
        ? std::numeric_limits<std::uint64_t>::max()
        : pipeline_start_offset_ns + pipeline_timed_ns;
    return std::max(compute_branch_host_ns, pipeline_branch);
}

std::int64_t CalculateReadyAhead(std::uint64_t deadline_time_ns,
                                 std::uint64_t gpu_ready_time_ns) noexcept {
    if (deadline_time_ns >= gpu_ready_time_ns) {
        const auto delta = deadline_time_ns - gpu_ready_time_ns;
        return delta > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                   ? std::numeric_limits<std::int64_t>::max()
                   : static_cast<std::int64_t>(delta);
    }
    const auto delta = gpu_ready_time_ns - deadline_time_ns;
    return delta > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
               ? std::numeric_limits<std::int64_t>::min()
               : -static_cast<std::int64_t>(delta);
}

std::vector<DeadlineProfile> EvaluatePipelineDeadlines(
    const std::vector<PipelineSample>& samples, const std::vector<std::uint64_t>& deadlines_ns) {
    std::vector<DeadlineProfile> result;
    for (const auto deadline : deadlines_ns) {
        DeadlineProfile profile;
        profile.deadline_ns = deadline;
        std::vector<double> ready, lateness;
        for (const auto& sample : samples) {
            if (sample.status != PipelineStatus::Success) continue;
            const auto supply = sample.pc_ns;
            const auto value = CalculateReadyAhead(deadline, supply);
            ready.push_back(static_cast<double>(value));
            lateness.push_back(static_cast<double>(std::max<std::int64_t>(0, -value)));
            if (value >= 0) ++profile.hits;
            else ++profile.misses;
        }
        const auto count = profile.hits + profile.misses;
        profile.success_rate = count ? static_cast<double>(profile.hits) / count : 0;
        profile.ready_ahead_ns = trace::CalculateDistribution(std::move(ready));
        profile.lateness_ns = trace::CalculateDistribution(std::move(lateness));
        if (count >= 1000) profile.p999_success_claim = profile.success_rate;
        result.push_back(profile);
    }
    return result;
}

void FinalizeHostCopy(HostCopyResult& result) {
    std::vector<double> wall, bandwidth, cpu, compute;
    for (const auto& sample : result.samples) {
        if (sample.status != PipelineStatus::Success) continue;
        wall.push_back(static_cast<double>(sample.wall_ns));
        bandwidth.push_back(sample.bytes_per_second);
        cpu.push_back(static_cast<double>(sample.process_cpu_ns));
        if (sample.compute_ns) compute.push_back(static_cast<double>(sample.compute_ns));
    }
    result.wall_ns = trace::CalculateDistribution(std::move(wall));
    result.bytes_per_second = trace::CalculateDistribution(std::move(bandwidth));
    result.process_cpu_ns = trace::CalculateDistribution(std::move(cpu));
    result.compute_ns = trace::CalculateDistribution(std::move(compute));
    if (result.samples.empty() && result.status == PipelineStatus::Success)
        result.status = PipelineStatus::InternalError;
}

void FinalizePipeline(PipelineResult& result) {
    result.end_to_end_ns = trace::CalculateDistribution(Values(result.samples, [](const auto& s) {
        return static_cast<double>(s.pc_ns);
    }));
    result.ready_ahead_ns = trace::CalculateDistribution(Values(result.samples, [](const auto& s) {
        return static_cast<double>(s.ready_ahead_ns);
    }));
    result.compute_path_added_percent = trace::CalculateDistribution(Values(result.samples, [](const auto& s) {
        return s.metrics.compute_path_added_percent;
    }));
    result.pipeline_slowdown = trace::CalculateDistribution(Values(result.samples, [](const auto& s) {
        return s.metrics.pipeline_slowdown;
    }));
    result.storage_ns = trace::CalculateDistribution(Values(result.samples, [](const auto& s) {
        return static_cast<double>(s.storage_ns);
    }));
    result.host_copy_ns = trace::CalculateDistribution(Values(result.samples, [](const auto& s) {
        return static_cast<double>(s.host_copy_ns);
    }));
    result.h2d_ns = trace::CalculateDistribution(Values(result.samples, [](const auto& s) {
        return static_cast<double>(s.h2d_ns);
    }));
    result.deadlines = EvaluatePipelineDeadlines(result.samples);
    if (result.samples.empty() && result.status == PipelineStatus::Success)
        result.status = PipelineStatus::InternalError;
}

std::optional<std::string> ValidateConfiguration(const PipelineConfiguration& c,
                                                 const storage::StorageTarget& target,
                                                 std::uint64_t dataset_bytes) {
    if (!c.aggregate_bytes || !c.chunk_bytes || c.aggregate_bytes % c.chunk_bytes)
        return "aggregate size must be an exact positive multiple of chunk size";
    if (c.buffer_depth < 1 || c.buffer_depth > 4) return "buffer depth must be 1..4";
    if (!c.repetitions && !c.stream_seconds) return "repetitions must be positive";
    if (c.aggregate_bytes > dataset_bytes) return "aggregate exceeds dataset";
    const auto alignment = std::max<std::uint32_t>(1, target.alignment.file_offset_alignment_bytes);
    if (c.chunk_bytes % alignment) return "chunk size violates storage alignment";
    if (c.chunk_bytes > std::numeric_limits<std::uint32_t>::max())
        return "chunk exceeds Windows ReadFile DWORD length";
    if (c.host_copy_workers < 1 || c.host_copy_workers > 4)
        return "host copy workers must be 1..4";
    if (!IsFinitePositive(c.compute_window_us)) return "compute window must be positive";
    return std::nullopt;
}

PipelinePlan BuildDefaultPlan(const storage::StorageTarget& target,
                              const storage::DatasetIdentity& dataset,
                              const memory::MemorySnapshot& host,
                              const cuda::DeviceMemoryInfo& device,
                              std::uint32_t repetitions) {
    PipelinePlan plan;
    plan.target = target;
    plan.dataset = dataset;
    plan.available_host_bytes = host.available_physical_bytes.value_or(0);
    plan.available_vram_bytes = device.free_bytes;
    constexpr std::uint64_t mib = 1024ULL * 1024ULL;
    const std::vector<std::pair<std::uint64_t, std::vector<std::uint64_t>>> sizes{
        {64 * mib, {64 * mib}}, {128 * mib, {128 * mib}},
        {256 * mib, {128 * mib, 64 * mib, 32 * mib}}};
    for (const auto type : {PipelineType::NvmePageablePinnedH2D,
                            PipelineType::NvmePinnedH2D}) {
        for (const auto& [aggregate, chunks] : sizes) {
            for (const auto chunk : chunks) {
                for (std::uint32_t depth = 1; depth <= 4; ++depth) {
                    for (const auto workload : {cuda::ComputeWorkload::SyntheticAlu,
                                                cuda::ComputeWorkload::MemoryBound,
                                                cuda::ComputeWorkload::Fp16Gemm}) {
                        for (const double window : {32'000.0, 64'000.0}) {
                            PipelinePlanEntry entry;
                            auto& c = entry.configuration;
                            c.pipeline_type = type;
                            c.aggregate_bytes = aggregate;
                            c.chunk_bytes = chunk;
                            c.buffer_depth = depth;
                            c.compute_workload = workload;
                            c.compute_window_us = window;
                            c.deadline_ns = static_cast<std::uint64_t>(window * 1000.0);
                            c.repetitions = repetitions;
                            c.contention_test = type == PipelineType::NvmePageablePinnedH2D
                                                    ? ContentionTest::I : ContentionTest::J;
                            entry.pageable_bytes = type == PipelineType::NvmePageablePinnedH2D
                                                       ? chunk * depth : 0;
                            entry.pinned_bytes = chunk * depth;
                            entry.vram_bytes = aggregate;
                            entry.outstanding_storage_bytes = chunk * depth;
                            const auto host_need = entry.pageable_bytes + entry.pinned_bytes;
                            if (plan.available_host_bytes && host_need > plan.available_host_bytes / 2) {
                                entry.disposition = PipelineStatus::SkippedSafetyLimit;
                                entry.reason = "host allocation exceeds half of available RAM";
                            } else if (plan.available_vram_bytes &&
                                       entry.vram_bytes + 1024ULL * mib > plan.available_vram_bytes) {
                                entry.disposition = PipelineStatus::SkippedSafetyLimit;
                                entry.reason = "VRAM allocation violates 1 GiB reserve";
                            }
                            plan.estimated_total_read_bytes +=
                                entry.disposition == PipelineStatus::Success
                                    ? aggregate * repetitions * 2ULL : 0;
                            plan.entries.push_back(std::move(entry));
                        }
                    }
                }
            }
        }
    }
    return plan;
}

std::vector<PipelineConfiguration> SelectRefinementConfigurations(
    const std::vector<PipelineResult>& coarse) {
    std::vector<PipelineConfiguration> selected;
    for (const auto& cell : coarse) {
        if (cell.status != PipelineStatus::Success || cell.samples.empty()) continue;
        const auto deadline = std::find_if(cell.deadlines.begin(), cell.deadlines.end(),
            [&](const auto& p) { return p.deadline_ns == cell.configuration.deadline_ns; });
        const double rate = deadline == cell.deadlines.end() ? 0 : deadline->success_rate;
        const double added = cell.compute_path_added_percent.p99;
        if ((rate >= 0.85 && rate < 1.0) || (added >= 1.5 && added <= 6.0)) {
            auto configuration = cell.configuration;
            configuration.phase = PipelinePhase::Refine;
            configuration.repetitions = std::max<std::uint32_t>(100, configuration.repetitions);
            configuration.refinement_reason = rate < 1.0
                ? "deadline transition" : "compute P99 crossed 2/5 percent band";
            selected.push_back(std::move(configuration));
        }
    }
    return selected;
}

std::string PipelineConfigurationIdentity(const PipelineConfiguration& c) {
    std::ostringstream material;
    material << "SIDECAR-PIPELINE-CONFIG-V1\n"
             << "type=" << ToString(c.pipeline_type) << '\n'
             << "aggregate_bytes=" << c.aggregate_bytes << '\n'
             << "chunk_bytes=" << c.chunk_bytes << '\n'
             << "depth=" << c.buffer_depth << '\n'
             << "compute=" << cuda::ToString(c.compute_workload) << '\n'
             << "compute_window_us=" << std::fixed << std::setprecision(3)
             << c.compute_window_us << '\n'
             << "deadline_ns=" << c.deadline_ns << '\n'
             << "phase=" << ToString(c.phase) << '\n'
             << "test=" << ToString(c.contention_test) << '\n'
             << "copy_workers=" << c.host_copy_workers << '\n';
    return core::Sha256Hex(material.str());
}

std::string HostCopyToJson(const HostCopyResult& r) {
    std::ostringstream out;
    out << std::setprecision(17) << "{\"affinity\":" << Escape(r.affinity)
        << ",\"bytes\":" << r.configuration.bytes
        << ",\"bytes_per_second\":" << DistributionJson(r.bytes_per_second)
        << ",\"compute_ns\":" << DistributionJson(r.compute_ns)
        << ",\"compute_window_us\":" << r.configuration.compute_window_us
        << ",\"message\":" << Escape(r.message)
        << ",\"phase\":" << Escape(ToString(r.configuration.phase))
        << ",\"process_cpu_ns\":" << DistributionJson(r.process_cpu_ns)
        << ",\"repetitions\":" << r.configuration.repetitions << ",\"samples\":[";
    for (std::size_t i = 0; i < r.samples.size(); ++i) {
        if (i) out << ',';
        const auto& s = r.samples[i];
        out << "{\"bytes_per_second\":" << s.bytes_per_second
            << ",\"compute_ns\":" << s.compute_ns << ",\"index\":" << s.sample_index
            << ",\"process_cpu_ns\":" << s.process_cpu_ns << ",\"status\":"
            << Escape(ToString(s.status)) << ",\"wall_ns\":" << s.wall_ns << '}';
    }
    out << "],\"status\":" << Escape(ToString(r.status))
        << ",\"wall_ns\":" << DistributionJson(r.wall_ns)
        << ",\"worker_count\":" << r.configuration.worker_count << '}';
    return out.str();
}

std::string PipelineResultToJson(const PipelineResult& r) {
    std::ostringstream out;
    out << std::setprecision(17) << "{\"aggregate_bytes\":"
        << r.configuration.aggregate_bytes << ",\"arena_allocation_id\":"
        << Escape(r.arena_allocation_id) << ",\"buffer_depth\":"
        << r.configuration.buffer_depth << ",\"chunk_bytes\":"
        << r.configuration.chunk_bytes << ",\"compute\":"
        << Escape(cuda::ToString(r.configuration.compute_workload))
        << ",\"compute_path_added_percent\":"
        << DistributionJson(r.compute_path_added_percent) << ",\"compute_window_us\":"
        << r.configuration.compute_window_us << ",\"contention_test\":"
        << Escape(ToString(r.configuration.contention_test)) << ",\"deadline_profiles\":[";
    for (std::size_t i = 0; i < r.deadlines.size(); ++i) {
        if (i) out << ',';
        const auto& d = r.deadlines[i];
        out << "{\"deadline_ns\":" << d.deadline_ns << ",\"hits\":" << d.hits
            << ",\"lateness_ns\":" << DistributionJson(d.lateness_ns)
            << ",\"misses\":" << d.misses << ",\"ready_ahead_ns\":"
            << DistributionJson(d.ready_ahead_ns) << ",\"success_rate\":"
            << d.success_rate << ",\"p999_success_rate\":";
        if (d.p999_success_claim) out << *d.p999_success_claim;
        else out << "null";
        out << '}';
    }
    out << "],\"end_to_end_ns\":" << DistributionJson(r.end_to_end_ns)
        << ",\"h2d_ns\":" << DistributionJson(r.h2d_ns)
        << ",\"host_copy_ns\":" << DistributionJson(r.host_copy_ns)
        << ",\"message\":" << Escape(r.message) << ",\"phase\":"
        << Escape(ToString(r.configuration.phase)) << ",\"pipeline_slowdown\":"
        << DistributionJson(r.pipeline_slowdown) << ",\"pipeline_type\":"
        << Escape(ToString(r.configuration.pipeline_type)) << ",\"ready_ahead_ns\":"
        << DistributionJson(r.ready_ahead_ns) << ",\"sample_count\":"
        << r.samples.size() << ",\"samples\":[";
    for (std::size_t i = 0; i < r.samples.size(); ++i) {
        if (i) out << ',';
        const auto& s = r.samples[i];
        out << "{\"c0_reference_ns\":" << s.c0_reference_ns
            << ",\"cc_ns\":" << s.cc_ns << ",\"compute_path_added_percent\":"
            << s.metrics.compute_path_added_percent << ",\"deadline_hit\":"
            << (s.deadline_hit ? "true" : "false") << ",\"h2d_ns\":" << s.h2d_ns
            << ",\"host_copy_ns\":" << s.host_copy_ns << ",\"index\":" << s.sample_index
            << ",\"makespan_ns\":" << s.makespan_ns << ",\"p0_reference_ns\":"
            << s.p0_reference_ns << ",\"pc_ns\":" << s.pc_ns
            << ",\"pipeline_slowdown\":" << s.metrics.pipeline_slowdown
            << ",\"ready_ahead_ns\":" << s.ready_ahead_ns << ",\"status\":"
            << Escape(ToString(s.status)) << ",\"storage_ns\":" << s.storage_ns
            << ",\"verified\":" << (s.verified ? "true" : "false") << '}';
    }
    out << "],\"slot_profiles\":[";
    for (std::size_t i = 0; i < r.slots.size(); ++i) {
        if (i) out << ',';
        const auto& s = r.slots[i];
        out << "{\"arena_offset\":" << s.arena_offset << ",\"bytes\":" << s.bytes
            << ",\"deadline_hits\":" << s.deadline_hits << ",\"deadline_misses\":"
            << s.deadline_misses << ",\"h2d_ns\":" << DistributionJson(s.h2d_ns)
            << ",\"host_copy_ns\":" << DistributionJson(s.host_copy_ns)
            << ",\"observation\":" << Escape(s.observation) << ",\"slot_id\":"
            << s.slot_id << ",\"transfer_count\":" << s.transfer_count
            << ",\"verification_failures\":" << s.verification_failures << '}';
    }
    out << "],\"status\":" << Escape(ToString(r.status))
        << ",\"storage_ns\":" << DistributionJson(r.storage_ns) << '}';
    return out.str();
}

std::string PipelinePlanToJson(const PipelinePlan& plan) {
    std::ostringstream out;
    out << "{\"available_host_bytes\":" << plan.available_host_bytes
        << ",\"available_vram_bytes\":" << plan.available_vram_bytes
        << ",\"dataset\":{\"identity_hash\":" << Escape(plan.dataset.identity_hash)
        << ",\"path\":" << Escape(plan.dataset.path.generic_string())
        << ",\"verified\":" << (plan.dataset.verified ? "true" : "false")
        << "},\"estimated_total_read_bytes\":" << plan.estimated_total_read_bytes
        << ",\"health_safety\":" << Escape(plan.health_safety) << ",\"entries\":[";
    for (std::size_t i = 0; i < plan.entries.size(); ++i) {
        if (i) out << ',';
        const auto& e = plan.entries[i];
        out << "{\"aggregate_bytes\":" << e.configuration.aggregate_bytes
            << ",\"buffer_depth\":" << e.configuration.buffer_depth
            << ",\"chunk_bytes\":" << e.configuration.chunk_bytes
            << ",\"compute\":" << Escape(cuda::ToString(e.configuration.compute_workload))
            << ",\"compute_window_us\":" << e.configuration.compute_window_us
            << ",\"disposition\":" << Escape(ToString(e.disposition))
            << ",\"outstanding_storage_bytes\":" << e.outstanding_storage_bytes
            << ",\"pageable_bytes\":" << e.pageable_bytes << ",\"pinned_bytes\":"
            << e.pinned_bytes << ",\"pipeline_type\":"
            << Escape(ToString(e.configuration.pipeline_type)) << ",\"reason\":"
            << Escape(e.reason) << ",\"vram_bytes\":" << e.vram_bytes << '}';
    }
    out << "],\"slot_rotation_policy\":" << Escape(plan.slot_rotation_policy)
        << ",\"target\":{\"model\":" << Escape(plan.target.model)
        << ",\"persistent_id\":" << Escape(plan.target.persistent_id)
        << "},\"telemetry_policy\":" << Escape(plan.telemetry_policy) << '}';
    return out.str();
}

std::string FormatPipelineResult(const PipelineResult& r) {
    std::ostringstream out;
    out << "SIDECAR MEMORY-HIGHWAY PIPELINE\n\n"
        << "Path: " << ToString(r.configuration.pipeline_type) << '\n'
        << "Test: " << ToString(r.configuration.contention_test) << '\n'
        << "Aggregate/chunk/depth: " << r.configuration.aggregate_bytes / 1048576
        << " MiB / " << r.configuration.chunk_bytes / 1048576 << " MiB / "
        << r.configuration.buffer_depth << '\n'
        << "Compute: " << cuda::ToString(r.configuration.compute_workload) << " / "
        << r.configuration.compute_window_us / 1000.0 << " ms\n"
        << "Status: " << ToString(r.status) << '\n'
        << "Samples: " << r.samples.size() << '\n'
        << "Pipeline P50/P95/P99: " << r.end_to_end_ns.p50 / 1e6 << " / "
        << r.end_to_end_ns.p95 / 1e6 << " / " << r.end_to_end_ns.p99 / 1e6 << " ms\n"
        << "Storage/host-copy/H2D P99: " << r.storage_ns.p99 / 1e6 << " / "
        << r.host_copy_ns.p99 / 1e6 << " / " << r.h2d_ns.p99 / 1e6 << " ms\n"
        << "Compute-path added P99: " << r.compute_path_added_percent.p99 << "%\n"
        << "Ready-ahead P50/P99: " << r.ready_ahead_ns.p50 / 1e6 << " / "
        << r.ready_ahead_ns.p99 / 1e6 << " ms\n";
    for (const auto& d : r.deadlines)
        out << "Deadline " << d.deadline_ns / 1e6 << " ms: "
            << d.success_rate * 100.0 << "% (" << d.hits << '/' << d.hits + d.misses << ")\n";
    if (!r.message.empty()) out << "Message: " << r.message << '\n';
    return out.str();
}

}  // namespace sidecar::pipeline
