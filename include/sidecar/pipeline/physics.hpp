#pragma once

#include "sidecar/cuda/overlap.hpp"
#include "sidecar/memory/host_memory.hpp"
#include "sidecar/storage/physics.hpp"
#include "sidecar/trace/benchmark.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sidecar::pipeline {

inline constexpr std::uint64_t kPipelineSeed = 0x5349444543415238ULL;

enum class PipelineType { NvmePageablePinnedH2D, NvmePinnedH2D };
enum class SlotState {
    Free,
    NvmeReading,
    HostCopyPending,
    HostCopyActive,
    H2DReady,
    H2DActive,
    GpuReady,
    VerifyPending,
};
enum class PipelinePhase { Validate, HostCopy, Baseline, Coarse, Refine, Repeat, Stream };
enum class ContentionTest { A, B, C, D, E, F, G, H, I, J };
enum class PipelineStatus {
    Success,
    SkippedUnsupported,
    SkippedSafetyLimit,
    InvalidConfiguration,
    StorageError,
    HostCopyError,
    H2DError,
    ComputeError,
    TimedOut,
    Cancelled,
    DataVerificationFailure,
    HardwareHealthChange,
    InternalError,
};

[[nodiscard]] const char* ToString(PipelineType value) noexcept;
[[nodiscard]] const char* ToString(SlotState value) noexcept;
[[nodiscard]] const char* ToString(PipelinePhase value) noexcept;
[[nodiscard]] const char* ToString(ContentionTest value) noexcept;
[[nodiscard]] const char* ToString(PipelineStatus value) noexcept;

class SlotStateMachine final {
public:
    explicit SlotStateMachine(std::uint32_t slot_id = 0) noexcept;
    [[nodiscard]] std::uint32_t slotId() const noexcept;
    [[nodiscard]] SlotState state() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] bool CanTransition(SlotState next) const noexcept;
    void Transition(SlotState next);
    void ResetAfterCancellation() noexcept;

private:
    std::uint32_t slot_id_{0};
    SlotState state_{SlotState::Free};
    std::uint64_t generation_{0};
};

struct HostCopyConfiguration {
    std::uint64_t bytes{64ULL * 1024ULL * 1024ULL};
    std::uint32_t worker_count{1};
    std::uint32_t warmups{2};
    std::uint32_t repetitions{20};
    std::optional<cuda::ComputeWorkload> concurrent_compute;
    double compute_window_us{0};
    PipelinePhase phase{PipelinePhase::HostCopy};
};

struct HostCopySample {
    std::uint32_t sample_index{0};
    std::uint64_t wall_ns{0};
    std::uint64_t process_cpu_ns{0};
    std::uint64_t compute_ns{0};
    double bytes_per_second{0};
    PipelineStatus status{PipelineStatus::Success};
    std::string message;
};

struct HostCopyResult {
    HostCopyConfiguration configuration;
    std::vector<HostCopySample> samples;
    trace::Distribution wall_ns;
    trace::Distribution bytes_per_second;
    trace::Distribution process_cpu_ns;
    trace::Distribution compute_ns;
    PipelineStatus status{PipelineStatus::Success};
    std::string affinity{"UNCONTROLLED"};
    std::string message;
};

struct PipelineConfiguration {
    PipelineType pipeline_type{PipelineType::NvmePinnedH2D};
    std::uint64_t aggregate_bytes{256ULL * 1024ULL * 1024ULL};
    std::uint64_t chunk_bytes{128ULL * 1024ULL * 1024ULL};
    std::uint32_t buffer_depth{2};
    cuda::ComputeWorkload compute_workload{cuda::ComputeWorkload::SyntheticAlu};
    double compute_window_us{64'000.0};
    std::uint64_t deadline_ns{64'000'000ULL};
    std::uint32_t repetitions{10};
    std::uint64_t ordering_seed{kPipelineSeed};
    std::uint64_t timeout_ms{120'000};
    PipelinePhase phase{PipelinePhase::Coarse};
    ContentionTest contention_test{ContentionTest::J};
    std::uint32_t host_copy_workers{1};
    int device_index{0};
    std::uint32_t stream_seconds{0};
    std::optional<std::uint32_t> cancel_after_blocks;
    std::string refinement_reason;
};

struct PipelineMetrics {
    double compute_path_delta_ns{0};
    double compute_path_added_ns{0};
    double compute_path_added_percent{0};
    double pipeline_slowdown{0};
    double compute_slowdown{0};
    double pipeline_overlap_raw{0};
    double pipeline_overlap_normalized{0};
    bool fit_1_percent{false};
    bool fit_2_percent{false};
    bool fit_5_percent{false};
};

struct StageSample {
    std::uint32_t sample_index{0};
    std::uint32_t block_id{0};
    std::uint32_t slot_id{0};
    std::string stage_type;
    std::uint64_t bytes{0};
    std::uint64_t file_offset{0};
    std::uint64_t host_submit_ns{0};
    std::uint64_t host_start_ns{0};
    std::uint64_t host_finish_ns{0};
    std::uint64_t device_duration_ns{0};
    std::uint64_t wait_ns{0};
    PipelineStatus status{PipelineStatus::Success};
    std::int64_t native_error{0};
};

struct PipelineSample {
    std::uint32_t sample_index{0};
    std::uint64_t c0_reference_ns{0};
    std::uint64_t p0_reference_ns{0};
    std::uint64_t cc_ns{0};
    std::uint64_t pc_ns{0};
    std::uint64_t makespan_ns{0};
    std::uint64_t storage_ns{0};
    std::uint64_t host_copy_ns{0};
    std::uint64_t h2d_ns{0};
    std::uint64_t fill_latency_ns{0};
    std::uint64_t steady_state_interval_ns{0};
    std::uint64_t drain_latency_ns{0};
    std::uint64_t gpu_waiting_for_data_ns{0};
    std::uint64_t h2d_waiting_for_source_ns{0};
    std::uint64_t pinned_slot_wait_ns{0};
    std::uint64_t pageable_slot_wait_ns{0};
    std::uint64_t nvme_queue_starved_ns{0};
    std::int64_t ready_ahead_ns{0};
    bool deadline_hit{false};
    bool verified{false};
    double bytes_per_second{0};
    PipelineMetrics metrics;
    std::vector<std::uint32_t> slot_ids;
    PipelineStatus status{PipelineStatus::Success};
    std::int64_t native_error{0};
    std::string message;
};

struct DeadlineProfile {
    std::uint64_t deadline_ns{0};
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    double success_rate{0};
    trace::Distribution ready_ahead_ns;
    trace::Distribution lateness_ns;
    std::optional<double> p999_success_claim;
};

struct SlotProfile {
    std::string arena_allocation_id;
    std::uint32_t slot_id{0};
    std::uint64_t arena_offset{0};
    std::uint64_t bytes{0};
    std::uint64_t transfer_count{0};
    std::uint64_t verification_failures{0};
    std::uint64_t deadline_hits{0};
    std::uint64_t deadline_misses{0};
    trace::Distribution host_copy_ns;
    trace::Distribution h2d_ns;
    std::string observation{"NO_REGION_VARIATION_OBSERVED"};
};

struct PipelineResult {
    PipelineConfiguration configuration;
    std::vector<PipelineSample> samples;
    std::vector<StageSample> stages;
    std::vector<DeadlineProfile> deadlines;
    std::vector<SlotProfile> slots;
    trace::Distribution end_to_end_ns;
    trace::Distribution ready_ahead_ns;
    trace::Distribution compute_path_added_percent;
    trace::Distribution pipeline_slowdown;
    trace::Distribution storage_ns;
    trace::Distribution host_copy_ns;
    trace::Distribution h2d_ns;
    storage::HealthSnapshot health_before;
    storage::HealthSnapshot health_after;
    cuda::TransferTelemetry gpu_before;
    cuda::TransferTelemetry gpu_after;
    memory::MemorySnapshot host_before;
    memory::MemorySnapshot host_after;
    std::string arena_allocation_id;
    PipelineStatus status{PipelineStatus::Success};
    std::string message;
};

struct PipelinePlanEntry {
    PipelineConfiguration configuration;
    std::uint64_t pageable_bytes{0};
    std::uint64_t pinned_bytes{0};
    std::uint64_t vram_bytes{0};
    std::uint64_t outstanding_storage_bytes{0};
    PipelineStatus disposition{PipelineStatus::Success};
    std::string reason;
};

struct PipelinePlan {
    storage::StorageTarget target;
    storage::DatasetIdentity dataset;
    std::vector<PipelinePlanEntry> entries;
    std::uint64_t available_host_bytes{0};
    std::uint64_t available_vram_bytes{0};
    std::uint64_t estimated_total_read_bytes{0};
    std::string telemetry_policy{"LOW_RATE_BOUNDARY_ONLY"};
    std::string health_safety{"STOP_ON_HEALTH_OR_REPEATED_VERIFICATION_CHANGE"};
    std::string slot_rotation_policy{"ROUND_ROBIN_SESSION_LOCAL"};
};

class IPipelineProvider {
public:
    virtual ~IPipelineProvider() = default;
    [[nodiscard]] virtual bool SupportsCuda() const noexcept = 0;
    virtual HostCopyResult RunHostCopy(const HostCopyConfiguration& configuration) = 0;
    virtual PipelineResult Run(const storage::StorageTarget& target,
                               const PipelineConfiguration& configuration) = 0;
};

[[nodiscard]] std::unique_ptr<IPipelineProvider> CreateNativePipelineProvider();
[[nodiscard]] PipelineMetrics CalculatePipelineMetrics(double c0_ns, double p0_ns,
                                                       double cc_ns, double pc_ns,
                                                       double makespan_ns) noexcept;
[[nodiscard]] std::uint64_t CalculateCommonTimelineMakespan(
    std::uint64_t compute_branch_host_ns,
    std::uint64_t pipeline_start_offset_ns,
    std::uint64_t pipeline_timed_ns) noexcept;
[[nodiscard]] std::int64_t CalculateReadyAhead(std::uint64_t deadline_time_ns,
                                               std::uint64_t gpu_ready_time_ns) noexcept;
[[nodiscard]] std::vector<DeadlineProfile> EvaluatePipelineDeadlines(
    const std::vector<PipelineSample>& samples,
    const std::vector<std::uint64_t>& deadlines_ns =
        {16'000'000ULL, 32'000'000ULL, 64'000'000ULL, 96'000'000ULL,
         128'000'000ULL, 160'000'000ULL});
void FinalizeHostCopy(HostCopyResult& result);
void FinalizePipeline(PipelineResult& result);
[[nodiscard]] std::optional<std::string> ValidateConfiguration(
    const PipelineConfiguration& configuration,
    const storage::StorageTarget& target,
    std::uint64_t dataset_bytes);
[[nodiscard]] PipelinePlan BuildDefaultPlan(const storage::StorageTarget& target,
                                            const storage::DatasetIdentity& dataset,
                                            const memory::MemorySnapshot& host,
                                            const cuda::DeviceMemoryInfo& device,
                                            std::uint32_t repetitions = 5);
[[nodiscard]] std::vector<PipelineConfiguration> SelectRefinementConfigurations(
    const std::vector<PipelineResult>& coarse);
[[nodiscard]] std::string PipelineConfigurationIdentity(
    const PipelineConfiguration& configuration);
[[nodiscard]] std::string HostCopyToJson(const HostCopyResult& result);
[[nodiscard]] std::string PipelineResultToJson(const PipelineResult& result);
[[nodiscard]] std::string PipelinePlanToJson(const PipelinePlan& plan);
[[nodiscard]] std::string FormatPipelineResult(const PipelineResult& result);

}  // namespace sidecar::pipeline
