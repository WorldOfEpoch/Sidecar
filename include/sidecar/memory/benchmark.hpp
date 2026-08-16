#pragma once

#include "sidecar/memory/host_memory.hpp"
#include "sidecar/memory/safety.hpp"
#include "sidecar/trace/benchmark.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace sidecar::memory {

struct TimerCalibration {
    std::string method{"steady_clock"};
    std::uint64_t bracket_overhead_ns{0};
    std::uint64_t observed_resolution_ns{0};
    std::vector<std::uint64_t> raw_bracket_samples_ns;
};

struct PhaseTiming {
    std::optional<std::uint64_t> raw_ns;
    std::optional<std::uint64_t> corrected_ns;
};

struct LifecycleSample {
    MemoryMethod method{MemoryMethod::Pageable};
    RegistrationMode mode{RegistrationMode::NotApplicable};
    std::uint64_t requested_bytes{0};
    std::uint32_t repetition{0};
    bool cold_setup{false};
    ResultStatus status{ResultStatus::Success};
    SafetyDecision safety;
    PhaseTiming backing_allocation;
    PhaseTiming first_touch;
    PhaseTiming warm_touch;
    PhaseTiming registration;
    PhaseTiming unregistration;
    PhaseTiming cleanup;
    MemorySnapshot before;
    MemorySnapshot after_setup;
    MemorySnapshot after_cleanup;
    std::int64_t available_recovery_delta_bytes{0};
    double available_recovery_percent{0};
    bool noisy{false};
    std::int64_t cuda_error{0};
    std::int64_t windows_error{0};
    std::string message;
};

struct PhaseStatistics {
    std::size_t count{0};
    trace::Distribution raw_ns;
    trace::Distribution corrected_ns;
    bool p99_meaningful{false};
};

struct LifecycleResult {
    MemoryMethod method{MemoryMethod::Pageable};
    RegistrationMode mode{RegistrationMode::NotApplicable};
    std::uint64_t requested_bytes{0};
    ResultStatus status{ResultStatus::Success};
    std::vector<LifecycleSample> samples;
    PhaseStatistics allocation;
    PhaseStatistics first_touch;
    PhaseStatistics warm_touch;
    PhaseStatistics registration;
    PhaseStatistics unregistration;
    PhaseStatistics cleanup;
    std::vector<std::pair<std::uint64_t, double>> amortized_setup_us;
    double allocation_latency_drift_percent{0};
    double cleanup_latency_drift_percent{0};
    std::string message;
};

struct LifecycleOptions {
    std::vector<MemoryMethod> methods;
    std::vector<std::uint64_t> sizes;
    SafetyPolicy safety;
    std::optional<std::uint32_t> repetitions;
    bool dry_run{false};
    bool include_cold_registration{true};
    bool include_pretouched_registration{true};
    std::uint32_t cooldown_checks{5};
    std::uint32_t cooldown_milliseconds{100};
};

struct LifecycleReport {
    TimerCalibration timer;
    MemorySnapshot initial_snapshot;
    std::vector<LifecycleResult> results;
    bool cuda_supported{false};
    bool cuda_warmup_performed{false};
    std::uint64_t cuda_warmup_ns{0};
    ProviderResult cuda_warmup_result;
    bool pressure_stop{false};
    std::string pressure_stop_reason;
};

struct ArenaTestResult {
    ArenaBackend backend{ArenaBackend::CudaHostAlloc};
    std::uint64_t capacity_bytes{0};
    std::uint32_t reuse_count{0};
    ResultStatus status{ResultStatus::Success};
    PhaseTiming setup;
    PhaseTiming cleanup;
    PhaseStatistics reuse;
    bool contents_verified{false};
    MemorySnapshot before;
    MemorySnapshot after_cleanup;
    std::vector<std::uint64_t> raw_reuse_ns;
    std::string message;
};

[[nodiscard]] TimerCalibration CalibrateTimer(std::uint32_t samples = 10000);
[[nodiscard]] std::uint32_t RepetitionsForSize(std::uint64_t bytes) noexcept;
[[nodiscard]] PhaseStatistics CalculatePhaseStatistics(
    const std::vector<LifecycleSample>& samples,
    const PhaseTiming LifecycleSample::* phase);
[[nodiscard]] LifecycleReport RunLifecycleLaboratory(IHostMemoryProvider& provider,
                                                     const LifecycleOptions& options);
[[nodiscard]] ArenaTestResult RunArenaTest(IHostMemoryProvider& provider,
                                           ArenaBackend backend,
                                           std::uint64_t bytes,
                                           std::uint32_t reuse_count,
                                           const SafetyPolicy& safety);
[[nodiscard]] std::string LifecycleReportToJson(const LifecycleReport& report);
[[nodiscard]] std::string FormatLifecycleReport(const LifecycleReport& report);
[[nodiscard]] std::string FormatMemoryInfo(const MemorySnapshot& snapshot);
[[nodiscard]] std::string MemoryInfoToJson(const MemorySnapshot& snapshot);
[[nodiscard]] std::string FormatMemoryPlan(const MemorySnapshot& snapshot,
                                           const SafetyPolicy& policy,
                                           MemoryMethod method,
                                           const std::vector<std::uint64_t>& sizes);
[[nodiscard]] std::string MemoryPlanToJson(const MemorySnapshot& snapshot,
                                           const SafetyPolicy& policy,
                                           MemoryMethod method,
                                           const std::vector<std::uint64_t>& sizes);
[[nodiscard]] std::string ArenaResultToJson(const ArenaTestResult& result);
[[nodiscard]] std::string FormatArenaResult(const ArenaTestResult& result);

}  // namespace sidecar::memory
