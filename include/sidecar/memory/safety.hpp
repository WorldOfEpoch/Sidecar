#pragma once

#include "sidecar/memory/host_memory.hpp"

#include <cstdint>
#include <vector>

namespace sidecar::memory {

enum class SafetyStatus {
    Safe,
    SkipInsufficientHeadroom,
    SkipPolicyLimit,
    SkipPreviousPressureSignal,
    SkipUnknownMemoryState,
};

struct SafetyPolicy {
    std::uint64_t absolute_reserve_bytes{0};  // zero selects the dynamic default
    double installed_reserve_fraction{0.15};
    double pageable_max_test_fraction{0.25};
    double pinned_max_test_fraction{0.10};
    std::uint64_t minimum_dynamic_reserve_bytes{4ULL * 1024 * 1024 * 1024};
    std::uint64_t cleanup_warning_bytes{512ULL * 1024 * 1024};
    std::uint64_t cleanup_hard_stop_bytes{2ULL * 1024 * 1024 * 1024};
};

struct SafetyDecision {
    SafetyStatus status{SafetyStatus::SkipUnknownMemoryState};
    std::uint64_t requested_bytes{0};
    std::uint64_t required_reserve_bytes{0};
    std::uint64_t projected_available_bytes{0};
    std::uint64_t method_policy_limit_bytes{0};
    std::string reason;
    [[nodiscard]] bool safe() const noexcept { return status == SafetyStatus::Safe; }
};

[[nodiscard]] std::uint64_t RequiredReserve(const MemorySnapshot& snapshot,
                                            const SafetyPolicy& policy) noexcept;
[[nodiscard]] SafetyDecision PlanAllocation(const MemorySnapshot& snapshot,
                                            const SafetyPolicy& policy,
                                            MemoryMethod method,
                                            std::uint64_t requested_bytes,
                                            bool previous_pressure_signal = false);
[[nodiscard]] std::vector<std::uint64_t> DefaultSizeSweep();
[[nodiscard]] const char* ToString(SafetyStatus status) noexcept;

}  // namespace sidecar::memory
