#include "sidecar/memory/safety.hpp"

#include <algorithm>
#include <cmath>

namespace sidecar::memory {

std::uint64_t RequiredReserve(const MemorySnapshot& snapshot,
                              const SafetyPolicy& policy) noexcept {
    const auto installed = snapshot.installed_physical_bytes.value_or(
        snapshot.visible_physical_bytes.value_or(0));
    const auto fractional = static_cast<std::uint64_t>(
        static_cast<long double>(installed) * policy.installed_reserve_fraction);
    const auto dynamic = std::max(policy.minimum_dynamic_reserve_bytes, fractional);
    return std::max(policy.absolute_reserve_bytes, dynamic);
}

SafetyDecision PlanAllocation(const MemorySnapshot& snapshot,
                              const SafetyPolicy& policy,
                              MemoryMethod method,
                              std::uint64_t requested_bytes,
                              bool previous_pressure_signal) {
    SafetyDecision decision;
    decision.requested_bytes = requested_bytes;
    decision.required_reserve_bytes = RequiredReserve(snapshot, policy);
    if (previous_pressure_signal) {
        decision.status = SafetyStatus::SkipPreviousPressureSignal;
        decision.reason = "a previous allocation or cleanup signaled pressure";
        return decision;
    }
    if (!snapshot.available_physical_bytes ||
        (!snapshot.installed_physical_bytes && !snapshot.visible_physical_bytes)) {
        decision.status = SafetyStatus::SkipUnknownMemoryState;
        decision.reason = "physical memory headroom is unavailable";
        return decision;
    }
    const auto installed = snapshot.installed_physical_bytes.value_or(
        *snapshot.visible_physical_bytes);
    const double fraction = method == MemoryMethod::Pageable
                                ? policy.pageable_max_test_fraction
                                : policy.pinned_max_test_fraction;
    decision.method_policy_limit_bytes = static_cast<std::uint64_t>(
        static_cast<long double>(installed) * fraction);
    if (requested_bytes == 0 || requested_bytes > decision.method_policy_limit_bytes) {
        decision.status = SafetyStatus::SkipPolicyLimit;
        decision.reason = "requested size exceeds the configured method fraction";
        return decision;
    }
    const auto available = *snapshot.available_physical_bytes;
    decision.projected_available_bytes = requested_bytes < available
                                             ? available - requested_bytes : 0;
    if (available <= decision.required_reserve_bytes ||
        requested_bytes > available - decision.required_reserve_bytes) {
        decision.status = SafetyStatus::SkipInsufficientHeadroom;
        decision.reason = "allocation would violate the emergency free-RAM reserve";
        return decision;
    }
    if (snapshot.system_commit_bytes && snapshot.system_commit_limit_bytes) {
        const auto commit = *snapshot.system_commit_bytes;
        const auto limit = *snapshot.system_commit_limit_bytes;
        if (commit >= limit || requested_bytes > limit - commit ||
            limit - commit - requested_bytes < decision.required_reserve_bytes / 2) {
            decision.status = SafetyStatus::SkipInsufficientHeadroom;
            decision.reason = "allocation would approach the system commit limit";
            return decision;
        }
    }
    decision.status = SafetyStatus::Safe;
    decision.reason = "headroom and method policy permit the allocation";
    return decision;
}

std::vector<std::uint64_t> DefaultSizeSweep() {
    constexpr std::uint64_t mib = 1024ULL * 1024;
    return {4 * mib, 16 * mib, 32 * mib, 64 * mib, 128 * mib, 256 * mib,
            512 * mib, 1024 * mib, 2048 * mib, 4096 * mib, 8192 * mib,
            12288 * mib, 16384 * mib};
}

const char* ToString(SafetyStatus status) noexcept {
    switch (status) {
        case SafetyStatus::Safe: return "SAFE";
        case SafetyStatus::SkipInsufficientHeadroom: return "SKIP_INSUFFICIENT_HEADROOM";
        case SafetyStatus::SkipPolicyLimit: return "SKIP_POLICY_LIMIT";
        case SafetyStatus::SkipPreviousPressureSignal: return "SKIP_PREVIOUS_PRESSURE_SIGNAL";
        case SafetyStatus::SkipUnknownMemoryState: return "SKIP_UNKNOWN_MEMORY_STATE";
    }
    return "SKIP_UNKNOWN_MEMORY_STATE";
}

}  // namespace sidecar::memory
