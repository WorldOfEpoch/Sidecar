#pragma once

#include <cstdint>
#include <type_traits>

namespace sidecar::trace {

enum class EventType : std::uint16_t {
    HostScheduleDecision = 1,
    HostPrefetchSubmit,
    HostIoSubmit,
    HostIoComplete,
    HostCudaLaunch,
    GpuTransferStart,
    GpuTransferEnd,
    GpuKernelStart,
    GpuKernelEnd,
    StallBegin,
    StallResolved,
    DeadlineExceeded,
    CacheHit,
    CacheMiss,
};

enum class MemoryTier : std::uint8_t {
    Unknown = 0,
    NvmeRaw,
    ColdPageableRam,
    WarmManagedRam,
    LockedHotRam,
    PinnedDmaRing,
    VramL1,
};

// The compact format uses session-local 32-bit IDs. Stable, full-width metadata
// is emitted once into a trace dictionary outside record(). This preserves parent
// correlation without allocating or performing indirection in the hot path.
struct TraceRecord32 {
    std::uint64_t host_timestamp_ns{0};
    std::uint32_t operation_id{0};
    std::uint32_t parent_operation_id{0};
    std::uint32_t subject_id{0};
    std::uint32_t payload_bytes{0};
    EventType event_type{EventType::HostScheduleDecision};
    MemoryTier source_tier{MemoryTier::Unknown};
    MemoryTier destination_tier{MemoryTier::Unknown};
    std::uint16_t producer_id{0};
    std::uint16_t auxiliary{0};
};

// The forensic format retains full-width IDs and two event-specific payloads.
struct TraceRecord64 {
    std::uint64_t host_timestamp_ns{0};
    std::uint64_t operation_id{0};
    std::uint64_t parent_operation_id{0};
    std::uint64_t subject_id{0};
    std::uint64_t payload_bytes{0};
    std::uint32_t producer_id{0};
    EventType event_type{EventType::HostScheduleDecision};
    MemoryTier source_tier{MemoryTier::Unknown};
    MemoryTier destination_tier{MemoryTier::Unknown};
    std::uint64_t auxiliary_0{0};
    std::uint64_t auxiliary_1{0};
};

static_assert(sizeof(TraceRecord32) == 32, "compact trace records must be exactly 32 bytes");
static_assert(sizeof(TraceRecord64) == 64, "forensic trace records must be exactly 64 bytes");
static_assert(std::is_trivially_copyable_v<TraceRecord32>);
static_assert(std::is_trivially_copyable_v<TraceRecord64>);

}  // namespace sidecar::trace

