#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace sidecar::trace {

inline constexpr std::size_t kCacheLineBytes = 64;

static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t));
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "Sidecar Flight Recorder requires lock-free 64-bit atomics");

// These states are deliberately separate types and separate cache lines. The
// ownership comments are part of the concurrency invariant, not decoration.
struct alignas(kCacheLineBytes) ProducerWriteState {
    // Written and read only by the producer.
    std::uint64_t value{0};
    std::array<std::byte, kCacheLineBytes - sizeof(value)> padding{};
};

struct alignas(kCacheLineBytes) PublishedSequenceState {
    // Release-written by the producer, acquire-read by the consumer.
    std::atomic<std::uint64_t> value{0};
    std::array<std::byte, kCacheLineBytes - sizeof(value)> padding{};
};

struct alignas(kCacheLineBytes) ConsumerReadState {
    // Written and read only by the consumer.
    std::uint64_t value{0};
    std::array<std::byte, kCacheLineBytes - sizeof(value)> padding{};
};

struct alignas(kCacheLineBytes) ConsumedSequenceState {
    // Release-written by the consumer, acquire-read by the producer.
    std::atomic<std::uint64_t> value{0};
    std::array<std::byte, kCacheLineBytes - sizeof(value)> padding{};
};

struct alignas(kCacheLineBytes) DroppedEventState {
    // Incremented only by the producer; relaxed reads are diagnostic.
    std::atomic<std::uint64_t> value{0};
    std::array<std::byte, kCacheLineBytes - sizeof(value)> padding{};
};

struct alignas(kCacheLineBytes) ProducerMetricsState {
    // Producer-private. Read only after the producer is quiescent.
    std::uint64_t high_water_mark{0};
    std::array<std::byte, kCacheLineBytes - sizeof(high_water_mark)> padding{};
};

static_assert(sizeof(ProducerWriteState) == kCacheLineBytes);
static_assert(sizeof(PublishedSequenceState) == kCacheLineBytes);
static_assert(sizeof(ConsumerReadState) == kCacheLineBytes);
static_assert(sizeof(ConsumedSequenceState) == kCacheLineBytes);
static_assert(sizeof(DroppedEventState) == kCacheLineBytes);
static_assert(sizeof(ProducerMetricsState) == kCacheLineBytes);

template <typename Record, std::size_t Capacity>
class SpscRing final {
    static_assert(std::is_trivially_copyable_v<Record>,
                  "Flight Recorder records must be trivially copyable");
    static_assert(Capacity >= 2, "an SPSC ring needs at least two slots");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "SPSC ring capacity must be a power of two");

public:
    using RecordType = Record;
    static constexpr std::size_t kCapacity = Capacity;

    SpscRing() : slots_(std::make_unique<Record[]>(Capacity)) {}

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    SpscRing(SpscRing&&) = delete;
    SpscRing& operator=(SpscRing&&) = delete;

    // Producer thread only. Record storage is fully written before the release
    // publication. No allocation, locks, blocking, I/O, or formatting occurs.
    [[nodiscard]] bool tryPush(const Record& record) noexcept {
        const std::uint64_t write_sequence = producer_write_.value;
        const std::uint64_t consumed_sequence =
            consumer_consumed_.value.load(std::memory_order_acquire);

        if (write_sequence - consumed_sequence >= Capacity) {
            dropped_events_.value.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        slots_[static_cast<std::size_t>(write_sequence) & (Capacity - 1)] = record;
        const std::uint64_t next_sequence = write_sequence + 1;
        producer_write_.value = next_sequence;

        const std::uint64_t occupancy = next_sequence - consumed_sequence;
        if (occupancy > producer_metrics_.high_water_mark) {
            producer_metrics_.high_water_mark = occupancy;
        }

        producer_published_.value.store(next_sequence, std::memory_order_release);
        return true;
    }

    // Consumer thread only. The acquire observes the complete record published
    // by tryPush; the release prevents the producer from reusing a slot early.
    [[nodiscard]] bool tryPop(Record& record) noexcept {
        const std::uint64_t read_sequence = consumer_read_.value;
        const std::uint64_t published_sequence =
            producer_published_.value.load(std::memory_order_acquire);

        if (read_sequence == published_sequence) {
            return false;
        }

        record = slots_[static_cast<std::size_t>(read_sequence) & (Capacity - 1)];
        const std::uint64_t next_sequence = read_sequence + 1;
        consumer_read_.value = next_sequence;
        consumer_consumed_.value.store(next_sequence, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::uint64_t droppedEvents() const noexcept {
        return dropped_events_.value.load(std::memory_order_relaxed);
    }

    // Call after the producer has stopped; this avoids adding another atomic
    // operation to the hot path solely for diagnostics.
    [[nodiscard]] std::uint64_t highWaterMarkQuiescent() const noexcept {
        return producer_metrics_.high_water_mark;
    }

    [[nodiscard]] std::uint64_t publishedSequence() const noexcept {
        return producer_published_.value.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::uint64_t consumedSequence() const noexcept {
        return consumer_consumed_.value.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::uint64_t producerSequenceQuiescent() const noexcept {
        return producer_write_.value;
    }

    [[nodiscard]] std::uint64_t consumerSequenceQuiescent() const noexcept {
        return consumer_read_.value;
    }

    [[nodiscard]] std::uint64_t sizeApproximate() const noexcept {
        // Diagnostic only: these atomics do not form a coherent snapshot. This
        // value must never control production backpressure, slot reuse, or
        // correctness decisions; tryPush/tryPop own those decisions. A
        // concurrent observation can appear reversed, so clamp instead of
        // exposing an unsigned underflow as a huge occupancy.
        const auto published = producer_published_.value.load(std::memory_order_acquire);
        const auto consumed = consumer_consumed_.value.load(std::memory_order_acquire);
        return consumed <= published ? published - consumed : 0;
    }

private:
    ProducerWriteState producer_write_;
    PublishedSequenceState producer_published_;
    ConsumerReadState consumer_read_;
    ConsumedSequenceState consumer_consumed_;
    DroppedEventState dropped_events_;
    ProducerMetricsState producer_metrics_;
    std::unique_ptr<Record[]> slots_;
};

}  // namespace sidecar::trace
