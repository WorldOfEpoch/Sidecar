#pragma once

#include "sidecar/trace/format.hpp"
#include "sidecar/trace/spsc_ring.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace sidecar::trace {

inline constexpr std::size_t kDefaultRingCapacity = 65536;
inline constexpr std::size_t kDefaultCollectorBatch = 4096;

[[nodiscard]] std::uint64_t CurrentThreadCpuTimeNs() noexcept;

class OperationIdAllocator final {
public:
    [[nodiscard]] std::uint64_t Allocate() {
        const auto id = next_.fetch_add(1, std::memory_order_relaxed);
        if (id == 0) {
            throw std::overflow_error("operation ID space exhausted");
        }
        return id;
    }

    [[nodiscard]] std::uint32_t AllocateCompact() {
        const auto id = Allocate();
        if (id > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("compact operation ID space exhausted");
        }
        return static_cast<std::uint32_t>(id);
    }

private:
    std::atomic<std::uint64_t> next_{1};
};

struct CollectorConfig {
    std::size_t batch_size{kDefaultCollectorBatch};
    std::uint32_t spin_iterations{256};
    std::uint32_t yield_iterations{64};
    std::chrono::microseconds idle_sleep{100};
    std::uint32_t flush_every_chunks{16};
};

struct RecorderMetrics {
    std::uint64_t records_written{0};
    std::uint64_t records_dropped{0};
    std::uint64_t high_water_mark{0};
    std::uint64_t chunks_written{0};
    std::uint64_t flush_count{0};
    std::uint64_t bytes_written{0};
    std::uint64_t collector_cpu_ns{0};
};

template <typename Record, std::size_t Capacity = kDefaultRingCapacity>
class FlightRecorder final {
public:
    class ProducerHandle final {
    public:
        [[nodiscard]] bool tryPush(const Record& record) noexcept {
            return ring_ != nullptr && ring_->tryPush(record);
        }
        [[nodiscard]] std::uint32_t producerId() const noexcept { return producer_id_; }

    private:
        friend class FlightRecorder;
        ProducerHandle(SpscRing<Record, Capacity>* ring, std::uint32_t producer_id)
            : ring_(ring), producer_id_(producer_id) {}
        SpscRing<Record, Capacity>* ring_{nullptr};
        std::uint32_t producer_id_{0};
    };

    explicit FlightRecorder(CollectorConfig config = {}) : config_(config) {
        if (config_.batch_size == 0) {
            throw std::invalid_argument("collector batch size must be nonzero");
        }
    }

    ~FlightRecorder() {
        if (running_.load(std::memory_order_relaxed)) {
            try { (void)Stop(); } catch (...) {}
        }
    }

    ProducerHandle RegisterProducer(std::string name, std::string type = "host") {
        if (started_) {
            throw std::logic_error("producer registration must precede Start");
        }
        const auto id = static_cast<std::uint32_t>(rings_.size());
        if constexpr (std::is_same_v<Record, TraceRecord32>) {
            if (id > std::numeric_limits<std::uint16_t>::max()) {
                throw std::overflow_error("compact trace producer ID space exhausted");
            }
        }
        rings_.push_back(std::make_unique<RingState>(id, std::move(name), std::move(type)));
        return ProducerHandle(&rings_.back()->ring, id);
    }

    void Start(const std::filesystem::path& path, TraceHeader header) {
        if (started_ || rings_.empty()) {
            throw std::logic_error("recorder requires registered producers and one Start");
        }
        header.producer_count = static_cast<std::uint32_t>(rings_.size());
        header.record_format = std::is_same_v<Record, TraceRecord32>
                                   ? RecordFormat::Compact32
                                   : RecordFormat::Forensic64;
        header.record_size = static_cast<std::uint16_t>(sizeof(Record));
        std::vector<MetadataEntry> metadata;
        metadata.reserve(rings_.size());
        for (const auto& state : rings_) {
            metadata.push_back({MetadataKind::Producer, state->id, state->name, state->type});
        }
        writer_ = std::make_unique<TraceWriter>(path, std::move(header), std::move(metadata));
        stop_requested_.store(false, std::memory_order_relaxed);
        running_.store(true, std::memory_order_release);
        started_ = true;
        collector_ = std::thread([this] {
            try { Collect(); }
            catch (...) { collector_error_ = std::current_exception(); }
        });
    }

    RecorderMetrics Stop() {
        if (!started_) {
            return metrics_;
        }
        stop_requested_.store(true, std::memory_order_release);
        if (collector_.joinable()) {
            collector_.join();
        }
        running_.store(false, std::memory_order_release);
        started_ = false;
        if (collector_error_) {
            auto error = collector_error_;
            collector_error_ = nullptr;
            std::rethrow_exception(error);
        }
        return metrics_;
    }

    [[nodiscard]] bool running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

private:
    struct RingState {
        RingState(std::uint32_t producer_id, std::string producer_name, std::string producer_type)
            : id(producer_id), name(std::move(producer_name)), type(std::move(producer_type)) {}
        std::uint32_t id;
        std::string name;
        std::string type;
        SpscRing<Record, Capacity> ring;
        std::uint64_t written_sequence{0};
    };

    void Collect() {
        const auto cpu_start = CurrentThreadCpuTimeNs();
        std::vector<Record> batch;
        batch.reserve(config_.batch_size);
        std::uint32_t idle = 0;
        std::uint64_t flushes = 0;
        bool draining_after_stop = false;
        for (;;) {
            bool any = false;
            for (auto& state : rings_) {
                batch.clear();
                Record record{};
                while (batch.size() < config_.batch_size && state->ring.tryPop(record)) {
                    batch.push_back(record);
                }
                if (!batch.empty()) {
                    any = true;
                    writer_->Append(state->id, state->written_sequence, batch);
                    state->written_sequence += batch.size();
                    ++flushes;
                    if (config_.flush_every_chunks != 0 &&
                        flushes % config_.flush_every_chunks == 0) {
                        writer_->Flush();
                    }
                }
            }

            if (stop_requested_.load(std::memory_order_acquire)) {
                if (!any) {
                    if (draining_after_stop) {
                        break;
                    }
                    // One authoritative empty pass after observing stop covers
                    // final publications without diagnostic occupancy checks.
                    draining_after_stop = true;
                } else {
                    draining_after_stop = false;
                }
            }
            if (any) {
                idle = 0;
            } else if (idle < config_.spin_iterations) {
                ++idle;
            } else if (idle < config_.spin_iterations + config_.yield_iterations) {
                ++idle;
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(config_.idle_sleep);
            }
        }

        metrics_.records_written = 0;
        metrics_.records_dropped = 0;
        metrics_.high_water_mark = 0;
        for (const auto& state : rings_) {
            metrics_.records_written += state->written_sequence;
            metrics_.records_dropped += state->ring.droppedEvents();
            metrics_.high_water_mark =
                std::max(metrics_.high_water_mark, state->ring.highWaterMarkQuiescent());
        }
        metrics_.chunks_written = writer_->chunksWritten();
        metrics_.flush_count = flushes;
        metrics_.collector_cpu_ns = CurrentThreadCpuTimeNs() - cpu_start;
        TraceFooter footer{metrics_.records_written,
                           metrics_.records_dropped,
                           metrics_.high_water_mark,
                           metrics_.chunks_written,
                           metrics_.flush_count,
                           writer_->bytesWritten(),
                           metrics_.collector_cpu_ns,
                           static_cast<std::uint64_t>(std::chrono::duration_cast<
                               std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch()).count())};
        writer_->Finalize(footer);
        metrics_.bytes_written = writer_->bytesWritten();
    }

    CollectorConfig config_;
    std::vector<std::unique_ptr<RingState>> rings_;
    std::unique_ptr<TraceWriter> writer_;
    std::thread collector_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};
    bool started_{false};
    RecorderMetrics metrics_;
    std::exception_ptr collector_error_;
};

}  // namespace sidecar::trace
