#include "sidecar/pipeline/physics.hpp"

#include "sidecar/core/sha256.hpp"
#include "sidecar/memory/host_memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace sidecar::pipeline {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}

std::uint64_t ProcessCpuNs() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    ULARGE_INTEGER k{}, u{};
    k.LowPart = kernel.dwLowDateTime; k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime; u.HighPart = user.dwHighDateTime;
    return (k.QuadPart + u.QuadPart) * 100ULL;
#else
    return 0;
#endif
}

PipelineStatus StorageStatus(storage::RunStatus status) {
    switch (status) {
        case storage::RunStatus::Success: return PipelineStatus::Success;
        case storage::RunStatus::SkippedUnsupported: return PipelineStatus::SkippedUnsupported;
        case storage::RunStatus::SkippedSafetyLimit: return PipelineStatus::SkippedSafetyLimit;
        case storage::RunStatus::InvalidConfiguration: return PipelineStatus::InvalidConfiguration;
        case storage::RunStatus::TimedOut: return PipelineStatus::TimedOut;
        case storage::RunStatus::Cancelled: return PipelineStatus::Cancelled;
        case storage::RunStatus::VerificationFailure:
            return PipelineStatus::DataVerificationFailure;
        case storage::RunStatus::AbortedHealth: return PipelineStatus::HardwareHealthChange;
        case storage::RunStatus::IoError: return PipelineStatus::StorageError;
    }
    return PipelineStatus::StorageError;
}

std::uint64_t ToNs(float milliseconds) {
    return static_cast<std::uint64_t>(std::llround(milliseconds * 1'000'000.0));
}

std::string CudaMessage(cudaError_t error, const char* operation) {
    return std::string(operation) + ": " + cudaGetErrorString(error);
}

bool UsesStorage(ContentionTest test) {
    return test == ContentionTest::B || test == ContentionTest::E ||
           test == ContentionTest::F || test == ContentionTest::H ||
           test == ContentionTest::I || test == ContentionTest::J;
}
bool UsesCopy(ContentionTest test) {
    return test == ContentionTest::C || test == ContentionTest::F ||
           test == ContentionTest::G || test == ContentionTest::I;
}
bool UsesH2D(ContentionTest test) {
    return test == ContentionTest::D || test == ContentionTest::G ||
           test == ContentionTest::H || test == ContentionTest::I ||
           test == ContentionTest::J;
}
bool UsesCompute(ContentionTest test) {
    return test == ContentionTest::A || test == ContentionTest::I ||
           test == ContentionTest::J;
}
bool UsesPageable(ContentionTest test) {
    return test == ContentionTest::B || test == ContentionTest::C ||
           test == ContentionTest::F || test == ContentionTest::G ||
           test == ContentionTest::I;
}

void ParallelCopy(void* destination, const void* source, std::uint64_t bytes,
                  std::uint32_t workers) {
    if (workers <= 1 || bytes < workers * 4096ULL) {
        std::memcpy(destination, source, static_cast<std::size_t>(bytes));
        return;
    }
    std::vector<std::jthread> threads;
    threads.reserve(workers);
    const auto portion = bytes / workers;
    for (std::uint32_t worker = 0; worker < workers; ++worker) {
        const auto begin = portion * worker;
        const auto length = worker + 1 == workers ? bytes - begin : portion;
        threads.emplace_back([=] {
            std::memcpy(static_cast<std::byte*>(destination) + begin,
                        static_cast<const std::byte*>(source) + begin,
                        static_cast<std::size_t>(length));
        });
    }
}

struct Arena {
    memory::IHostMemoryProvider& host{memory::NativeHostMemoryProvider()};
    memory::MemoryRegion pageable;
    memory::MemoryRegion pinned;
    memory::MemoryRegion verify;
    void* device{nullptr};
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> starts;
    std::vector<cudaEvent_t> ends;

    ~Arena() { Release(); }
    void Release() noexcept {
        for (auto event : starts) if (event) (void)cudaEventDestroy(event);
        for (auto event : ends) if (event) (void)cudaEventDestroy(event);
        for (auto stream : streams) if (stream) (void)cudaStreamDestroy(stream);
        starts.clear(); ends.clear(); streams.clear();
        if (device) (void)cudaFree(device);
        device = nullptr;
        (void)host.FreePageable(verify);
        (void)host.FreeCudaHost(pinned);
        (void)host.FreePageable(pageable);
    }
};

struct RuntimeSlot {
    SlotStateMachine machine;
    std::uint32_t block{0};
    std::uint64_t file_offset{0};
    std::uint64_t storage_submit_relative{0};
    std::uint64_t storage_complete_relative{0};
    std::uint64_t h2d_submit_relative{0};
    bool occupied{false};
    explicit RuntimeSlot(std::uint32_t id) : machine(id) {}
};

struct OneRun {
    PipelineSample sample;
    std::vector<StageSample> stages;
    std::vector<std::vector<double>> slot_copy;
    std::vector<std::vector<double>> slot_h2d;
    std::vector<std::uint64_t> slot_transfers;
    std::vector<std::uint64_t> slot_hits;
    std::vector<std::uint64_t> slot_misses;
};

class NativePipelineProvider final : public IPipelineProvider {
public:
    ~NativePipelineProvider() override { (void)cuda::NativeOverlapProvider().Release(); }
    bool SupportsCuda() const noexcept override { return true; }

    HostCopyResult RunHostCopy(const HostCopyConfiguration& configuration) override {
        HostCopyResult result;
        result.configuration = configuration;
        if (!configuration.bytes || !configuration.repetitions ||
            configuration.worker_count < 1 || configuration.worker_count > 4) {
            result.status = PipelineStatus::InvalidConfiguration;
            result.message = "invalid host-copy configuration";
            return result;
        }
        auto& host = memory::NativeHostMemoryProvider();
        memory::MemoryRegion source, destination;
        auto allocation = host.AllocatePageable(configuration.bytes, source);
        if (allocation.ok()) allocation = host.AllocateCudaHost(configuration.bytes, destination);
        if (!allocation.ok()) {
            (void)host.FreePageable(source);
            result.status = PipelineStatus::SkippedUnsupported;
            result.message = allocation.message;
            return result;
        }
        storage::FillDatasetBytes(source.data, 0, configuration.bytes);
        std::uint64_t checksum = 0;
        (void)host.Touch(destination, 0, checksum);

        cuda::WorkloadProfile profile;
        auto& compute = cuda::NativeOverlapProvider();
        if (configuration.concurrent_compute) {
            const auto prepared = compute.Prepare(0, cuda::HostMemoryClass::PinnedHostAlloc, 4096);
            if (!prepared.ok()) {
                result.status = PipelineStatus::ComputeError;
                result.message = prepared.message;
                (void)host.FreeCudaHost(destination); (void)host.FreePageable(source);
                return result;
            }
            profile = compute.Calibrate(*configuration.concurrent_compute,
                                        configuration.compute_window_us, 3, 7);
            if (profile.status != cuda::OverlapStatus::Success) {
                result.status = PipelineStatus::ComputeError;
                result.message = profile.message;
                (void)host.FreeCudaHost(destination); (void)host.FreePageable(source);
                return result;
            }
        }
        const auto copy_once = [&](std::uint32_t index, bool record) {
            std::future<cuda::RawOverlapTiming> compute_future;
            std::promise<void> entered;
            auto entered_future = entered.get_future();
            if (configuration.concurrent_compute) {
                compute_future = std::async(std::launch::async, [&] {
                    entered.set_value();
                    return compute.RunTopology(profile, cuda::TransferDirection::H2D, 1,
                        cuda::SampleMode::ComputeOnly, cuda::InstrumentationMode::Full,
                        5'000'000, 500'000, index);
                });
                entered_future.wait();
                // WU6's dependency gate is five milliseconds. Start the host copy
                // shortly before release so the copy and calibrated kernel overlap.
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
            const auto cpu_before = ProcessCpuNs();
            const auto started = NowNs();
            ParallelCopy(destination.data, source.data, configuration.bytes,
                         configuration.worker_count);
            const auto finished = NowNs();
            const auto cpu_after = ProcessCpuNs();
            cuda::RawOverlapTiming compute_timing;
            if (configuration.concurrent_compute) compute_timing = compute_future.get();
            if (record) {
                HostCopySample sample;
                sample.sample_index = index;
                sample.wall_ns = finished - started;
                sample.process_cpu_ns = cpu_after >= cpu_before ? cpu_after - cpu_before : 0;
                sample.compute_ns = compute_timing.compute_ns;
                sample.bytes_per_second = sample.wall_ns
                    ? static_cast<double>(configuration.bytes) * 1e9 / sample.wall_ns : 0;
                if (configuration.concurrent_compute &&
                    compute_timing.status != cuda::OverlapStatus::Success) {
                    sample.status = PipelineStatus::ComputeError;
                    sample.message = compute_timing.message;
                }
                result.samples.push_back(std::move(sample));
            }
        };
        for (std::uint32_t i = 0; i < configuration.warmups; ++i) copy_once(i, false);
        for (std::uint32_t i = 0; i < configuration.repetitions; ++i) copy_once(i, true);
        if (!storage::VerifyDatasetBytes(destination.data, 0, configuration.bytes)) {
            result.status = PipelineStatus::DataVerificationFailure;
            result.message = "pageable-to-pinned post-timing verification failed";
        }
        (void)host.FreeCudaHost(destination); (void)host.FreePageable(source);
        FinalizeHostCopy(result);
        return result;
    }

    PipelineResult Run(const storage::StorageTarget& target,
                       const PipelineConfiguration& configuration) override {
        PipelineResult result;
        result.configuration = configuration;
        if (const auto issue = ValidateConfiguration(configuration, target,
                                                      storage::kDefaultDatasetBytes)) {
            result.status = PipelineStatus::InvalidConfiguration;
            result.message = *issue;
            return result;
        }
        if (cudaSetDevice(configuration.device_index) != cudaSuccess) {
            result.status = PipelineStatus::SkippedUnsupported;
            result.message = "cudaSetDevice failed";
            (void)cudaGetLastError();
            return result;
        }
        auto storage_provider = storage::CreateNativeStorageProvider();
        result.health_before = storage_provider->CaptureHealth(target, "WU8_BEFORE");
        result.host_before = memory::NativeHostMemoryProvider().Snapshot();
        auto& compute = cuda::NativeOverlapProvider();
        const bool compute_required = UsesCompute(configuration.contention_test) &&
                                      configuration.phase != PipelinePhase::Baseline;
        cuda::WorkloadProfile profile;
        if (compute_required) {
            const auto prepared = compute.Prepare(configuration.device_index,
                cuda::HostMemoryClass::PinnedHostAlloc, 4096);
            if (!prepared.ok()) {
                result.status = PipelineStatus::ComputeError;
                result.message = prepared.message;
                return result;
            }
            profile = compute.Calibrate(configuration.compute_workload,
                                        configuration.compute_window_us, 3, 9);
            if (profile.status != cuda::OverlapStatus::Success) {
                result.status = PipelineStatus::ComputeError;
                result.message = profile.message;
                return result;
            }
        }
        result.gpu_before = compute.ReadTelemetry(configuration.device_index);

        Arena arena;
        const auto slot_bytes = configuration.chunk_bytes * configuration.buffer_depth;
        auto allocation = memory::ProviderResult{};
        if (UsesPageable(configuration.contention_test))
            allocation = arena.host.AllocatePageable(slot_bytes, arena.pageable);
        if (allocation.ok()) allocation = arena.host.AllocateCudaHost(slot_bytes, arena.pinned);
        if (allocation.ok())
            allocation = arena.host.AllocatePageable(configuration.aggregate_bytes, arena.verify);
        if (!allocation.ok()) {
            result.status = PipelineStatus::SkippedSafetyLimit;
            result.message = allocation.message;
            return result;
        }
        const auto device_error = cudaMalloc(&arena.device,
            static_cast<std::size_t>(configuration.aggregate_bytes));
        if (device_error != cudaSuccess) {
            result.status = PipelineStatus::SkippedSafetyLimit;
            result.message = CudaMessage(device_error, "cudaMalloc(pipeline destination)");
            return result;
        }
        arena.streams.resize(configuration.buffer_depth);
        arena.starts.resize(configuration.buffer_depth);
        arena.ends.resize(configuration.buffer_depth);
        for (std::uint32_t slot = 0; slot < configuration.buffer_depth; ++slot) {
            auto error = cudaStreamCreateWithFlags(&arena.streams[slot], cudaStreamNonBlocking);
            if (error == cudaSuccess) error = cudaEventCreate(&arena.starts[slot]);
            if (error == cudaSuccess) error = cudaEventCreate(&arena.ends[slot]);
            if (error != cudaSuccess) {
                result.status = PipelineStatus::H2DError;
                result.message = CudaMessage(error, "pipeline stream/event allocation");
                return result;
            }
        }
        std::uint64_t touch = 0;
        if (arena.pageable.data) (void)arena.host.Touch(arena.pageable, 0, touch);
        (void)arena.host.Touch(arena.pinned, 0, touch);
        (void)arena.host.Touch(arena.verify, 0, touch);
        {
            std::ostringstream material;
            material << "WU8-ARENA-V1:" << NowNs() << ':' << configuration.chunk_bytes << ':'
                     << configuration.buffer_depth;
            result.arena_allocation_id = core::Sha256Hex(material.str()).substr(0, 24);
        }
        result.slots.resize(configuration.buffer_depth);
        for (std::uint32_t slot = 0; slot < configuration.buffer_depth; ++slot) {
            result.slots[slot].arena_allocation_id = result.arena_allocation_id;
            result.slots[slot].slot_id = slot;
            result.slots[slot].arena_offset = configuration.chunk_bytes * slot;
            result.slots[slot].bytes = configuration.chunk_bytes;
        }
        std::vector<std::vector<double>> aggregate_copy(configuration.buffer_depth);
        std::vector<std::vector<double>> aggregate_h2d(configuration.buffer_depth);

        const auto run_compute_only = [&](std::uint32_t repetition) {
            return compute.RunTopology(profile, cuda::TransferDirection::H2D, 1,
                cuda::SampleMode::ComputeOnly, cuda::InstrumentationMode::Full,
                5'000'000, 500'000, repetition);
        };
        const auto repetitions = configuration.stream_seconds ?
            std::numeric_limits<std::uint32_t>::max() : configuration.repetitions;
        const auto stream_deadline = configuration.stream_seconds
            ? Clock::now() + std::chrono::seconds(configuration.stream_seconds)
            : Clock::time_point::max();

        for (std::uint32_t repetition = 0;
             repetition < repetitions && Clock::now() < stream_deadline; ++repetition) {
            cuda::RawOverlapTiming c0;
            if (compute_required) c0 = run_compute_only(repetition);
            if (compute_required && c0.status != cuda::OverlapStatus::Success) {
                result.status = PipelineStatus::ComputeError;
                result.message = c0.message;
                break;
            }
            auto p0 = ExecuteStages(target, configuration, arena, repetition, false, nullptr);
            if (p0.sample.status != PipelineStatus::Success) {
                result.status = p0.sample.status;
                result.message = p0.sample.message;
                result.samples.push_back(std::move(p0.sample));
                break;
            }
            OneRun concurrent;
            cuda::RawOverlapTiming cc;
            const bool run_compute = compute_required;
            std::uint64_t makespan = 0;
            if (run_compute) {
                std::atomic_bool compute_gate_released{false};
                std::atomic_bool compute_failed_before_gate{false};
                auto future = std::async(std::launch::async, [&] {
                    auto timing=compute.RunTopology(profile, cuda::TransferDirection::H2D, 1,
                        cuda::SampleMode::ComputeOnly, cuda::InstrumentationMode::Full,
                        5'000'000, 500'000, repetition, &compute_gate_released);
                    if (!compute_gate_released.load(std::memory_order_acquire)) {
                        compute_failed_before_gate.store(true, std::memory_order_release);
                        compute_gate_released.store(true, std::memory_order_release);
                        compute_gate_released.notify_all();
                    }
                    return timing;
                });
                compute_gate_released.wait(false, std::memory_order_acquire);
                if (compute_failed_before_gate.load(std::memory_order_acquire)) {
                    cc = future.get();
                    result.status = PipelineStatus::ComputeError;
                    result.message = cc.message.empty()
                        ? "compute gate failed before release" : cc.message;
                    break;
                }
                // The CUDA host gate publishes immediately before releasing the
                // compute dependency event. This is the shared host-clock origin
                // for compute and the external NVMe/host/H2D pipeline.
                const auto makespan_start = NowNs();
                const auto pipeline_start = NowNs();
                concurrent = ExecuteStages(target, configuration, arena, repetition, true, nullptr);
                cc = future.get();
                if (cc.status != cuda::OverlapStatus::Success) {
                    result.status = PipelineStatus::ComputeError;
                    result.message = cc.message.empty()
                        ? "concurrent compute failed" : cc.message;
                    break;
                }
                // ExecuteStages performs full D2H verification after recording
                // sample.pc_ns. Verification must not contaminate M, so combine
                // the two branch completion durations from the common gate
                // rather than timing the function return.
                makespan = CalculateCommonTimelineMakespan(cc.makespan_host_ns,
                    pipeline_start - makespan_start, concurrent.sample.pc_ns);
                const auto expected_floor = std::max(cc.compute_ns, concurrent.sample.pc_ns);
                if (static_cast<double>(makespan) <
                    static_cast<double>(expected_floor) * 0.90) {
                    result.status = PipelineStatus::InternalError;
                    result.message = "common host timeline ended before a measured branch";
                    break;
                }
            } else {
                concurrent = std::move(p0);
                cc = c0;
                makespan = concurrent.sample.pc_ns;
            }
            auto& sample = concurrent.sample;
            sample.sample_index = repetition;
            sample.c0_reference_ns = c0.compute_ns;
            sample.p0_reference_ns = run_compute ? p0.sample.pc_ns : sample.pc_ns;
            sample.cc_ns = run_compute ? cc.compute_ns : c0.compute_ns;
            sample.makespan_ns = makespan;
            sample.metrics = CalculatePipelineMetrics(static_cast<double>(sample.c0_reference_ns),
                static_cast<double>(sample.p0_reference_ns), static_cast<double>(sample.cc_ns),
                static_cast<double>(sample.pc_ns), static_cast<double>(sample.makespan_ns));
            sample.ready_ahead_ns = CalculateReadyAhead(configuration.deadline_ns, sample.pc_ns);
            sample.deadline_hit = sample.ready_ahead_ns >= 0;
            for (std::size_t slot = 0; slot < result.slots.size(); ++slot) {
                result.slots[slot].transfer_count += concurrent.slot_transfers[slot];
                result.slots[slot].deadline_hits += concurrent.slot_hits[slot];
                result.slots[slot].deadline_misses += concurrent.slot_misses[slot];
                aggregate_copy[slot].insert(aggregate_copy[slot].end(),
                    concurrent.slot_copy[slot].begin(), concurrent.slot_copy[slot].end());
                aggregate_h2d[slot].insert(aggregate_h2d[slot].end(),
                    concurrent.slot_h2d[slot].begin(), concurrent.slot_h2d[slot].end());
            }
            result.stages.insert(result.stages.end(), concurrent.stages.begin(),
                                 concurrent.stages.end());
            result.samples.push_back(std::move(sample));
            if (result.samples.back().status != PipelineStatus::Success) {
                result.status = result.samples.back().status;
                result.message = result.samples.back().message;
                break;
            }
        }
        for (std::size_t slot = 0; slot < result.slots.size(); ++slot) {
            result.slots[slot].host_copy_ns = trace::CalculateDistribution(aggregate_copy[slot]);
            result.slots[slot].h2d_ns = trace::CalculateDistribution(aggregate_h2d[slot]);
        }
        std::vector<double> all_h2d;
        for (const auto& values : aggregate_h2d)
            all_h2d.insert(all_h2d.end(), values.begin(), values.end());
        const auto overall_h2d = trace::CalculateDistribution(std::move(all_h2d));
        for (auto& slot : result.slots) {
            if (slot.transfer_count >= 20 && overall_h2d.median > 0 &&
                std::abs(slot.h2d_ns.median / overall_h2d.median - 1.0) > 0.20)
                slot.observation = "REGION_VARIATION_OBSERVED";
        }
        result.gpu_after = compute.ReadTelemetry(configuration.device_index);
        result.host_after = memory::NativeHostMemoryProvider().Snapshot();
        result.health_after = storage_provider->CaptureHealth(target, "WU8_AFTER");
        const auto increased = [](const auto& before, const auto& after) {
            return before && after && *after > *before;
        };
        if ((result.health_after.critical_warning && *result.health_after.critical_warning) ||
            increased(result.health_before.media_data_errors_low64,
                      result.health_after.media_data_errors_low64) ||
            increased(result.health_before.error_log_entries_low64,
                      result.health_after.error_log_entries_low64)) {
            result.status = PipelineStatus::HardwareHealthChange;
            result.message = "serious NVMe health counter changed during WU8 run";
        }
        FinalizePipeline(result);
        return result;
    }

private:
    OneRun ExecuteStages(const storage::StorageTarget& target,
                         const PipelineConfiguration& configuration, Arena& arena,
                         std::uint32_t repetition, bool concurrent,
                         std::atomic_bool* cancellation) {
        OneRun run;
        run.slot_copy.resize(configuration.buffer_depth);
        run.slot_h2d.resize(configuration.buffer_depth);
        run.slot_transfers.resize(configuration.buffer_depth);
        run.slot_hits.resize(configuration.buffer_depth);
        run.slot_misses.resize(configuration.buffer_depth);
        auto& sample = run.sample;
        sample.sample_index = repetition;
        const bool storage_enabled = UsesStorage(configuration.contention_test);
        const bool copy_enabled = UsesCopy(configuration.contention_test);
        const bool h2d_enabled = UsesH2D(configuration.contention_test);
        const auto block_count = static_cast<std::uint32_t>(
            configuration.aggregate_bytes / configuration.chunk_bytes);
        if (!storage_enabled && !copy_enabled && !h2d_enabled) {
            sample.status = PipelineStatus::Success;
            sample.verified = true;
            return run;
        }
        std::vector<RuntimeSlot> slots;
        for (std::uint32_t slot = 0; slot < configuration.buffer_depth; ++slot)
            slots.emplace_back(slot);
        if (!storage_enabled) {
            for (std::uint32_t slot = 0; slot < configuration.buffer_depth; ++slot) {
                const auto offset = static_cast<std::uint64_t>(slot) * configuration.chunk_bytes;
                void* destination = UsesPageable(configuration.contention_test)
                    ? static_cast<std::byte*>(arena.pageable.data) + offset
                    : static_cast<std::byte*>(arena.pinned.data) + offset;
                storage::FillDatasetBytes(destination, offset, configuration.chunk_bytes);
            }
        }
        std::unique_ptr<storage::IAsyncStorageReader> reader;
        if (storage_enabled) {
            try {
                reader = storage::CreateNativeAsyncStorageReader(target,
                                                                 configuration.buffer_depth);
            } catch (const std::exception& error) {
                sample.status = PipelineStatus::StorageError;
                sample.message = error.what();
                return run;
            }
        }
        std::uint32_t next_block = 0, ready_blocks = 0;
        std::uint64_t last_storage_submit = 0, first_ready = 0, previous_ready = 0;
        std::vector<std::uint64_t> block_offsets(block_count);
        std::vector<std::uint64_t> block_ready(block_count);
        const auto dataset_span = storage::kDefaultDatasetBytes - configuration.aggregate_bytes;
        const auto base_offset = dataset_span
            ? (static_cast<std::uint64_t>(repetition) * configuration.aggregate_bytes) % dataset_span
            : 0;
        const auto wall_start = NowNs();

        auto finish_without_h2d = [&](RuntimeSlot& slot, std::uint64_t now) {
            slot.machine.Transition(SlotState::VerifyPending);
            block_ready[slot.block] = now - wall_start;
            if (!first_ready) first_ready = block_ready[slot.block];
            if (previous_ready) sample.steady_state_interval_ns += block_ready[slot.block] - previous_ready;
            previous_ready = block_ready[slot.block];
            ++ready_blocks;
            const bool hit = block_ready[slot.block] <= configuration.deadline_ns;
            if (hit) ++run.slot_hits[slot.machine.slotId()]; else ++run.slot_misses[slot.machine.slotId()];
            slot.machine.Transition(SlotState::Free);
            slot.occupied = false;
        };

        auto process_source = [&](RuntimeSlot& slot, std::uint64_t source_ready) {
            const auto id = slot.machine.slotId();
            const auto slot_offset = static_cast<std::uint64_t>(id) * configuration.chunk_bytes;
            void* pinned = static_cast<std::byte*>(arena.pinned.data) + slot_offset;
            if (copy_enabled) {
                slot.machine.Transition(SlotState::HostCopyPending);
                slot.machine.Transition(SlotState::HostCopyActive);
                const auto begin = NowNs();
                ParallelCopy(pinned, static_cast<std::byte*>(arena.pageable.data) + slot_offset,
                             configuration.chunk_bytes, configuration.host_copy_workers);
                const auto end = NowNs();
                const auto duration = end - begin;
                sample.host_copy_ns += duration;
                run.slot_copy[id].push_back(static_cast<double>(duration));
                run.stages.push_back({repetition, slot.block, id, "HOST_COPY",
                    configuration.chunk_bytes, slot.file_offset, begin - wall_start,
                    begin - wall_start, end - wall_start, 0, begin > source_ready ? begin-source_ready:0,
                    PipelineStatus::Success, 0});
                slot.machine.Transition(h2d_enabled ? SlotState::H2DReady
                                                   : SlotState::VerifyPending);
            } else if (h2d_enabled) {
                slot.machine.Transition(SlotState::H2DReady);
            }
            if (!h2d_enabled) {
                if (!copy_enabled) slot.machine.Transition(SlotState::VerifyPending);
                block_ready[slot.block] = NowNs() - wall_start;
                if (!first_ready) first_ready = block_ready[slot.block];
                if (previous_ready)
                    sample.steady_state_interval_ns += block_ready[slot.block] - previous_ready;
                previous_ready = block_ready[slot.block];
                ++ready_blocks;
                const bool hit = block_ready[slot.block] <= configuration.deadline_ns;
                if (hit) ++run.slot_hits[id]; else ++run.slot_misses[id];
                slot.machine.Transition(SlotState::Free);
                slot.occupied = false;
                return;
            }
            const auto submit = NowNs();
            auto error = cudaEventRecord(arena.starts[id], arena.streams[id]);
            if (error == cudaSuccess)
                error = cudaMemcpyAsync(static_cast<std::byte*>(arena.device) +
                    static_cast<std::uint64_t>(slot.block) * configuration.chunk_bytes,
                    pinned, static_cast<std::size_t>(configuration.chunk_bytes),
                    cudaMemcpyHostToDevice, arena.streams[id]);
            if (error == cudaSuccess) error = cudaEventRecord(arena.ends[id], arena.streams[id]);
            if (error != cudaSuccess) {
                sample.status = PipelineStatus::H2DError;
                sample.native_error = error;
                sample.message = CudaMessage(error, "cudaMemcpyAsync(pipeline)");
                return;
            }
            slot.h2d_submit_relative = submit - wall_start;
            slot.machine.Transition(SlotState::H2DActive);
        };

        auto submit_available = [&] {
            bool submitted = false;
            for (auto& slot : slots) {
                if (next_block >= block_count) break;
                if (slot.machine.state() != SlotState::Free) continue;
                slot.block = next_block++;
                slot.file_offset = base_offset +
                    static_cast<std::uint64_t>(slot.block) * configuration.chunk_bytes;
                if (!storage_enabled)
                    slot.file_offset = static_cast<std::uint64_t>(slot.machine.slotId()) *
                                       configuration.chunk_bytes;
                block_offsets[slot.block] = slot.file_offset;
                slot.occupied = true;
                sample.slot_ids.push_back(slot.machine.slotId());
                if (storage_enabled) {
                    slot.machine.Transition(SlotState::NvmeReading);
                    const auto slot_offset = static_cast<std::uint64_t>(slot.machine.slotId()) *
                                             configuration.chunk_bytes;
                    void* destination = UsesPageable(configuration.contention_test)
                        ? static_cast<std::byte*>(arena.pageable.data) + slot_offset
                        : static_cast<std::byte*>(arena.pinned.data) + slot_offset;
                    const auto submitted_at = NowNs();
                    auto submitted_io = reader->Submit({slot.machine.slotId(), slot.file_offset,
                                                        configuration.chunk_bytes, destination});
                    if (submitted_io.status != storage::RunStatus::Success) {
                        sample.status = StorageStatus(submitted_io.status);
                        sample.native_error = submitted_io.native_error;
                        sample.message = "IOCP pipeline submission failed";
                        return submitted;
                    }
                    slot.storage_submit_relative = submitted_at - wall_start;
                    last_storage_submit = slot.storage_submit_relative;
                } else {
                    // A deterministic prefilled control enters at the same logical
                    // source-ready boundary as an IOCP completion.
                    slot.machine.Transition(SlotState::NvmeReading);
                    process_source(slot, NowNs());
                }
                submitted = true;
                if (sample.status != PipelineStatus::Success) return submitted;
            }
            return submitted;
        };

        submit_available();
        while (ready_blocks < block_count && sample.status == PipelineStatus::Success) {
            if ((cancellation && cancellation->load(std::memory_order_relaxed)) ||
                (configuration.cancel_after_blocks &&
                 ready_blocks >= *configuration.cancel_after_blocks)) {
                sample.status = PipelineStatus::Cancelled;
                sample.message = "pipeline cancellation requested";
                if (reader) reader->Cancel();
                break;
            }
            bool progressed = false;
            for (auto& slot : slots) {
                if (slot.machine.state() != SlotState::H2DActive) continue;
                const auto query = cudaEventQuery(arena.ends[slot.machine.slotId()]);
                if (query == cudaErrorNotReady) { (void)cudaGetLastError(); continue; }
                if (query != cudaSuccess) {
                    sample.status = PipelineStatus::H2DError;
                    sample.native_error = query;
                    sample.message = CudaMessage(query, "cudaEventQuery(pipeline H2D)");
                    break;
                }
                float milliseconds = 0;
                const auto elapsed = cudaEventElapsedTime(&milliseconds,
                    arena.starts[slot.machine.slotId()], arena.ends[slot.machine.slotId()]);
                if (elapsed != cudaSuccess) {
                    sample.status = PipelineStatus::H2DError;
                    sample.message = CudaMessage(elapsed, "cudaEventElapsedTime(pipeline H2D)");
                    break;
                }
                const auto duration = ToNs(milliseconds);
                sample.h2d_ns += duration;
                run.slot_h2d[slot.machine.slotId()].push_back(static_cast<double>(duration));
                run.stages.push_back({repetition, slot.block, slot.machine.slotId(), "H2D",
                    configuration.chunk_bytes, slot.file_offset, slot.h2d_submit_relative,
                    slot.h2d_submit_relative, NowNs() - wall_start, duration,
                    slot.h2d_submit_relative > slot.storage_complete_relative
                        ? slot.h2d_submit_relative - slot.storage_complete_relative : 0,
                    PipelineStatus::Success, 0});
                slot.machine.Transition(SlotState::GpuReady);
                ++run.slot_transfers[slot.machine.slotId()];
                finish_without_h2d(slot, NowNs());
                progressed = true;
            }
            if (sample.status != PipelineStatus::Success) break;
            if (submit_available()) progressed = true;
            if (sample.status != PipelineStatus::Success) break;
            if (storage_enabled && reader->Outstanding()) {
                const auto wait_start = NowNs();
                auto completion = reader->Wait(configuration.timeout_ms);
                const auto wait_end = NowNs();
                if (completion.status != storage::RunStatus::Success) {
                    sample.status = StorageStatus(completion.status);
                    sample.native_error = completion.native_error;
                    sample.message = "IOCP pipeline completion failed";
                    break;
                }
                if (completion.token >= slots.size()) {
                    sample.status = PipelineStatus::StorageError;
                    sample.message = "IOCP returned invalid pipeline slot token";
                    break;
                }
                auto& slot = slots[static_cast<std::size_t>(completion.token)];
                slot.storage_complete_relative = completion.completed_ns - wall_start;
                const auto duration = completion.completed_ns - completion.submitted_ns;
                sample.storage_ns += duration;
                run.stages.push_back({repetition, slot.block, slot.machine.slotId(), "NVME",
                    configuration.chunk_bytes, slot.file_offset, slot.storage_submit_relative,
                    slot.storage_submit_relative, slot.storage_complete_relative, 0,
                    wait_end - wait_start, PipelineStatus::Success, 0});
                process_source(slot, completion.completed_ns);
                progressed = true;
            }
            if (!progressed) {
                RuntimeSlot* active = nullptr;
                for (auto& slot : slots)
                    if (slot.machine.state() == SlotState::H2DActive) { active = &slot; break; }
                if (active) {
                    const auto wait_begin = NowNs();
                    const auto error = cudaEventSynchronize(arena.ends[active->machine.slotId()]);
                    sample.pinned_slot_wait_ns += NowNs() - wait_begin;
                    if (error != cudaSuccess) {
                        sample.status = PipelineStatus::H2DError;
                        sample.message = CudaMessage(error, "cudaEventSynchronize(pipeline)");
                    }
                } else if (ready_blocks < block_count) {
                    sample.status = PipelineStatus::InternalError;
                    sample.message = "pipeline made no progress";
                }
            }
        }
        if (reader && reader->Outstanding()) reader->Cancel();
        if (block_count > 1) sample.steady_state_interval_ns /= block_count - 1;
        sample.fill_latency_ns = first_ready;
        sample.pc_ns = NowNs() - wall_start;
        sample.drain_latency_ns = sample.pc_ns > last_storage_submit
            ? sample.pc_ns - last_storage_submit : 0;
        sample.gpu_waiting_for_data_ns = sample.pc_ns > configuration.deadline_ns
            ? sample.pc_ns - configuration.deadline_ns : 0;
        sample.nvme_queue_starved_ns = sample.pinned_slot_wait_ns;
        sample.bytes_per_second = sample.pc_ns
            ? static_cast<double>(configuration.aggregate_bytes) * 1e9 / sample.pc_ns : 0;

        if (sample.status == PipelineStatus::Success) {
            bool verified = true;
            if (h2d_enabled) {
                const auto error = cudaMemcpy(arena.verify.data, arena.device,
                    static_cast<std::size_t>(configuration.aggregate_bytes),
                    cudaMemcpyDeviceToHost);
                if (error != cudaSuccess) {
                    sample.status = PipelineStatus::H2DError;
                    sample.native_error = error;
                    sample.message = CudaMessage(error, "cudaMemcpy(pipeline verification)");
                    verified = false;
                } else {
                    for (std::uint32_t block = 0; block < block_count; ++block) {
                        if (!storage::VerifyDatasetBytes(
                                static_cast<std::byte*>(arena.verify.data) +
                                    static_cast<std::uint64_t>(block) * configuration.chunk_bytes,
                                block_offsets[block], configuration.chunk_bytes)) {
                            verified = false;
                            break;
                        }
                    }
                }
            } else {
                for (const auto& slot : slots) {
                    if (!slot.machine.generation()) continue;
                    const auto slot_offset = static_cast<std::uint64_t>(slot.machine.slotId()) *
                                             configuration.chunk_bytes;
                    const void* source = copy_enabled
                        ? static_cast<const std::byte*>(arena.pinned.data) + slot_offset
                        : UsesPageable(configuration.contention_test)
                            ? static_cast<const std::byte*>(arena.pageable.data) + slot_offset
                            : static_cast<const std::byte*>(arena.pinned.data) + slot_offset;
                    verified = verified && storage::VerifyDatasetBytes(
                        source, slot.file_offset, configuration.chunk_bytes);
                }
            }
            sample.verified = verified;
            if (!verified) {
                sample.status = PipelineStatus::DataVerificationFailure;
                sample.message = "post-timing end-to-end deterministic verification failed";
            }
        }
        (void)concurrent;
        return run;
    }
};

}  // namespace

std::unique_ptr<IPipelineProvider> CreateNativePipelineProvider() {
    return std::make_unique<NativePipelineProvider>();
}

}  // namespace sidecar::pipeline
