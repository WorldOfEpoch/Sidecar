#include "sidecar/trace/benchmark.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <map>
#include <cstring>
#include <numeric>
#include <sstream>
#include <thread>
#include <tuple>
#include <type_traits>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace sidecar::trace {
namespace {

std::atomic<std::uint64_t> g_benchmark_sink{0};

std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::uint64_t ThreadCpuNowNs() {
#ifdef _WIN32
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user)) {
        ULARGE_INTEGER k{}, u{};
        k.LowPart = kernel.dwLowDateTime; k.HighPart = kernel.dwHighDateTime;
        u.LowPart = user.dwLowDateTime; u.HighPart = user.dwHighDateTime;
        return (k.QuadPart + u.QuadPart) * 100ULL;
    }
#endif
    return 0;
}

std::uint64_t DoWork(std::uint64_t value, unsigned rounds) {
    for (unsigned index = 0; index < rounds; ++index) {
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        value *= 0x2545F4914F6CDD1DULL;
    }
    return value;
}

unsigned WorkRounds(const std::string& workload) {
    if (workload == "SATURATION") return 0;
    if (workload == "LIGHT_COMPUTE") return 12;
    return 128;
}

std::uint64_t RunBaseline(const std::string& workload,
                          std::uint32_t producers,
                          std::uint64_t events) {
    const unsigned rounds = WorkRounds(workload);
    const auto start = NowNs();
    std::vector<std::thread> threads;
    std::atomic<std::uint64_t> sink{0};
    for (std::uint32_t producer = 0; producer < producers; ++producer) {
        threads.emplace_back([&, producer] {
            std::uint64_t local = producer + 1;
            for (std::uint64_t index = 0; index < events; ++index) {
                local = DoWork(local + index, rounds);
            }
            sink.fetch_xor(local, std::memory_order_relaxed);
        });
    }
    for (auto& thread : threads) thread.join();
    g_benchmark_sink.fetch_xor(sink.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return NowNs() - start;
}

template <typename Record>
Record MakeBenchmarkRecord(std::uint32_t producer, std::uint64_t index, std::uint64_t timestamp) {
    Record record{};
    record.host_timestamp_ns = timestamp;
    record.operation_id = static_cast<decltype(record.operation_id)>(index + 1);
    record.parent_operation_id = static_cast<decltype(record.parent_operation_id)>(index);
    record.subject_id = static_cast<decltype(record.subject_id)>(producer * 1000 + 7);
    record.payload_bytes = static_cast<decltype(record.payload_bytes)>(4096);
    record.event_type = EventType::HostIoSubmit;
    record.source_tier = MemoryTier::NvmeRaw;
    record.destination_tier = MemoryTier::WarmManagedRam;
    record.producer_id = static_cast<decltype(record.producer_id)>(producer);
    if constexpr (std::is_same_v<Record, TraceRecord32>) record.auxiliary = 1;
    else { record.auxiliary_0 = index; record.auxiliary_1 = producer; }
    return record;
}

template <typename Record, std::size_t Capacity>
ObserverSample RunTraced(const std::string& workload,
                         std::uint32_t producers,
                         std::uint64_t events,
                         std::uint32_t repetition,
                         std::size_t batch_size,
                         const std::filesystem::path& path,
                         TraceHeader header,
                         std::uint64_t baseline_ns) {
    CollectorConfig config;
    config.batch_size = batch_size;
    FlightRecorder<Record, Capacity> recorder(config);
    using Handle = typename FlightRecorder<Record, Capacity>::ProducerHandle;
    std::vector<Handle> handles;
    for (std::uint32_t producer = 0; producer < producers; ++producer) {
        handles.push_back(recorder.RegisterProducer("producer-" + std::to_string(producer),
                                                    "synthetic"));
    }
    header.trace_mode = "LIGHT_SPSC";
    header.monotonic_start_ns = NowNs();
    recorder.Start(path, header);
    const unsigned rounds = WorkRounds(workload);
    std::vector<double> latency_samples;
    std::mutex latency_mutex;
    std::atomic<std::uint64_t> cpu_sum{0};
    const auto start = NowNs();
    std::vector<std::thread> threads;
    for (std::uint32_t producer = 0; producer < producers; ++producer) {
        threads.emplace_back([&, producer] {
            std::vector<double> local_samples;
            std::uint64_t local = producer + 1;
            const auto cpu_start = ThreadCpuNowNs();
            for (std::uint64_t index = 0; index < events; ++index) {
                local = DoWork(local + index, rounds);
                const auto timestamp = NowNs();
                const auto record = MakeBenchmarkRecord<Record>(producer, index, timestamp);
                if ((index & 1023ULL) == 0) {
                    const auto push_start = NowNs();
                    (void)handles[producer].tryPush(record);
                    local_samples.push_back(static_cast<double>(NowNs() - push_start));
                } else {
                    (void)handles[producer].tryPush(record);
                }
            }
            cpu_sum.fetch_add(ThreadCpuNowNs() - cpu_start, std::memory_order_relaxed);
            g_benchmark_sink.fetch_xor(local, std::memory_order_relaxed);
            std::lock_guard lock(latency_mutex);
            latency_samples.insert(latency_samples.end(), local_samples.begin(), local_samples.end());
        });
    }
    for (auto& thread : threads) thread.join();
    const auto producer_wall = NowNs() - start;
    const auto metrics = recorder.Stop();
    const auto trace_total = NowNs() - start;
    ObserverSample sample;
    sample.workload = workload;
    sample.record_size = sizeof(Record);
    sample.producer_count = producers;
    sample.ring_capacity = Capacity;
    sample.collector_batch_size = batch_size;
    sample.repetition = repetition;
    sample.baseline_ns = baseline_ns;
    sample.traced_ns = producer_wall;
    sample.trace_total_ns = trace_total;
    sample.producer_cpu_ns = cpu_sum.load(std::memory_order_relaxed);
    sample.recorder = metrics;
    sample.overhead_raw = ObserverOverheadRaw(baseline_ns, producer_wall);
    const double seconds = static_cast<double>(producer_wall) / 1e9;
    sample.events_per_second = seconds > 0 ? (events * producers) / seconds : 0;
    const double trace_seconds = static_cast<double>(trace_total) / 1e9;
    sample.disk_bytes_per_second = trace_seconds > 0 ? metrics.bytes_written / trace_seconds : 0;
    sample.sampled_try_push_ns = CalculateDistribution(latency_samples);
    sample.raw_try_push_ns = std::move(latency_samples);
    sample.trace_path = path;
    return sample;
}

template <std::size_t Capacity>
CapacitySweepPoint CapacityPoint(std::uint64_t events) {
    SpscRing<TraceRecord32, Capacity> ring;
    const auto start = NowNs();
    for (std::uint64_t index = 0; index < events; ++index) {
        (void)ring.tryPush(MakeBenchmarkRecord<TraceRecord32>(0, index, index));
        TraceRecord32 output;
        (void)ring.tryPop(output);
    }
    const auto elapsed = NowNs() - start;
    return {Capacity, Capacity * sizeof(TraceRecord32),
            elapsed ? events * 1e9 / static_cast<double>(elapsed) : 0,
            ring.droppedEvents(), ring.highWaterMarkQuiescent()};
}

template <typename Record>
double CopyCost(std::uint64_t iterations) {
    constexpr std::size_t kBlockRecords = 4096;
    std::vector<Record> source(kBlockRecords);
    std::vector<Record> destination(kBlockRecords);
    for (std::size_t index = 0; index < kBlockRecords; ++index)
        source[index] = MakeBenchmarkRecord<Record>(1, index + 1, index + 7);
    const auto start = NowNs();
    std::uint64_t sink = 0;
    std::uint64_t copied = 0;
    while (copied < iterations) {
        const auto count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kBlockRecords, iterations - copied));
        source[copied % kBlockRecords].operation_id =
            static_cast<decltype(source[0].operation_id)>(copied + 1);
        std::memcpy(destination.data(), source.data(), count * sizeof(Record));
        std::atomic_signal_fence(std::memory_order_seq_cst);
        sink ^= static_cast<std::uint64_t>(destination[(copied / kBlockRecords) % count].operation_id);
        copied += count;
    }
    const auto elapsed = NowNs() - start;
    g_benchmark_sink.fetch_xor(sink, std::memory_order_relaxed);
    return static_cast<double>(elapsed) / iterations;
}

}  // namespace

double ObserverOverheadRaw(std::uint64_t baseline_ns, std::uint64_t traced_ns) noexcept {
    if (baseline_ns == 0) return 0;
    return (static_cast<double>(traced_ns) - static_cast<double>(baseline_ns)) /
           static_cast<double>(baseline_ns);
}

Distribution CalculateDistribution(std::vector<double> samples) {
    Distribution result;
    result.count = samples.size();
    if (samples.empty()) return result;
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double p) {
        const auto index = static_cast<std::size_t>(
            std::ceil(p * static_cast<double>(samples.size())) - 1.0);
        return samples[std::min(index, samples.size() - 1)];
    };
    result.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    result.median = percentile(0.5);
    result.p50 = result.median;
    result.p90 = percentile(0.90);
    result.p95 = percentile(0.95);
    result.p99 = percentile(0.99);
    if (samples.size() >= 1000U) result.p999 = percentile(0.999);
    result.minimum = samples.front();
    result.maximum = samples.back();
    double variance = 0;
    for (const auto sample : samples) variance += (sample - result.mean) * (sample - result.mean);
    result.stddev = std::sqrt(variance / samples.size());
    return result;
}

ObserverBenchmarkReport RunObserverBenchmark(const ObserverBenchmarkOptions& options,
                                             const TraceHeader& provenance_header) {
    std::filesystem::create_directories(options.output_directory);
    ObserverBenchmarkReport report;
    std::vector<double> timestamp_samples;
    timestamp_samples.reserve(10000);
    const auto timestamp_start = NowNs();
    for (std::uint64_t index = 0; index < 1000000; ++index) {
        if ((index % 100) == 0) {
            const auto sample_start = NowNs();
            g_benchmark_sink.fetch_xor(NowNs(), std::memory_order_relaxed);
            timestamp_samples.push_back(static_cast<double>(NowNs() - sample_start));
        } else {
            g_benchmark_sink.fetch_xor(NowNs(), std::memory_order_relaxed);
        }
    }
    const auto timestamp_elapsed = NowNs() - timestamp_start;
    report.timestamp_calls_per_second = 1000000.0 * 1e9 / timestamp_elapsed;
    report.timestamp_call_ns = CalculateDistribution(std::move(timestamp_samples));
    report.record32_copy_ns = CopyCost<TraceRecord32>(5000000);
    report.record64_copy_ns = CopyCost<TraceRecord64>(5000000);

    const std::array<std::string, 3> workloads{"SATURATION", "LIGHT_COMPUTE", "MEDIUM_COMPUTE"};
    for (const auto& workload : workloads) {
        for (const auto producers : {1U, 2U, 4U, 8U}) {
            for (std::uint32_t repetition = 0; repetition < options.repetitions; ++repetition) {
                const auto baseline = RunBaseline(workload, producers, options.events_per_producer);
                const auto stem = workload + "-p" + std::to_string(producers) + "-r" +
                                  std::to_string(repetition);
                report.samples.push_back(RunTraced<TraceRecord32, kDefaultRingCapacity>(
                    workload, producers, options.events_per_producer, repetition,
                    kDefaultCollectorBatch, options.output_directory / (stem + "-32.sidecartrace"),
                    provenance_header, baseline));
                report.samples.push_back(RunTraced<TraceRecord64, kDefaultRingCapacity>(
                    workload, producers, options.events_per_producer, repetition,
                    kDefaultCollectorBatch, options.output_directory / (stem + "-64.sidecartrace"),
                    provenance_header, baseline));
            }
        }
    }
    if (options.full_sweeps) {
        const auto sweep_events = std::min<std::uint64_t>(options.events_per_producer, 200000);
        report.capacity_sweep = {CapacityPoint<4096>(sweep_events),
                                 CapacityPoint<16384>(sweep_events),
                                 CapacityPoint<65536>(sweep_events),
                                 CapacityPoint<262144>(sweep_events),
                                 CapacityPoint<1048576>(sweep_events)};
        for (const auto batch : {256U, 1024U, 4096U, 16384U, 65536U}) {
            const auto baseline = RunBaseline("SATURATION", 1, sweep_events);
            auto sample = RunTraced<TraceRecord32, kDefaultRingCapacity>(
                "BATCH_SWEEP", 1, sweep_events, 0, batch,
                options.output_directory / ("batch-" + std::to_string(batch) + ".sidecartrace"),
                provenance_header, baseline);
            const auto end_to_end_events_per_second = sample.trace_total_ns
                ? sample.recorder.records_written * 1e9 /
                    static_cast<double>(sample.trace_total_ns) : 0.0;
            report.batch_sweep.push_back({batch, end_to_end_events_per_second,
                                          sample.recorder.records_dropped,
                                          sample.recorder.collector_cpu_ns,
                                          sample.disk_bytes_per_second});
        }
    }
    return report;
}

std::string ObserverBenchmarkToJson(const ObserverBenchmarkReport& report) {
    std::ostringstream out;
    out << "{\"timestamp\":{\"calls_per_second\":" << report.timestamp_calls_per_second
        << ",\"mean_ns\":" << report.timestamp_call_ns.mean << ",\"p95_ns\":"
        << report.timestamp_call_ns.p95 << ",\"p99_ns\":" << report.timestamp_call_ns.p99
        << "},\"copy\":{\"record32_ns\":" << report.record32_copy_ns
        << ",\"record64_ns\":" << report.record64_copy_ns << "},\"samples\":[";
    for (std::size_t index = 0; index < report.samples.size(); ++index) {
        const auto& sample = report.samples[index];
        if (index) out << ',';
        out << "{\"workload\":\"" << sample.workload << "\",\"record_size\":"
            << sample.record_size << ",\"producers\":" << sample.producer_count
            << ",\"repetition\":" << sample.repetition << ",\"baseline_ns\":"
            << sample.baseline_ns << ",\"traced_ns\":" << sample.traced_ns
            << ",\"trace_total_ns\":" << sample.trace_total_ns
            << ",\"overhead_raw\":" << sample.overhead_raw
            << ",\"events_per_second\":" << sample.events_per_second
            << ",\"written\":" << sample.recorder.records_written
            << ",\"dropped\":" << sample.recorder.records_dropped
            << ",\"collector_cpu_ns\":" << sample.recorder.collector_cpu_ns
            << ",\"trace_bytes\":" << sample.recorder.bytes_written
            << ",\"try_push_p95_ns\":" << sample.sampled_try_push_ns.p95
            << ",\"try_push_p99_ns\":" << sample.sampled_try_push_ns.p99
            << ",\"raw_try_push_ns\":[";
        for (std::size_t raw = 0; raw < sample.raw_try_push_ns.size(); ++raw) {
            if (raw) out << ',';
            out << sample.raw_try_push_ns[raw];
        }
        out << "]}";
    }
    out << "],\"overhead_aggregates\":[";
    std::map<std::tuple<std::string, std::uint32_t, std::uint32_t>, std::vector<double>> groups;
    for (const auto& sample : report.samples)
        groups[{sample.workload, sample.record_size, sample.producer_count}].push_back(
            sample.overhead_raw);
    bool first = true;
    for (const auto& [key, values] : groups) {
        if (!first) out << ',';
        first = false;
        const auto stats = CalculateDistribution(values);
        out << "{\"workload\":\"" << std::get<0>(key) << "\",\"record_size\":"
            << std::get<1>(key) << ",\"producers\":" << std::get<2>(key)
            << ",\"mean_raw\":" << stats.mean << ",\"median_raw\":" << stats.median
            << ",\"stddev_raw\":" << stats.stddev << ",\"p95_raw\":" << stats.p95
            << ",\"p99_raw\":" << stats.p99 << ",\"min_raw\":" << stats.minimum
            << ",\"max_raw\":" << stats.maximum << '}';
    }
    out << "]}";
    return out.str();
}

std::string FormatObserverBenchmark(const ObserverBenchmarkReport& report) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "SIDECAR OBSERVER-EFFECT LABORATORY\n\n"
        << "Timestamp steady_clock calls/s: "
        << report.timestamp_calls_per_second << "\nTimestamp sampled mean/p95/p99 ns: "
        << report.timestamp_call_ns.mean << '/' << report.timestamp_call_ns.p95 << '/'
        << report.timestamp_call_ns.p99 << "\nRecord copy ns (32/64): "
        << report.record32_copy_ns << '/' << report.record64_copy_ns << "\n\nSamples\n";
    for (const auto& sample : report.samples) {
        out << "  " << sample.workload << " record=" << sample.record_size
            << " producers=" << sample.producer_count << " rep=" << sample.repetition
            << " baseline_ms=" << sample.baseline_ns / 1e6
            << " traced_ms=" << sample.traced_ns / 1e6
            << " overhead_pct=" << sample.overhead_raw * 100.0
            << " events_s=" << sample.events_per_second
            << " written=" << sample.recorder.records_written
            << " dropped=" << sample.recorder.records_dropped
            << " push_p95_ns=" << sample.sampled_try_push_ns.p95
            << " push_p99_ns=" << sample.sampled_try_push_ns.p99 << '\n';
    }
    out << "\nOverhead aggregates (raw ratio)\n";
    std::map<std::tuple<std::string, std::uint32_t, std::uint32_t>, std::vector<double>> groups;
    for (const auto& sample : report.samples)
        groups[{sample.workload, sample.record_size, sample.producer_count}].push_back(
            sample.overhead_raw);
    for (const auto& [key, values] : groups) {
        const auto stats = CalculateDistribution(values);
        out << "  " << std::get<0>(key) << " record=" << std::get<1>(key)
            << " producers=" << std::get<2>(key) << " mean=" << stats.mean
            << " median=" << stats.median << " stddev=" << stats.stddev
            << " p95=" << stats.p95 << " p99=" << stats.p99
            << " min=" << stats.minimum << " max=" << stats.maximum << '\n';
    }
    out << "\nCapacity sweep\n";
    for (const auto& point : report.capacity_sweep) {
        out << "  capacity=" << point.capacity << " memory_bytes=" << point.memory_bytes
            << " events_s=" << point.events_per_second << " dropped=" << point.dropped << '\n';
    }
    out << "\nBatch sweep\n";
    for (const auto& point : report.batch_sweep) {
        out << "  batch=" << point.batch_size << " events_s=" << point.events_per_second
            << " dropped=" << point.dropped << " collector_cpu_ns=" << point.collector_cpu_ns
            << " disk_bytes_s=" << point.disk_bytes_per_second << '\n';
    }
    return out.str();
}

}  // namespace sidecar::trace
