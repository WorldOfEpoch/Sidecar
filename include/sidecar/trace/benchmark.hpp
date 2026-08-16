#pragma once

#include "sidecar/trace/recorder.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace sidecar::trace {

struct Distribution {
    std::uint64_t count{0};
    double mean{0};
    double median{0};
    double stddev{0};
    double p50{0};
    double p90{0};
    double p95{0};
    double p99{0};
    std::optional<double> p999;
    double minimum{0};
    double maximum{0};
};

struct ObserverSample {
    std::string workload;
    std::uint32_t record_size{0};
    std::uint32_t producer_count{0};
    std::uint64_t ring_capacity{0};
    std::uint64_t collector_batch_size{0};
    std::uint32_t repetition{0};
    std::uint64_t baseline_ns{0};
    std::uint64_t traced_ns{0};
    std::uint64_t trace_total_ns{0};
    std::uint64_t producer_cpu_ns{0};
    RecorderMetrics recorder;
    double overhead_raw{0};
    double events_per_second{0};
    double disk_bytes_per_second{0};
    Distribution sampled_try_push_ns;
    std::vector<double> raw_try_push_ns;
    std::filesystem::path trace_path;
};

struct CapacitySweepPoint {
    std::uint64_t capacity{0};
    std::uint64_t memory_bytes{0};
    double events_per_second{0};
    std::uint64_t dropped{0};
    std::uint64_t high_water{0};
};

struct BatchSweepPoint {
    std::uint64_t batch_size{0};
    double events_per_second{0};
    std::uint64_t dropped{0};
    std::uint64_t collector_cpu_ns{0};
    double disk_bytes_per_second{0};
};

struct ObserverBenchmarkReport {
    std::vector<ObserverSample> samples;
    std::vector<CapacitySweepPoint> capacity_sweep;
    std::vector<BatchSweepPoint> batch_sweep;
    Distribution timestamp_call_ns;
    double timestamp_calls_per_second{0};
    double record32_copy_ns{0};
    double record64_copy_ns{0};
};

struct ObserverBenchmarkOptions {
    std::filesystem::path output_directory{std::filesystem::path("traces") / "benchmark"};
    std::uint64_t events_per_producer{100000};
    std::uint32_t repetitions{3};
    bool full_sweeps{true};
};

[[nodiscard]] double ObserverOverheadRaw(std::uint64_t baseline_ns,
                                         std::uint64_t traced_ns) noexcept;
[[nodiscard]] Distribution CalculateDistribution(std::vector<double> samples);
[[nodiscard]] ObserverBenchmarkReport RunObserverBenchmark(
    const ObserverBenchmarkOptions& options,
    const TraceHeader& provenance_header);
[[nodiscard]] std::string ObserverBenchmarkToJson(const ObserverBenchmarkReport& report);
[[nodiscard]] std::string FormatObserverBenchmark(const ObserverBenchmarkReport& report);

}  // namespace sidecar::trace
