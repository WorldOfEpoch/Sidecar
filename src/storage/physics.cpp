#include "sidecar/storage/physics.hpp"

#include "sidecar/core/sha256.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace sidecar::storage {
namespace {

constexpr std::uint64_t MiB(std::uint64_t value) { return value * 1024ULL * 1024ULL; }

std::uint64_t SplitMix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

std::uint64_t DatasetWord(std::uint64_t word_index) noexcept {
    return SplitMix64(kDatasetSeed ^ word_index);
}

double Percentile(const std::vector<double>& sorted, double probability) {
    if (sorted.empty()) return 0.0;
    const double position = probability * static_cast<double>(sorted.size() - 1U);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

std::string Escape(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char character : value) {
        switch (character) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (character < 0x20U) {
                    out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<unsigned>(character) << std::dec;
                } else {
                    out << static_cast<char>(character);
                }
        }
    }
    out << '"';
    return out.str();
}

template <typename T>
void OptionalJson(std::ostringstream& out, const std::optional<T>& value) {
    if (value) out << *value;
    else out << "null";
}

std::string DistributionJson(const Distribution& value) {
    std::ostringstream out;
    out << "{\"count\":" << value.count << ",\"mean\":" << value.mean
        << ",\"min\":" << value.minimum << ",\"max\":" << value.maximum
        << ",\"stddev\":" << value.standard_deviation << ",\"p50\":" << value.p50
        << ",\"p90\":" << value.p90 << ",\"p95\":" << value.p95
        << ",\"p99\":" << value.p99 << ",\"p999\":";
    OptionalJson(out, value.p999);
    out << '}';
    return out.str();
}

}  // namespace

const char* ToString(BackendKind value) noexcept {
    switch (value) {
        case BackendKind::OverlappedUnbuffered: return "WINDOWS_OVERLAPPED_UNBUFFERED";
        case BackendKind::Buffered: return "WINDOWS_BUFFERED";
        case BackendKind::MemoryMapped: return "WINDOWS_MEMORY_MAPPED";
        case BackendKind::DirectStorage: return "DIRECTSTORAGE";
    }
    return "UNKNOWN";
}

const char* ToString(AccessPattern value) noexcept {
    switch (value) {
        case AccessPattern::Sequential: return "SEQUENTIAL_ALIGNED";
        case AccessPattern::Pseudorandom: return "DETERMINISTIC_PSEUDORANDOM";
        case AccessPattern::Windowed: return "WINDOWED_STRIDED";
    }
    return "UNKNOWN";
}

const char* ToString(DestinationKind value) noexcept {
    switch (value) {
        case DestinationKind::PageablePretouched: return "PAGEABLE_PRETOUCHED";
        case DestinationKind::CudaHostAlloc: return "CUDA_HOST_ALLOC";
    }
    return "UNKNOWN";
}

const char* ToString(RunStatus value) noexcept {
    switch (value) {
        case RunStatus::Success: return "SUCCESS";
        case RunStatus::SkippedUnsupported: return "SKIPPED_UNSUPPORTED";
        case RunStatus::SkippedSafetyLimit: return "SKIPPED_SAFETY_LIMIT";
        case RunStatus::InvalidConfiguration: return "INVALID_CONFIGURATION";
        case RunStatus::IoError: return "IO_ERROR";
        case RunStatus::TimedOut: return "TIMED_OUT";
        case RunStatus::Cancelled: return "CANCELLED";
        case RunStatus::VerificationFailure: return "VERIFICATION_FAILURE";
        case RunStatus::AbortedHealth: return "ABORTED_HEALTH";
    }
    return "IO_ERROR";
}

void FillDatasetBytes(void* destination, std::uint64_t file_offset, std::uint64_t bytes) {
    auto* output = static_cast<std::uint8_t*>(destination);
    std::uint64_t written = 0;
    while (written < bytes && ((file_offset + written) & 7ULL) != 0ULL) {
        const auto absolute = file_offset + written;
        output[written++] = static_cast<std::uint8_t>(DatasetWord(absolute / 8ULL) >>
                                                      ((absolute & 7ULL) * 8ULL));
    }
    while (bytes - written >= 8ULL) {
        const std::uint64_t word = DatasetWord((file_offset + written) / 8ULL);
        std::memcpy(output + written, &word, sizeof(word));
        written += 8ULL;
    }
    while (written < bytes) {
        const auto absolute = file_offset + written;
        output[written++] = static_cast<std::uint8_t>(DatasetWord(absolute / 8ULL) >>
                                                      ((absolute & 7ULL) * 8ULL));
    }
}

bool VerifyDatasetBytes(const void* source, std::uint64_t file_offset, std::uint64_t bytes) {
    constexpr std::uint64_t kChunk = 64ULL * 1024ULL;
    std::array<std::uint8_t, kChunk> expected{};
    const auto* observed = static_cast<const std::uint8_t*>(source);
    for (std::uint64_t offset = 0; offset < bytes; offset += kChunk) {
        const auto count = std::min(kChunk, bytes - offset);
        FillDatasetBytes(expected.data(), file_offset + offset, count);
        if (std::memcmp(observed + offset, expected.data(), static_cast<std::size_t>(count)) != 0)
            return false;
    }
    return true;
}

std::string DatasetIdentityHash(std::uint64_t size_bytes) {
    std::ostringstream material;
    material << "SIDECAR-STORAGE-DATASET-V1\n"
             << "generator=SIDECAR-SPLITMIX64-BYTE-V1\n"
             << "seed=" << kDatasetSeed << "\nsize_bytes=" << size_bytes << '\n';
    return core::Sha256Hex(material.str());
}

std::vector<std::uint64_t> GenerateOffsets(const BenchmarkConfig& config,
                                           std::uint64_t dataset_bytes) {
    std::vector<std::uint64_t> offsets;
    if (config.block_bytes == 0 || dataset_bytes < config.block_bytes) return offsets;
    const std::uint64_t blocks = dataset_bytes / config.block_bytes;
    offsets.reserve(static_cast<std::size_t>(config.request_count));
    if (config.pattern == AccessPattern::Sequential) {
        for (std::uint64_t index = 0; index < config.request_count; ++index)
            offsets.push_back((index % blocks) * config.block_bytes);
        return offsets;
    }

    std::uint64_t domain_blocks = blocks;
    std::uint64_t domain_start = 0;
    if (config.pattern == AccessPattern::Windowed) {
        domain_blocks = std::max<std::uint64_t>(1, std::min(blocks,
            std::max(config.block_bytes, config.window_bytes) / config.block_bytes));
        const auto windows = std::max<std::uint64_t>(1, blocks / domain_blocks);
        domain_start = (SplitMix64(config.ordering_seed) % windows) * domain_blocks;
    }
    const std::uint64_t start = SplitMix64(config.ordering_seed ^ config.block_bytes) % domain_blocks;
    std::uint64_t stride = (SplitMix64(config.ordering_seed ^ config.queue_depth) |
                            1ULL) % domain_blocks;
    if (stride == 0) stride = 1;
    while (std::gcd(stride, domain_blocks) != 1ULL) {
        stride += 2ULL;
        if (stride >= domain_blocks) stride = 1;
    }
    for (std::uint64_t index = 0; index < config.request_count; ++index) {
        const auto cycle = index / domain_blocks;
        const auto local = (start + (index % domain_blocks) * stride + cycle) % domain_blocks;
        offsets.push_back((domain_start + local) * config.block_bytes);
    }
    return offsets;
}

std::optional<std::string> ValidateConfig(const BenchmarkConfig& config,
                                          const StorageTarget& target,
                                          std::uint64_t dataset_bytes) {
    if (config.block_bytes == 0 || config.queue_depth == 0 || config.request_count == 0)
        return "block size, queue depth, and request count must be non-zero";
    if (config.block_bytes > dataset_bytes) return "block size exceeds dataset";
    if (config.queue_depth > config.request_count) return "queue depth exceeds request count";
    if (config.block_bytes > UINT64_MAX / config.queue_depth ||
        config.block_bytes * config.queue_depth > config.outstanding_byte_cap)
        return "outstanding bytes exceed safety cap";
    if (config.backend == BackendKind::OverlappedUnbuffered) {
        const auto offset_alignment = target.alignment.file_offset_alignment_bytes;
        const auto buffer_alignment = target.alignment.buffer_alignment_bytes;
        if (offset_alignment == 0 || buffer_alignment == 0)
            return "unbuffered alignment is unknown";
        if ((config.block_bytes % offset_alignment) != 0)
            return "block size violates unbuffered file-offset alignment";
    }
    if (config.backend == BackendKind::MemoryMapped && config.queue_depth != 1)
        return "memory-mapped control supports queue depth 1 only";
    return std::nullopt;
}

Distribution Summarize(const std::vector<double>& values) {
    Distribution result;
    result.count = values.size();
    if (values.empty()) return result;
    std::vector<double> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    result.minimum = sorted.front();
    result.maximum = sorted.back();
    result.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
                  static_cast<double>(sorted.size());
    double sum_squares = 0.0;
    for (double value : sorted) {
        const double delta = value - result.mean;
        sum_squares += delta * delta;
    }
    result.standard_deviation = std::sqrt(sum_squares / static_cast<double>(sorted.size()));
    result.p50 = Percentile(sorted, .50);
    result.p90 = Percentile(sorted, .90);
    result.p95 = Percentile(sorted, .95);
    result.p99 = Percentile(sorted, .99);
    if (sorted.size() >= 1000U) result.p999 = Percentile(sorted, .999);
    return result;
}

std::vector<DeadlineResult> EvaluateDeadlines(const std::vector<RequestSample>& samples,
                                              std::uint64_t wall_time_ns) {
    constexpr std::array<std::uint64_t, 9> deadlines{
        500000ULL, 1000000ULL, 2000000ULL, 4000000ULL, 8000000ULL,
        16000000ULL, 32000000ULL, 64000000ULL, 128000000ULL};
    std::vector<DeadlineResult> results;
    for (auto deadline : deadlines) {
        DeadlineResult result;
        result.deadline_ns = deadline;
        std::uint64_t compliant_bytes = 0;
        for (const auto& sample : samples) {
            if (sample.status == RunStatus::Success && sample.completion_latency_ns <= deadline) {
                ++result.hits;
                compliant_bytes += sample.completed_bytes;
            } else {
                ++result.misses;
            }
        }
        const auto total = result.hits + result.misses;
        if (total) result.success_rate = static_cast<double>(result.hits) / total;
        if (wall_time_ns) result.compliant_bytes_per_second =
            static_cast<double>(compliant_bytes) * 1.0e9 / wall_time_ns;
        results.push_back(result);
    }
    return results;
}

void FinalizeBenchmark(BenchmarkResult& result) {
    std::vector<double> submission, latency, processing;
    result.completed_bytes = 0;
    std::uint64_t successes = 0;
    for (const auto& sample : result.samples) {
        submission.push_back(static_cast<double>(sample.submission_cost_ns));
        latency.push_back(static_cast<double>(sample.completion_latency_ns));
        processing.push_back(static_cast<double>(sample.completion_processing_ns));
        if (sample.status == RunStatus::Success) {
            result.completed_bytes += sample.completed_bytes;
            ++successes;
        }
    }
    result.submission_cost = Summarize(submission);
    result.completion_latency = Summarize(latency);
    result.completion_processing = Summarize(processing);
    result.deadlines = EvaluateDeadlines(result.samples, result.wall_time_ns);
    if (result.wall_time_ns) {
        result.bytes_per_second = static_cast<double>(result.completed_bytes) * 1.0e9 /
                                  result.wall_time_ns;
        result.iops = static_cast<double>(successes) * 1.0e9 / result.wall_time_ns;
    }
    const double gib = static_cast<double>(result.completed_bytes) / (1024.0 * 1024.0 * 1024.0);
    const auto process_ns = result.cpu.process_user_ns + result.cpu.process_kernel_ns;
    if (gib > 0) result.cpu.process_cpu_seconds_per_gib = process_ns / 1.0e9 / gib;
    if (!result.samples.empty()) {
        result.cpu.completion_cpu_ns_per_request =
            static_cast<double>(result.cpu.completion_thread_user_ns +
                                result.cpu.completion_thread_kernel_ns) /
            result.samples.size();
    }
}

StoragePlan BuildDefaultPlan(const StorageTarget& target, const DatasetIdentity& dataset,
                             std::uint64_t available_memory_bytes, bool include_controls) {
    StoragePlan plan;
    plan.target = target;
    plan.dataset = dataset;
    plan.available_memory_bytes = available_memory_bytes;
    plan.outstanding_byte_cap = std::min(MiB(1024), available_memory_bytes / 8ULL);
    if (plan.outstanding_byte_cap < MiB(128)) plan.outstanding_byte_cap = MiB(128);
    constexpr std::array<std::uint64_t, 10> blocks{
        64ULL * 1024ULL, 256ULL * 1024ULL, MiB(1), MiB(2), MiB(4),
        MiB(8), MiB(16), MiB(32), MiB(64), MiB(128)};
    constexpr std::array<std::uint32_t, 8> depths{1, 2, 4, 8, 16, 32, 64, 128};
    constexpr std::array<AccessPattern, 2> patterns{
        AccessPattern::Sequential, AccessPattern::Pseudorandom};
    auto add = [&](BenchmarkConfig config) {
        config.outstanding_byte_cap = plan.outstanding_byte_cap;
        PlanEntry entry{config};
        if (config.backend == BackendKind::DirectStorage) {
            entry.disposition = RunStatus::SkippedUnsupported;
            entry.reason = "optional DirectStorage provider unavailable";
        } else if (const auto issue = ValidateConfig(config, target, dataset.size_bytes)) {
            entry.disposition = RunStatus::SkippedSafetyLimit;
            entry.reason = *issue;
        } else {
            plan.estimated_read_bytes += config.block_bytes * config.request_count;
        }
        plan.entries.push_back(std::move(entry));
    };
    for (auto pattern : patterns) {
        for (auto block : blocks) {
            for (auto depth : depths) {
                BenchmarkConfig config;
                config.pattern = pattern;
                config.block_bytes = block;
                config.queue_depth = depth;
                config.request_count = std::max<std::uint64_t>(depth,
                    std::min<std::uint64_t>(8192, std::max<std::uint64_t>(8, MiB(512) / block)));
                add(config);
            }
        }
    }
    for (auto block : {MiB(1), MiB(4), MiB(16), MiB(64), MiB(128)}) {
        for (auto depth : {1U, 4U, 16U}) {
            BenchmarkConfig pinned;
            pinned.destination = DestinationKind::CudaHostAlloc;
            pinned.block_bytes = block;
            pinned.queue_depth = depth;
            pinned.request_count = std::max<std::uint64_t>(depth, MiB(256) / block);
            pinned.phase = "PINNED_CROSSCHECK";
            add(pinned);
        }
    }
    for (auto block : {MiB(1), MiB(16)}) {
        for (auto depth : {4U, 16U}) {
            BenchmarkConfig windowed;
            windowed.pattern = AccessPattern::Windowed;
            windowed.block_bytes = block;
            windowed.queue_depth = depth;
            windowed.request_count = std::max<std::uint64_t>(depth, MiB(256) / block);
            windowed.phase = "WINDOWED_CROSSCHECK";
            add(windowed);
        }
    }
    if (include_controls) {
        for (auto block : {MiB(1), MiB(64)}) {
            BenchmarkConfig buffered;
            buffered.backend = BackendKind::Buffered;
            buffered.block_bytes = block;
            buffered.request_count = std::max<std::uint64_t>(8, MiB(256) / block);
            buffered.phase = "BUFFERED_CONTROL";
            add(buffered);
            for (const auto* phase : {"MMAP_FIRST_ACCESS", "MMAP_WARM_ACCESS"}) {
                BenchmarkConfig mapped = buffered;
                mapped.backend = BackendKind::MemoryMapped;
                mapped.phase = phase;
                add(mapped);
            }
            BenchmarkConfig direct = buffered;
            direct.backend = BackendKind::DirectStorage;
            direct.phase = "DIRECTSTORAGE_CONTROL";
            add(direct);
        }
    }
    return plan;
}

std::string PlanToJson(const StoragePlan& plan) {
    std::ostringstream out;
    out << "{\"dataset\":{\"exists\":" << (plan.dataset.exists ? "true" : "false")
        << ",\"format_version\":" << plan.dataset.format_version
        << ",\"identity_hash\":" << Escape(plan.dataset.identity_hash)
        << ",\"path\":" << Escape(plan.dataset.path.generic_string())
        << ",\"size_bytes\":" << plan.dataset.size_bytes << "},\"entries\":[";
    for (std::size_t i = 0; i < plan.entries.size(); ++i) {
        if (i) out << ',';
        const auto& e = plan.entries[i];
        out << "{\"backend\":" << Escape(ToString(e.config.backend))
            << ",\"block_bytes\":" << e.config.block_bytes
            << ",\"destination\":" << Escape(ToString(e.config.destination))
            << ",\"disposition\":" << Escape(ToString(e.disposition))
            << ",\"pattern\":" << Escape(ToString(e.config.pattern))
            << ",\"queue_depth\":" << e.config.queue_depth
            << ",\"reason\":" << Escape(e.reason)
            << ",\"request_count\":" << e.config.request_count << '}';
    }
    out << "],\"estimated_read_bytes\":" << plan.estimated_read_bytes
        << ",\"outstanding_byte_cap\":" << plan.outstanding_byte_cap
        << ",\"target\":{\"alignment\":{\"buffer_bytes\":"
        << plan.target.alignment.buffer_alignment_bytes << ",\"file_offset_bytes\":"
        << plan.target.alignment.file_offset_alignment_bytes << ",\"logical_sector_bytes\":"
        << plan.target.alignment.logical_sector_bytes << ",\"physical_sector_bytes\":"
        << plan.target.alignment.physical_sector_bytes << "},\"bus_type\":"
        << Escape(plan.target.bus_type) << ",\"mapping_confidence\":"
        << Escape(plan.target.mapping_confidence) << ",\"model\":"
        << Escape(plan.target.model) << ",\"persistent_id\":"
        << Escape(plan.target.persistent_id) << ",\"physical_disk_number\":"
        << plan.target.physical_disk_number << ",\"volume_name\":"
        << Escape(plan.target.volume_name) << "},\"temperature_cap_c\":"
        << plan.temperature_cap_c << '}';
    return out.str();
}

std::string ResultToJson(const BenchmarkResult& result) {
    std::ostringstream out;
    out << "{\"backend\":" << Escape(ToString(result.config.backend))
        << ",\"block_bytes\":" << result.config.block_bytes
        << ",\"bytes_per_second\":" << result.bytes_per_second
        << ",\"cache_classification\":" << Escape(result.cache_classification)
        << ",\"completed_bytes\":" << result.completed_bytes
        << ",\"completion_latency_ns\":" << DistributionJson(result.completion_latency)
        << ",\"completion_processing_ns\":" << DistributionJson(result.completion_processing)
        << ",\"destination\":" << Escape(ToString(result.config.destination))
        << ",\"iops\":" << result.iops << ",\"message\":" << Escape(result.message)
        << ",\"pattern\":" << Escape(ToString(result.config.pattern))
        << ",\"queue_depth\":" << result.config.queue_depth
        << ",\"sample_count\":" << result.samples.size()
        << ",\"status\":" << Escape(ToString(result.status))
        << ",\"submission_cost_ns\":" << DistributionJson(result.submission_cost)
        << ",\"verified\":" << (result.verified ? "true" : "false")
        << ",\"wall_time_ns\":" << result.wall_time_ns << '}';
    return out.str();
}

std::string HealthToJson(const HealthSnapshot& h) {
    std::ostringstream out;
    out << "{\"available_spare_percent\":"; OptionalJson(out, h.available_spare_percent);
    out << ",\"available_spare_threshold_percent\":"; OptionalJson(out, h.available_spare_threshold_percent);
    out << ",\"controller_busy_minutes_low64\":"; OptionalJson(out, h.controller_busy_minutes_low64);
    out << ",\"critical_warning\":"; OptionalJson(out, h.critical_warning);
    out << ",\"data_units_read_low64\":"; OptionalJson(out, h.data_units_read_low64);
    out << ",\"data_units_written_low64\":"; OptionalJson(out, h.data_units_written_low64);
    out << ",\"error_log_entries_low64\":"; OptionalJson(out, h.error_log_entries_low64);
    out << ",\"host_read_commands_low64\":"; OptionalJson(out, h.host_read_commands_low64);
    out << ",\"host_write_commands_low64\":"; OptionalJson(out, h.host_write_commands_low64);
    out << ",\"media_data_errors_low64\":"; OptionalJson(out, h.media_data_errors_low64);
    out << ",\"message\":" << Escape(h.message) << ",\"percentage_used\":";
    OptionalJson(out, h.percentage_used);
    out << ",\"phase\":" << Escape(h.phase) << ",\"power_cycles_low64\":";
    OptionalJson(out, h.power_cycles_low64);
    out << ",\"power_on_hours_low64\":"; OptionalJson(out, h.power_on_hours_low64);
    out << ",\"provider\":" << Escape(h.provider) << ",\"raw_evidence\":"
        << h.raw_evidence_json << ",\"status\":" << Escape(h.status)
        << ",\"temperature_c\":"; OptionalJson(out, h.temperature_c);
    out << ",\"unsafe_shutdowns_low64\":"; OptionalJson(out, h.unsafe_shutdowns_low64);
    out << '}';
    return out.str();
}

}  // namespace sidecar::storage
