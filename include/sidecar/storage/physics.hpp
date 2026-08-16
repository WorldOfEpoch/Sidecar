#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sidecar::storage {

inline constexpr std::uint32_t kDatasetFormatVersion = 1;
inline constexpr std::uint64_t kDefaultDatasetBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kDatasetSeed = 0x5349444543415237ULL;

enum class BackendKind { OverlappedUnbuffered, Buffered, MemoryMapped, DirectStorage };
enum class AccessPattern { Sequential, Pseudorandom, Windowed };
enum class DestinationKind { PageablePretouched, CudaHostAlloc };
enum class RunStatus { Success, SkippedUnsupported, SkippedSafetyLimit, InvalidConfiguration,
                       IoError, TimedOut, Cancelled, VerificationFailure, AbortedHealth };

[[nodiscard]] const char* ToString(BackendKind value) noexcept;
[[nodiscard]] const char* ToString(AccessPattern value) noexcept;
[[nodiscard]] const char* ToString(DestinationKind value) noexcept;
[[nodiscard]] const char* ToString(RunStatus value) noexcept;

struct AlignmentInfo {
    std::uint32_t logical_sector_bytes{0};
    std::uint32_t physical_sector_bytes{0};
    std::uint32_t file_offset_alignment_bytes{0};
    std::uint32_t buffer_alignment_bytes{0};
    std::string source;
};

struct StorageTarget {
    std::string machine_hash;
    std::string persistent_id;
    std::string model;
    std::string firmware;
    std::string bus_type;
    std::uint64_t capacity_bytes{0};
    std::uint32_t physical_disk_number{0};
    std::filesystem::path mount_point;
    std::filesystem::path dataset_path;
    std::string volume_name;
    std::vector<std::string> location_paths;
    AlignmentInfo alignment;
    std::string mapping_confidence{"UNKNOWN"};
    std::string mapping_source;
};

struct DatasetIdentity {
    std::uint32_t format_version{kDatasetFormatVersion};
    std::uint64_t size_bytes{kDefaultDatasetBytes};
    std::uint64_t seed{kDatasetSeed};
    std::string generator{"SIDECAR-SPLITMIX64-BYTE-V1"};
    std::string identity_hash;
    std::filesystem::path path;
    bool exists{false};
    bool verified{false};
    bool full_verification{false};
    std::uint64_t verified_bytes{0};
    std::string message;
};

struct HealthSnapshot {
    std::string provider{"WINDOWS_STORAGE_PROTOCOL"};
    std::string phase;
    std::optional<std::uint32_t> critical_warning;
    std::optional<double> temperature_c;
    std::optional<std::uint32_t> available_spare_percent;
    std::optional<std::uint32_t> available_spare_threshold_percent;
    std::optional<std::uint32_t> percentage_used;
    std::optional<std::uint64_t> data_units_read_low64;
    std::optional<std::uint64_t> data_units_written_low64;
    std::optional<std::uint64_t> host_read_commands_low64;
    std::optional<std::uint64_t> host_write_commands_low64;
    std::optional<std::uint64_t> controller_busy_minutes_low64;
    std::optional<std::uint64_t> power_cycles_low64;
    std::optional<std::uint64_t> power_on_hours_low64;
    std::optional<std::uint64_t> unsafe_shutdowns_low64;
    std::optional<std::uint64_t> media_data_errors_low64;
    std::optional<std::uint64_t> error_log_entries_low64;
    std::string raw_evidence_json{"{}"};
    std::string status{"UNSUPPORTED"};
    std::string message;
};

struct BenchmarkConfig {
    BackendKind backend{BackendKind::OverlappedUnbuffered};
    AccessPattern pattern{AccessPattern::Sequential};
    DestinationKind destination{DestinationKind::PageablePretouched};
    std::uint64_t block_bytes{1024ULL * 1024ULL};
    std::uint32_t queue_depth{1};
    std::uint64_t request_count{64};
    std::uint64_t outstanding_byte_cap{1024ULL * 1024ULL * 1024ULL};
    std::uint64_t timeout_ms{120000};
    std::uint64_t ordering_seed{kDatasetSeed};
    std::uint64_t window_bytes{1024ULL * 1024ULL * 1024ULL};
    std::string phase{"COARSE"};
};

struct RequestSample {
    std::uint64_t request_index{0};
    std::uint64_t batch_id{0};
    std::uint64_t file_offset{0};
    std::uint64_t requested_bytes{0};
    std::uint64_t completed_bytes{0};
    std::uint64_t submission_reference_ns{0};
    std::uint64_t submission_cost_ns{0};
    std::uint64_t completion_latency_ns{0};
    std::uint64_t completion_processing_ns{0};
    RunStatus status{RunStatus::Success};
    std::uint32_t native_error{0};
};

struct Distribution {
    std::uint64_t count{0};
    double mean{0};
    double minimum{0};
    double maximum{0};
    double standard_deviation{0};
    double p50{0};
    double p90{0};
    double p95{0};
    double p99{0};
    std::optional<double> p999;
};

struct DeadlineResult {
    std::uint64_t deadline_ns{0};
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    double success_rate{0};
    double compliant_bytes_per_second{0};
};

struct CpuUsage {
    std::uint64_t process_user_ns{0};
    std::uint64_t process_kernel_ns{0};
    std::uint64_t completion_thread_user_ns{0};
    std::uint64_t completion_thread_kernel_ns{0};
    double process_cpu_seconds_per_gib{0};
    double completion_cpu_ns_per_request{0};
};

struct BenchmarkResult {
    BenchmarkConfig config;
    RunStatus status{RunStatus::Success};
    std::vector<RequestSample> samples;
    Distribution submission_cost;
    Distribution completion_latency;
    Distribution completion_processing;
    std::vector<DeadlineResult> deadlines;
    CpuUsage cpu;
    std::uint64_t completed_bytes{0};
    std::uint64_t wall_time_ns{0};
    double bytes_per_second{0};
    double iops{0};
    bool verified{false};
    std::string cache_classification{"CACHE_BYPASSED"};
    std::string message;
};

struct PlanEntry {
    BenchmarkConfig config;
    RunStatus disposition{RunStatus::Success};
    std::string reason;
};

struct StoragePlan {
    StorageTarget target;
    DatasetIdentity dataset;
    std::vector<PlanEntry> entries;
    std::uint64_t estimated_read_bytes{0};
    std::uint64_t outstanding_byte_cap{0};
    std::uint64_t available_memory_bytes{0};
    double temperature_cap_c{75.0};
};

class IStorageProvider {
public:
    virtual ~IStorageProvider() = default;
    [[nodiscard]] virtual StorageTarget ResolveTarget(
        const std::optional<std::string>& selector,
        const std::optional<std::filesystem::path>& dataset_path) = 0;
    [[nodiscard]] virtual DatasetIdentity InspectDataset(const StorageTarget& target) = 0;
    virtual DatasetIdentity CreateDataset(const StorageTarget& target, std::uint64_t bytes) = 0;
    virtual DatasetIdentity VerifyDataset(const StorageTarget& target, bool full) = 0;
    [[nodiscard]] virtual HealthSnapshot CaptureHealth(const StorageTarget& target,
                                                       std::string phase) = 0;
    [[nodiscard]] virtual BenchmarkResult Run(const StorageTarget& target,
                                              const BenchmarkConfig& config) = 0;
    [[nodiscard]] virtual bool DirectStorageAvailable() const noexcept = 0;
};

struct ExternalReadRequest {
    std::uint64_t token{0};
    std::uint64_t file_offset{0};
    std::uint64_t bytes{0};
    void* destination{nullptr};
};

struct ExternalReadCompletion {
    std::uint64_t token{0};
    std::uint64_t file_offset{0};
    std::uint64_t requested_bytes{0};
    std::uint64_t completed_bytes{0};
    std::uint64_t submitted_ns{0};
    std::uint64_t completed_ns{0};
    std::uint64_t submission_cost_ns{0};
    std::uint32_t native_error{0};
    RunStatus status{RunStatus::Success};
};

class IAsyncStorageReader {
public:
    virtual ~IAsyncStorageReader() = default;
    [[nodiscard]] virtual ExternalReadCompletion Submit(
        const ExternalReadRequest& request) = 0;
    [[nodiscard]] virtual ExternalReadCompletion Wait(std::uint64_t timeout_ms) = 0;
    virtual void Cancel() noexcept = 0;
    [[nodiscard]] virtual std::uint32_t Outstanding() const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<IAsyncStorageReader> CreateNativeAsyncStorageReader(
    const StorageTarget& target, std::uint32_t maximum_outstanding);

[[nodiscard]] std::unique_ptr<IStorageProvider> CreateNativeStorageProvider();
[[nodiscard]] std::vector<std::uint64_t> GenerateOffsets(const BenchmarkConfig& config,
                                                         std::uint64_t dataset_bytes);
[[nodiscard]] std::optional<std::string> ValidateConfig(const BenchmarkConfig& config,
                                                        const StorageTarget& target,
                                                        std::uint64_t dataset_bytes);
[[nodiscard]] Distribution Summarize(const std::vector<double>& values);
[[nodiscard]] std::vector<DeadlineResult> EvaluateDeadlines(
    const std::vector<RequestSample>& samples, std::uint64_t wall_time_ns);
void FinalizeBenchmark(BenchmarkResult& result);
[[nodiscard]] StoragePlan BuildDefaultPlan(const StorageTarget& target,
                                           const DatasetIdentity& dataset,
                                           std::uint64_t available_memory_bytes,
                                           bool include_controls = true);
void FillDatasetBytes(void* destination, std::uint64_t file_offset, std::uint64_t bytes);
[[nodiscard]] bool VerifyDatasetBytes(const void* source, std::uint64_t file_offset,
                                      std::uint64_t bytes);
[[nodiscard]] std::string DatasetIdentityHash(std::uint64_t size_bytes);
[[nodiscard]] std::string PlanToJson(const StoragePlan& plan);
[[nodiscard]] std::string ResultToJson(const BenchmarkResult& result);
[[nodiscard]] std::string HealthToJson(const HealthSnapshot& health);

}  // namespace sidecar::storage
