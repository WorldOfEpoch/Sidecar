#pragma once

#include "sidecar/memory/benchmark.hpp"
#include "sidecar/trace/benchmark.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sidecar::cuda {

enum class TransferDirection { H2D, D2H };
enum class HostMemoryClass {
  PageablePretouched,
  PinnedHostAlloc,
  PinnedRegistered
};
enum class CopyApiMode { Synchronous, Asynchronous };
enum class TransferStatus {
  Success,
  SkippedSafetyLimit,
  SkippedUnsupported,
  CudaOutOfMemory,
  CudaError,
  HostAllocationFailure,
  VerificationFailure,
  CleanupFailure,
  InternalError,
};

struct TransferError {
  TransferStatus status{TransferStatus::Success};
  std::int64_t native_error{0};
  std::string message;
  [[nodiscard]] bool ok() const noexcept {
    return status == TransferStatus::Success;
  }
};

struct DeviceMemoryInfo {
  bool cuda_supported{false};
  int device_index{0};
  std::string name;
  std::uint64_t free_bytes{0};
  std::uint64_t total_bytes{0};
  std::uint32_t async_engine_count{0};
};

struct TransferTelemetry {
  bool available{false};
  std::optional<std::uint32_t> temperature_c;
  std::optional<std::uint32_t> graphics_clock_mhz;
  std::optional<std::uint32_t> memory_clock_mhz;
  std::optional<double> power_watts;
  std::optional<double> power_limit_watts;
  std::optional<std::uint32_t> pcie_generation;
  std::optional<std::uint32_t> pcie_width;
};

struct DeviceSafetyPolicy {
  std::uint64_t absolute_reserve_bytes{2ULL * 1024 * 1024 * 1024};
  double total_reserve_fraction{0.20};
};

struct DeviceSafetyDecision {
  bool safe{false};
  std::uint64_t required_bytes{0};
  std::uint64_t reserve_bytes{0};
  std::string reason;
};

struct TransferConfiguration {
  TransferDirection direction{TransferDirection::H2D};
  HostMemoryClass memory_class{HostMemoryClass::PinnedHostAlloc};
  CopyApiMode api_mode{CopyApiMode::Asynchronous};
  std::uint64_t transfer_bytes{0};
  std::uint32_t warmup_count{3};
  std::uint32_t repetitions{0};
  std::uint32_t batch_count{1};
  std::uint32_t chunk_count{1};
  int device_index{0};
  std::string stream_mode{"NON_BLOCKING"};
  std::string validation_mode{"SAMPLED_DETERMINISTIC"};
  std::string experiment{"ISOLATED"};
};

struct CopyRequest {
  TransferDirection direction{TransferDirection::H2D};
  CopyApiMode api_mode{CopyApiMode::Asynchronous};
  std::uint64_t segment_bytes{0};
  std::uint32_t copy_count{1};
  bool contiguous_segments{false};
  bool synchronize_each{false};
};

struct TransferSample {
  std::uint32_t repetition{0};
  std::uint64_t payload_bytes{0};
  std::uint64_t host_api_raw_ns{0};
  std::uint64_t device_duration_ns{0};
  std::uint64_t end_to_end_raw_ns{0};
  TransferStatus status{TransferStatus::Success};
  std::int64_t cuda_error{0};
  std::string message;
};

struct TransferStatistics {
  std::size_t count{0};
  trace::Distribution host_api_ns;
  trace::Distribution device_ns;
  trace::Distribution end_to_end_ns;
  trace::Distribution bytes_per_second;
  bool p99_meaningful{false};
};

struct TransferResult {
  TransferConfiguration configuration;
  TransferStatus status{TransferStatus::Success};
  std::vector<TransferSample> samples;
  TransferStatistics statistics;
  bool verified{false};
  bool noisy{false};
  std::string async_behavior{"NOT_APPLICABLE"};
  std::string message;
};

struct SaturationProfile {
  TransferDirection direction{TransferDirection::H2D};
  HostMemoryClass memory_class{HostMemoryClass::PinnedHostAlloc};
  CopyApiMode api_mode{CopyApiMode::Asynchronous};
  double peak_bytes_per_second{0};
  std::uint64_t peak_size_bytes{0};
  std::uint64_t knee_80_bytes{0};
  std::uint64_t knee_90_bytes{0};
  std::uint64_t knee_95_bytes{0};
};

struct TransferRunOptions {
  std::vector<TransferDirection> directions;
  std::vector<HostMemoryClass> memory_classes;
  std::vector<CopyApiMode> api_modes;
  std::vector<std::uint64_t> sizes;
  std::optional<std::uint32_t> repetitions;
  std::uint32_t warmup_count{3};
  int device_index{0};
  std::uint64_t ordering_seed{0x5349444543415235ULL};
  memory::SafetyPolicy host_safety;
  DeviceSafetyPolicy device_safety;
  bool dry_run{false};
};

struct TransferRunReport {
  memory::TimerCalibration host_timer;
  DeviceMemoryInfo device;
  std::vector<TransferResult> results;
  std::vector<SaturationProfile> profiles;
  std::vector<std::uint64_t> wu6_candidate_sizes;
  std::uint64_t ordering_seed{0};
  TransferStatus status{TransferStatus::Success};
  std::string message;
};

struct BidirectionalSample {
  std::uint32_t repetition{0};
  std::uint64_t h2d_device_ns{0};
  std::uint64_t d2h_device_ns{0};
  std::uint64_t makespan_ns{0};
  std::uint64_t host_submission_ns{0};
  TransferStatus status{TransferStatus::Success};
};

struct BidirectionalResult {
  HostMemoryClass memory_class{HostMemoryClass::PinnedHostAlloc};
  std::uint64_t transfer_bytes{0};
  std::uint32_t repetitions{0};
  double isolated_h2d_median_ns{0};
  double isolated_d2h_median_ns{0};
  double concurrent_h2d_median_ns{0};
  double concurrent_d2h_median_ns{0};
  double makespan_median_ns{0};
  double aggregate_bytes_per_second{0};
  double concurrency_benefit{0};
  bool verified{false};
  TransferStatus status{TransferStatus::Success};
  std::vector<BidirectionalSample> samples;
  std::string message;
};

struct SustainedResult {
  TransferConfiguration configuration;
  std::uint64_t total_payload_bytes{0};
  std::uint64_t total_duration_ns{0};
  std::uint64_t host_submission_ns{0};
  std::uint64_t device_duration_ns{0};
  double bytes_per_second{0};
  TransferTelemetry before;
  TransferTelemetry maximum_during;
  TransferTelemetry after;
  bool verified{false};
  bool noisy{false};
  TransferStatus status{TransferStatus::Success};
  std::string message;
};

class ITransferProvider {
public:
  virtual ~ITransferProvider() = default;
  [[nodiscard]] virtual bool SupportsCuda() const noexcept = 0;
  virtual DeviceMemoryInfo InspectDevice(int device_index) = 0;
  virtual TransferTelemetry ReadTelemetry(int device_index) = 0;
  virtual TransferError Prepare(int device_index, HostMemoryClass memory_class,
                                std::uint64_t capacity_bytes,
                                std::uint32_t buffer_count) = 0;
  virtual TransferError Warmup(TransferDirection direction,
                               std::uint64_t bytes) = 0;
  virtual TransferSample RunCopy(const CopyRequest &request,
                                 std::uint32_t repetition) = 0;
  virtual BidirectionalSample RunBidirectional(std::uint64_t bytes,
                                               std::uint32_t repetition) = 0;
  virtual TransferError Verify(TransferDirection direction, std::uint64_t bytes,
                               std::uint32_t buffer_index = 0) = 0;
  virtual TransferError Release() noexcept = 0;
};

[[nodiscard]] ITransferProvider &NativeTransferProvider();
[[nodiscard]] std::vector<std::uint64_t> DefaultTransferSizeSweep();
[[nodiscard]] std::uint32_t AdaptiveRepetitions(std::uint64_t bytes) noexcept;
[[nodiscard]] double BytesPerSecond(std::uint64_t bytes,
                                    std::uint64_t duration_ns) noexcept;
[[nodiscard]] double
DecimalGigabytesPerSecond(double bytes_per_second) noexcept;
[[nodiscard]] double BinaryGibibytesPerSecond(double bytes_per_second) noexcept;
[[nodiscard]] DeviceSafetyDecision
PlanDeviceMemory(const DeviceMemoryInfo &device, std::uint64_t required_bytes,
                 const DeviceSafetyPolicy &policy);
[[nodiscard]] std::string
TransferConfigurationIdentity(const TransferConfiguration &configuration);
[[nodiscard]] TransferStatistics
CalculateTransferStatistics(const std::vector<TransferSample> &samples);
[[nodiscard]] SaturationProfile
CalculateSaturationProfile(TransferDirection direction,
                           HostMemoryClass memory_class, CopyApiMode api_mode,
                           const std::vector<TransferResult> &results);
[[nodiscard]] double
CalculateConcurrencyBenefit(double isolated_h2d_ns, double isolated_d2h_ns,
                            double concurrent_makespan_ns) noexcept;
[[nodiscard]] std::uint64_t ChunkSize(std::uint64_t total_bytes,
                                      std::uint32_t chunk_count);
[[nodiscard]] std::uint64_t BatchPayloadBytes(std::uint64_t copy_bytes,
                                              std::uint32_t batch_count);
[[nodiscard]] TransferRunReport
RunTransferSweep(ITransferProvider &provider,
                 const memory::MemorySnapshot &host,
                 const TransferRunOptions &options);
[[nodiscard]] std::vector<BidirectionalResult> RunBidirectionalSweep(
    ITransferProvider &provider, const DeviceMemoryInfo &device,
    const memory::MemorySnapshot &host, const std::vector<std::uint64_t> &sizes,
    HostMemoryClass memory_class, std::uint32_t repetitions,
    const memory::SafetyPolicy &host_safety,
    const DeviceSafetyPolicy &device_safety);
[[nodiscard]] SustainedResult RunSustainedTransfer(
    ITransferProvider &provider, const DeviceMemoryInfo &device,
    const memory::MemorySnapshot &host, TransferConfiguration configuration,
    std::uint64_t target_payload_bytes, const memory::SafetyPolicy &host_safety,
    const DeviceSafetyPolicy &device_safety);
[[nodiscard]] std::string
TransferRunReportToJson(const TransferRunReport &report);
[[nodiscard]] std::string
FormatTransferRunReport(const TransferRunReport &report);
[[nodiscard]] std::string
BidirectionalResultsToJson(const std::vector<BidirectionalResult> &results);
[[nodiscard]] std::string
FormatBidirectionalResults(const std::vector<BidirectionalResult> &results);
[[nodiscard]] std::string SustainedResultToJson(const SustainedResult &result);
[[nodiscard]] std::string FormatSustainedResult(const SustainedResult &result);
[[nodiscard]] const char *ToString(TransferDirection direction) noexcept;
[[nodiscard]] const char *ToString(HostMemoryClass memory_class) noexcept;
[[nodiscard]] const char *ToString(CopyApiMode api_mode) noexcept;
[[nodiscard]] const char *ToString(TransferStatus status) noexcept;

} // namespace sidecar::cuda
