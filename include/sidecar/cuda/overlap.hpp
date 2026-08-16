#pragma once

#include "sidecar/cuda/transfer.hpp"
#include "sidecar/trace/benchmark.hpp"

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sidecar::cuda {

enum class ComputeWorkload { SyntheticAlu, MemoryBound, Fp16Gemm };
enum class OverlapPhase { Calibration, Instrumentation, Coarse, Refine, Repeat };
enum class SampleMode { ComputeOnly, TransferOnly, Concurrent };
enum class InstrumentationMode { Minimal, MatchedGateDependency, Full };
enum class OverlapStatus {
  Success,
  SkippedUnsupported,
  SkippedSafetyLimit,
  OutsideFullHideRegion,
  InvalidGate,
  InstrumentationBias,
  ComputeValidationFailure,
  TransferValidationFailure,
  CudaError,
  CublasError,
  Noisy,
  InvalidEventOrder,
  BaselineDrift,
  MeasurementSensitive,
  NanOrInf,
  ImpossibleDuration,
  InternalError,
};

struct WorkloadProfile {
  ComputeWorkload workload{ComputeWorkload::SyntheticAlu};
  double target_us{0};
  trace::Distribution calibrated_us;
  std::uint32_t calibration_samples{0};
  std::uint64_t alu_iterations{0};
  std::uint64_t memory_working_set_bytes{0};
  std::uint32_t memory_passes{0};
  std::uint32_t memory_block_size{0};
  std::uint32_t memory_elements_per_thread{0};
  std::uint32_t gemm_m{0}, gemm_n{0}, gemm_k{0}, gemm_repetitions{0};
  std::string gemm_a_type{"CUDA_R_16F"};
  std::string gemm_b_type{"CUDA_R_16F"};
  std::string gemm_c_type{"CUDA_R_32F"};
  std::string gemm_compute_type{"CUBLAS_COMPUTE_32F"};
  std::string gemm_math_mode{"CUBLAS_DEFAULT_MATH"};
  std::string gemm_algorithm{"CUBLAS_GEMM_DEFAULT"};
  std::optional<std::int64_t> cublas_version;
  bool validated{false};
  OverlapStatus status{OverlapStatus::Success};
  std::string message;
};

struct OverlapConfiguration {
  ComputeWorkload workload{ComputeWorkload::SyntheticAlu};
  TransferDirection direction{TransferDirection::H2D};
  HostMemoryClass memory_class{HostMemoryClass::PinnedHostAlloc};
  std::uint64_t transfer_bytes{0};
  double target_compute_us{0};
  OverlapPhase phase{OverlapPhase::Coarse};
  std::uint32_t repetitions{0};
  std::uint64_t ordering_seed{0x5349444543415236ULL};
  std::uint64_t gate_delay_ns{0};
  std::uint64_t gate_margin_ns{0};
  int device_index{0};
};

struct RawOverlapTiming {
  std::uint64_t compute_ns{0};
  std::uint64_t transfer_ns{0};
  std::uint64_t makespan_device_primary_ns{0};
  std::uint64_t makespan_device_crosscheck_ns{0};
  std::uint64_t makespan_host_ns{0};
  std::uint64_t host_submission_ns{0};
  std::uint64_t gate_actual_ns{0};
  bool gate_valid{true};
  bool compute_valid{true};
  bool transfer_valid{true};
  std::int64_t native_error{0};
  OverlapStatus status{OverlapStatus::Success};
  std::string message;
};

struct OverlapMetrics {
  double critical_path_delta_ns{0};
  double critical_path_added_ns{0};
  double compute_slowdown{0};
  double transfer_slowdown{0};
  double overlap_efficiency_raw{0};
  double overlap_efficiency_normalized{0};
  double compute_path_delta_ns{0};
  double compute_path_added_ns{0};
  double compute_path_added_percent{0};
  double hidden_fraction_raw{0};
  double hidden_fraction_normalized{0};
  double compute_retention{0};
  bool fit_1_percent{false};
  bool fit_2_percent{false};
  bool fit_5_percent{false};
};

struct OverlapSample {
  std::uint32_t repetition{0};
  std::uint64_t baseline_block_id{0};
  std::uint64_t c0_reference_ns{0};
  std::uint64_t t0_reference_ns{0};
  std::uint64_t cc_ns{0};
  std::uint64_t tc_ns{0};
  std::uint64_t makespan_device_primary_ns{0};
  std::uint64_t makespan_device_crosscheck_ns{0};
  std::uint64_t makespan_host_ns{0};
  std::uint64_t host_submission_ns{0};
  std::uint64_t gate_delay_ns{0};
  std::uint64_t gate_actual_ns{0};
  std::uint64_t gate_margin_ns{0};
  bool gate_valid{false};
  OverlapMetrics metrics;
  OverlapStatus instrumentation_status{OverlapStatus::Success};
  OverlapStatus status{OverlapStatus::Success};
  std::int64_t native_error{0};
  std::string message;
};

struct OverlapStatistics {
  std::size_t valid_samples{0};
  trace::Distribution c0_ns, t0_ns, cc_ns, tc_ns, makespan_ns;
  trace::Distribution added_ns, added_percent, compute_slowdown;
  trace::Distribution transfer_slowdown, hidden_fraction;
  trace::Distribution primary_crosscheck_delta_ns;
  double fit_rate_1_percent{0}, fit_rate_2_percent{0}, fit_rate_5_percent{0};
  bool p99_meaningful{false};
};

struct OverlapCellResult {
  OverlapConfiguration configuration;
  WorkloadProfile profile;
  std::vector<OverlapSample> samples;
  OverlapStatistics statistics;
  TransferTelemetry telemetry_before, telemetry_after;
  bool refined{false};
  std::string refinement_reason;
  OverlapStatus status{OverlapStatus::Success};
  std::string message;
};

struct InstrumentationObservation {
  ComputeWorkload workload{ComputeWorkload::SyntheticAlu};
  TransferDirection direction{TransferDirection::H2D};
  std::uint64_t transfer_bytes{0};
  double target_compute_us{0};
  InstrumentationMode mode{InstrumentationMode::Minimal};
  trace::Distribution compute_ns, transfer_ns, host_ns, topology_overhead_ns;
  double compute_bias_percent{0};
  double transfer_bias_percent{0};
  bool rejected{false};
  OverlapStatus status{OverlapStatus::Success};
};

struct SafeTransferEnvelope {
  ComputeWorkload workload{ComputeWorkload::SyntheticAlu};
  TransferDirection direction{TransferDirection::H2D};
  HostMemoryClass memory_class{HostMemoryClass::PinnedHostAlloc};
  double target_compute_us{0};
  double tolerance_percent{0};
  std::uint64_t largest_measured_bytes{0};
  double p99_added_percent{0};
  double fit_rate{0};
  std::string confidence{"UNKNOWN"};
};

struct OverlapPlan {
  std::vector<ComputeWorkload> workloads;
  std::vector<double> compute_windows_us;
  std::vector<std::uint64_t> h2d_sizes, d2h_sizes;
  std::vector<HostMemoryClass> memory_classes;
  std::uint32_t coarse_repetitions{9};
  std::uint32_t refine_repetitions{101};
  std::uint32_t calibration_repetitions{21};
  std::uint32_t warmups{3};
  std::uint64_t maximum_host_bytes{0};
  std::uint64_t maximum_device_bytes{0};
};

struct OverlapRunOptions {
  OverlapPlan plan;
  bool calibrate{true};
  bool instrumentation_control{false};
  bool run_coarse{true};
  bool run_refinement{false};
  bool dry_run{false};
  bool include_pageable_controls{false};
  bool include_registered_crosscheck{false};
  int device_index{0};
  memory::SafetyPolicy host_safety;
  DeviceSafetyPolicy device_safety;
};

struct OverlapRunReport {
  OverlapPlan plan;
  DeviceMemoryInfo device;
  std::vector<WorkloadProfile> profiles;
  std::vector<InstrumentationObservation> instrumentation;
  std::vector<OverlapCellResult> cells;
  std::vector<SafeTransferEnvelope> envelopes;
  OverlapStatus status{OverlapStatus::Success};
  std::string message;
};

class IOverlapProvider {
public:
  virtual ~IOverlapProvider() = default;
  [[nodiscard]] virtual bool SupportsCuda() const noexcept = 0;
  virtual DeviceMemoryInfo InspectDevice(int device_index) = 0;
  virtual TransferTelemetry ReadTelemetry(int device_index) = 0;
  virtual TransferError Prepare(int device_index, HostMemoryClass memory_class,
                                std::uint64_t transfer_capacity_bytes) = 0;
  virtual WorkloadProfile Calibrate(ComputeWorkload workload, double target_us,
                                    std::uint32_t warmups,
                                    std::uint32_t repetitions) = 0;
  virtual RawOverlapTiming RunTopology(const WorkloadProfile &profile,
                                       TransferDirection direction,
                                       std::uint64_t transfer_bytes,
                                       SampleMode sample_mode,
                                       InstrumentationMode instrumentation,
                                       std::uint64_t gate_delay_ns,
                                       std::uint64_t gate_margin_ns,
                                       std::uint32_t repetition,
                                       std::atomic_bool *external_gate_released = nullptr) = 0;
  virtual TransferError Validate(const WorkloadProfile &profile,
                                 TransferDirection direction,
                                 std::uint64_t transfer_bytes) = 0;
  virtual TransferError Release() noexcept = 0;
};

[[nodiscard]] IOverlapProvider &NativeOverlapProvider();
[[nodiscard]] OverlapPlan DefaultOverlapPlan();
[[nodiscard]] std::vector<std::uint64_t> SelectRefinementSizes(
    const std::vector<OverlapCellResult> &coarse_cells);
[[nodiscard]] bool ShouldPruneOutsideFullHide(double c0_ns, double t0_ns,
                                              double ratio = 4.0) noexcept;
[[nodiscard]] std::uint64_t AdaptiveGateDelayNs(
    const std::vector<std::uint64_t> &submission_ns,
    std::uint64_t minimum_delay_ns = 2'000'000) noexcept;
[[nodiscard]] OverlapMetrics CalculateOverlapMetrics(
    double c0_ns, double t0_ns, double cc_ns, double tc_ns,
    double makespan_ns) noexcept;
[[nodiscard]] OverlapStatistics CalculateOverlapStatistics(
    const std::vector<OverlapSample> &samples);
[[nodiscard]] std::vector<SafeTransferEnvelope> CalculateSafeEnvelopes(
    const std::vector<OverlapCellResult> &cells);
[[nodiscard]] OverlapRunReport RunOverlapLaboratory(
    IOverlapProvider &provider, const memory::MemorySnapshot &host,
    const OverlapRunOptions &options);
[[nodiscard]] std::string OverlapConfigurationIdentity(
    const OverlapConfiguration &configuration);
[[nodiscard]] std::string OverlapRunReportToJson(const OverlapRunReport &report);
[[nodiscard]] std::string FormatOverlapRunReport(const OverlapRunReport &report);
[[nodiscard]] const char *ToString(ComputeWorkload value) noexcept;
[[nodiscard]] const char *ToString(OverlapPhase value) noexcept;
[[nodiscard]] const char *ToString(SampleMode value) noexcept;
[[nodiscard]] const char *ToString(InstrumentationMode value) noexcept;
[[nodiscard]] const char *ToString(OverlapStatus value) noexcept;

} // namespace sidecar::cuda
