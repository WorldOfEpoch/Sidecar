#include "sidecar/cuda/overlap.hpp"
#include "sidecar/database/database.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace sidecar::cuda;

void Expect(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

class MockProvider final : public IOverlapProvider {
public:
  bool SupportsCuda() const noexcept override { return supported; }
  DeviceMemoryInfo InspectDevice(int index) override {
    return {supported, index, "mock-overlap", 12ULL << 30, 16ULL << 30, 2};
  }
  TransferTelemetry ReadTelemetry(int) override {
    TransferTelemetry value;
    value.available = true;
    value.pcie_generation = 4;
    value.pcie_width = 16;
    return value;
  }
  TransferError Prepare(int, HostMemoryClass, std::uint64_t bytes) override {
    prepared = true;
    capacity = bytes;
    return fail_prepare
               ? TransferError{TransferStatus::CudaOutOfMemory, 2, "mock OOM"}
               : TransferError{};
  }
  WorkloadProfile Calibrate(ComputeWorkload workload, double target,
                            std::uint32_t, std::uint32_t repetitions) override {
    WorkloadProfile value;
    value.workload = workload;
    value.target_us = target;
    value.calibration_samples = repetitions;
    value.calibrated_us = sidecar::trace::CalculateDistribution(
        std::vector<double>(repetitions, target));
    value.alu_iterations = 100;
    value.memory_working_set_bytes = 256ULL << 20;
    value.memory_passes = 2;
    value.memory_block_size = 256;
    value.gemm_m = value.gemm_n = value.gemm_k = 2048;
    value.gemm_repetitions = 4;
    value.cublas_version = 120800;
    value.validated = !fail_calibration;
    value.status = fail_calibration ? OverlapStatus::ComputeValidationFailure
                                    : OverlapStatus::Success;
    return value;
  }
  RawOverlapTiming RunTopology(const WorkloadProfile &, TransferDirection,
                               std::uint64_t bytes, SampleMode sample_mode,
                               InstrumentationMode instrumentation,
                               std::uint64_t gate_delay,
                               std::uint64_t gate_margin,
                               std::uint32_t repetition,
                               std::atomic_bool *external_gate_released) override {
    if (external_gate_released)
      external_gate_released->store(true, std::memory_order_release);
    if (external_gate_released) external_gate_released->notify_all();
    RawOverlapTiming value;
    const auto compute = 1'000'000ULL + repetition * 100;
    const auto transfer = 80'000ULL + bytes / 1024;
    value.compute_ns = sample_mode == SampleMode::TransferOnly ? 0 : compute;
    value.transfer_ns = sample_mode == SampleMode::ComputeOnly ? 0 : transfer;
    if (inject_gate_bias && instrumentation != InstrumentationMode::Minimal)
      value.compute_ns += 100'000;
    if (inject_full_bias && instrumentation == InstrumentationMode::Full)
      value.compute_ns += 100'000;
    if (inject_unresolved_full_bias &&
        instrumentation == InstrumentationMode::Full) {
      if (repetition < 20)
        value.compute_ns += 100'000;
      else
        value.compute_ns -= 100'000;
    }
    value.makespan_device_primary_ns =
        sample_mode == SampleMode::Concurrent
            ? compute + 10'000
            : std::max(value.compute_ns, value.transfer_ns);
    value.makespan_device_crosscheck_ns = value.makespan_device_primary_ns + 64;
    value.makespan_host_ns = value.makespan_device_primary_ns + 1000;
    value.host_submission_ns = 20'000;
    value.gate_actual_ns = gate_delay;
    value.gate_valid = !invalid_gate &&
                       value.host_submission_ns + gate_margin < gate_delay;
    value.status = value.gate_valid ? OverlapStatus::Success
                                    : OverlapStatus::InvalidGate;
    return value;
  }
  TransferError Validate(const WorkloadProfile &, TransferDirection,
                         std::uint64_t) override {
    return fail_validation
               ? TransferError{TransferStatus::VerificationFailure, 0,
                               "mock validation failure"}
               : TransferError{};
  }
  TransferError Release() noexcept override {
    prepared = false;
    ++releases;
    return {};
  }

  bool supported{true}, prepared{false}, fail_prepare{false};
  bool fail_calibration{false}, fail_validation{false}, invalid_gate{false};
  bool inject_gate_bias{false}, inject_full_bias{false};
  bool inject_unresolved_full_bias{false};
  int releases{0};
  std::uint64_t capacity{0};
};

sidecar::memory::MemorySnapshot Host() {
  sidecar::memory::MemorySnapshot value;
  value.installed_physical_bytes = 16ULL << 30;
  value.visible_physical_bytes = 16ULL << 30;
  value.available_physical_bytes = 12ULL << 30;
  value.system_commit_bytes = 2ULL << 30;
  value.system_commit_limit_bytes = 24ULL << 30;
  return value;
}

void TestPlanAndGate() {
  const auto plan = DefaultOverlapPlan();
  Expect(plan.workloads.size() == 3 && plan.compute_windows_us.size() == 10,
         "default plan coverage");
  Expect(plan.h2d_sizes.front() == (1ULL << 20) &&
             plan.h2d_sizes.back() == (1024ULL << 20) &&
             std::find(plan.h2d_sizes.begin(), plan.h2d_sizes.end(),
                       512ULL << 20) != plan.h2d_sizes.end(),
         "H2D plan boundaries");
  Expect(plan.d2h_sizes.size() == 4, "reduced D2H plan");
  Expect(AdaptiveGateDelayNs({100, 200, 300}, 1000) == 1200,
         "adaptive gate safety factor");
  Expect(AdaptiveGateDelayNs({}, 5000) == 5000, "empty gate history");
  Expect(ShouldPruneOutsideFullHide(100, 500), "impossible region pruning");
  Expect(!ShouldPruneOutsideFullHide(100, 400), "pruning threshold changed");
  OverlapConfiguration configuration;
  configuration.transfer_bytes = 2ULL << 20;
  const auto identity = OverlapConfigurationIdentity(configuration);
  Expect(identity.size() == 64 &&
             identity == OverlapConfigurationIdentity(configuration),
         "overlap configuration identity is unstable");
  configuration.transfer_bytes = 64ULL << 20;
  Expect(identity != OverlapConfigurationIdentity(configuration),
         "overlap identity omitted transfer size");
}

void TestOverlapMath() {
  const auto value = CalculateOverlapMetrics(1000, 500, 1100, 600, 1200);
  Expect(value.critical_path_delta_ns == 200 &&
             value.compute_path_added_ns == 200 &&
             value.compute_path_added_percent == 20,
         "compute path math");
  Expect(std::abs(value.compute_slowdown - 0.1) < 1e-12 &&
             std::abs(value.transfer_slowdown - 0.2) < 1e-12,
         "slowdown math");
  Expect(std::abs(value.overlap_efficiency_raw - 0.6) < 1e-12 &&
             std::abs(value.hidden_fraction_raw - 0.6) < 1e-12,
         "overlap math");
  const auto above = CalculateOverlapMetrics(1000, 500, 1000, 500, 900);
  Expect(above.overlap_efficiency_raw > 1 &&
             above.overlap_efficiency_normalized == 1 &&
             above.hidden_fraction_raw > 1 &&
             above.hidden_fraction_normalized == 1,
         "raw values above one were not preserved");
  const auto negative = CalculateOverlapMetrics(1000, 500, 1300, 800, 1700);
  Expect(negative.overlap_efficiency_raw < 0 &&
             negative.overlap_efficiency_normalized == 0,
         "negative raw overlap was not preserved");
}

OverlapSample Sample(std::uint32_t repetition, double added_percent) {
  OverlapSample value;
  value.repetition = repetition;
  value.baseline_block_id = repetition + 1;
  value.c0_reference_ns = 1'000'000;
  value.t0_reference_ns = 100'000;
  value.cc_ns = 1'000'000;
  value.tc_ns = 100'000;
  value.makespan_device_primary_ns =
      static_cast<std::uint64_t>(1'000'000 * (1.0 + added_percent / 100.0));
  value.makespan_device_crosscheck_ns = value.makespan_device_primary_ns + 64;
  value.gate_valid = true;
  value.status = OverlapStatus::Success;
  value.metrics = CalculateOverlapMetrics(
      static_cast<double>(value.c0_reference_ns),
      static_cast<double>(value.t0_reference_ns),
      static_cast<double>(value.cc_ns), static_cast<double>(value.tc_ns),
      static_cast<double>(value.makespan_device_primary_ns));
  return value;
}

void TestStatisticsFitAndEnvelope() {
  OverlapCellResult small, large;
  for (std::uint32_t i = 0; i < 100; ++i) {
    small.samples.push_back(Sample(i, i >= 98 ? 1.5 : 0.5));
    large.samples.push_back(Sample(i, i >= 98 ? 3.0 : 1.5));
  }
  small.statistics = CalculateOverlapStatistics(small.samples);
  large.statistics = CalculateOverlapStatistics(large.samples);
  Expect(small.statistics.p99_meaningful &&
             small.statistics.fit_rate_2_percent == 1.0,
         "P99/fit calculation");
  small.configuration = {ComputeWorkload::SyntheticAlu,
                         TransferDirection::H2D,
                         HostMemoryClass::PinnedHostAlloc,
                         2ULL << 20,
                         1000};
  large.configuration = small.configuration;
  large.configuration.transfer_bytes = 64ULL << 20;
  const auto envelopes = CalculateSafeEnvelopes({small, large});
  auto two = std::find_if(envelopes.begin(), envelopes.end(), [](const auto &e) {
    return e.tolerance_percent == 2.0;
  });
  Expect(two != envelopes.end() && two->largest_measured_bytes == (2ULL << 20),
         "safe envelope did not use largest qualifying measured point");
  const auto selected = SelectRefinementSizes({small, large});
  Expect(!selected.empty(), "classification boundary was not refined");
}

void TestMockLaboratoryAndErrors() {
  MockProvider provider;
  OverlapRunOptions options;
  options.plan = DefaultOverlapPlan();
  options.plan.workloads = {ComputeWorkload::SyntheticAlu};
  options.plan.compute_windows_us = {1000};
  options.plan.h2d_sizes = {2ULL << 20};
  options.plan.d2h_sizes.clear();
  options.plan.memory_classes = {HostMemoryClass::PinnedHostAlloc};
  options.plan.coarse_repetitions = 4;
  options.plan.calibration_repetitions = 3;
  options.plan.maximum_host_bytes = 2ULL << 20;
  options.plan.maximum_device_bytes = 770ULL << 20;
  const auto report = RunOverlapLaboratory(provider, Host(), options);
  Expect(report.status == OverlapStatus::Success && report.profiles.size() == 1 &&
             report.cells.size() == 1 && report.cells[0].samples.size() == 4,
         "mock overlap laboratory coverage");
  Expect(report.cells[0].samples[0].baseline_block_id == 1 &&
             report.cells[0].samples[1].baseline_block_id == 2,
         "baseline pairing IDs missing");
  Expect(report.cells[0].samples[0].gate_valid,
         "valid pending gate was rejected");
  const auto json_one = OverlapRunReportToJson(report);
  const auto json_two = OverlapRunReportToJson(report);
  Expect(json_one == json_two && json_one.find("sidecar.cuda-overlap.v1") !=
                                    std::string::npos,
         "overlap JSON is unstable");

  options.plan.d2h_sizes = options.plan.h2d_sizes;
  options.plan.h2d_sizes.clear();
  const auto d2h_only = RunOverlapLaboratory(provider, Host(), options);
  Expect(d2h_only.status == OverlapStatus::Success &&
             d2h_only.cells.size() == 1 &&
             d2h_only.cells.front().configuration.direction ==
                 TransferDirection::D2H,
         "D2H-only plan incorrectly executed an empty H2D pass");

  options.plan.h2d_sizes = options.plan.d2h_sizes;
  options.plan.d2h_sizes.clear();

  options.plan.memory_classes = {HostMemoryClass::PageablePretouched};
  const auto pageable = RunOverlapLaboratory(provider, Host(), options);
  Expect(pageable.status == OverlapStatus::SkippedUnsupported &&
             pageable.cells.empty() &&
             pageable.message.find("pending release gate") !=
                 std::string::npos,
         "pageable pending-gate overlap was not rejected safely");
  options.plan.memory_classes = {HostMemoryClass::PinnedHostAlloc};

  options.run_refinement = true;
  options.plan.refine_repetitions = 5;
  const auto refined = RunOverlapLaboratory(provider, Host(), options);
  Expect(refined.cells.size() == 2 && refined.cells.back().refined &&
             refined.cells.back().samples.size() == 5 &&
             !refined.cells.back().refinement_reason.empty(),
         "automated coarse-to-refinement phase");

  MockProvider unsupported;
  unsupported.supported = false;
  Expect(RunOverlapLaboratory(unsupported, Host(), options).status ==
             OverlapStatus::SkippedUnsupported,
         "unsupported provider status");
  MockProvider failed;
  failed.fail_prepare = true;
  Expect(RunOverlapLaboratory(failed, Host(), options).status ==
             OverlapStatus::CudaError,
         "CUDA allocation error propagation");
}

void TestInstrumentationBiasAndInvalidGate() {
  OverlapRunOptions options;
  options.plan = DefaultOverlapPlan();
  options.plan.workloads = {ComputeWorkload::SyntheticAlu};
  options.plan.compute_windows_us = {1000};
  options.plan.h2d_sizes = {2ULL << 20};
  options.plan.d2h_sizes.clear();
  options.plan.maximum_host_bytes = 2ULL << 20;
  options.plan.maximum_device_bytes = 770ULL << 20;
  options.instrumentation_control = true;
  options.run_coarse = false;
  MockProvider biased;
  biased.inject_full_bias = true;
  const auto report = RunOverlapLaboratory(biased, Host(), options);
  Expect(report.status == OverlapStatus::InstrumentationBias &&
             report.instrumentation.size() == 3 &&
             report.instrumentation.back().rejected,
         "instrumentation bias gate");

  MockProvider matched_gate;
  matched_gate.inject_gate_bias = true;
  const auto matched_report =
      RunOverlapLaboratory(matched_gate, Host(), options);
  Expect(matched_report.status == OverlapStatus::Success &&
             !matched_report.instrumentation[1].rejected &&
             matched_report.instrumentation[1].compute_bias_percent > 9.0 &&
             std::abs(matched_report.instrumentation[2]
                          .compute_bias_percent) < 0.1,
         "common matched gate effect is reported but not confused with full "
         "topology observer bias");

  MockProvider unresolved;
  unresolved.inject_unresolved_full_bias = true;
  const auto unresolved_report =
      RunOverlapLaboratory(unresolved, Host(), options);
  Expect(unresolved_report.status == OverlapStatus::Success &&
             !unresolved_report.instrumentation.back().rejected &&
             unresolved_report.instrumentation.back().status ==
                 OverlapStatus::MeasurementSensitive,
         "unresolved paired instrumentation delta classification");

  options.instrumentation_control = false;
  options.run_coarse = true;
  options.plan.coarse_repetitions = 3;
  MockProvider invalid;
  invalid.invalid_gate = true;
  const auto invalid_report = RunOverlapLaboratory(invalid, Host(), options);
  Expect(!invalid_report.cells.empty() &&
             invalid_report.cells.front().statistics.valid_samples == 0 &&
             invalid_report.cells.front().status == OverlapStatus::InvalidGate,
         "invalid gate samples were included");
}

std::filesystem::path TemporaryDatabase() {
  return std::filesystem::temp_directory_path() /
         ("sidecar-overlap-" +
          std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".db");
}

void TestMigration6Persistence() {
  const auto path = TemporaryDatabase();
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
      std::filesystem::remove(path.string() + "-wal", ignored);
      std::filesystem::remove(path.string() + "-shm", ignored);
    }
  } cleanup{path};
  auto database = sidecar::database::Database::Open(
      path, sidecar::database::OpenMode::CreateOrOpen);
  database.Initialize();
  Expect(database.Status().schema_version == sidecar::database::kCurrentSchemaVersion &&
             database.Status().latest_migration == sidecar::database::kCurrentSchemaVersion,
         "current migration status");
  database.UpsertHardwareProfile({"overlap-machine", "host", "Windows", "11",
                                  "1", "CPU", 1, 1, 1, 1024, 512, "{}"});
  const auto session = database.StartBenchmarkSession(
      {"overlap-machine", "M0-FROZEN-1", "commit", std::nullopt,
       "CUDA_OVERLAP", std::nullopt});
  const auto profile = database.InsertComputeWorkloadProfile(
      {session, "SYNTHETIC_ALU", 1000, "{}", 3, 100, 0, 0, 0, 0,
       0, 0, 0, 0, "", "", "", "", "", "", std::nullopt, true,
       "SUCCESS", std::nullopt});
  const auto configuration = database.InsertCudaOverlapConfiguration(
      {session, profile, "identity", "COARSE", "H2D", "PINNED_HOSTALLOC",
       2LL << 20, 1000, 1, 1, 2'000'000, 250'000, 0, "FULL", "",
       "SUCCESS"});
  const auto benchmark = database.InsertCudaOverlapBenchmark(
      {configuration, "SUCCESS", "{}", "{}", "{}", false, std::nullopt});
  sidecar::database::CudaOverlapSampleInput sample;
  sample.benchmark_id = benchmark;
  sample.gate_valid = true;
  sample.fit_1_percent = sample.fit_2_percent = sample.fit_5_percent = true;
  sample.instrumentation_status = sample.status = "SUCCESS";
  database.InsertCudaOverlapSample(sample);
  database.InsertCudaOverlapFitProfile(
      {session, "SYNTHETIC_ALU", "H2D", "PINNED_HOSTALLOC", 1000, 2,
       2LL << 20, 1, 1, "TRUSTED"});
  database.InsertInstrumentationControl(
      {session, "SYNTHETIC_ALU", "H2D", "FULL", 2LL << 20, 1000,
       "{}", "{}", "{}", 0, 0, false, "SUCCESS"});
  const auto counts = database.OverlapCounts();
  Expect(counts.profiles == 1 && counts.configurations == 1 &&
             counts.benchmarks == 1 && counts.samples == 1 &&
             counts.fit_profiles == 1 && counts.instrumentation_controls == 1,
         "overlap persistence counts");
  const auto envelopes = database.LatestOverlapEnvelopes();
  Expect(envelopes.size() == 1 &&
             envelopes.front().largest_measured_bytes == (2LL << 20),
         "overlap envelope report persistence");
}

} // namespace

int main() {
  try {
    TestPlanAndGate();
    TestOverlapMath();
    TestStatisticsFitAndEnvelope();
    TestMockLaboratoryAndErrors();
    TestInstrumentationBiasAndInvalidGate();
    TestMigration6Persistence();
    std::cout << "all CUDA overlap unit tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CUDA overlap unit test failure: " << error.what() << '\n';
    return 1;
  }
}
