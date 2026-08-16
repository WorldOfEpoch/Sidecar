#include "sidecar/cuda/overlap_cli.hpp"

#include "sidecar/cuda/overlap.hpp"
#include "sidecar/database/database.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/memory/host_memory.hpp"
#include "sidecar/version.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>
#include <tuple>

namespace sidecar::cuda {
namespace {
constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

std::string Lower(std::string_view value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return result;
}

std::uint64_t ParseSize(std::string value) {
  std::uint64_t multiplier = 1;
  const auto lower = Lower(value);
  if (lower.ends_with("mib")) {
    multiplier = kMiB;
    value.resize(value.size() - 3);
  } else if (lower.ends_with("mb")) {
    multiplier = 1'000'000;
    value.resize(value.size() - 2);
  } else if (lower.ends_with('m')) {
    multiplier = kMiB;
    value.pop_back();
  } else if (lower.ends_with("gib")) {
    multiplier = 1024ULL * kMiB;
    value.resize(value.size() - 3);
  } else if (lower.ends_with('g')) {
    multiplier = 1024ULL * kMiB;
    value.pop_back();
  }
  const auto number = std::stoull(value);
  if (!number || number > UINT64_MAX / multiplier)
    throw std::invalid_argument("invalid overlap transfer size");
  return number * multiplier;
}

template <typename T, typename Parser>
std::vector<T> ParseList(const std::string &text, Parser parser) {
  std::vector<T> result;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const auto end = text.find(',', begin);
    result.push_back(parser(text.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin)));
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return result;
}

double ParseWindow(std::string value) {
  const auto lower = Lower(value);
  double multiplier = 1000.0;
  if (lower.ends_with("us")) {
    multiplier = 1.0;
    value.resize(value.size() - 2);
  } else if (lower.ends_with("ms")) {
    multiplier = 1000.0;
    value.resize(value.size() - 2);
  }
  const auto parsed = std::stod(value) * multiplier;
  if (!(parsed > 0))
    throw std::invalid_argument("invalid compute window");
  return parsed;
}

ComputeWorkload ParseWorkload(std::string_view value) {
  const auto lower = Lower(value);
  if (lower == "alu" || lower == "synthetic_alu")
    return ComputeWorkload::SyntheticAlu;
  if (lower == "memory" || lower == "memory_bound")
    return ComputeWorkload::MemoryBound;
  if (lower == "gemm" || lower == "fp16_gemm")
    return ComputeWorkload::Fp16Gemm;
  throw std::invalid_argument("compute must be alu, memory, gemm, or all");
}

TransferDirection ParseDirection(std::string_view value) {
  const auto lower = Lower(value);
  if (lower == "h2d")
    return TransferDirection::H2D;
  if (lower == "d2h")
    return TransferDirection::D2H;
  throw std::invalid_argument("direction must be h2d or d2h");
}

HostMemoryClass ParseBackend(std::string_view value) {
  const auto lower = Lower(value);
  if (lower == "hostalloc" || lower == "pinned")
    return HostMemoryClass::PinnedHostAlloc;
  if (lower == "registered" || lower == "hostregister")
    return HostMemoryClass::PinnedRegistered;
  if (lower == "pageable")
    return HostMemoryClass::PageablePretouched;
  throw std::invalid_argument(
      "backend must be hostalloc, registered, or pageable");
}

struct CliOptions {
  bool json{false}, dry_run{false}, persist{true};
  bool refine{false}, instrumentation{false};
  int device_index{0};
  std::optional<std::uint32_t> repetitions;
  std::filesystem::path database{std::filesystem::path("data") / "sidecar.db"};
  OverlapPlan plan{DefaultOverlapPlan()};
};

CliOptions ParseOptions(int argc, char **argv) {
  CliOptions options;
  std::optional<TransferDirection> direction;
  for (int index = 4; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--json")
      options.json = true;
    else if (argument == "--dry-run")
      options.dry_run = true;
    else if (argument == "--no-persist")
      options.persist = false;
    else if (argument == "--coarse")
      options.refine = false;
    else if (argument == "--refine")
      options.refine = true;
    else if (argument == "--compute" && index + 1 < argc) {
      const auto value = Lower(argv[++index]);
      if (value == "all")
        options.plan.workloads = {ComputeWorkload::SyntheticAlu,
                                  ComputeWorkload::MemoryBound,
                                  ComputeWorkload::Fp16Gemm};
      else
        options.plan.workloads = {ParseWorkload(value)};
    } else if ((argument == "--compute-window" ||
                argument == "--compute-windows") &&
               index + 1 < argc) {
      options.plan.compute_windows_us = ParseList<double>(argv[++index],
                                                          ParseWindow);
    } else if (argument == "--direction" && index + 1 < argc) {
      direction = ParseDirection(argv[++index]);
    } else if ((argument == "--transfer-size" || argument == "--size" ||
                argument == "--sizes") &&
               index + 1 < argc) {
      const auto sizes =
          ParseList<std::uint64_t>(argv[++index], ParseSize);
      options.plan.h2d_sizes = sizes;
      options.plan.d2h_sizes = sizes;
    } else if (argument == "--backend" && index + 1 < argc) {
      options.plan.memory_classes = {ParseBackend(argv[++index])};
    } else if (argument == "--repetitions" && index + 1 < argc) {
      const auto value = std::stoull(argv[++index]);
      if (!value || value > UINT32_MAX)
        throw std::invalid_argument("repetitions is out of range");
      options.repetitions = static_cast<std::uint32_t>(value);
    } else if (argument == "--device" && index + 1 < argc) {
      options.device_index = std::stoi(argv[++index]);
    } else if (argument == "--database" && index + 1 < argc) {
      options.database = argv[++index];
    } else {
      throw std::invalid_argument("invalid CUDA overlap option: " +
                                  std::string(argument));
    }
  }
  if (direction == TransferDirection::H2D)
    options.plan.d2h_sizes.clear();
  else if (direction == TransferDirection::D2H) {
    options.plan.d2h_sizes = options.plan.h2d_sizes;
    options.plan.h2d_sizes.clear();
  }
  std::uint64_t maximum = 0;
  for (const auto bytes : options.plan.h2d_sizes)
    maximum = std::max(maximum, bytes);
  for (const auto bytes : options.plan.d2h_sizes)
    maximum = std::max(maximum, bytes);
  options.plan.maximum_host_bytes = maximum ? maximum : kMiB;
  options.plan.maximum_device_bytes = options.plan.maximum_host_bytes +
                                      768ULL * kMiB;
  if (options.repetitions) {
    options.plan.coarse_repetitions = *options.repetitions;
    options.plan.refine_repetitions = *options.repetitions;
    options.plan.calibration_repetitions = *options.repetitions;
  }
  return options;
}

std::string DistributionJson(const trace::Distribution &value) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"mean\":" << value.mean
      << ",\"median\":" << value.median << ",\"p50\":" << value.p50
      << ",\"p90\":" << value.p90 << ",\"p95\":" << value.p95
      << ",\"p99\":" << value.p99 << ",\"minimum\":" << value.minimum
      << ",\"maximum\":" << value.maximum << ",\"stddev\":"
      << value.stddev << '}';
  return out.str();
}

std::string StatisticsJson(const OverlapStatistics &value) {
  std::ostringstream out;
  out << "{\"valid_samples\":" << value.valid_samples
      << ",\"c0_ns\":" << DistributionJson(value.c0_ns)
      << ",\"t0_ns\":" << DistributionJson(value.t0_ns)
      << ",\"cc_ns\":" << DistributionJson(value.cc_ns)
      << ",\"tc_ns\":" << DistributionJson(value.tc_ns)
      << ",\"makespan_ns\":" << DistributionJson(value.makespan_ns)
      << ",\"added_ns\":" << DistributionJson(value.added_ns)
      << ",\"added_percent\":" << DistributionJson(value.added_percent)
      << ",\"compute_slowdown\":" << DistributionJson(value.compute_slowdown)
      << ",\"transfer_slowdown\":" << DistributionJson(value.transfer_slowdown)
      << ",\"hidden_fraction\":" << DistributionJson(value.hidden_fraction)
      << ",\"primary_crosscheck_delta_ns\":"
      << DistributionJson(value.primary_crosscheck_delta_ns)
      << ",\"fit_rate_1pct\":" << value.fit_rate_1_percent
      << ",\"fit_rate_2pct\":" << value.fit_rate_2_percent
      << ",\"fit_rate_5pct\":" << value.fit_rate_5_percent
      << ",\"p99_meaningful\":"
      << (value.p99_meaningful ? "true" : "false") << '}';
  return out.str();
}

std::string TelemetryJson(const TransferTelemetry &value) {
  const auto number = [](const auto &optional) {
    return optional ? std::to_string(*optional) : std::string("null");
  };
  std::ostringstream out;
  out << "{\"available\":" << (value.available ? "true" : "false")
      << ",\"temperature_c\":" << number(value.temperature_c)
      << ",\"graphics_clock_mhz\":" << number(value.graphics_clock_mhz)
      << ",\"memory_clock_mhz\":" << number(value.memory_clock_mhz)
      << ",\"power_watts\":" << number(value.power_watts)
      << ",\"power_limit_watts\":" << number(value.power_limit_watts)
      << ",\"pcie_generation\":" << number(value.pcie_generation)
      << ",\"pcie_width\":" << number(value.pcie_width) << '}';
  return out.str();
}

hardware::DiscoveryReport DiscoverMachine() {
  auto provider = hardware::CreateNativeDiscoveryProvider();
  hardware::HardwareDiscoveryService service(*provider);
  return service.Discover();
}

std::int64_t StartSession(database::Database &database,
                          const hardware::DiscoveryReport &machine,
                          std::string_view command) {
  hardware::PersistDiscovery(database, machine);
  const auto version = CurrentVersionInfo();
  return database.StartBenchmarkSession(
      {machine.identity.machine_hash,
       std::string(version.spec_version),
       std::string(version.git_commit),
       std::nullopt,
       "CUDA_OVERLAP",
       std::string("WU6 ") + std::string(command),
       machine.snapshot.cuda_runtime_version
           ? std::optional<std::int64_t>(*machine.snapshot.cuda_runtime_version)
           : std::nullopt,
       machine.snapshot.cuda_driver_version
           ? std::optional<std::int64_t>(*machine.snapshot.cuda_driver_version)
           : std::nullopt,
       std::nullopt,
       machine.snapshot.nvidia_driver_version.empty()
           ? std::nullopt
           : std::optional<std::string>(
                 machine.snapshot.nvidia_driver_version),
       machine.snapshot.operating_system.version});
}

void PersistReport(database::Database &database, std::int64_t session,
                   const OverlapRunReport &report) {
  using Key = std::tuple<ComputeWorkload, double>;
  std::map<Key, std::int64_t> profile_ids;
  database.BeginWriteTransaction();
  try {
    for (const auto &profile : report.profiles) {
      const auto id = database.InsertComputeWorkloadProfile(
          {session,
           ToString(profile.workload),
           profile.target_us,
           DistributionJson(profile.calibrated_us),
           profile.calibration_samples,
           static_cast<std::int64_t>(profile.alu_iterations),
           static_cast<std::int64_t>(profile.memory_working_set_bytes),
           profile.memory_passes,
           profile.memory_block_size,
           profile.memory_elements_per_thread,
           profile.gemm_m,
           profile.gemm_n,
           profile.gemm_k,
           profile.gemm_repetitions,
           profile.gemm_a_type,
           profile.gemm_b_type,
           profile.gemm_c_type,
           profile.gemm_compute_type,
           profile.gemm_math_mode,
           profile.gemm_algorithm,
           profile.cublas_version,
           profile.validated,
           ToString(profile.status),
           profile.message.empty()
               ? std::nullopt
               : std::optional<std::string>(profile.message)});
      profile_ids[{profile.workload, profile.target_us}] = id;
    }
    for (const auto &cell : report.cells) {
      const auto &c = cell.configuration;
      const auto found = profile_ids.find({c.workload, c.target_compute_us});
      if (found == profile_ids.end())
        continue;
      const auto configuration_id = database.InsertCudaOverlapConfiguration(
          {session,
           found->second,
           OverlapConfigurationIdentity(c),
           ToString(c.phase),
           ToString(c.direction),
           ToString(c.memory_class),
           static_cast<std::int64_t>(c.transfer_bytes),
           c.target_compute_us,
           c.repetitions,
           static_cast<std::int64_t>(c.ordering_seed),
           static_cast<std::int64_t>(c.gate_delay_ns),
           static_cast<std::int64_t>(c.gate_margin_ns),
           c.device_index,
           ToString(InstrumentationMode::Full),
           cell.refinement_reason,
           ToString(cell.status)});
      const auto benchmark_id = database.InsertCudaOverlapBenchmark(
          {configuration_id,
           ToString(cell.status),
           StatisticsJson(cell.statistics),
           TelemetryJson(cell.telemetry_before),
           TelemetryJson(cell.telemetry_after),
           cell.refined,
           cell.message.empty() ? std::nullopt
                                : std::optional<std::string>(cell.message)});
      for (const auto &sample : cell.samples) {
        const auto &m = sample.metrics;
        database.InsertCudaOverlapSample(
            {benchmark_id,
             sample.repetition,
             static_cast<std::int64_t>(sample.baseline_block_id),
             static_cast<std::int64_t>(sample.c0_reference_ns),
             static_cast<std::int64_t>(sample.t0_reference_ns),
             static_cast<std::int64_t>(sample.cc_ns),
             static_cast<std::int64_t>(sample.tc_ns),
             static_cast<std::int64_t>(sample.makespan_device_primary_ns),
             static_cast<std::int64_t>(sample.makespan_device_crosscheck_ns),
             static_cast<std::int64_t>(sample.makespan_host_ns),
             static_cast<std::int64_t>(sample.host_submission_ns),
             static_cast<std::int64_t>(sample.gate_delay_ns),
             static_cast<std::int64_t>(sample.gate_actual_ns),
             static_cast<std::int64_t>(sample.gate_margin_ns),
             m.critical_path_delta_ns,
             m.compute_path_added_ns,
             m.compute_path_added_percent,
             m.compute_slowdown,
             m.transfer_slowdown,
             m.overlap_efficiency_raw,
             m.overlap_efficiency_normalized,
             m.hidden_fraction_raw,
             m.hidden_fraction_normalized,
             m.compute_retention,
             sample.gate_valid,
             m.fit_1_percent,
             m.fit_2_percent,
             m.fit_5_percent,
             ToString(sample.instrumentation_status),
             ToString(sample.status),
             sample.native_error
                 ? std::optional<std::int64_t>(sample.native_error)
                 : std::nullopt,
             sample.message.empty()
                 ? std::nullopt
                 : std::optional<std::string>(sample.message)});
      }
    }
    for (const auto &envelope : report.envelopes)
      database.InsertCudaOverlapFitProfile(
          {session,
           ToString(envelope.workload),
           ToString(envelope.direction),
           ToString(envelope.memory_class),
           envelope.target_compute_us,
           envelope.tolerance_percent,
           static_cast<std::int64_t>(envelope.largest_measured_bytes),
           envelope.p99_added_percent,
           envelope.fit_rate,
           envelope.confidence});
    for (const auto &control : report.instrumentation)
      database.InsertInstrumentationControl(
          {session,
           ToString(control.workload),
           ToString(control.direction),
           ToString(control.mode),
           static_cast<std::int64_t>(control.transfer_bytes),
           control.target_compute_us,
           DistributionJson(control.compute_ns),
           DistributionJson(control.transfer_ns),
           std::string("{\"host_ns\":") + DistributionJson(control.host_ns) +
               ",\"topology_overhead_ns\":" +
               DistributionJson(control.topology_overhead_ns) + "}",
           control.compute_bias_percent,
           control.transfer_bias_percent,
           control.rejected,
           ToString(control.status)});
    database.CommitWriteTransaction();
  } catch (...) {
    database.RollbackWriteTransaction();
    throw;
  }
}

std::string PlanJson(const OverlapPlan &plan) {
  std::ostringstream out;
  out << "{\"schema\":\"sidecar.cuda-overlap-plan.v1\",\"workloads\":[";
  for (std::size_t i = 0; i < plan.workloads.size(); ++i) {
    if (i)
      out << ',';
    out << '"' << ToString(plan.workloads[i]) << '"';
  }
  out << "],\"compute_windows_us\":[";
  for (std::size_t i = 0; i < plan.compute_windows_us.size(); ++i) {
    if (i)
      out << ',';
    out << plan.compute_windows_us[i];
  }
  const auto sizes = [&](const auto &values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i)
        out << ',';
      out << values[i];
    }
    out << ']';
  };
  out << "],\"h2d_sizes_bytes\":";
  sizes(plan.h2d_sizes);
  out << ",\"d2h_sizes_bytes\":";
  sizes(plan.d2h_sizes);
  std::size_t directional_cells = 0;
  for (const auto target : plan.compute_windows_us) {
    for (const auto bytes : plan.h2d_sizes)
      if (bytes != 1024ULL * kMiB || target == 1000 || target >= 32000)
        ++directional_cells;
    directional_cells += plan.d2h_sizes.size();
  }
  const auto matrix_size = plan.workloads.size() * directional_cells *
                           plan.memory_classes.size();
  out << ",\"coarse_repetitions\":" << plan.coarse_repetitions
      << ",\"refine_repetitions\":" << plan.refine_repetitions
      << ",\"calibration_repetitions\":" << plan.calibration_repetitions
      << ",\"matrix_cells\":" << matrix_size
      << ",\"maximum_host_bytes\":" << plan.maximum_host_bytes
      << ",\"maximum_device_bytes\":" << plan.maximum_device_bytes
      << ",\"primary_backend\":\"PERSISTENT_CUDA_HOST_ALLOC\""
      << ",\"flight_recorder\":\"OFF_PER_SAMPLE\""
      << ",\"scope\":\"COMPUTE_TRANSFER_OVERLAP_ONLY\"}";
  return out.str();
}

int Info(const CliOptions &options) {
  const auto device = NativeOverlapProvider().InspectDevice(options.device_index);
  if (options.json) {
    std::cout << "{\"schema\":\"sidecar.cuda-overlap-info.v1\",\"status\":\""
              << (device.cuda_supported ? "SUCCESS" : "SKIPPED_UNSUPPORTED")
              << "\",\"device\":\"" << device.name
              << "\",\"async_engines\":" << device.async_engine_count
              << ",\"workloads\":[\"SYNTHETIC_ALU\",\"MEMORY_BOUND\",\"FP16_GEMM\"]}"
              << '\n';
  } else {
    std::cout << "SIDECAR CUDA OVERLAP INFORMATION\n\nStatus: "
              << (device.cuda_supported ? "SUCCESS" : "SKIPPED_UNSUPPORTED")
              << "\nDevice: " << (device.name.empty() ? "unknown" : device.name)
              << "\nWorkloads: SYNTHETIC_ALU, MEMORY_BOUND, FP16_GEMM\n"
              << "Primary backend: persistent cudaHostAlloc\n";
  }
  return 0;
}

int Plan(const CliOptions &options) {
  if (options.json)
    std::cout << PlanJson(options.plan) << '\n';
  else
    std::cout << "SIDECAR CUDA OVERLAP DRY-RUN PLAN\n\nWorkloads: "
              << options.plan.workloads.size()
              << "\nCompute windows: " << options.plan.compute_windows_us.size()
              << "\nH2D sizes: " << options.plan.h2d_sizes.size()
              << "\nD2H sizes: " << options.plan.d2h_sizes.size()
              << "\nCoarse repetitions: " << options.plan.coarse_repetitions
              << "\nRefine repetitions: " << options.plan.refine_repetitions
              << "\nMaximum pinned bytes: " << options.plan.maximum_host_bytes
              << "\nFlight Recorder per sample: OFF\n";
  return 0;
}

int Report(const CliOptions &options) {
  if (!std::filesystem::exists(options.database)) {
    std::cerr << "database does not exist: " << options.database.string() << '\n';
    return 2;
  }
  auto database = database::Database::Open(
      options.database, database::OpenMode::ExistingReadWrite);
  database.Initialize();
  const auto counts = database.OverlapCounts();
  const auto envelopes = database.LatestOverlapEnvelopes();
  if (options.json) {
    std::cout << "{\"schema\":\"sidecar.cuda-overlap-report.v1\",\"counts\":{"
              << "\"profiles\":" << counts.profiles
              << ",\"configurations\":" << counts.configurations
              << ",\"benchmarks\":" << counts.benchmarks
              << ",\"samples\":" << counts.samples
              << ",\"fit_profiles\":" << counts.fit_profiles
              << ",\"instrumentation_controls\":"
              << counts.instrumentation_controls << "},\"envelopes\":[";
    for (std::size_t i = 0; i < envelopes.size(); ++i) {
      if (i)
        std::cout << ',';
      const auto &e = envelopes[i];
      std::cout << "{\"session_id\":" << e.session_id
                << ",\"workload\":\"" << e.workload
                << "\",\"direction\":\"" << e.direction
                << "\",\"memory\":\"" << e.host_memory_class
                << "\",\"target_us\":" << e.target_compute_us
                << ",\"tolerance_percent\":" << e.tolerance_percent
                << ",\"largest_measured_bytes\":"
                << e.largest_measured_bytes << ",\"p99_added_percent\":"
                << e.p99_added_percent << ",\"fit_rate\":" << e.fit_rate
                << ",\"confidence\":\"" << e.confidence << "\"}";
    }
    std::cout << "]}\n";
  } else {
    std::cout << "SIDECAR CUDA OVERLAP DATABASE REPORT\n\nProfiles: "
              << counts.profiles << "\nConfigurations: " << counts.configurations
              << "\nBenchmarks: " << counts.benchmarks
              << "\nSamples: " << counts.samples
              << "\nSafe envelopes: " << counts.fit_profiles
              << "\nInstrumentation controls: "
              << counts.instrumentation_controls << '\n';
  }
  return 0;
}

int Execute(std::string_view command, CliOptions options) {
  if (command == "info")
    return Info(options);
  if (command == "plan" || options.dry_run)
    return Plan(options);
  if (command == "report")
    return Report(options);

  OverlapRunOptions run;
  run.plan = options.plan;
  run.device_index = options.device_index;
  run.instrumentation_control = command == "instrumentation";
  run.run_coarse = command == "run" || command == "matrix" ||
                   command == "validate";
  run.run_refinement = options.refine;
  if (command == "calibrate") {
    run.run_coarse = false;
    run.run_refinement = false;
  }
  if (command == "validate") {
    run.plan.workloads = {ComputeWorkload::SyntheticAlu,
                          ComputeWorkload::MemoryBound,
                          ComputeWorkload::Fp16Gemm};
    run.plan.compute_windows_us = {1000};
    run.plan.h2d_sizes = {2 * kMiB};
    run.plan.d2h_sizes.clear();
    run.plan.coarse_repetitions = options.repetitions.value_or(3);
    run.plan.calibration_repetitions = options.repetitions.value_or(5);
    run.plan.maximum_host_bytes = 2 * kMiB;
    run.plan.maximum_device_bytes = 770 * kMiB;
  }
  auto report = RunOverlapLaboratory(
      NativeOverlapProvider(), memory::NativeHostMemoryProvider().Snapshot(),
      run);
  std::optional<std::int64_t> session;
  if (options.persist && report.status != OverlapStatus::SkippedUnsupported) {
    if (options.database.has_parent_path())
      std::filesystem::create_directories(options.database.parent_path());
    auto database = database::Database::Open(
        options.database, database::OpenMode::CreateOrOpen);
    database.Initialize();
    const auto machine = DiscoverMachine();
    session = StartSession(database, machine, command);
    PersistReport(database, *session, report);
    database.CompleteBenchmarkSession(
        *session,
        report.status == OverlapStatus::Success ? "COMPLETE" : "FAILED");
  }
  if (options.json)
    std::cout << OverlapRunReportToJson(report) << '\n';
  else {
    std::cout << FormatOverlapRunReport(report);
    if (session)
      std::cout << "Session ID: " << *session << '\n';
  }
  return report.status == OverlapStatus::CudaError ||
                 report.status == OverlapStatus::CublasError ||
                 report.status == OverlapStatus::InternalError
             ? 4
             : 0;
}

} // namespace

int RunOverlapCli(int argc, char **argv) {
  if (argc < 4)
    throw std::invalid_argument(
        "cuda overlap requires info|calibrate|instrumentation|plan|run|matrix|report|validate");
  const std::string_view command(argv[3]);
  if (command != "info" && command != "calibrate" &&
      command != "instrumentation" && command != "plan" &&
      command != "run" && command != "matrix" && command != "report" &&
      command != "validate")
    throw std::invalid_argument("unknown CUDA overlap command");
  return Execute(command, ParseOptions(argc, argv));
}

} // namespace sidecar::cuda
