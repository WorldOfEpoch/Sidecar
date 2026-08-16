#include "sidecar/cuda/transfer_cli.hpp"
#include "sidecar/cuda/transfer.hpp"
#include "sidecar/database/database.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/memory/host_memory.hpp"
#include "sidecar/version.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>

namespace sidecar::cuda {
namespace {
constexpr std::uint64_t kKiB = 1024ULL, kMiB = 1024ULL * 1024,
                        kGiB = 1024ULL * 1024 * 1024;

std::string Lower(std::string_view value) {
  std::string r(value);
  std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return r;
}
std::string Escape(std::string_view value) {
  std::ostringstream out;
  for (const unsigned char character : value) {
    if (character == '"')
      out << "\\\"";
    else if (character == '\\')
      out << "\\\\";
    else if (character == '\n')
      out << "\\n";
    else if (character < 0x20)
      out << '?';
    else
      out << static_cast<char>(character);
  }
  return out.str();
}
std::uint64_t ParseSize(std::string value) {
  std::uint64_t m = 1;
  if (value.size() >= 2) {
    auto s = Lower(value.substr(value.size() - 2));
    if (s == "kb") {
      m = 1000;
      value.resize(value.size() - 2);
    } else if (s == "mb") {
      m = 1000000;
      value.resize(value.size() - 2);
    } else if (s == "gb") {
      m = 1000000000;
      value.resize(value.size() - 2);
    }
  }
  if (m == 1 && !value.empty()) {
    char s = static_cast<char>(
        std::tolower(static_cast<unsigned char>(value.back())));
    if (s == 'k') {
      m = kKiB;
      value.pop_back();
    } else if (s == 'm') {
      m = kMiB;
      value.pop_back();
    } else if (s == 'g') {
      m = kGiB;
      value.pop_back();
    }
  }
  auto n = std::stoull(value);
  if (!n || n > UINT64_MAX / m)
    throw std::invalid_argument("invalid transfer size");
  return n * m;
}
std::vector<std::uint64_t> ParseSizes(const std::string &text) {
  std::vector<std::uint64_t> v;
  std::size_t b = 0;
  while (b <= text.size()) {
    auto e = text.find(',', b);
    v.push_back(ParseSize(
        text.substr(b, e == std::string::npos ? std::string::npos : e - b)));
    if (e == std::string::npos)
      break;
    b = e + 1;
  }
  return v;
}
std::uint32_t ParseCount(const char *value, bool allow_zero,
                         std::string_view name) {
  const auto parsed = std::stoull(value);
  if ((!allow_zero && parsed == 0) || parsed > UINT32_MAX)
    throw std::invalid_argument(std::string(name) + " is out of range");
  return static_cast<std::uint32_t>(parsed);
}
TransferDirection ParseDirection(std::string_view v) {
  auto n = Lower(v);
  if (n == "h2d")
    return TransferDirection::H2D;
  if (n == "d2h")
    return TransferDirection::D2H;
  throw std::invalid_argument("direction must be h2d or d2h");
}
HostMemoryClass ParseMemory(std::string_view v) {
  auto n = Lower(v);
  if (n == "pageable")
    return HostMemoryClass::PageablePretouched;
  if (n == "hostalloc")
    return HostMemoryClass::PinnedHostAlloc;
  if (n == "registered" || n == "hostregister")
    return HostMemoryClass::PinnedRegistered;
  throw std::invalid_argument(
      "memory must be pageable, hostalloc, or registered");
}
CopyApiMode ParseApi(std::string_view v) {
  auto n = Lower(v);
  if (n == "sync")
    return CopyApiMode::Synchronous;
  if (n == "async")
    return CopyApiMode::Asynchronous;
  throw std::invalid_argument("API must be sync or async");
}

struct CliOptions {
  bool json = false, dry = false, persist = true;
  std::vector<TransferDirection> directions;
  std::vector<HostMemoryClass> memories;
  std::vector<CopyApiMode> apis;
  std::vector<std::uint64_t> sizes;
  std::optional<std::uint32_t> repetitions;
  std::uint32_t warmups = 3;
  int device = 0;
  std::filesystem::path database = std::filesystem::path("data") / "sidecar.db";
};
CliOptions ParseOptions(int argc, char **argv, int start) {
  CliOptions o;
  for (int i = start; i < argc; ++i) {
    std::string_view a(argv[i]);
    if (a == "--json")
      o.json = true;
    else if (a == "--dry-run")
      o.dry = true;
    else if (a == "--no-persist")
      o.persist = false;
    else if (a == "--direction" && i + 1 < argc)
      o.directions = {ParseDirection(argv[++i])};
    else if (a == "--memory" && i + 1 < argc)
      o.memories = {ParseMemory(argv[++i])};
    else if (a == "--api" && i + 1 < argc)
      o.apis = {ParseApi(argv[++i])};
    else if ((a == "--size" || a == "--sizes") && i + 1 < argc)
      o.sizes = ParseSizes(argv[++i]);
    else if (a == "--repetitions" && i + 1 < argc)
      o.repetitions = ParseCount(argv[++i], false, "repetitions");
    else if (a == "--warmups" && i + 1 < argc)
      o.warmups = ParseCount(argv[++i], true, "warmups");
    else if (a == "--device" && i + 1 < argc)
      o.device = std::stoi(argv[++i]);
    else if (a == "--database" && i + 1 < argc)
      o.database = argv[++i];
    else
      throw std::invalid_argument("invalid CUDA transfer option: " +
                                  std::string(a));
  }
  return o;
}

std::int64_t StartSession(database::Database &db,
                          const hardware::DiscoveryReport &machine,
                          std::string notes) {
  hardware::PersistDiscovery(db, machine);
  auto v = CurrentVersionInfo();
  return db.StartBenchmarkSession(
      {machine.identity.machine_hash, std::string(v.spec_version),
       std::string(v.git_commit), std::nullopt, "CUDA_TRANSFER",
       std::move(notes),
       machine.snapshot.cuda_runtime_version
           ? std::optional<std::int64_t>(*machine.snapshot.cuda_runtime_version)
           : std::nullopt,
       machine.snapshot.cuda_driver_version
           ? std::optional<std::int64_t>(*machine.snapshot.cuda_driver_version)
           : std::nullopt,
       std::nullopt,
       machine.snapshot.nvidia_driver_version.empty()
           ? std::nullopt
           : std::optional<std::string>(machine.snapshot.nvidia_driver_version),
       machine.snapshot.operating_system.version});
}
std::string StatsJson(const TransferStatistics &s) {
  auto d = [](const trace::Distribution &v) {
    std::ostringstream o;
    o << "{\"mean\":" << v.mean << ",\"median\":" << v.median
      << ",\"p90\":" << v.p90 << ",\"p95\":" << v.p95 << ",\"p99\":" << v.p99
      << ",\"minimum\":" << v.minimum << ",\"maximum\":" << v.maximum
      << ",\"stddev\":" << v.stddev << '}';
    return o.str();
  };
  std::ostringstream o;
  o << "{\"count\":" << s.count
    << ",\"p99_meaningful\":" << (s.p99_meaningful ? "true" : "false")
    << ",\"host_api_ns\":" << d(s.host_api_ns)
    << ",\"device_ns\":" << d(s.device_ns)
    << ",\"end_to_end_ns\":" << d(s.end_to_end_ns)
    << ",\"bytes_per_second\":" << d(s.bytes_per_second) << '}';
  return o.str();
}
std::string ValuesJson(const std::vector<std::uint64_t> &v) {
  std::ostringstream o;
  o << '[';
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i)
      o << ',';
    o << v[i];
  }
  o << ']';
  return o.str();
}

void PersistResults(database::Database &db, std::int64_t session,
                    const std::vector<TransferResult> &results,
                    const std::vector<SaturationProfile> &profiles,
                    const std::vector<std::uint64_t> &candidates,
                    const memory::TimerCalibration &timer, std::uint64_t seed) {
  db.BeginWriteTransaction();
  try {
    for (const auto &r : results) {
      const auto &c = r.configuration;
      const auto buffer_bytes =
          c.experiment == "BATCH_SYNC_EACH" || c.experiment == "BATCH_ONE_SYNC"
              ? c.transfer_bytes * c.batch_count
              : c.transfer_bytes;
      auto cid = db.InsertCudaTransferConfiguration(
          {session,
           TransferConfigurationIdentity(c),
           c.experiment,
           ToString(c.direction),
           ToString(c.memory_class),
           ToString(c.api_mode),
           static_cast<std::int64_t>(c.transfer_bytes),
           c.stream_mode,
           c.batch_count,
           c.chunk_count,
           c.warmup_count,
           c.repetitions,
           c.device_index,
           static_cast<std::int64_t>(buffer_bytes),
           static_cast<std::int64_t>(buffer_bytes),
           c.validation_mode,
           "steady_clock_raw_ns",
           static_cast<std::int64_t>(timer.bracket_overhead_ns),
           static_cast<std::int64_t>(timer.observed_resolution_ns),
           "REUSED_CUDA_EVENTS",
           static_cast<std::int64_t>(seed),
           ToString(r.status)});
      auto bid = db.InsertCudaTransferBenchmark(
          {cid, ToString(r.status), r.verified, r.noisy, r.async_behavior,
           StatsJson(r.statistics),
           r.message.empty() ? std::nullopt
                             : std::optional<std::string>(r.message)});
      for (const auto &s : r.samples)
        db.InsertCudaTransferSample(
            {bid, s.repetition, static_cast<std::int64_t>(s.payload_bytes),
             static_cast<std::int64_t>(s.host_api_raw_ns),
             static_cast<std::int64_t>(s.device_duration_ns),
             static_cast<std::int64_t>(s.end_to_end_raw_ns), ToString(s.status),
             s.cuda_error,
             s.message.empty() ? std::nullopt
                               : std::optional<std::string>(s.message)});
    }
    auto cj = ValuesJson(candidates);
    for (const auto &p : profiles)
      db.InsertCudaTransferProfile(
          {session, ToString(p.direction), ToString(p.memory_class),
           ToString(p.api_mode), p.peak_bytes_per_second,
           static_cast<std::int64_t>(p.peak_size_bytes),
           static_cast<std::int64_t>(p.knee_80_bytes),
           static_cast<std::int64_t>(p.knee_90_bytes),
           static_cast<std::int64_t>(p.knee_95_bytes), cj});
    db.CommitWriteTransaction();
  } catch (...) {
    db.RollbackWriteTransaction();
    throw;
  }
}

std::vector<TransferResult>
SegmentExperiment(ITransferProvider &provider, HostMemoryClass memory,
                  TransferDirection direction, std::uint64_t capacity,
                  const std::vector<std::uint64_t> &sizes,
                  const std::vector<std::uint32_t> &counts, bool chunks,
                  bool sync_each, std::uint32_t repetitions, int device) {
  std::vector<TransferResult> results;
  const auto prepared = provider.Prepare(device, memory, capacity, 1);
  if (!prepared.ok()) {
    TransferResult failed;
    failed.configuration.memory_class = memory;
    failed.configuration.direction = direction;
    failed.configuration.api_mode = CopyApiMode::Asynchronous;
    failed.configuration.transfer_bytes = capacity;
    failed.configuration.device_index = device;
    failed.configuration.experiment =
        chunks ? "CONTIGUOUS_CHUNKING" : "SMALL_COPY_BATCHING";
    failed.status = prepared.status;
    failed.message = prepared.message;
    results.push_back(std::move(failed));
    return results;
  }
  for (auto size : sizes)
    for (auto count : counts) {
      auto segment = chunks ? ChunkSize(size, count) : size;
      auto total = chunks ? size : BatchPayloadBytes(size, count);
      TransferResult r;
      r.configuration.direction = direction;
      r.configuration.memory_class = memory;
      r.configuration.api_mode = CopyApiMode::Asynchronous;
      r.configuration.transfer_bytes = chunks ? size : segment;
      r.configuration.batch_count = chunks ? 1 : count;
      r.configuration.chunk_count = chunks ? count : 1;
      r.configuration.repetitions = repetitions;
      r.configuration.warmup_count = 1;
      r.configuration.device_index = device;
      r.configuration.experiment =
          chunks ? "CONTIGUOUS_CHUNKING"
                 : (sync_each ? "BATCH_SYNC_EACH" : "BATCH_ONE_SYNC");
      const auto warmup = provider.Warmup(direction, std::min(total, capacity));
      if (!warmup.ok()) {
        r.status = warmup.status;
        r.message = warmup.message;
        results.push_back(std::move(r));
        continue;
      }
      CopyRequest request{direction, CopyApiMode::Asynchronous,
                          segment,   count,
                          true,      sync_each};
      for (std::uint32_t i = 0; i < repetitions; ++i)
        r.samples.push_back(provider.RunCopy(request, i));
      auto verified = provider.Verify(direction, total);
      r.verified = verified.ok();
      r.status = verified.status;
      r.message = verified.message;
      r.statistics = CalculateTransferStatistics(r.samples);
      r.async_behavior = "BATCHED_SUBMISSION";
      for (const auto &sample : r.samples) {
        if (sample.status != TransferStatus::Success) {
          r.status = sample.status;
          r.message = sample.message;
          break;
        }
      }
      results.push_back(std::move(r));
    }
  const auto released = provider.Release();
  if (!released.ok()) {
    if (results.empty())
      results.push_back({});
    results.back().status = released.status;
    results.back().message = released.message;
  }
  return results;
}

int Info(const CliOptions &o) {
  auto d = NativeTransferProvider().InspectDevice(o.device);
  if (o.json)
    std::cout << "{\"supported\":" << (d.cuda_supported ? "true" : "false")
              << ",\"device_index\":" << d.device_index << ",\"name\":\""
              << Escape(d.name) << "\",\"free_bytes\":" << d.free_bytes
              << ",\"total_bytes\":" << d.total_bytes
              << ",\"async_engine_count\":" << d.async_engine_count << "}\n";
  else
    std::cout << "SIDECAR CUDA TRANSFER INFORMATION\n\nSupported: "
              << (d.cuda_supported ? "yes" : "no") << "\nDevice: " << d.name
              << "\nVRAM free/total: " << d.free_bytes << '/' << d.total_bytes
              << " bytes\nAsync engines: " << d.async_engine_count << '\n';
  return 0;
}
int Plan(const CliOptions &o) {
  auto sizes = o.sizes.empty() ? DefaultTransferSizeSweep() : o.sizes;
  auto memories_selected =
      o.memories.empty()
          ? std::vector<HostMemoryClass>{HostMemoryClass::PageablePretouched,
                                         HostMemoryClass::PinnedHostAlloc,
                                         HostMemoryClass::PinnedRegistered}
          : o.memories;
  std::uint64_t memories = o.memories.empty() ? 3 : o.memories.size(),
                directions = o.directions.empty() ? 2 : o.directions.size(),
                apis = o.apis.empty() ? 2 : o.apis.size(), bytes = 0,
                configs = 0;
  for (auto size : sizes) {
    auto reps = o.repetitions.value_or(AdaptiveRepetitions(size));
    bytes += size * reps * memories * directions * apis;
    configs += memories * directions * apis;
  }
  auto d = NativeTransferProvider().InspectDevice(o.device);
  auto maximum = *std::max_element(sizes.begin(), sizes.end());
  auto safe = PlanDeviceMemory(d, maximum, {});
  const auto host_snapshot = memory::NativeHostMemoryProvider().Snapshot();
  const auto host_method = [](HostMemoryClass memory_class) {
    switch (memory_class) {
    case HostMemoryClass::PageablePretouched:
      return memory::MemoryMethod::Pageable;
    case HostMemoryClass::PinnedHostAlloc:
      return memory::MemoryMethod::CudaHostAlloc;
    case HostMemoryClass::PinnedRegistered:
      return memory::MemoryMethod::CudaHostRegister;
    }
    return memory::MemoryMethod::Pageable;
  };
  std::vector<std::pair<HostMemoryClass, memory::SafetyDecision>> host_plans;
  bool all_host_safe = true;
  for (const auto memory_class : memories_selected) {
    auto decision = memory::PlanAllocation(host_snapshot, {},
                                           host_method(memory_class), maximum);
    all_host_safe = all_host_safe && decision.safe();
    host_plans.emplace_back(memory_class, std::move(decision));
  }
  if (o.json) {
    std::cout << "{\"dry_run\":true,\"configurations\":" << configs
              << ",\"estimated_payload_bytes\":" << bytes
              << ",\"host_buffer_bytes\":" << maximum
              << ",\"device_buffer_bytes\":" << maximum
              << ",\"device_safety\":\"" << (safe.safe ? "SAFE" : "SKIP")
              << "\",\"device_safety_reason\":\"" << Escape(safe.reason)
              << "\",\"host_safety\":[";
    for (std::size_t index = 0; index < host_plans.size(); ++index) {
      if (index)
        std::cout << ',';
      const auto &[memory_class, decision] = host_plans[index];
      std::cout << "{\"memory_class\":\"" << ToString(memory_class)
                << "\",\"status\":\"" << (decision.safe() ? "RUN" : "SKIP")
                << "\",\"reason\":\"" << Escape(decision.reason) << "\"}";
    }
    std::cout << "],\"sizes\":[";
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << "{\"bytes\":" << sizes[index] << ",\"repetitions\":"
                << o.repetitions.value_or(AdaptiveRepetitions(sizes[index]))
                << '}';
    }
    std::cout << "],\"configuration_status\":\""
              << (safe.safe && all_host_safe ? "RUN" : "PARTIAL_OR_SKIP")
              << "\"}\n";
  } else {
    std::cout << "SIDECAR CUDA TRANSFER PLAN\n\nConfigurations: " << configs
              << "\nEstimated payload bytes: " << bytes
              << "\nPersistent host/device buffer: " << maximum
              << " bytes each\nDevice safety: " << (safe.safe ? "SAFE" : "SKIP")
              << " - " << safe.reason << '\n';
    for (const auto &[memory_class, decision] : host_plans)
      std::cout << "Host safety " << ToString(memory_class) << ": "
                << (decision.safe() ? "RUN" : "SKIP") << " - "
                << decision.reason << '\n';
    std::cout << "\nTransfer sizes/repetitions:\n";
    for (const auto size : sizes)
      std::cout << "  " << size << " bytes x "
                << o.repetitions.value_or(AdaptiveRepetitions(size)) << '\n';
    std::cout
        << "No transfer-buffer allocation or payload transfer performed.\n";
  }
  return 0;
}

int Run(const CliOptions &o) {
  TransferRunOptions options;
  options.directions = o.directions;
  options.memory_classes = o.memories;
  options.api_modes = o.apis;
  options.sizes = o.sizes;
  options.repetitions = o.repetitions;
  options.warmup_count = o.warmups;
  options.device_index = o.device;
  options.dry_run = o.dry;
  auto report =
      RunTransferSweep(NativeTransferProvider(),
                       memory::NativeHostMemoryProvider().Snapshot(), options);
  std::int64_t session = 0;
  if (o.persist && !o.dry && !report.results.empty()) {
    auto p = hardware::CreateNativeDiscoveryProvider();
    hardware::HardwareDiscoveryService service(*p);
    auto machine = service.Discover();
    if (o.database.has_parent_path())
      std::filesystem::create_directories(o.database.parent_path());
    auto db =
        database::Database::Open(o.database, database::OpenMode::CreateOrOpen);
    db.Initialize();
    session = StartSession(db, machine, "WU5 isolated transfer sweep");
    PersistResults(db, session, report.results, report.profiles,
                   report.wu6_candidate_sizes, report.host_timer,
                   report.ordering_seed);
    db.CompleteBenchmarkSession(
        session,
        report.status == TransferStatus::Success ? "COMPLETE" : "FAILED");
  }
  std::cout << (o.json ? TransferRunReportToJson(report)
                       : FormatTransferRunReport(report));
  if (session && !o.json)
    std::cout << "\nDatabase session: " << session << '\n';
  return report.status == TransferStatus::Success ||
                 report.status == TransferStatus::SkippedUnsupported
             ? 0
             : 7;
}

int Bidirectional(const CliOptions &o) {
  auto &provider = NativeTransferProvider();
  if (!provider.SupportsCuda()) {
    if (o.json)
      std::cout << "{\"status\":\"SKIPPED_UNSUPPORTED\",\"results\":[]}\n";
    else
      std::cout << "SIDECAR CUDA BIDIRECTIONAL COPIES\n\nStatus: "
                   "SKIPPED_UNSUPPORTED\n";
    return 0;
  }
  auto sizes =
      o.sizes.empty()
          ? std::vector<std::uint64_t>{1 * kMiB,   4 * kMiB,  16 * kMiB,
                                       32 * kMiB,  64 * kMiB, 128 * kMiB,
                                       256 * kMiB, 512 * kMiB}
          : o.sizes;
  auto memory = o.memories.empty() ? HostMemoryClass::PinnedHostAlloc
                                   : o.memories.front();
  auto device = provider.InspectDevice(o.device);
  auto results = RunBidirectionalSweep(
      provider, device, memory::NativeHostMemoryProvider().Snapshot(), sizes,
      memory, o.repetitions.value_or(10), {}, {});
  const auto failed =
      std::any_of(results.begin(), results.end(), [](const auto &r) {
        return r.status != TransferStatus::Success;
      });
  std::int64_t session = 0;
  if (o.persist && !results.empty()) {
    auto p = hardware::CreateNativeDiscoveryProvider();
    hardware::HardwareDiscoveryService service(*p);
    auto machine = service.Discover();
    auto db =
        database::Database::Open(o.database, database::OpenMode::CreateOrOpen);
    db.Initialize();
    session = StartSession(db, machine, "WU5 bidirectional copies");
    for (const auto &r : results) {
      std::ostringstream raw;
      raw << '[';
      for (std::size_t i = 0; i < r.samples.size(); ++i) {
        if (i)
          raw << ',';
        auto &s = r.samples[i];
        raw << "{\"h2d_ns\":" << s.h2d_device_ns
            << ",\"d2h_ns\":" << s.d2h_device_ns
            << ",\"makespan_ns\":" << s.makespan_ns << '}';
      }
      raw << ']';
      (void)db.InsertCudaBidirectionalBenchmark(
          {session, ToString(r.memory_class),
           static_cast<std::int64_t>(r.transfer_bytes), r.repetitions,
           r.isolated_h2d_median_ns, r.isolated_d2h_median_ns,
           r.concurrent_h2d_median_ns, r.concurrent_d2h_median_ns,
           r.makespan_median_ns, r.aggregate_bytes_per_second,
           r.concurrency_benefit, r.verified, ToString(r.status), raw.str(),
           r.message.empty() ? std::nullopt
                             : std::optional<std::string>(r.message)});
    }
    db.CompleteBenchmarkSession(session, failed ? "FAILED" : "COMPLETE");
  }
  std::cout << (o.json ? BidirectionalResultsToJson(results)
                       : FormatBidirectionalResults(results));
  if (session && !o.json)
    std::cout << "\nDatabase session: " << session << '\n';
  return failed ? 7 : 0;
}

int Sustained(const CliOptions &o) {
  TransferConfiguration c;
  c.direction =
      o.directions.empty() ? TransferDirection::H2D : o.directions.front();
  c.memory_class = o.memories.empty() ? HostMemoryClass::PinnedHostAlloc
                                      : o.memories.front();
  c.api_mode = CopyApiMode::Asynchronous;
  c.transfer_bytes = o.sizes.empty() ? 64 * kMiB : o.sizes.front();
  c.device_index = o.device;
  c.experiment = "SUSTAINED";
  auto &provider = NativeTransferProvider();
  auto result = RunSustainedTransfer(
      provider, provider.InspectDevice(o.device),
      memory::NativeHostMemoryProvider().Snapshot(), c, 8 * kGiB, {}, {});
  std::int64_t session = 0;
  if (o.persist && result.status != TransferStatus::SkippedUnsupported) {
    auto p = hardware::CreateNativeDiscoveryProvider();
    hardware::HardwareDiscoveryService service(*p);
    auto machine = service.Discover();
    auto db =
        database::Database::Open(o.database, database::OpenMode::CreateOrOpen);
    db.Initialize();
    session = StartSession(db, machine, "WU5 sustained link state");
    TransferResult persisted;
    persisted.configuration = result.configuration;
    persisted.configuration.repetitions = 1;
    persisted.status = result.status;
    persisted.verified = result.verified;
    persisted.noisy = result.noisy;
    persisted.async_behavior = "SUSTAINED_BATCH";
    if (result.total_payload_bytes)
      persisted.samples.push_back(
          {0, result.total_payload_bytes, result.host_submission_ns,
           result.device_duration_ns, result.total_duration_ns, result.status,
           0, result.message});
    persisted.statistics = CalculateTransferStatistics(persisted.samples);
    const auto timer = memory::CalibrateTimer();
    PersistResults(db, session, {persisted}, {}, {}, timer, 0);
    auto persist = [&](const char *phase, const TransferTelemetry &t) {
      db.InsertCudaLinkState(
          {session, phase,
           t.temperature_c ? std::optional<std::int64_t>(*t.temperature_c)
                           : std::nullopt,
           t.graphics_clock_mhz
               ? std::optional<std::int64_t>(*t.graphics_clock_mhz)
               : std::nullopt,
           t.memory_clock_mhz ? std::optional<std::int64_t>(*t.memory_clock_mhz)
                              : std::nullopt,
           t.power_watts, t.power_limit_watts,
           t.pcie_generation ? std::optional<std::int64_t>(*t.pcie_generation)
                             : std::nullopt,
           t.pcie_width ? std::optional<std::int64_t>(*t.pcie_width)
                        : std::nullopt});
    };
    persist("BEFORE", result.before);
    persist("MAXIMUM_DURING", result.maximum_during);
    persist("AFTER_COOLDOWN", result.after);
    db.CompleteBenchmarkSession(
        session,
        result.status == TransferStatus::Success ? "COMPLETE" : "FAILED");
  }
  std::cout << (o.json ? SustainedResultToJson(result)
                       : FormatSustainedResult(result));
  if (session && !o.json)
    std::cout << "\nDatabase session: " << session << '\n';
  return result.status == TransferStatus::Success ||
                 result.status == TransferStatus::SkippedUnsupported
             ? 0
             : 7;
}

int Segments(const CliOptions &o, bool chunks) {
  auto &provider = NativeTransferProvider();
  if (!provider.SupportsCuda()) {
    TransferRunReport unsupported;
    unsupported.status = TransferStatus::SkippedUnsupported;
    unsupported.message = "CUDA transfer support is unavailable";
    std::cout << (o.json ? TransferRunReportToJson(unsupported)
                         : FormatTransferRunReport(unsupported));
    return 0;
  }
  auto memory = o.memories.empty() ? HostMemoryClass::PinnedHostAlloc
                                   : o.memories.front();
  auto direction =
      o.directions.empty() ? TransferDirection::H2D : o.directions.front();
  std::vector<TransferResult> results;
  if (chunks)
    results = SegmentExperiment(provider, memory, direction, 128 * kMiB,
                                {128 * kMiB}, {1, 2, 4, 8, 16}, true, false,
                                o.repetitions.value_or(30), o.device);
  else {
    std::vector<std::uint64_t> sizes{64 * kKiB, 256 * kKiB, 1 * kMiB, 4 * kMiB};
    std::vector<std::uint32_t> counts{2, 4, 8, 16, 32};
    results = SegmentExperiment(provider, memory, direction, 128 * kMiB, sizes,
                                counts, false, true, o.repetitions.value_or(30),
                                o.device);
    auto batched = SegmentExperiment(provider, memory, direction, 128 * kMiB,
                                     sizes, counts, false, false,
                                     o.repetitions.value_or(30), o.device);
    results.insert(results.end(), std::make_move_iterator(batched.begin()),
                   std::make_move_iterator(batched.end()));
  }
  TransferRunReport report;
  report.host_timer = memory::CalibrateTimer();
  report.device = provider.InspectDevice(o.device);
  report.results = std::move(results);
  for (const auto &result : report.results) {
    if (result.status != TransferStatus::Success) {
      report.status = result.status;
      report.message = result.message;
      break;
    }
  }
  std::int64_t session = 0;
  if (o.persist && !report.results.empty()) {
    auto p = hardware::CreateNativeDiscoveryProvider();
    hardware::HardwareDiscoveryService service(*p);
    auto machine = service.Discover();
    auto db =
        database::Database::Open(o.database, database::OpenMode::CreateOrOpen);
    db.Initialize();
    session = StartSession(db, machine,
                           chunks ? "WU5 contiguous chunking"
                                  : "WU5 small-copy batching");
    PersistResults(db, session, report.results, {}, {}, report.host_timer, 0);
    db.CompleteBenchmarkSession(
        session,
        report.status == TransferStatus::Success ? "COMPLETE" : "FAILED");
  }
  std::cout << (o.json ? TransferRunReportToJson(report)
                       : FormatTransferRunReport(report));
  if (session && !o.json)
    std::cout << "\nDatabase session: " << session << '\n';
  return report.status == TransferStatus::Success ? 0 : 7;
}

int Report(const CliOptions &o) {
  auto db = database::Database::Open(o.database,
                                     database::OpenMode::ExistingReadWrite);
  db.Initialize();
  auto report = db.TransferReport();
  const auto &c = report.counts;
  if (o.json) {
    std::cout << "{\"configurations\":" << c.configurations
              << ",\"benchmarks\":" << c.benchmarks
              << ",\"samples\":" << c.samples
              << ",\"bidirectional\":" << c.bidirectional
              << ",\"link_observations\":" << c.link_observations
              << ",\"profile_count\":" << c.profiles
              << ",\"best_bidirectional_bytes_per_second\":"
              << report.best_bidirectional_bytes_per_second
              << ",\"latest_profiles\":[";
    for (std::size_t index = 0; index < report.latest_profiles.size();
         ++index) {
      if (index)
        std::cout << ',';
      const auto &profile = report.latest_profiles[index];
      std::cout << "{\"session_id\":" << profile.session_id
                << ",\"direction\":\"" << profile.direction
                << "\",\"host_memory_class\":\"" << profile.host_memory_class
                << "\",\"api_mode\":\"" << profile.api_mode
                << "\",\"peak_bytes_per_second\":"
                << profile.peak_bytes_per_second
                << ",\"peak_size_bytes\":" << profile.peak_size_bytes
                << ",\"knee_80_bytes\":" << profile.knee_80_bytes
                << ",\"knee_90_bytes\":" << profile.knee_90_bytes
                << ",\"knee_95_bytes\":" << profile.knee_95_bytes
                << ",\"host_api_median_ns\":" << profile.host_api_median_ns
                << ",\"device_median_ns\":" << profile.device_median_ns
                << ",\"end_to_end_median_ns\":" << profile.end_to_end_median_ns
                << ",\"device_p95_ns\":" << profile.device_p95_ns
                << ",\"device_p99_ns\":" << profile.device_p99_ns
                << ",\"p99_meaningful\":"
                << (profile.p99_meaningful ? "true" : "false") << '}';
    }
    std::cout << "],\"best_sustained\":";
    if (report.best_sustained) {
      const auto &sustained = *report.best_sustained;
      std::cout << "{\"session_id\":" << sustained.session_id
                << ",\"direction\":\"" << Escape(sustained.direction)
                << "\",\"host_memory_class\":\""
                << Escape(sustained.host_memory_class)
                << "\",\"transfer_bytes\":" << sustained.transfer_bytes
                << ",\"payload_bytes\":" << sustained.payload_bytes
                << ",\"bytes_per_second\":" << sustained.bytes_per_second
                << '}';
    } else {
      std::cout << "null";
    }
    std::cout << ",\"latest_link_states\":[";
    for (std::size_t index = 0; index < report.latest_link_states.size();
         ++index) {
      if (index)
        std::cout << ',';
      const auto &link = report.latest_link_states[index];
      const auto optional = [](const auto &value) {
        return value ? std::to_string(*value) : std::string("null");
      };
      std::cout << "{\"session_id\":" << link.session_id << ",\"phase\":\""
                << Escape(link.phase)
                << "\",\"temperature_c\":" << optional(link.temperature_c)
                << ",\"graphics_clock_mhz\":"
                << optional(link.graphics_clock_mhz)
                << ",\"memory_clock_mhz\":" << optional(link.memory_clock_mhz)
                << ",\"power_watts\":" << optional(link.power_watts)
                << ",\"power_limit_watts\":" << optional(link.power_limit_watts)
                << ",\"pcie_generation\":" << optional(link.pcie_generation)
                << ",\"pcie_width\":" << optional(link.pcie_width) << '}';
    }
    std::cout << "],\"latest_supplemental\":[";
    for (std::size_t index = 0; index < report.latest_supplemental.size();
         ++index) {
      if (index)
        std::cout << ',';
      const auto &summary = report.latest_supplemental[index];
      std::cout << "{\"session_id\":" << summary.session_id
                << ",\"experiment\":\"" << Escape(summary.experiment)
                << "\",\"direction\":\"" << Escape(summary.direction)
                << "\",\"host_memory_class\":\""
                << Escape(summary.host_memory_class)
                << "\",\"transfer_bytes\":" << summary.transfer_bytes
                << ",\"batch_count\":" << summary.batch_count
                << ",\"chunk_count\":" << summary.chunk_count
                << ",\"sample_count\":" << summary.sample_count
                << ",\"mean_host_api_ns\":" << summary.mean_host_api_ns
                << ",\"mean_device_ns\":" << summary.mean_device_ns
                << ",\"mean_end_to_end_ns\":" << summary.mean_end_to_end_ns
                << ",\"mean_bytes_per_second\":"
                << summary.mean_bytes_per_second << '}';
    }
    std::cout << "]}\n";
  } else {
    std::cout << "SIDECAR CUDA TRANSFER DATABASE REPORT\n\nConfigurations: "
              << c.configurations << "\nBenchmarks: " << c.benchmarks
              << "\nRaw samples: " << c.samples
              << "\nBidirectional: " << c.bidirectional
              << "\nLink observations: " << c.link_observations
              << "\nProfiles: " << c.profiles << "\nBest bidirectional: "
              << DecimalGigabytesPerSecond(
                     report.best_bidirectional_bytes_per_second)
              << " GB/s\n\nLATEST DIRECTIONAL PROFILES\n";
    for (const auto &profile : report.latest_profiles) {
      std::cout << profile.direction << ' ' << profile.host_memory_class << ' '
                << profile.api_mode << " peak="
                << DecimalGigabytesPerSecond(profile.peak_bytes_per_second)
                << " GB/s size=" << profile.peak_size_bytes
                << " knees80/90/95=" << profile.knee_80_bytes << '/'
                << profile.knee_90_bytes << '/' << profile.knee_95_bytes
                << " bytes api/device/e2e="
                << profile.host_api_median_ns / 1000.0 << '/'
                << profile.device_median_ns / 1000.0 << '/'
                << profile.end_to_end_median_ns / 1000.0
                << " us device_p95=" << profile.device_p95_ns / 1000.0
                << " us device_p99=";
      if (profile.p99_meaningful)
        std::cout << profile.device_p99_ns / 1000.0 << " us\n";
      else
        std::cout << "insufficient-samples\n";
    }
    if (report.best_sustained) {
      const auto &sustained = *report.best_sustained;
      std::cout << "\nBEST SUSTAINED\n"
                << sustained.direction << ' ' << sustained.host_memory_class
                << " size=" << sustained.transfer_bytes
                << " bytes payload=" << sustained.payload_bytes
                << " bytes rate="
                << DecimalGigabytesPerSecond(sustained.bytes_per_second)
                << " GB/s session=" << sustained.session_id << '\n';
    }
    if (!report.latest_link_states.empty()) {
      std::cout << "\nLATEST LINK STATE\n";
      for (const auto &link : report.latest_link_states)
        std::cout << link.phase << " Gen" << link.pcie_generation.value_or(0)
                  << " x" << link.pcie_width.value_or(0)
                  << " temp=" << link.temperature_c.value_or(0)
                  << " C power=" << link.power_watts.value_or(0) << " W\n";
    }
    if (!report.latest_supplemental.empty()) {
      std::cout << "\nLATEST BATCHING / CHUNKING\n";
      for (const auto &summary : report.latest_supplemental)
        std::cout << summary.experiment << ' ' << summary.direction << ' '
                  << summary.host_memory_class
                  << " size=" << summary.transfer_bytes
                  << " batch=" << summary.batch_count
                  << " chunks=" << summary.chunk_count
                  << " mean_device=" << summary.mean_device_ns / 1000.0
                  << " us mean_api=" << summary.mean_host_api_ns / 1000.0
                  << " us mean_e2e=" << summary.mean_end_to_end_ns / 1000.0
                  << " us rate="
                  << DecimalGigabytesPerSecond(summary.mean_bytes_per_second)
                  << " GB/s\n";
    }
  }
  return 0;
}
} // namespace

int RunTransferCli(int argc, char **argv) {
  if (argc < 4 || std::string_view(argv[1]) != "cuda" ||
      std::string_view(argv[2]) != "transfer")
    throw std::invalid_argument("usage: sidecar-lab cuda transfer <command>");
  auto command = Lower(argv[3]);
  auto options = ParseOptions(argc, argv, 4);
  if (command == "info")
    return Info(options);
  if (command == "plan")
    return Plan(options);
  if (command == "run")
    return Run(options);
  if (command == "sustained" || command == "link-state")
    return Sustained(options);
  if (command == "bidirectional")
    return Bidirectional(options);
  if (command == "batching")
    return Segments(options, false);
  if (command == "chunking")
    return Segments(options, true);
  if (command == "report")
    return Report(options);
  throw std::invalid_argument("unknown CUDA transfer command");
}
} // namespace sidecar::cuda
