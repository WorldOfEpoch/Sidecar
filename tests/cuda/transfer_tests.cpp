#include "sidecar/cuda/transfer.hpp"
#include "sidecar/database/database.hpp"

#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace sidecar::cuda;
void Expect(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

class MockProvider final : public ITransferProvider {
public:
  bool SupportsCuda() const noexcept override { return supported; }
  DeviceMemoryInfo InspectDevice(int index) override {
    return {supported, index, "mock", 12ULL << 30, 16ULL << 30, 2};
  }
  TransferTelemetry ReadTelemetry(int) override {
    TransferTelemetry t;
    t.available = true;
    t.pcie_generation = 4;
    t.pcie_width = 16;
    return t;
  }
  TransferError Prepare(int, HostMemoryClass, std::uint64_t capacity,
                        std::uint32_t buffers) override {
    ++prepares;
    prepared = true;
    capacity_ = capacity;
    buffer_count_ = buffers;
    return fail_prepare
               ? TransferError{TransferStatus::CudaOutOfMemory, 2, "mock OOM"}
               : TransferError{};
  }
  TransferError Warmup(TransferDirection, std::uint64_t) override {
    return prepared ? TransferError{}
                    : TransferError{TransferStatus::InternalError, 0,
                                    "not prepared"};
  }
  TransferSample RunCopy(const CopyRequest &request,
                         std::uint32_t repetition) override {
    TransferSample s;
    s.repetition = repetition;
    s.payload_bytes = request.segment_bytes * request.copy_count;
    s.host_api_raw_ns = 100 + request.copy_count;
    s.device_duration_ns = request.segment_bytes / 1024 + 1000;
    s.end_to_end_raw_ns = s.device_duration_ns + 200;
    s.status = TransferStatus::Success;
    return s;
  }
  BidirectionalSample RunBidirectional(std::uint64_t bytes,
                                       std::uint32_t repetition) override {
    BidirectionalSample s;
    s.repetition = repetition;
    s.h2d_device_ns = bytes / 1024 + 1000;
    s.d2h_device_ns = bytes / 1024 + 1100;
    s.makespan_ns = bytes / 1024 + 1200;
    s.host_submission_ns = 200;
    s.status = TransferStatus::Success;
    return s;
  }
  TransferError Verify(TransferDirection, std::uint64_t,
                       std::uint32_t = 0) override {
    return {};
  }
  TransferError Release() noexcept override {
    prepared = false;
    ++releases;
    return {};
  }
  bool supported = true, fail_prepare = false, prepared = false;
  int prepares = 0, releases = 0;
  std::uint64_t capacity_ = 0;
  std::uint32_t buffer_count_ = 0;
};

sidecar::memory::MemorySnapshot Host() {
  sidecar::memory::MemorySnapshot s;
  s.installed_physical_bytes = 16ULL << 30;
  s.visible_physical_bytes = 16ULL << 30;
  s.available_physical_bytes = 12ULL << 30;
  s.system_commit_bytes = 2ULL << 30;
  s.system_commit_limit_bytes = 24ULL << 30;
  return s;
}

void TestSweepRepetitionsAndUnits() {
  auto s = DefaultTransferSizeSweep();
  Expect(s.front() == 4096 && s.back() == (1ULL << 30), "sweep endpoints");
  Expect(std::find(s.begin(), s.end(), 96ULL << 20) != s.end() &&
             std::find(s.begin(), s.end(), 192ULL << 20) != s.end(),
         "intermediate sizes missing");
  Expect(AdaptiveRepetitions(4096) > AdaptiveRepetitions(1ULL << 30),
         "adaptive repetitions");
  Expect(BytesPerSecond(1000, 1000) == 1e9, "bytes/s");
  Expect(DecimalGigabytesPerSecond(1e9) == 1.0, "GB/s");
  Expect(BinaryGibibytesPerSecond(static_cast<double>(1ULL << 30)) == 1.0,
         "GiB/s");
}
void TestSafety() {
  DeviceMemoryInfo d{true, 0, "mock", 12ULL << 30, 16ULL << 30, 2};
  Expect(PlanDeviceMemory(d, 1ULL << 30, {}).safe, "safe VRAM rejected");
  Expect(!PlanDeviceMemory(d, 10ULL << 30, {}).safe, "VRAM reserve ignored");
}
void TestIdentity() {
  TransferConfiguration a;
  a.transfer_bytes = 4096;
  auto one = TransferConfigurationIdentity(a),
       two = TransferConfigurationIdentity(a);
  Expect(one == two && one.size() == 64, "configuration identity unstable");
  a.transfer_bytes = 8192;
  Expect(TransferConfigurationIdentity(a) != one, "identity omitted size");
}
void TestStatisticsAndKnees() {
  std::vector<TransferSample> samples{{0, 1000, 10, 100, 120},
                                      {1, 1000, 12, 110, 130},
                                      {2, 1000, 11, 105, 125}};
  auto stats = CalculateTransferStatistics(samples);
  Expect(stats.count == 3 && stats.device_ns.median == 105 &&
             !stats.p99_meaningful,
         "statistics");
  std::vector<TransferResult> results;
  for (auto pair :
       {std::pair<std::uint64_t, double>{1, 50}, {2, 80}, {4, 95}, {8, 100}}) {
    TransferResult r;
    r.configuration.transfer_bytes = pair.first;
    r.configuration.memory_class = HostMemoryClass::PinnedHostAlloc;
    r.configuration.direction = TransferDirection::H2D;
    r.configuration.api_mode = CopyApiMode::Asynchronous;
    r.status = TransferStatus::Success;
    r.statistics.count = 1;
    r.statistics.bytes_per_second.median = pair.second;
    results.push_back(r);
  }
  auto p = CalculateSaturationProfile(TransferDirection::H2D,
                                      HostMemoryClass::PinnedHostAlloc,
                                      CopyApiMode::Asynchronous, results);
  Expect(p.peak_size_bytes == 8 && p.knee_80_bytes == 2 && p.knee_95_bytes == 4,
         "saturation knees");
}
void TestBatchChunkBidiMath() {
  Expect(BatchPayloadBytes(4096, 32) == 131072, "batch math");
  Expect(ChunkSize(128ULL << 20, 16) == 8ULL << 20, "chunk math");
  Expect(CalculateConcurrencyBenefit(10, 10, 12) > 1.66, "concurrency math");
  bool rejected = false;
  try {
    (void)ChunkSize(10, 3);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  Expect(rejected, "invalid chunks accepted");
}
void TestMockSweepAndJson() {
  MockProvider p;
  TransferRunOptions o;
  o.memory_classes = {HostMemoryClass::PinnedHostAlloc};
  o.directions = {TransferDirection::H2D};
  o.api_modes = {CopyApiMode::Asynchronous};
  o.sizes = {4096, 1ULL << 20};
  o.repetitions = 3;
  o.warmup_count = 1;
  o.host_safety.minimum_dynamic_reserve_bytes = 1ULL << 20;
  auto report = RunTransferSweep(p, Host(), o);
  Expect(report.results.size() == 2 && report.results[0].verified &&
             p.prepares == 1 && p.releases == 1,
         "mock sweep lifetime");
  Expect(std::all_of(report.wu6_candidate_sizes.begin(),
                     report.wu6_candidate_sizes.end(),
                     [&](auto candidate) {
                       return std::find(o.sizes.begin(), o.sizes.end(),
                                        candidate) != o.sizes.end();
                     }),
         "WU6 candidates included an unmeasured size");
  auto json = TransferRunReportToJson(report);
  Expect(json.find("host_api_raw_ns") != std::string::npos, "raw JSON missing");
  Expect(json == TransferRunReportToJson(report), "transfer JSON is unstable");
  o.dry_run = true;
  MockProvider dry_provider;
  (void)RunTransferSweep(dry_provider, Host(), o);
  Expect(dry_provider.prepares == 0, "dry run allocated transfer buffers");
  o.dry_run = false;
  MockProvider failing_provider;
  failing_provider.fail_prepare = true;
  auto failed = RunTransferSweep(failing_provider, Host(), o);
  Expect(failed.status == TransferStatus::CudaOutOfMemory,
         "mock CUDA failure was not classified");
}

std::filesystem::path Temp() {
  return std::filesystem::temp_directory_path() /
         ("sidecar-transfer-" +
          std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".db");
}
void TestMigration5Persistence() {
  auto path = Temp();
  struct Cleanup {
    std::filesystem::path p;
    ~Cleanup() {
      std::error_code e;
      std::filesystem::remove(p, e);
      std::filesystem::remove(p.string() + "-wal", e);
      std::filesystem::remove(p.string() + "-shm", e);
    }
  } cleanup{path};
  auto db = sidecar::database::Database::Open(
      path, sidecar::database::OpenMode::CreateOrOpen);
  db.Initialize();
  Expect(db.Status().schema_version == sidecar::database::kCurrentSchemaVersion &&
             db.Status().latest_migration == sidecar::database::kCurrentSchemaVersion,
         "current migration preserves WU5 tables");
  db.UpsertHardwareProfile({"transfer-machine", "host", "Windows", "11", "1",
                            "CPU", 1, 1, 1, 1024, 512, "{}"});
  auto session =
      db.StartBenchmarkSession({"transfer-machine", "M0-FROZEN-1", "commit",
                                std::nullopt, "CUDA_TRANSFER", std::nullopt});
  TransferConfiguration c;
  c.transfer_bytes = 4096;
  auto cid =
      db.InsertCudaTransferConfiguration({session,
                                          TransferConfigurationIdentity(c),
                                          "ISOLATED",
                                          "H2D",
                                          "PINNED_HOSTALLOC",
                                          "ASYNCHRONOUS",
                                          4096,
                                          "NON_BLOCKING",
                                          1,
                                          1,
                                          3,
                                          10,
                                          0,
                                          4096,
                                          4096,
                                          "SAMPLED",
                                          "steady",
                                          0,
                                          100,
                                          "events",
                                          1,
                                          "SUCCESS"});
  auto bid = db.InsertCudaTransferBenchmark({cid, "SUCCESS", true, false,
                                             "HOST_RETURNED_QUICKLY", "{}",
                                             std::nullopt});
  db.InsertCudaTransferSample(
      {bid, 0, 4096, 100, 200, 300, "SUCCESS", 0, std::nullopt});
  db.InsertCudaTransferProfile({session, "H2D", "PINNED_HOSTALLOC",
                                "ASYNCHRONOUS", 1e9, 4096, 4096, 4096, 4096,
                                "[4096]"});
  auto counts = db.TransferCounts();
  Expect(counts.configurations == 1 && counts.benchmarks == 1 &&
             counts.samples == 1 && counts.profiles == 1,
         "transfer persistence");
}

struct Test {
  std::string_view name;
  std::function<void()> run;
};
} // namespace
int main() {
  std::vector<Test> tests{
      {"sweep_repetitions_units", TestSweepRepetitionsAndUnits},
      {"device_safety", TestSafety},
      {"configuration_identity", TestIdentity},
      {"statistics_knees", TestStatisticsAndKnees},
      {"batch_chunk_bidi_math", TestBatchChunkBidiMath},
      {"mock_sweep_json", TestMockSweepAndJson},
      {"migration5_persistence", TestMigration5Persistence}};
  int failures = 0;
  for (auto &t : tests)
    try {
      t.run();
      std::cout << "[PASS] " << t.name << '\n';
    } catch (const std::exception &e) {
      ++failures;
      std::cerr << "[FAIL] " << t.name << ": " << e.what() << '\n';
    }
  std::cout << "tests=" << tests.size() << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
