#include "sidecar/database/database.hpp"
#include "sidecar/memory/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace {

using namespace sidecar::memory;

void Expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

class MockProvider final : public IHostMemoryProvider {
public:
    MemorySnapshot Snapshot() const override {
        MemorySnapshot snapshot;
        snapshot.installed_physical_bytes = 16ULL << 30;
        snapshot.visible_physical_bytes = 16ULL << 30;
        snapshot.available_physical_bytes = 12ULL << 30;
        snapshot.system_commit_bytes = 4ULL << 30;
        snapshot.system_commit_limit_bytes = 24ULL << 30;
        snapshot.process_working_set_bytes = live_bytes_;
        snapshot.process_private_bytes = live_bytes_;
        snapshot.process_page_fault_count = touches_;
        snapshot.memory_load_percent = 25;
        return snapshot;
    }
    std::uint64_t PageSize() const noexcept override { return 4096; }
    bool SupportsCudaHostMemory() const noexcept override { return cuda_supported; }
    ProviderResult WarmupCuda() override { return cuda_supported ? ProviderResult{} :
        ProviderResult{ResultStatus::SkippedUnsupported, 0, "mock no CUDA"}; }

    ProviderResult AllocatePageable(std::uint64_t bytes, MemoryRegion& region) override {
        events.emplace_back("allocate_pageable");
        ++pageable_allocations;
        if (fail_pageable) return {ResultStatus::OsAllocationFailure, 8, "mock pageable fail"};
        return Allocate(bytes, region);
    }
    ProviderResult FreePageable(MemoryRegion& region) noexcept override {
        events.emplace_back("free_pageable");
        ++pageable_frees; return Free(region);
    }
    ProviderResult AllocateCudaHost(std::uint64_t bytes, MemoryRegion& region) override {
        events.emplace_back("allocate_host");
        ++host_allocations;
        if (!cuda_supported) return {ResultStatus::SkippedUnsupported, 0, "mock no CUDA"};
        if (fail_hostalloc) return {ResultStatus::CudaOutOfMemory, 2, "mock hostalloc fail"};
        return Allocate(bytes, region);
    }
    ProviderResult FreeCudaHost(MemoryRegion& region) noexcept override {
        events.emplace_back("free_host");
        ++host_frees; return Free(region);
    }
    ProviderResult RegisterCudaHost(MemoryRegion&) override {
        events.emplace_back("register");
        ++registrations;
        if (fail_register) return {ResultStatus::RegistrationFailure, 3, "mock register fail"};
        registered = true; return {};
    }
    ProviderResult UnregisterCudaHost(MemoryRegion&) noexcept override {
        events.emplace_back("unregister");
        ++unregistrations;
        if (fail_unregister_attempts > 0) {
            --fail_unregister_attempts;
            return {ResultStatus::CleanupFailure, 4, "mock unregister fail"};
        }
        registered = false; return {};
    }
    ProviderResult Touch(MemoryRegion& region, std::uint64_t pass,
                         std::uint64_t& checksum) noexcept override {
        events.emplace_back("touch");
        ++touches_;
        if (!region.data) return {ResultStatus::VerificationFailure, 0, "empty"};
        auto* bytes = static_cast<std::byte*>(region.data);
        for (std::uint64_t offset = 0; offset < region.size_bytes; offset += 4096) {
            bytes[offset] = static_cast<std::byte>((pass + offset / 4096) & 0xff);
            checksum ^= static_cast<std::uint8_t>(bytes[offset]);
        }
        return {};
    }

    bool cuda_supported{true};
    bool fail_pageable{false};
    bool fail_hostalloc{false};
    bool fail_register{false};
    int fail_unregister_attempts{0};
    bool registered{false};
    int pageable_allocations{0}, pageable_frees{0};
    int host_allocations{0}, host_frees{0};
    int registrations{0}, unregistrations{0};
    std::vector<std::string> events;

private:
    ProviderResult Allocate(std::uint64_t bytes, MemoryRegion& region) {
        auto* memory = new (std::nothrow) std::byte[static_cast<std::size_t>(bytes)];
        if (!memory) return {ResultStatus::OsAllocationFailure, 0, "new failed"};
        allocations_[memory] = bytes; live_bytes_ += bytes; region = {memory, bytes}; return {};
    }
    ProviderResult Free(MemoryRegion& region) noexcept {
        if (!region.data) return {};
        auto found = allocations_.find(region.data);
        if (found == allocations_.end()) return {ResultStatus::CleanupFailure, 0, "unknown block"};
        live_bytes_ -= found->second; allocations_.erase(found);
        delete[] static_cast<std::byte*>(region.data); region = {}; return {};
    }
    mutable std::uint64_t live_bytes_{0};
    mutable std::uint64_t touches_{0};
    std::unordered_map<void*, std::uint64_t> allocations_;
};

void TestSafetyPlanner() {
    MockProvider provider;
    const auto snapshot = provider.Snapshot();
    SafetyPolicy policy;
    policy.minimum_dynamic_reserve_bytes = 1ULL << 30;
    Expect(RequiredReserve(snapshot, policy) == static_cast<std::uint64_t>(2.4 * (1ULL << 30)),
           "dynamic reserve calculation changed");
    Expect(PlanAllocation(snapshot, policy, MemoryMethod::Pageable, 1ULL << 30).safe(),
           "safe pageable allocation was rejected");
    Expect(PlanAllocation(snapshot, policy, MemoryMethod::CudaHostAlloc, 2ULL << 30).status ==
               SafetyStatus::SkipPolicyLimit,
           "pinned policy fraction was not enforced");
    Expect(PlanAllocation(snapshot, policy, MemoryMethod::Pageable, 1ULL << 30, true).status ==
               SafetyStatus::SkipPreviousPressureSignal,
           "previous pressure did not stop escalation");
    auto low = snapshot; low.available_physical_bytes = 2ULL << 30;
    Expect(PlanAllocation(low, policy, MemoryMethod::Pageable, 1ULL << 30).status ==
               SafetyStatus::SkipInsufficientHeadroom,
           "emergency headroom was not protected");
}

void TestSweepAndRepetitions() {
    const auto sweep = DefaultSizeSweep();
    Expect(sweep.front() == 4ULL * 1024 * 1024 && sweep.back() == 16ULL * 1024 * 1024 * 1024,
           "default sweep endpoints changed");
    Expect(std::is_sorted(sweep.begin(), sweep.end()), "default sweep is not monotonic");
    Expect(RepetitionsForSize(64ULL << 20) > RepetitionsForSize(8ULL << 30),
           "large allocations did not reduce repetitions");
}

void TestStatisticsAndRawCorrected() {
    std::vector<LifecycleSample> samples(3);
    samples[0].cleanup = {110, 100}; samples[1].cleanup = {210, 200};
    samples[2].cleanup = {310, 300};
    const auto stats = CalculatePhaseStatistics(samples, &LifecycleSample::cleanup);
    Expect(stats.count == 3 && stats.raw_ns.median == 210 &&
               stats.corrected_ns.median == 200,
           "raw/corrected statistics were not preserved");
    Expect(!stats.p99_meaningful, "three samples must not claim meaningful P99");
    const auto timer = CalibrateTimer(100);
    Expect(timer.raw_bracket_samples_ns.size() == 100,
           "timer calibration did not retain raw samples");
}

LifecycleOptions TinyOptions(MemoryMethod method) {
    LifecycleOptions options;
    options.methods = {method}; options.sizes = {8192}; options.repetitions = 2;
    options.safety.minimum_dynamic_reserve_bytes = 1ULL << 20;
    options.cooldown_checks = 1; options.cooldown_milliseconds = 0;
    return options;
}

void TestMockPageableSuccessFailure() {
    MockProvider provider;
    auto report = RunLifecycleLaboratory(provider, TinyOptions(MemoryMethod::Pageable));
    Expect(report.results[0].status == ResultStatus::Success &&
               provider.pageable_allocations == 2 && provider.pageable_frees == 2,
           "mock pageable lifecycle failed");
    provider.fail_pageable = true;
    report = RunLifecycleLaboratory(provider, TinyOptions(MemoryMethod::Pageable));
    Expect(report.results[0].status == ResultStatus::OsAllocationFailure,
           "pageable allocation failure was not classified");
}

void TestMockCudaSuccessFailureAndCleanup() {
    MockProvider provider;
    auto report = RunLifecycleLaboratory(provider, TinyOptions(MemoryMethod::CudaHostAlloc));
    Expect(report.results[0].status == ResultStatus::Success && provider.host_frees == 2,
           "mock cudaHostAlloc lifecycle failed");
    provider.fail_hostalloc = true;
    report = RunLifecycleLaboratory(provider, TinyOptions(MemoryMethod::CudaHostAlloc));
    Expect(report.results[0].status == ResultStatus::CudaOutOfMemory,
           "cudaHostAlloc failure classification changed");

    MockProvider registration_provider;
    registration_provider.fail_register = true;
    report = RunLifecycleLaboratory(registration_provider,
                                    TinyOptions(MemoryMethod::CudaHostRegister));
    Expect(report.results[0].status == ResultStatus::RegistrationFailure &&
               registration_provider.pageable_frees >= 1,
           "registration failure leaked pageable backing memory");

    MockProvider transient_cleanup_provider;
    transient_cleanup_provider.fail_unregister_attempts = 1;
    auto cleanup_options = TinyOptions(MemoryMethod::CudaHostRegister);
    cleanup_options.repetitions = 1;
    cleanup_options.include_pretouched_registration = false;
    report = RunLifecycleLaboratory(transient_cleanup_provider, cleanup_options);
    Expect(report.results[0].status == ResultStatus::CleanupFailure &&
               transient_cleanup_provider.unregistrations == 2 &&
               transient_cleanup_provider.pageable_frees == 1 &&
               !transient_cleanup_provider.registered,
           "transient unregister failure did not recover without leaking backing memory");
}

void TestColdVersusPretouchedCallOrder() {
    MockProvider provider;
    auto options = TinyOptions(MemoryMethod::CudaHostRegister);
    options.repetitions = 1;
    const auto report = RunLifecycleLaboratory(provider, options);
    Expect(report.results.size() == 2 &&
               report.results[0].mode == RegistrationMode::Cold &&
               report.results[1].mode == RegistrationMode::Pretouched,
           "cold and pre-touched registration modes were not separated");
    Expect(provider.registrations == 2 && provider.unregistrations == 2,
           "registration cleanup counts did not reconcile");
    const std::vector<std::string> expected{
        "allocate_pageable", "register", "touch", "touch", "unregister", "free_pageable",
        "allocate_pageable", "touch", "register", "touch", "unregister", "free_pageable"};
    Expect(provider.events == expected,
           "cold and pre-touched registration operation order changed");
}

void TestArenaBoundsReuseAndCleanup() {
    MockProvider provider;
    {
        PersistentPinnedArena arena(provider, ArenaBackend::CudaHostAlloc, 8192);
        Expect(arena.View(4096, 4096).size() == 4096, "valid arena view failed");
        bool rejected = false;
        try { (void)arena.View(8192, 1); } catch (const std::out_of_range&) { rejected = true; }
        Expect(rejected, "arena out-of-bounds view was accepted");
    }
    Expect(provider.host_frees == 1, "arena destructor did not release memory");
    const auto test = RunArenaTest(provider, ArenaBackend::RegisteredPageable,
                                   8192, 5, TinyOptions(MemoryMethod::Pageable).safety);
    Expect(test.status == ResultStatus::Success && test.contents_verified &&
               test.raw_reuse_ns.size() == 5,
           "persistent arena reuse experiment failed");
}

void TestJsonStability() {
    MockProvider provider;
    const auto report = RunLifecycleLaboratory(provider, TinyOptions(MemoryMethod::Pageable));
    Expect(LifecycleReportToJson(report) == LifecycleReportToJson(report),
           "lifecycle JSON is not deterministic");
    Expect(MemoryInfoToJson(provider.Snapshot()).find("available_physical_bytes") !=
               std::string::npos,
           "memory snapshot JSON omitted required state");
}

std::filesystem::path TempDb() {
    return std::filesystem::temp_directory_path() /
           ("sidecar-memory-test-" + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()) + ".db");
}

void TestMigration4Persistence() {
    const auto path = TempDb();
    struct Cleanup { std::filesystem::path path; ~Cleanup() {
        std::error_code ignored; std::filesystem::remove(path, ignored);
        std::filesystem::remove(path.string() + "-wal", ignored);
        std::filesystem::remove(path.string() + "-shm", ignored);
    }} cleanup{path};
    auto database = sidecar::database::Database::Open(
        path, sidecar::database::OpenMode::CreateOrOpen);
    database.Initialize();
    Expect(database.Status().schema_version == sidecar::database::kCurrentSchemaVersion &&
               database.Status().latest_migration == sidecar::database::kCurrentSchemaVersion,
           "database did not retain memory tables through current migration");
    database.UpsertHardwareProfile({"memory-machine", "host", "Windows", "11", "1", "CPU",
                                    1, 1, 1, 1024, 512, "{}"});
    const auto session = database.StartBenchmarkSession(
        {"memory-machine", "M0-FROZEN-1", "commit", std::nullopt,
         "MEMORY_LIFECYCLE", std::nullopt});
    const auto benchmark = database.InsertHostMemoryBenchmark(
        {session, "PAGEABLE", "NOT_APPLICABLE", 4096, 1, "SUCCESS", "SAFE",
         "steady_clock", 0, 100, std::nullopt, std::nullopt, "{}", "[]", std::nullopt});
    database.InsertHostMemorySample(
        {benchmark, 0, true, "SUCCESS", 10, 10, 20, 20, 5, 5,
         std::nullopt, std::nullopt, std::nullopt, std::nullopt, 10, 10,
         "{}", "{}", "{}", 0, 0.0, false, std::nullopt, std::nullopt, std::nullopt});
    (void)database.InsertPersistentArenaTest(
        {session, "CUDA_HOST_ALLOC", 4096, 10, 100, 50, "{}", "[]", true,
         "SUCCESS", std::nullopt});
    const auto counts = database.MemoryCounts();
    Expect(counts.benchmarks == 1 && counts.samples == 1 && counts.arena_tests == 1,
           "memory laboratory rows were not persisted");
}

struct Test { std::string_view name; std::function<void()> run; };

}  // namespace

int main() {
    const std::vector<Test> tests{
        {"safety_planner", TestSafetyPlanner},
        {"sweep_repetitions", TestSweepAndRepetitions},
        {"statistics_raw_corrected", TestStatisticsAndRawCorrected},
        {"mock_pageable", TestMockPageableSuccessFailure},
        {"mock_cuda_cleanup", TestMockCudaSuccessFailureAndCleanup},
        {"cold_vs_pretouched", TestColdVersusPretouchedCallOrder},
        {"arena_bounds_reuse", TestArenaBoundsReuseAndCleanup},
        {"json_stability", TestJsonStability},
        {"migration4_persistence", TestMigration4Persistence},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try { test.run(); std::cout << "[PASS] " << test.name << '\n'; }
        catch (const std::exception& error) {
            ++failures; std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }
    std::cout << "tests=" << tests.size() << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
