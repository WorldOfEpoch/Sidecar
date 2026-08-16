#include "sidecar/memory/cli.hpp"

#include "sidecar/database/database.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/memory/benchmark.hpp"
#include "sidecar/version.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>

namespace sidecar::memory {
namespace {

constexpr std::uint64_t kMiB = 1024ULL * 1024;
constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;

std::string Lower(std::string_view value) {
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return result;
}

MemoryMethod ParseMethod(std::string_view value) {
    const auto normalized = Lower(value);
    if (normalized == "pageable") return MemoryMethod::Pageable;
    if (normalized == "hostalloc" || normalized == "cudahostalloc")
        return MemoryMethod::CudaHostAlloc;
    if (normalized == "hostregister" || normalized == "cudahostregister")
        return MemoryMethod::CudaHostRegister;
    throw std::invalid_argument("memory method must be pageable, hostalloc, or hostregister");
}

std::uint64_t ParseSize(std::string value) {
    std::uint64_t multiplier = 1;
    if (!value.empty() && (value.back() == 'M' || value.back() == 'm')) {
        multiplier = kMiB; value.pop_back();
    } else if (!value.empty() && (value.back() == 'G' || value.back() == 'g')) {
        multiplier = kGiB; value.pop_back();
    }
    const auto amount = std::stoull(value);
    if (amount == 0 || amount > UINT64_MAX / multiplier)
        throw std::invalid_argument("invalid memory size");
    return amount * multiplier;
}

std::vector<std::uint64_t> ParseSizes(const std::string& value) {
    std::vector<std::uint64_t> sizes;
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto end = value.find(',', start);
        sizes.push_back(ParseSize(value.substr(start, end == std::string::npos
                                                         ? std::string::npos : end - start)));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return sizes;
}

std::optional<std::int64_t> Int64(std::optional<std::uint64_t> value) {
    return value ? std::optional<std::int64_t>(static_cast<std::int64_t>(*value)) : std::nullopt;
}

std::string StatisticsJson(const LifecycleResult& result) {
    const auto phase = [](const PhaseStatistics& stats) {
        std::ostringstream out;
        out << "{\"count\":" << stats.count << ",\"raw_mean_ns\":" << stats.raw_ns.mean
            << ",\"raw_median_ns\":" << stats.raw_ns.median
            << ",\"raw_p90_ns\":" << stats.raw_ns.p90
            << ",\"raw_p95_ns\":" << stats.raw_ns.p95
            << ",\"raw_p99_ns\":" << (stats.p99_meaningful
                                                 ? std::to_string(stats.raw_ns.p99) : "null")
            << ",\"raw_min_ns\":" << stats.raw_ns.minimum
            << ",\"raw_max_ns\":" << stats.raw_ns.maximum
            << ",\"raw_stddev_ns\":" << stats.raw_ns.stddev << '}';
        return out.str();
    };
    std::ostringstream out;
    out << "{\"allocation\":" << phase(result.allocation)
        << ",\"first_touch\":" << phase(result.first_touch)
        << ",\"warm_touch\":" << phase(result.warm_touch)
        << ",\"registration\":" << phase(result.registration)
        << ",\"unregistration\":" << phase(result.unregistration)
        << ",\"cleanup\":" << phase(result.cleanup)
        << ",\"allocation_drift_percent\":" << result.allocation_latency_drift_percent
        << ",\"cleanup_drift_percent\":" << result.cleanup_latency_drift_percent << '}';
    return out.str();
}

std::string AmortizationJson(const LifecycleResult& result) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < result.amortized_setup_us.size(); ++index) {
        if (index) out << ',';
        out << "{\"reuse_count\":" << result.amortized_setup_us[index].first
            << ",\"calculated_setup_us_per_reuse\":"
            << result.amortized_setup_us[index].second << '}';
    }
    out << ']';
    return out.str();
}

std::int64_t StartMemorySession(sidecar::database::Database& database,
                                const sidecar::hardware::DiscoveryReport& machine,
                                std::string notes) {
    sidecar::hardware::PersistDiscovery(database, machine);
    const auto version = sidecar::CurrentVersionInfo();
    return database.StartBenchmarkSession({machine.identity.machine_hash,
                                           std::string(version.spec_version),
                                           std::string(version.git_commit),
                                           std::nullopt, "MEMORY_LIFECYCLE",
                                           std::move(notes),
                                           machine.snapshot.cuda_runtime_version
                                               ? std::optional<std::int64_t>(
                                                     *machine.snapshot.cuda_runtime_version)
                                               : std::nullopt,
                                           machine.snapshot.cuda_driver_version
                                               ? std::optional<std::int64_t>(
                                                     *machine.snapshot.cuda_driver_version)
                                               : std::nullopt,
                                           std::nullopt,
                                           machine.snapshot.nvidia_driver_version.empty()
                                               ? std::nullopt
                                               : std::optional<std::string>(
                                                     machine.snapshot.nvidia_driver_version),
                                           machine.snapshot.operating_system.version});
}

void PersistLifecycle(sidecar::database::Database& database,
                      std::int64_t session,
                      const LifecycleReport& report) {
    for (const auto& result : report.results) {
        const auto safety = result.samples.empty() ? SafetyStatus::SkipUnknownMemoryState
                                                   : result.samples.front().safety.status;
        const auto benchmark_id = database.InsertHostMemoryBenchmark({
            session, ToString(result.method), ToString(result.mode),
            static_cast<std::int64_t>(result.requested_bytes),
            static_cast<std::int64_t>(result.samples.size()), ToString(result.status),
            ToString(safety), report.timer.method,
            static_cast<std::int64_t>(report.timer.bracket_overhead_ns),
            static_cast<std::int64_t>(report.timer.observed_resolution_ns),
            result.method == MemoryMethod::CudaHostAlloc
                ? std::optional<std::string>("cudaHostAllocDefault") : std::nullopt,
            result.method == MemoryMethod::CudaHostRegister
                ? std::optional<std::string>("cudaHostRegisterDefault") : std::nullopt,
            StatisticsJson(result), AmortizationJson(result),
            result.message.empty() ? std::nullopt
                                   : std::optional<std::string>(result.message)});
        for (const auto& sample : result.samples) {
            database.InsertHostMemorySample({
                benchmark_id, sample.repetition, sample.cold_setup, ToString(sample.status),
                Int64(sample.backing_allocation.raw_ns), Int64(sample.backing_allocation.corrected_ns),
                Int64(sample.first_touch.raw_ns), Int64(sample.first_touch.corrected_ns),
                Int64(sample.warm_touch.raw_ns), Int64(sample.warm_touch.corrected_ns),
                Int64(sample.registration.raw_ns), Int64(sample.registration.corrected_ns),
                Int64(sample.unregistration.raw_ns), Int64(sample.unregistration.corrected_ns),
                Int64(sample.cleanup.raw_ns), Int64(sample.cleanup.corrected_ns),
                MemoryInfoToJson(sample.before), MemoryInfoToJson(sample.after_setup),
                MemoryInfoToJson(sample.after_cleanup), sample.available_recovery_delta_bytes,
                sample.available_recovery_percent, sample.noisy, sample.cuda_error,
                sample.windows_error, sample.message.empty() ? std::nullopt
                    : std::optional<std::string>(sample.message)});
        }
    }
}

struct CommonOptions {
    bool json{false};
    bool dry_run{false};
    bool persist{true};
    std::vector<MemoryMethod> methods;
    std::vector<std::uint64_t> sizes;
    std::optional<std::uint32_t> repetitions;
    std::filesystem::path database{std::filesystem::path("data") / "sidecar.db"};
    SafetyPolicy safety;
};

CommonOptions ParseCommon(int argc, char** argv, int start) {
    CommonOptions options;
    for (int index = start; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--json") options.json = true;
        else if (argument == "--dry-run") options.dry_run = true;
        else if (argument == "--no-persist") options.persist = false;
        else if (argument == "--method" && index + 1 < argc)
            options.methods = {ParseMethod(argv[++index])};
        else if (argument == "--sizes" && index + 1 < argc)
            options.sizes = ParseSizes(argv[++index]);
        else if (argument == "--repetitions" && index + 1 < argc)
            options.repetitions = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        else if (argument == "--database" && index + 1 < argc)
            options.database = argv[++index];
        else if (argument == "--reserve" && index + 1 < argc)
            options.safety.absolute_reserve_bytes = ParseSize(argv[++index]);
        else if (argument == "--max-fraction" && index + 1 < argc) {
            const auto fraction = std::stod(argv[++index]);
            options.safety.pageable_max_test_fraction = fraction;
            options.safety.pinned_max_test_fraction = fraction;
        } else throw std::invalid_argument("invalid memory option: " + std::string(argument));
    }
    return options;
}

int MemoryInfo(int argc, char** argv) {
    bool json = false;
    if (argc == 4 && std::string_view(argv[3]) == "--json") json = true;
    else if (argc != 3) throw std::invalid_argument("usage: sidecar-lab memory info [--json]");
    const auto snapshot = NativeHostMemoryProvider().Snapshot();
    std::cout << (json ? MemoryInfoToJson(snapshot) : FormatMemoryInfo(snapshot)) << '\n';
    return 0;
}

int MemoryPlan(int argc, char** argv) {
    auto options = ParseCommon(argc, argv, 3);
    const auto method = options.methods.empty() ? MemoryMethod::CudaHostAlloc : options.methods[0];
    const auto sizes = options.sizes.empty() ? DefaultSizeSweep() : options.sizes;
    const auto snapshot = NativeHostMemoryProvider().Snapshot();
    std::cout << (options.json ? MemoryPlanToJson(snapshot, options.safety, method, sizes)
                              : FormatMemoryPlan(snapshot, options.safety, method, sizes)) << '\n';
    return 0;
}

int Lifecycle(int argc, char** argv, bool stress) {
    auto parsed = ParseCommon(argc, argv, 3);
    LifecycleOptions options;
    options.methods = parsed.methods.empty()
                          ? std::vector<MemoryMethod>{MemoryMethod::Pageable,
                                                     MemoryMethod::CudaHostAlloc,
                                                     MemoryMethod::CudaHostRegister}
                          : parsed.methods;
    if (!parsed.sizes.empty()) options.sizes = parsed.sizes;
    else if (stress) options.sizes = {64 * kMiB, 256 * kMiB, 1024 * kMiB,
                                     512 * kMiB, 2048 * kMiB};
    else options.sizes = {64 * kMiB, 256 * kMiB, 1024 * kMiB};
    options.safety = parsed.safety;
    options.repetitions = parsed.repetitions;
    options.dry_run = parsed.dry_run;
    const auto report = RunLifecycleLaboratory(NativeHostMemoryProvider(), options);
    std::int64_t session = 0;
    std::optional<sidecar::database::Database> database;
    if (parsed.persist && !parsed.dry_run) {
        auto provider = sidecar::hardware::CreateNativeDiscoveryProvider();
        sidecar::hardware::HardwareDiscoveryService service(*provider);
        const auto machine = service.Discover();
        if (parsed.database.has_parent_path())
            std::filesystem::create_directories(parsed.database.parent_path());
        database.emplace(sidecar::database::Database::Open(
            parsed.database, sidecar::database::OpenMode::CreateOrOpen));
        database->Initialize();
        session = StartMemorySession(*database, machine,
                                     stress ? "WU4 bounded churn" : "WU4 lifecycle laboratory");
        PersistLifecycle(*database, session, report);
        database->CompleteBenchmarkSession(session,
            report.pressure_stop ? "ABORTED" : "COMPLETE");
    }
    std::cout << (parsed.json ? LifecycleReportToJson(report)
                             : FormatLifecycleReport(report));
    if (session && !parsed.json) std::cout << "\nDatabase session: " << session << '\n';
    return 0;
}

int Arena(int argc, char** argv) {
    bool json = false, persist = true;
    ArenaBackend backend = ArenaBackend::CudaHostAlloc;
    std::uint64_t bytes = 64 * kMiB;
    std::uint32_t reuses = 100;
    std::filesystem::path database_path = std::filesystem::path("data") / "sidecar.db";
    SafetyPolicy safety;
    for (int index = 3; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--json") json = true;
        else if (argument == "--no-persist") persist = false;
        else if (argument == "--backend" && index + 1 < argc) {
            const auto value = Lower(argv[++index]);
            if (value == "hostalloc" || value == "cudahostalloc")
                backend = ArenaBackend::CudaHostAlloc;
            else if (value == "hostregister" || value == "cudahostregister" ||
                     value == "register")
                backend = ArenaBackend::RegisteredPageable;
            else throw std::invalid_argument("arena backend must be hostalloc or hostregister");
        } else if (argument == "--size" && index + 1 < argc) bytes = ParseSize(argv[++index]);
        else if (argument == "--reuses" && index + 1 < argc)
            reuses = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        else if (argument == "--database" && index + 1 < argc) database_path = argv[++index];
        else throw std::invalid_argument("invalid arena option: " + std::string(argument));
    }
    auto& provider = NativeHostMemoryProvider();
    if (provider.SupportsCudaHostMemory()) (void)provider.WarmupCuda();
    const auto result = RunArenaTest(provider, backend, bytes, reuses, safety);
    std::int64_t session = 0;
    if (persist) {
        auto hardware_provider = sidecar::hardware::CreateNativeDiscoveryProvider();
        sidecar::hardware::HardwareDiscoveryService service(*hardware_provider);
        const auto machine = service.Discover();
        auto database = sidecar::database::Database::Open(
            database_path, sidecar::database::OpenMode::CreateOrOpen);
        database.Initialize();
        session = StartMemorySession(database, machine, "WU4 persistent arena test");
        std::ostringstream stats, raw;
        stats << "{\"count\":" << result.reuse.count << ",\"median_ns\":"
              << result.reuse.raw_ns.median << ",\"p95_ns\":" << result.reuse.raw_ns.p95 << '}';
        raw << '[';
        for (std::size_t index = 0; index < result.raw_reuse_ns.size(); ++index) {
            if (index) raw << ','; raw << result.raw_reuse_ns[index];
        }
        raw << ']';
        (void)database.InsertPersistentArenaTest({
            session, backend == ArenaBackend::CudaHostAlloc ? "CUDA_HOST_ALLOC"
                                                            : "REGISTERED_PAGEABLE",
            static_cast<std::int64_t>(bytes), reuses, Int64(result.setup.raw_ns),
            Int64(result.cleanup.raw_ns), stats.str(), raw.str(), result.contents_verified,
            ToString(result.status), result.message.empty() ? std::nullopt
                : std::optional<std::string>(result.message)});
        database.CompleteBenchmarkSession(session,
            result.status == ResultStatus::Success ? "COMPLETE" : "FAILED");
    }
    std::cout << (json ? ArenaResultToJson(result) : FormatArenaResult(result)) << '\n';
    if (session && !json) std::cout << "Database session: " << session << '\n';
    return result.status == ResultStatus::Success ||
                   result.status == ResultStatus::SkippedUnsupported ? 0 : 6;
}

int Report(int argc, char** argv) {
    bool json = false;
    std::filesystem::path path = std::filesystem::path("data") / "sidecar.db";
    for (int index = 3; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--json") json = true;
        else if (argument == "--database" && index + 1 < argc) path = argv[++index];
        else throw std::invalid_argument("invalid memory report option");
    }
    auto database = sidecar::database::Database::Open(
        path, sidecar::database::OpenMode::ExistingReadWrite);
    database.Initialize();
    const auto counts = database.MemoryCounts();
    if (json) {
        std::cout << "{\"benchmarks\":" << counts.benchmarks
                  << ",\"samples\":" << counts.samples
                  << ",\"pressure_events\":" << counts.pressure_events
                  << ",\"arena_tests\":" << counts.arena_tests << "}\n";
    } else {
        std::cout << "SIDECAR HOST MEMORY DATABASE REPORT\n\n"
                  << "Benchmarks: " << counts.benchmarks << '\n'
                  << "Raw samples: " << counts.samples << '\n'
                  << "Pressure events: " << counts.pressure_events << '\n'
                  << "Arena tests: " << counts.arena_tests << '\n';
    }
    return 0;
}

}  // namespace

int RunMemoryCommand(int argc, char** argv) {
    if (argc < 3 || std::string_view(argv[1]) != "memory") return -1;
    const std::string_view command(argv[2]);
    if (command == "info") return MemoryInfo(argc, argv);
    if (command == "plan") return MemoryPlan(argc, argv);
    if (command == "lifecycle") return Lifecycle(argc, argv, false);
    if (command == "stress") return Lifecycle(argc, argv, true);
    if (command == "arena") return Arena(argc, argv);
    if (command == "report") return Report(argc, argv);
    return -1;
}

}  // namespace sidecar::memory
