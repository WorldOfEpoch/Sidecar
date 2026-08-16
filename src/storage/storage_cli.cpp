#include "sidecar/storage/cli.hpp"

#include "sidecar/core/sha256.hpp"
#include "sidecar/database/database.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/memory/host_memory.hpp"
#include "sidecar/storage/physics.hpp"
#include "sidecar/version.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace sidecar::storage {
namespace {

struct Options {
    std::string command;
    bool json{false}, persist{true}, dry_run{false}, full{false};
    std::filesystem::path database{std::filesystem::path("data") / "sidecar.db"};
    std::optional<std::string> device;
    std::optional<std::filesystem::path> dataset;
    std::optional<BackendKind> backend;
    std::optional<AccessPattern> pattern;
    std::optional<DestinationKind> destination;
    std::optional<std::uint64_t> block_bytes;
    std::optional<std::uint32_t> queue_depth;
    std::optional<std::uint64_t> requests;
    std::uint64_t timeout_ms{120000};
    double temperature_cap_c{75.0};
    std::optional<std::string> phase;
};

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::uint64_t ParseSize(std::string value) {
    const auto lowered = Lower(value);
    std::uint64_t multiplier = 1;
    std::string number = lowered;
    for (const auto& [suffix, scale] :
         std::initializer_list<std::pair<std::string_view, std::uint64_t>>{
             {"gib", 1ULL << 30U}, {"gb", 1000000000ULL}, {"g", 1ULL << 30U},
             {"mib", 1ULL << 20U}, {"mb", 1000000ULL}, {"m", 1ULL << 20U},
             {"kib", 1ULL << 10U}, {"kb", 1000ULL}, {"k", 1ULL << 10U}}) {
        if (lowered.ends_with(suffix)) {
            multiplier = scale;
            number.resize(number.size() - suffix.size());
            break;
        }
    }
    std::size_t consumed = 0;
    const auto base = std::stoull(number, &consumed);
    if (consumed != number.size() || base > UINT64_MAX / multiplier)
        throw std::invalid_argument("invalid byte size: " + value);
    return base * multiplier;
}

BackendKind ParseBackend(const std::string& value) {
    const auto v = Lower(value);
    if (v == "unbuffered" || v == "overlapped" || v == "iocp")
        return BackendKind::OverlappedUnbuffered;
    if (v == "buffered") return BackendKind::Buffered;
    if (v == "mapped" || v == "mmap") return BackendKind::MemoryMapped;
    if (v == "directstorage") return BackendKind::DirectStorage;
    throw std::invalid_argument("unknown storage backend: " + value);
}

AccessPattern ParsePattern(const std::string& value) {
    const auto v = Lower(value);
    if (v == "sequential" || v == "seq") return AccessPattern::Sequential;
    if (v == "random" || v == "pseudorandom") return AccessPattern::Pseudorandom;
    if (v == "windowed" || v == "strided") return AccessPattern::Windowed;
    throw std::invalid_argument("unknown access pattern: " + value);
}

DestinationKind ParseDestination(const std::string& value) {
    const auto v = Lower(value);
    if (v == "pageable" || v == "pretouched") return DestinationKind::PageablePretouched;
    if (v == "pinned" || v == "hostalloc" || v == "cudahostalloc")
        return DestinationKind::CudaHostAlloc;
    throw std::invalid_argument("unknown destination: " + value);
}

Options Parse(int argc, char** argv) {
    Options value;
    if (argc < 3) throw std::invalid_argument("storage subcommand required");
    value.command = Lower(argv[2]);
    int index = 3;
    if (value.command == "dataset") {
        if (argc < 4) throw std::invalid_argument("storage dataset requires create or verify");
        value.command += "." + Lower(argv[3]);
        index = 4;
    }
    for (; index < argc; ++index) {
        const std::string argument(argv[index]);
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::invalid_argument(argument + " requires a value");
            return argv[index];
        };
        if (argument == "--json") value.json = true;
        else if (argument == "--no-persist") value.persist = false;
        else if (argument == "--dry-run") value.dry_run = true;
        else if (argument == "--full") value.full = true;
        else if (argument == "--database") value.database = next();
        else if (argument == "--device") value.device = next();
        else if (argument == "--dataset") value.dataset = std::filesystem::path(next());
        else if (argument == "--backend") value.backend = ParseBackend(next());
        else if (argument == "--pattern") value.pattern = ParsePattern(next());
        else if (argument == "--destination") value.destination = ParseDestination(next());
        else if (argument == "--block" || argument == "--size")
            value.block_bytes = ParseSize(next());
        else if (argument == "--qd") value.queue_depth =
            static_cast<std::uint32_t>(std::stoul(next()));
        else if (argument == "--requests") value.requests = std::stoull(next());
        else if (argument == "--timeout-ms") value.timeout_ms = std::stoull(next());
        else if (argument == "--temperature-cap") value.temperature_cap_c = std::stod(next());
        else if (argument == "--phase") value.phase = next();
        else throw std::invalid_argument("unknown storage option: " + argument);
    }
    return value;
}

std::optional<std::int64_t> DbInt(const std::optional<std::uint64_t>& value) {
    if (!value || *value > static_cast<std::uint64_t>(INT64_MAX)) return std::nullopt;
    return static_cast<std::int64_t>(*value);
}
std::optional<std::int64_t> DbInt(const std::optional<std::uint32_t>& value) {
    if (!value) return std::nullopt;
    return static_cast<std::int64_t>(*value);
}

database::StorageHealthSnapshotInput HealthInput(
    const StorageTarget& target, const HealthSnapshot& health,
    std::optional<std::int64_t> session = std::nullopt) {
    database::StorageHealthSnapshotInput input;
    input.machine_hash = target.machine_hash; input.persistent_id = target.persistent_id;
    input.session_id = session; input.phase = health.phase; input.provider = health.provider;
    input.status = health.status; input.critical_warning = DbInt(health.critical_warning);
    input.temperature_c = health.temperature_c;
    input.available_spare_percent = DbInt(health.available_spare_percent);
    input.available_spare_threshold_percent = DbInt(health.available_spare_threshold_percent);
    input.percentage_used = DbInt(health.percentage_used);
    input.data_units_read_low64 = DbInt(health.data_units_read_low64);
    input.data_units_written_low64 = DbInt(health.data_units_written_low64);
    input.host_read_commands_low64 = DbInt(health.host_read_commands_low64);
    input.host_write_commands_low64 = DbInt(health.host_write_commands_low64);
    input.controller_busy_minutes_low64 = DbInt(health.controller_busy_minutes_low64);
    input.power_cycles_low64 = DbInt(health.power_cycles_low64);
    input.power_on_hours_low64 = DbInt(health.power_on_hours_low64);
    input.unsafe_shutdowns_low64 = DbInt(health.unsafe_shutdowns_low64);
    input.media_data_errors_low64 = DbInt(health.media_data_errors_low64);
    input.error_log_entries_low64 = DbInt(health.error_log_entries_low64);
    input.raw_evidence_json = health.raw_evidence_json;
    if (!health.message.empty()) input.message = health.message;
    return input;
}

hardware::DiscoveryReport DiscoverAndPersist(database::Database* database) {
    auto provider = hardware::CreateNativeDiscoveryProvider();
    hardware::HardwareDiscoveryService service(*provider);
    auto report = service.Discover();
    if (database) hardware::PersistDiscovery(*database, report);
    return report;
}

std::optional<database::Database> OpenDatabase(const Options& options) {
    if (!options.persist) return std::nullopt;
    if (options.database.has_parent_path())
        std::filesystem::create_directories(options.database.parent_path());
    auto database = database::Database::Open(options.database,
                                             database::OpenMode::CreateOrOpen);
    database.Initialize();
    return database;
}

std::int64_t StartSession(database::Database& database,
                          const hardware::DiscoveryReport& machine) {
    const auto version = CurrentVersionInfo();
    database::BenchmarkSessionInput input;
    input.machine_hash = machine.identity.machine_hash;
    input.sidecar_spec_version = std::string(version.spec_version);
    input.sidecar_git_commit = std::string(version.git_commit);
    input.trace_mode = "STORAGE_PHYSICS_WU7";
    input.notes = "NVMe to host-memory discovery lab";
    input.cuda_runtime_version = machine.snapshot.cuda_runtime_version;
    input.cuda_driver_version = machine.snapshot.cuda_driver_version;
    if (!machine.snapshot.nvidia_driver_version.empty())
        input.nvidia_driver_version = machine.snapshot.nvidia_driver_version;
    input.os_version = machine.snapshot.operating_system.version;
    return database.StartBenchmarkSession(input);
}

database::StorageDatasetInput DatasetInput(const StorageTarget& target,
                                           const DatasetIdentity& dataset) {
    return {target.machine_hash, target.persistent_id, dataset.generator,
            dataset.identity_hash, dataset.path.generic_string(), target.volume_name,
            dataset.format_version, static_cast<std::int64_t>(dataset.seed),
            static_cast<std::int64_t>(dataset.size_bytes), target.physical_disk_number,
            dataset.verified, dataset.full_verification,
            static_cast<std::int64_t>(dataset.verified_bytes)};
}

bool HealthStop(const HealthSnapshot& value, double temperature_cap) {
    return (value.critical_warning && *value.critical_warning != 0) ||
           (value.temperature_c && *value.temperature_c >= temperature_cap);
}

std::string ConfigHash(const BenchmarkConfig& config) {
    std::ostringstream material;
    material << ToString(config.backend) << '|' << ToString(config.pattern) << '|'
             << ToString(config.destination) << '|' << config.block_bytes << '|'
             << config.queue_depth << '|' << config.request_count << '|'
             << config.outstanding_byte_cap << '|' << config.ordering_seed << '|'
             << config.window_bytes << '|' << config.phase;
    return core::Sha256Hex(material.str());
}

void PersistRun(database::Database& database, std::int64_t session,
                std::int64_t dataset_id, const std::vector<BenchmarkResult>& results,
                std::optional<std::int64_t> before_id, std::optional<std::int64_t> after_id) {
    database.BeginWriteTransaction();
    try {
        struct Best {
            double throughput{0};
            std::int64_t block{0}, depth{0};
        };
        std::map<std::string, Best> best;
        for (const auto& result : results) {
            database::StorageConfigurationInput configuration;
            configuration.session_id = session; configuration.storage_dataset_id = dataset_id;
            configuration.configuration_hash = ConfigHash(result.config);
            configuration.backend = ToString(result.config.backend);
            configuration.access_pattern = ToString(result.config.pattern);
            configuration.destination_kind = ToString(result.config.destination);
            configuration.block_bytes = static_cast<std::int64_t>(result.config.block_bytes);
            configuration.queue_depth = result.config.queue_depth;
            configuration.request_count = static_cast<std::int64_t>(result.config.request_count);
            configuration.outstanding_byte_cap =
                static_cast<std::int64_t>(result.config.outstanding_byte_cap);
            configuration.timeout_ms = static_cast<std::int64_t>(result.config.timeout_ms);
            configuration.ordering_seed = static_cast<std::int64_t>(result.config.ordering_seed);
            configuration.window_bytes = static_cast<std::int64_t>(result.config.window_bytes);
            configuration.phase = result.config.phase;
            configuration.cache_classification = result.cache_classification;
            configuration.status = ToString(result.status);
            if (!result.message.empty()) configuration.skip_reason = result.message;
            const auto configuration_id = database.InsertStorageConfiguration(configuration);

            std::ostringstream cpu;
            cpu << "{\"completion_thread_kernel_ns\":" << result.cpu.completion_thread_kernel_ns
                << ",\"completion_thread_user_ns\":" << result.cpu.completion_thread_user_ns
                << ",\"process_cpu_seconds_per_gib\":" << result.cpu.process_cpu_seconds_per_gib
                << ",\"process_kernel_ns\":" << result.cpu.process_kernel_ns
                << ",\"process_user_ns\":" << result.cpu.process_user_ns << '}';
            database::StorageBenchmarkInput benchmark;
            benchmark.configuration_id = configuration_id;
            benchmark.health_before_id = before_id;
            benchmark.health_after_id = after_id;
            benchmark.status = ToString(result.status);
            benchmark.statistics_json = ResultToJson(result);
            benchmark.cpu_json = cpu.str();
            benchmark.verified = result.verified;
            benchmark.completed_bytes = static_cast<std::int64_t>(result.completed_bytes);
            benchmark.wall_time_ns = static_cast<std::int64_t>(result.wall_time_ns);
            benchmark.bytes_per_second = result.bytes_per_second;
            benchmark.iops = result.iops;
            if (!result.message.empty()) benchmark.message = result.message;
            const auto benchmark_id = database.InsertStorageBenchmark(benchmark);
            for (const auto& sample : result.samples) {
                database.InsertStorageSample({
                    benchmark_id, static_cast<std::int64_t>(sample.request_index),
                    static_cast<std::int64_t>(sample.batch_id),
                    static_cast<std::int64_t>(sample.file_offset),
                    static_cast<std::int64_t>(sample.requested_bytes),
                    static_cast<std::int64_t>(sample.completed_bytes),
                    static_cast<std::int64_t>(sample.submission_reference_ns),
                    static_cast<std::int64_t>(sample.submission_cost_ns),
                    static_cast<std::int64_t>(sample.completion_latency_ns),
                    static_cast<std::int64_t>(sample.completion_processing_ns),
                    ToString(sample.status),
                    sample.native_error ? std::optional<std::int64_t>(sample.native_error)
                                        : std::nullopt});
            }
            for (const auto& deadline : result.deadlines)
                database.InsertStorageDeadline({
                    benchmark_id, static_cast<std::int64_t>(deadline.deadline_ns),
                    static_cast<std::int64_t>(deadline.hits),
                    static_cast<std::int64_t>(deadline.misses), deadline.success_rate,
                    deadline.compliant_bytes_per_second});
            const std::string key = configuration.backend + "|" + configuration.access_pattern +
                                    "|" + configuration.destination_kind;
            auto& candidate = best[key];
            if (result.status == RunStatus::Success &&
                result.bytes_per_second > candidate.throughput) {
                candidate = {result.bytes_per_second, configuration.block_bytes,
                             configuration.queue_depth};
            }
        }
        for (const auto& [key, peak] : best) {
            const auto first = key.find('|');
            const auto second = key.find('|', first + 1);
            database.InsertStorageBackendProfile({
                session, key.substr(0, first), key.substr(first + 1, second - first - 1),
                key.substr(second + 1), peak.throughput, peak.block, peak.depth,
                std::nullopt, std::nullopt, std::nullopt,
                "{\"criterion\":\"deadline tables retain measured 95/99/99.9 success\"}",
                "{\"aggregate_bytes\":[67108864,134217728,268435456],"
                "\"interpretation\":\"NVMe-to-host supply only; no GPU contention\"}"});
        }
        database.CommitWriteTransaction();
    } catch (...) {
        database.RollbackWriteTransaction();
        throw;
    }
}

void PrintTarget(const StorageTarget& target, const DatasetIdentity& dataset,
                 bool direct_storage) {
    std::cout << "SIDECAR NVME STORAGE INFORMATION\n\n"
              << "Target\n"
              << "  Model: " << target.model << "\n"
              << "  Persistent id: " << target.persistent_id << "\n"
              << "  Firmware: " << target.firmware << "\n"
              << "  Bus: " << target.bus_type << "\n"
              << "  Physical disk: " << target.physical_disk_number << "\n"
              << "  Volume: " << target.volume_name << "\n"
              << "  Mount: " << target.mount_point.string() << "\n"
              << "  Mapping: " << target.mapping_confidence << " (" << target.mapping_source
              << ")\n\nAlignment\n"
              << "  Logical sector: " << target.alignment.logical_sector_bytes << " bytes\n"
              << "  Physical sector: " << target.alignment.physical_sector_bytes << " bytes\n"
              << "  File offset: " << target.alignment.file_offset_alignment_bytes << " bytes\n"
              << "  Destination: " << target.alignment.buffer_alignment_bytes << " bytes\n\n"
              << "Dataset\n"
              << "  Path: " << dataset.path.string() << "\n"
              << "  Identity: " << dataset.identity_hash << "\n"
              << "  Size: " << dataset.size_bytes << " bytes\n"
              << "  Exists: " << (dataset.exists ? "yes" : "no") << "\n"
              << "DirectStorage: " << (direct_storage ? "available" : "SKIPPED_UNSUPPORTED")
              << '\n';
}

std::string InfoJson(const StorageTarget& target, const DatasetIdentity& dataset,
                     bool direct_storage) {
    const auto plan = BuildDefaultPlan(target, dataset, 16ULL << 30U, false);
    auto json = PlanToJson(plan);
    json.pop_back();
    json += ",\"directstorage_available\":";
    json += direct_storage ? "true}" : "false}";
    return json;
}

std::vector<PlanEntry> FilterPlan(const StoragePlan& plan, const Options& options,
                                  bool single) {
    std::vector<PlanEntry> values;
    for (const auto& entry : plan.entries) {
        if (options.backend && entry.config.backend != *options.backend) continue;
        if (options.pattern && entry.config.pattern != *options.pattern) continue;
        if (options.destination && entry.config.destination != *options.destination) continue;
        if (options.block_bytes && entry.config.block_bytes != *options.block_bytes) continue;
        if (options.queue_depth && entry.config.queue_depth != *options.queue_depth) continue;
        values.push_back(entry);
    }
    if (single) {
        BenchmarkConfig requested;
        requested.backend = options.backend.value_or(BackendKind::OverlappedUnbuffered);
        requested.pattern = options.pattern.value_or(AccessPattern::Sequential);
        requested.destination = options.destination.value_or(DestinationKind::PageablePretouched);
        requested.block_bytes = options.block_bytes.value_or(1024ULL * 1024ULL);
        requested.queue_depth = options.queue_depth.value_or(1);
        requested.request_count = options.requests.value_or(
            std::max<std::uint64_t>(requested.queue_depth,
                                    (256ULL << 20U) / requested.block_bytes));
        requested.timeout_ms = options.timeout_ms;
        requested.outstanding_byte_cap = plan.outstanding_byte_cap;
        requested.phase = options.phase.value_or(
            options.requests ? "REFINE_EXPLICIT_REQUEST_COUNT" : "COARSE");
        values = {{requested}};
        if (const auto issue = ValidateConfig(requested, plan.target, plan.dataset.size_bytes)) {
            values.front().disposition = RunStatus::SkippedSafetyLimit;
            values.front().reason = *issue;
        }
    } else if (options.requests) {
        for (auto& entry : values) entry.config.request_count =
            std::max<std::uint64_t>(*options.requests, entry.config.queue_depth);
    }
    if (!single && options.phase)
        for (auto& entry : values) entry.config.phase = *options.phase;
    return values;
}

int RunMeasurement(const Options& options, IStorageProvider& provider,
                   const StorageTarget& target, const DatasetIdentity& dataset,
                   bool single) {
    const auto snapshot = memory::NativeHostMemoryProvider().Snapshot();
    auto plan = BuildDefaultPlan(target, dataset,
        snapshot.available_physical_bytes.value_or(16ULL << 30U));
    plan.temperature_cap_c = options.temperature_cap_c;
    auto entries = FilterPlan(plan, options, single);
    if (entries.empty()) throw std::runtime_error("storage filters selected no configurations");
    if (options.dry_run) {
        plan.entries = entries;
        plan.estimated_read_bytes = 0;
        for (const auto& e : entries)
            if (e.disposition == RunStatus::Success)
                plan.estimated_read_bytes += e.config.block_bytes * e.config.request_count;
        std::cout << (options.json ? PlanToJson(plan) : "Storage dry-run plan\n");
        if (!options.json) {
            PrintTarget(target, dataset, provider.DirectStorageAvailable());
            std::cout << "Configurations: " << entries.size()
                      << "\nEstimated reads: " << plan.estimated_read_bytes << " bytes\n";
        } else std::cout << '\n';
        return 0;
    }
    if (!dataset.verified)
        throw std::runtime_error("dataset must pass verification before measurement");

    auto database = OpenDatabase(options);
    hardware::DiscoveryReport machine;
    std::int64_t session = 0, dataset_id = 0;
    if (database) {
        machine = DiscoverAndPersist(&*database);
        dataset_id = database->UpsertStorageDataset(DatasetInput(target, dataset));
        session = StartSession(*database, machine);
    }
    const auto before = provider.CaptureHealth(target, "BEFORE");
    if (HealthStop(before, options.temperature_cap_c)) {
        if (database) {
            (void)database->InsertStorageHealthSnapshot(
                HealthInput(target, before, session));
            database->CompleteBenchmarkSession(session, "ABORTED");
        }
        throw std::runtime_error("health gate rejected measurement before start");
    }
    std::vector<BenchmarkResult> results;
    results.reserve(entries.size());
    std::vector<HealthSnapshot> during;
    bool health_aborted = false;
    for (std::size_t index = 0; index < entries.size(); ++index) {
        auto entry = entries[index];
        if (entry.disposition != RunStatus::Success) {
            BenchmarkResult skipped;
            skipped.config = entry.config; skipped.status = entry.disposition;
            skipped.message = entry.reason;
            results.push_back(std::move(skipped));
            continue;
        }
        auto result = provider.Run(target, entry.config);
        results.push_back(std::move(result));
        if ((index + 1U) % 16U == 0U) {
            during.push_back(provider.CaptureHealth(target, "DURING"));
            if (HealthStop(during.back(), options.temperature_cap_c)) {
                health_aborted = true;
                break;
            }
        }
    }
    const auto after = provider.CaptureHealth(target, "AFTER");
    const bool counter_increase =
        before.media_data_errors_low64 && after.media_data_errors_low64 &&
        *after.media_data_errors_low64 > *before.media_data_errors_low64;
    if (counter_increase)
        for (auto& result : results)
            if (result.status == RunStatus::Success) result.message +=
                " POST_RUN_MEDIA_ERROR_COUNTER_INCREASE";

    if (database) {
        const auto before_id = database->InsertStorageHealthSnapshot(
            HealthInput(target, before, session));
        for (const auto& observation : during)
            (void)database->InsertStorageHealthSnapshot(
                HealthInput(target, observation, session));
        const auto after_id = database->InsertStorageHealthSnapshot(
            HealthInput(target, after, session));
        PersistRun(*database, session, dataset_id, results, before_id, after_id);
        const bool failed = health_aborted || counter_increase ||
            std::any_of(results.begin(), results.end(),
            [](const auto& result) {
                return result.status != RunStatus::Success &&
                       result.status != RunStatus::SkippedUnsupported &&
                       result.status != RunStatus::SkippedSafetyLimit;
            });
        database->CompleteBenchmarkSession(
            session, health_aborted ? "ABORTED" : (failed ? "FAILED" : "COMPLETE"));
    }
    if (options.json) {
        std::cout << "{\"health_after\":" << HealthToJson(after)
                  << ",\"health_before\":" << HealthToJson(before) << ",\"results\":[";
        for (std::size_t index = 0; index < results.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << ResultToJson(results[index]);
        }
        std::cout << "],\"session_id\":" << session << "}\n";
    } else {
        std::cout << "Storage measurement results\n";
        for (const auto& result : results)
            std::cout << ToString(result.config.backend) << ' '
                      << ToString(result.config.pattern) << " block=" << result.config.block_bytes
                      << " qd=" << result.config.queue_depth << " status="
                      << ToString(result.status) << " throughput_GiB_s="
                      << result.bytes_per_second / (1024.0 * 1024.0 * 1024.0)
                      << " p99_ms=" << result.completion_latency.p99 / 1.0e6 << '\n';
        if (session) std::cout << "Database session: " << session << '\n';
    }
    return counter_increase ? 6 : 0;
}

int Report(const Options& options) {
    auto database = database::Database::Open(options.database,
                                             database::OpenMode::ExistingReadWrite);
    database.Initialize();
    const auto counts = database.StorageCounts();
    const auto rows = database.LatestStorageBenchmarks();
    if (options.json) {
        std::cout << "{\"counts\":{\"backend_profiles\":" << counts.backend_profiles
                  << ",\"benchmarks\":" << counts.benchmarks
                  << ",\"configurations\":" << counts.configurations
                  << ",\"datasets\":" << counts.datasets
                  << ",\"deadline_profiles\":" << counts.deadline_profiles
                  << ",\"health_snapshots\":" << counts.health_snapshots
                  << ",\"samples\":" << counts.samples << "},\"latest\":[";
        for (std::size_t index = 0; index < rows.size(); ++index) {
            if (index) std::cout << ',';
            const auto& row = rows[index];
            std::cout << "{\"backend\":\"" << row.backend << "\",\"block_bytes\":"
                      << row.block_bytes << ",\"bytes_per_second\":" << row.bytes_per_second
                      << ",\"destination\":\"" << row.destination_kind << "\",\"p99_ns\":"
                      << row.p99_ns << ",\"pattern\":\"" << row.access_pattern
                      << "\",\"queue_depth\":" << row.queue_depth << ",\"session_id\":"
                      << row.session_id << ",\"status\":\"" << row.status << "\"}";
        }
        std::cout << "]}\n";
    } else {
        std::cout << "Storage physics: datasets=" << counts.datasets
                  << " health=" << counts.health_snapshots
                  << " configurations=" << counts.configurations
                  << " benchmarks=" << counts.benchmarks << " samples=" << counts.samples << '\n';
        for (const auto& row : rows)
            std::cout << row.backend << ' ' << row.access_pattern << " block=" << row.block_bytes
                      << " qd=" << row.queue_depth << " GiB/s="
                      << row.bytes_per_second / (1024.0 * 1024.0 * 1024.0)
                      << " p99_ms=" << row.p99_ns / 1.0e6 << '\n';
    }
    return 0;
}

}  // namespace

int RunStorageCli(int argc, char** argv) {
    if (argc < 2 || std::string_view(argv[1]) != "storage") return -1;
    const auto options = Parse(argc, argv);
    if (options.command == "report") return Report(options);
    auto provider = CreateNativeStorageProvider();
    const auto target = provider->ResolveTarget(options.device, options.dataset);
    auto dataset = provider->InspectDataset(target);

    if (options.command == "info") {
        if (options.json) std::cout << InfoJson(target, dataset,
                                               provider->DirectStorageAvailable()) << '\n';
        else PrintTarget(target, dataset, provider->DirectStorageAvailable());
        return 0;
    }
    if (options.command == "health") {
        const auto health = provider->CaptureHealth(target, "ON_DEMAND");
        if (options.persist) {
            auto database = OpenDatabase(options);
            (void)DiscoverAndPersist(&*database);
            const auto id = database->InsertStorageHealthSnapshot(HealthInput(target, health));
            if (!options.json) std::cout << "Health snapshot id: " << id << '\n';
        }
        std::cout << HealthToJson(health) << '\n';
        return health.status == "SUCCESS" ? 0 : 5;
    }
    if (options.command == "dataset.create") {
        dataset = provider->CreateDataset(target, kDefaultDatasetBytes);
    } else if (options.command == "dataset.verify") {
        dataset = provider->VerifyDataset(target, options.full);
    }
    if (options.command == "dataset.create" || options.command == "dataset.verify") {
        std::int64_t id = 0;
        if (options.persist) {
            auto database = OpenDatabase(options);
            (void)DiscoverAndPersist(&*database);
            id = database->UpsertStorageDataset(DatasetInput(target, dataset));
        }
        if (options.json)
            std::cout << "{\"dataset_id\":" << id << ",\"exists\":"
                      << (dataset.exists ? "true" : "false") << ",\"identity_hash\":\""
                      << dataset.identity_hash << "\",\"message\":\"" << dataset.message
                      << "\",\"path\":\"" << dataset.path.generic_string()
                      << "\",\"size_bytes\":" << dataset.size_bytes << ",\"verified\":"
                      << (dataset.verified ? "true" : "false") << ",\"verified_bytes\":"
                      << dataset.verified_bytes << "}\n";
        else
            std::cout << "Dataset " << dataset.path.string() << "\nIdentity "
                      << dataset.identity_hash << "\nVerified "
                      << (dataset.verified ? "yes" : "no") << " (" << dataset.verified_bytes
                      << " bytes)\n" << dataset.message << '\n';
        return dataset.verified ? 0 : 4;
    }
    if (options.command == "plan") {
        auto plan = BuildDefaultPlan(target, dataset,
            memory::NativeHostMemoryProvider().Snapshot().available_physical_bytes
                .value_or(16ULL << 30U));
        plan.entries = FilterPlan(plan, options, false);
        plan.estimated_read_bytes = 0;
        for (const auto& entry : plan.entries)
            if (entry.disposition == RunStatus::Success)
                plan.estimated_read_bytes +=
                    entry.config.block_bytes * entry.config.request_count;
        if (options.json) std::cout << PlanToJson(plan) << '\n';
        else {
            PrintTarget(target, dataset, provider->DirectStorageAvailable());
            std::cout << "Configurations: " << plan.entries.size()
                      << "\nOutstanding cap: " << plan.outstanding_byte_cap
                      << "\nEstimated reads: " << plan.estimated_read_bytes << " bytes\n";
        }
        return 0;
    }
    if (options.command == "run" || options.command == "deadline")
        return RunMeasurement(options, *provider, target,
                              provider->VerifyDataset(target, false), true);
    if (options.command == "matrix")
        return RunMeasurement(options, *provider, target,
                              provider->VerifyDataset(target, false), false);
    if (options.command == "validate") {
        auto validation = options;
        validation.persist = false;
        validation.block_bytes = 64ULL * 1024ULL;
        validation.queue_depth = 2;
        validation.requests = 16;
        return RunMeasurement(validation, *provider, target,
                              provider->VerifyDataset(target, false), true);
    }
    throw std::invalid_argument("unknown storage command: " + options.command);
}

}  // namespace sidecar::storage
