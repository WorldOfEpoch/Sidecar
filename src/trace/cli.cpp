#include "sidecar/trace/cli.hpp"

#include "sidecar/database/database.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/trace/benchmark.hpp"
#include "sidecar/trace/format.hpp"
#include "sidecar/trace/recorder.hpp"
#include "sidecar/version.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace sidecar::trace {
namespace {

std::uint64_t SteadyNowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::uint64_t WallNowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

TraceHeader MakeHeader(const std::string& machine_hash, const std::string& session_id) {
    const auto version = sidecar::CurrentVersionInfo();
    TraceHeader header;
    header.sidecar_version = version.version;
    header.spec_version = version.spec_version;
    header.git_commit = version.git_commit;
    header.machine_hash = machine_hash;
    header.session_id = session_id;
    header.trace_mode = "LIGHT_SPSC";
    header.wall_clock_created_ns = WallNowNs();
    header.monotonic_start_ns = SteadyNowNs();
    return header;
}

sidecar::hardware::DiscoveryReport DiscoverMachine() {
    auto provider = sidecar::hardware::CreateNativeDiscoveryProvider();
    sidecar::hardware::HardwareDiscoveryService service(*provider);
    return service.Discover();
}

template <typename Record>
Record MakeTestRecord(std::uint32_t producer, std::uint64_t index) {
    Record record{};
    record.host_timestamp_ns = SteadyNowNs();
    record.operation_id = static_cast<decltype(record.operation_id)>(index + 1);
    record.parent_operation_id = static_cast<decltype(record.parent_operation_id)>(index);
    record.subject_id = static_cast<decltype(record.subject_id)>(42 + producer);
    record.payload_bytes = static_cast<decltype(record.payload_bytes)>(4096 + index);
    record.event_type = EventType::HostIoComplete;
    record.source_tier = MemoryTier::NvmeRaw;
    record.destination_tier = MemoryTier::WarmManagedRam;
    record.producer_id = static_cast<decltype(record.producer_id)>(producer);
    if constexpr (std::is_same_v<Record, TraceRecord32>) record.auxiliary = 7;
    else { record.auxiliary_0 = index; record.auxiliary_1 = ~index; }
    return record;
}

template <typename Record>
RecorderMetrics GenerateTrace(const std::filesystem::path& path,
                              const TraceHeader& header,
                              std::uint32_t producers,
                              std::uint64_t records_per_producer) {
    FlightRecorder<Record> recorder;
    using Handle = typename FlightRecorder<Record>::ProducerHandle;
    std::vector<Handle> handles;
    for (std::uint32_t producer = 0; producer < producers; ++producer) {
        handles.push_back(recorder.RegisterProducer("test-producer-" +
                                                    std::to_string(producer), "synthetic"));
    }
    recorder.Start(path, header);
    std::vector<std::thread> threads;
    for (std::uint32_t producer = 0; producer < producers; ++producer) {
        threads.emplace_back([&, producer] {
            for (std::uint64_t index = 0; index < records_per_producer; ++index) {
                (void)handles[producer].tryPush(MakeTestRecord<Record>(producer, index));
            }
        });
    }
    for (auto& thread : threads) thread.join();
    return recorder.Stop();
}

int GenerateTest(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        throw std::invalid_argument(
            "usage: sidecar-lab trace generate-test <file> [--record-size 32|64]");
    }
    std::string record_size = "32";
    if (argc == 6) {
        if (std::string_view(argv[4]) != "--record-size") {
            throw std::invalid_argument("expected --record-size");
        }
        record_size = argv[5];
    }
    const auto report = DiscoverMachine();
    const auto header = MakeHeader(report.identity.machine_hash, "generated-test");
    const auto path = std::filesystem::path(argv[3]);
    RecorderMetrics metrics;
    if (record_size == "32") metrics = GenerateTrace<TraceRecord32>(path, header, 4, 25000);
    else if (record_size == "64") metrics = GenerateTrace<TraceRecord64>(path, header, 4, 25000);
    else throw std::invalid_argument("record size must be 32 or 64");
    std::cout << "generated " << std::filesystem::absolute(path).string() << '\n'
              << "records_written " << metrics.records_written << '\n'
              << "records_dropped " << metrics.records_dropped << '\n'
              << "trace_bytes " << metrics.bytes_written << '\n';
    return metrics.records_written + metrics.records_dropped == 100000 ? 0 : 4;
}

struct FileCommandOptions {
    std::filesystem::path path;
    bool json{false};
};

FileCommandOptions ParseFileCommand(int argc, char** argv) {
    if (argc < 4 || argc > 5) throw std::invalid_argument("trace file path is required");
    FileCommandOptions options{argv[3], false};
    if (argc == 5) {
        if (std::string_view(argv[4]) != "--json") throw std::invalid_argument("expected --json");
        options.json = true;
    }
    return options;
}

int Summary(int argc, char** argv) {
    const auto options = ParseFileCommand(argc, argv);
    const auto trace = TraceReader::Read(options.path, false);
    std::cout << (options.json ? TraceSummaryToJson(trace) : FormatTraceSummary(trace)) << '\n';
    return trace.status == TraceValidationStatus::Valid ? 0 : 5;
}

template <typename Record>
void PrintRecords(std::vector<Record> records) {
    std::stable_sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        return std::tie(left.host_timestamp_ns, left.producer_id, left.operation_id) <
               std::tie(right.host_timestamp_ns, right.producer_id, right.operation_id);
    });
    const auto count = std::min<std::size_t>(20, records.size());
    for (std::size_t index = 0; index < count; ++index) {
        const auto& record = records[index];
        std::cout << "  timestamp=" << record.host_timestamp_ns
                  << " producer=" << record.producer_id
                  << " operation=" << record.operation_id
                  << " parent=" << record.parent_operation_id
                  << " event=" << static_cast<unsigned>(record.event_type) << '\n';
    }
}

int Inspect(int argc, char** argv) {
    const auto options = ParseFileCommand(argc, argv);
    const auto trace = TraceReader::Read(options.path, true);
    if (options.json) {
        std::cout << TraceSummaryToJson(trace) << '\n';
    } else {
        std::cout << FormatTraceSummary(trace) << "\nMetadata\n";
        for (const auto& entry : trace.metadata) {
            std::cout << "  kind=" << static_cast<unsigned>(entry.kind) << " id=" << entry.id
                      << " name=" << entry.name << " type=" << entry.type << '\n';
        }
        std::cout << "\nFirst records (best-effort timestamp merge; per-producer order is canonical)\n";
        if (trace.header.record_format == RecordFormat::Compact32) PrintRecords(trace.records32);
        else PrintRecords(trace.records64);
    }
    return trace.status == TraceValidationStatus::Valid ? 0 : 5;
}

int Validate(int argc, char** argv) {
    if (argc != 4) throw std::invalid_argument("usage: sidecar-lab trace validate <file>");
    const auto trace = TraceReader::Read(argv[3], false);
    std::cout << "status " << ToString(trace.status) << '\n'
              << "message " << trace.message << '\n'
              << "valid_records " << trace.valid_event_records << '\n'
              << "valid_bytes " << trace.valid_bytes << '\n';
    return trace.status == TraceValidationStatus::Valid ? 0 : 5;
}

struct BenchmarkCliOptions {
    bool json{false};
    bool persist{true};
    ObserverBenchmarkOptions benchmark;
    std::filesystem::path database{std::filesystem::path("data") / "sidecar.db"};
};

std::string RawLatencyJson(const ObserverSample& sample) {
    std::string json = "{\"latency_sampling_interval\":1024,\"try_push_ns\":[";
    for (std::size_t index = 0; index < sample.raw_try_push_ns.size(); ++index) {
        if (index) json += ',';
        json += std::to_string(sample.raw_try_push_ns[index]);
    }
    json += "]}";
    return json;
}

BenchmarkCliOptions ParseBenchmarkOptions(int argc, char** argv) {
    BenchmarkCliOptions options;
    for (int index = 3; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--json") options.json = true;
        else if (argument == "--no-persist") options.persist = false;
        else if (argument == "--quick") {
            options.benchmark.events_per_producer = 20000;
            options.benchmark.repetitions = 1;
            options.benchmark.full_sweeps = false;
        } else if (argument == "--events" && index + 1 < argc) {
            options.benchmark.events_per_producer = std::stoull(argv[++index]);
        } else if (argument == "--repetitions" && index + 1 < argc) {
            options.benchmark.repetitions = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--database" && index + 1 < argc) {
            options.database = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            options.benchmark.output_directory = argv[++index];
        } else throw std::invalid_argument("invalid trace benchmark option: " + std::string(argument));
    }
    return options;
}

int Benchmark(int argc, char** argv) {
    const auto options = ParseBenchmarkOptions(argc, argv);
    const auto machine = DiscoverMachine();
    const auto version = sidecar::CurrentVersionInfo();
    std::optional<sidecar::database::Database> database;
    std::int64_t session_id = 0;
    if (options.persist) {
        if (options.database.has_parent_path())
            std::filesystem::create_directories(options.database.parent_path());
        database.emplace(sidecar::database::Database::Open(
            options.database, sidecar::database::OpenMode::CreateOrOpen));
        database->Initialize();
        sidecar::hardware::PersistDiscovery(*database, machine);
        session_id = database->StartBenchmarkSession({machine.identity.machine_hash,
                                                       std::string(version.spec_version),
                                                       std::string(version.git_commit),
                                                       std::nullopt,
                                                       "LIGHT_SPSC",
                                                       std::string("WU3 observer-effect laboratory")});
    }
    auto header = MakeHeader(machine.identity.machine_hash,
                             session_id ? std::to_string(session_id) : "unpersisted");
    const auto report = RunObserverBenchmark(options.benchmark, header);
    bool reconciled = true;
    if (database) {
        for (const auto& sample : report.samples) {
            const auto configuration_id = database->InsertTraceConfiguration({
                session_id, sample.record_size, static_cast<std::int64_t>(sample.ring_capacity),
                sample.producer_count, static_cast<std::int64_t>(sample.collector_batch_size),
                "spin_yield_sleep", "steady_clock_ns"});
            const auto generated = static_cast<std::int64_t>(
                options.benchmark.events_per_producer * sample.producer_count);
            reconciled = reconciled && generated == static_cast<std::int64_t>(
                sample.recorder.records_written + sample.recorder.records_dropped);
            (void)database->InsertTraceBenchmark({
                session_id, configuration_id, sample.workload, sample.repetition,
                static_cast<std::int64_t>(sample.baseline_ns),
                static_cast<std::int64_t>(sample.traced_ns), generated,
                static_cast<std::int64_t>(sample.recorder.records_written),
                static_cast<std::int64_t>(sample.recorder.records_dropped), sample.events_per_second,
                static_cast<std::int64_t>(sample.producer_cpu_ns),
                static_cast<std::int64_t>(sample.recorder.collector_cpu_ns), sample.overhead_raw,
                static_cast<std::int64_t>(sample.recorder.high_water_mark),
                static_cast<std::int64_t>(sample.recorder.bytes_written),
                sample.disk_bytes_per_second, sample.sampled_try_push_ns.p50,
                sample.sampled_try_push_ns.p95, sample.sampled_try_push_ns.p99,
                RawLatencyJson(sample)});
            (void)database->InsertTraceFile({
                session_id, std::filesystem::absolute(sample.trace_path).generic_string(), 1,
                sample.record_size, "LIGHT_SPSC",
                static_cast<std::int64_t>(sample.recorder.bytes_written),
                static_cast<std::int64_t>(sample.recorder.records_written),
                static_cast<std::int64_t>(sample.recorder.records_dropped),
                static_cast<std::int64_t>(sample.recorder.high_water_mark), true, std::nullopt});
        }
        database->CompleteBenchmarkSession(session_id, reconciled ? "COMPLETE" : "FAILED");
    }
    if (options.json) std::cout << ObserverBenchmarkToJson(report) << '\n';
    else {
        std::cout << FormatObserverBenchmark(report);
        if (session_id) std::cout << "\nDatabase session: " << session_id << '\n';
    }
    return reconciled ? 0 : 6;
}

}  // namespace

int RunTraceCommand(int argc, char** argv) {
    if (argc < 3 || std::string_view(argv[1]) != "trace") return -1;
    const std::string_view command(argv[2]);
    if (command == "generate-test") return GenerateTest(argc, argv);
    if (command == "summary") return Summary(argc, argv);
    if (command == "inspect") return Inspect(argc, argv);
    if (command == "validate") return Validate(argc, argv);
    if (command == "benchmark") return Benchmark(argc, argv);
    return -1;
}

}  // namespace sidecar::trace
