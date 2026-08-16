#include "sidecar/database/database.hpp"
#include "sidecar/cuda/overlap_cli.hpp"
#include "sidecar/cuda/transfer_cli.hpp"
#include "sidecar/hardware/format.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/memory/cli.hpp"
#include "sidecar/llama/cli.hpp"
#include "sidecar/pipeline/cli.hpp"
#include "sidecar/storage/cli.hpp"
#include "sidecar/trace/record.hpp"
#include "sidecar/trace/cli.hpp"
#include "sidecar/trace/spsc_ring.hpp"
#include "sidecar/version.hpp"

#include <sqlite3.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>

namespace {

using sidecar::trace::EventType;
using sidecar::trace::MemoryTier;
using sidecar::trace::TraceRecord32;
using sidecar::trace::TraceRecord64;

constexpr std::size_t kSelfTestCapacity = 131072;
constexpr std::uint64_t kSelfTestRecords = 100000;

void PrintUsage() {
    std::cout
        << "Usage:\n"
        << "  sidecar-lab version\n"
        << "  sidecar-lab db init [database-path]\n"
        << "  sidecar-lab db status [database-path]\n"
        << "  sidecar-lab trace selftest [--record-size 32|64|both]\n"
        << "  sidecar-lab trace generate-test <file> [--record-size 32|64]\n"
        << "  sidecar-lab trace inspect <file> [--json]\n"
        << "  sidecar-lab trace summary <file> [--json]\n"
        << "  sidecar-lab trace validate <file>\n"
        << "  sidecar-lab trace benchmark [--json] [--quick] [--database path]\n"
        << "  sidecar-lab info [--json] [--database path] [--no-persist]\n"
        << "  sidecar-lab topology [--json] [--database path] [--no-persist]\n";
    std::cout
        << "  sidecar-lab memory info [--json]\n"
        << "  sidecar-lab memory plan --dry-run [--method pageable|hostalloc|hostregister]\n"
        << "  sidecar-lab memory lifecycle [--method ...] [--sizes 64M,1G] [--dry-run]\n"
        << "  sidecar-lab memory arena [--backend hostalloc|hostregister] [--size 64M]\n"
        << "  sidecar-lab memory stress [--method ...] [--sizes ...]\n"
        << "  sidecar-lab memory report [--json]\n"
        << "  sidecar-lab cuda transfer info|plan|run [filters]\n"
        << "  sidecar-lab cuda transfer sustained|link-state [filters]\n"
        << "  sidecar-lab cuda transfer bidirectional [filters]\n"
        << "  sidecar-lab cuda transfer batching|chunking [filters]\n"
        << "  sidecar-lab cuda transfer report [--json] [--database path]\n";
    std::cout
        << "  sidecar-lab cuda overlap info|calibrate|instrumentation|plan [filters]\n"
        << "  sidecar-lab cuda overlap run|matrix|report|validate [filters]\n";
    std::cout
        << "  sidecar-lab storage info|health|plan|run|matrix|deadline|report|validate [filters]\n"
        << "  sidecar-lab storage dataset create|verify [--full] [--json]\n"
        << "  sidecar-lab pipeline info|host-copy|plan|baseline|run|matrix|stream|report|validate [filters]\n";
    std::cout << "  sidecar-lab llama info|model|baseline|observe|overhead|demand|shadow|report|validate [filters]\n";
}

std::filesystem::path DatabasePath(int argc, char** argv) {
    if (argc >= 4) {
        return std::filesystem::path(argv[3]);
    }
    return std::filesystem::path("data") / "sidecar.db";
}

template <typename Record>
Record MakeSelfTestRecord(std::uint64_t index) {
    Record record{};
    record.host_timestamp_ns = index * 17 + 3;
    record.operation_id = static_cast<decltype(record.operation_id)>(index);
    record.parent_operation_id = static_cast<decltype(record.parent_operation_id)>(
        index == 0 ? 0 : index - 1);
    record.subject_id = static_cast<decltype(record.subject_id)>(index ^ 0x55AA55AAULL);
    record.payload_bytes = static_cast<decltype(record.payload_bytes)>(index * 4096);
    record.event_type = EventType::HostCudaLaunch;
    record.source_tier = MemoryTier::PinnedDmaRing;
    record.destination_tier = MemoryTier::VramL1;
    record.producer_id = static_cast<decltype(record.producer_id)>(7);
    if constexpr (std::is_same_v<Record, TraceRecord32>) {
        record.auxiliary = static_cast<std::uint16_t>((index ^ 0xA55AULL) & 0xFFFFULL);
    } else {
        record.auxiliary_0 = index ^ 0xA55AA55AA55AA55AULL;
        record.auxiliary_1 = ~record.auxiliary_0;
    }
    return record;
}

template <typename Record>
bool IsExpectedRecord(const Record& record, std::uint64_t index) {
    const Record expected = MakeSelfTestRecord<Record>(index);
    bool equal = record.host_timestamp_ns == expected.host_timestamp_ns &&
                 record.operation_id == expected.operation_id &&
                 record.parent_operation_id == expected.parent_operation_id &&
                 record.subject_id == expected.subject_id &&
                 record.payload_bytes == expected.payload_bytes &&
                 record.event_type == expected.event_type &&
                 record.source_tier == expected.source_tier &&
                 record.destination_tier == expected.destination_tier &&
                 record.producer_id == expected.producer_id;
    if constexpr (std::is_same_v<Record, TraceRecord32>) {
        equal = equal && record.auxiliary == expected.auxiliary;
    } else {
        equal = equal && record.auxiliary_0 == expected.auxiliary_0 &&
                record.auxiliary_1 == expected.auxiliary_1 &&
                record.auxiliary_1 == ~record.auxiliary_0;
    }
    return equal;
}

template <typename Record>
bool RunTraceSelfTest(std::string_view label) {
    sidecar::trace::SpscRing<Record, kSelfTestCapacity> ring;
    std::atomic<bool> producer_done{false};
    std::atomic<bool> producer_ok{true};

    const auto started = std::chrono::steady_clock::now();
    std::thread producer([&] {
        for (std::uint64_t index = 0; index < kSelfTestRecords; ++index) {
            if (!ring.tryPush(MakeSelfTestRecord<Record>(index))) {
                producer_ok.store(false, std::memory_order_relaxed);
                break;
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::uint64_t consumed = 0;
    bool consumer_ok = true;
    Record record{};
    while (consumed < kSelfTestRecords) {
        if (ring.tryPop(record)) {
            if (!IsExpectedRecord(record, consumed)) {
                consumer_ok = false;
                break;
            }
            ++consumed;
        } else if (producer_done.load(std::memory_order_acquire)) {
            if (!producer_ok.load(std::memory_order_relaxed)) {
                break;
            }
            // The acquire above observes all producer publications. Retry the
            // authoritative tryPop protocol instead of consulting diagnostic
            // occupancy; the next acquire sees the final published sequence.
            continue;
        } else {
            std::this_thread::yield();
        }
    }
    producer.join();
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    const bool passed = producer_ok.load(std::memory_order_relaxed) && consumer_ok &&
                        consumed == kSelfTestRecords && ring.droppedEvents() == 0 &&
                        ring.publishedSequence() == kSelfTestRecords &&
                        ring.consumedSequence() == kSelfTestRecords;

    std::cout << "trace selftest record_size=" << label
              << " records=" << consumed
              << " dropped=" << ring.droppedEvents()
              << " high_water=" << ring.highWaterMarkQuiescent()
              << " elapsed_us=" << elapsed_us
              << " result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

int RunVersion() {
    const auto version = sidecar::CurrentVersionInfo();
    std::cout << "Sidecar " << version.version << '\n'
              << "spec " << version.spec_version << '\n'
              << "git " << version.git_commit << '\n'
              << "sqlite " << sqlite3_libversion() << '\n'
              << "cuda discovery-" << (SIDECAR_CUDA_ENABLED ? "enabled" : "disabled") << '\n';
    return 0;
}

int RunDbInit(int argc, char** argv) {
    const auto path = DatabasePath(argc, argv);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    auto database = sidecar::database::Database::Open(
        path, sidecar::database::OpenMode::CreateOrOpen);
    database.Initialize();
    const auto status = database.Status();
    std::cout << "database initialized\n"
              << "path " << std::filesystem::absolute(status.path).string() << '\n'
              << "schema_version " << status.schema_version << '\n'
              << "latest_migration " << status.latest_migration << '\n'
              << "foreign_keys " << (status.foreign_keys_enabled ? "ON" : "OFF") << '\n'
              << "foreign_key_violations " << status.foreign_key_violations << '\n';
    return 0;
}

int RunDbStatus(int argc, char** argv) {
    const auto path = DatabasePath(argc, argv);
    if (!std::filesystem::exists(path)) {
        std::cerr << "database does not exist: " << std::filesystem::absolute(path).string() << '\n';
        return 2;
    }
    auto database = sidecar::database::Database::Open(
        path, sidecar::database::OpenMode::ExistingReadWrite);
    const auto status = database.Status();
    std::cout << "path " << std::filesystem::absolute(status.path).string() << '\n'
              << "initialized " << (status.initialized ? "yes" : "no") << '\n'
              << "application_id " << status.application_id << '\n'
              << "schema_version " << status.schema_version << '\n'
              << "latest_migration " << status.latest_migration << '\n'
              << "user_tables " << status.user_table_count << '\n'
              << "foreign_keys " << (status.foreign_keys_enabled ? "ON" : "OFF") << '\n'
              << "foreign_key_violations " << status.foreign_key_violations << '\n';
    return status.initialized ? 0 : 3;
}

int RunTraceSelfTests(int argc, char** argv) {
    std::string record_size = "both";
    if (argc == 5 && std::string_view(argv[3]) == "--record-size") {
        record_size = argv[4];
    } else if (argc != 3) {
        PrintUsage();
        return 1;
    }

    bool passed = true;
    if (record_size == "32" || record_size == "both") {
        passed = RunTraceSelfTest<TraceRecord32>("32") && passed;
    }
    if (record_size == "64" || record_size == "both") {
        passed = RunTraceSelfTest<TraceRecord64>("64") && passed;
    }
    if (record_size != "32" && record_size != "64" && record_size != "both") {
        std::cerr << "invalid record size: " << record_size << '\n';
        return 1;
    }
    return passed ? 0 : 4;
}

struct DiscoveryOptions {
    bool json{false};
    bool persist{true};
    std::filesystem::path database_path{std::filesystem::path("data") / "sidecar.db"};
};

DiscoveryOptions ParseDiscoveryOptions(int argc, char** argv) {
    DiscoveryOptions options;
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--json") {
            options.json = true;
        } else if (argument == "--no-persist") {
            options.persist = false;
        } else if (argument == "--database" && index + 1 < argc) {
            options.database_path = argv[++index];
        } else {
            throw std::invalid_argument("invalid discovery option: " + std::string(argument));
        }
    }
    return options;
}

int RunDiscoveryCommand(bool topology, int argc, char** argv) {
    const DiscoveryOptions options = ParseDiscoveryOptions(argc, argv);
    auto provider = sidecar::hardware::CreateNativeDiscoveryProvider();
    sidecar::hardware::HardwareDiscoveryService service(*provider);
    const sidecar::hardware::DiscoveryReport report = service.Discover();

    if (options.persist) {
        if (options.database_path.has_parent_path()) {
            std::filesystem::create_directories(options.database_path.parent_path());
        }
        auto database = sidecar::database::Database::Open(
            options.database_path, sidecar::database::OpenMode::CreateOrOpen);
        database.Initialize();
        sidecar::hardware::PersistDiscovery(database, report);
    }

    if (options.json) {
        std::cout << (topology ? sidecar::hardware::TopologyToJson(report)
                              : sidecar::hardware::DiscoveryReportToJson(report))
                  << '\n';
    } else {
        std::cout << (topology ? sidecar::hardware::FormatTopology(report)
                              : sidecar::hardware::FormatSystemInformation(report));
    }
    return 0;
}

int Run(int argc, char** argv) {
    const int llama_result = sidecar::llama::RunLlamaCli(argc, argv);
    if (llama_result >= 0) {
        return llama_result;
    }
    if (argc >= 2 && std::string_view(argv[1]) == "pipeline") {
        return sidecar::pipeline::RunPipelineCli(argc, argv);
    }
    const int storage_result = sidecar::storage::RunStorageCli(argc, argv);
    if (storage_result >= 0) {
        return storage_result;
    }
    if (argc >= 3 && std::string_view(argv[1]) == "cuda" &&
        std::string_view(argv[2]) == "overlap") {
        return sidecar::cuda::RunOverlapCli(argc, argv);
    }
    if (argc >= 3 && std::string_view(argv[1]) == "cuda" &&
        std::string_view(argv[2]) == "transfer") {
        return sidecar::cuda::RunTransferCli(argc, argv);
    }
    const int memory_result = sidecar::memory::RunMemoryCommand(argc, argv);
    if (memory_result >= 0) {
        return memory_result;
    }
    const int trace_result = sidecar::trace::RunTraceCommand(argc, argv);
    if (trace_result >= 0) {
        return trace_result;
    }
    if (argc == 2 && std::string_view(argv[1]) == "version") {
        return RunVersion();
    }
    if (argc >= 3 && std::string_view(argv[1]) == "db" &&
        std::string_view(argv[2]) == "init") {
        return RunDbInit(argc, argv);
    }
    if (argc >= 3 && std::string_view(argv[1]) == "db" &&
        std::string_view(argv[2]) == "status") {
        return RunDbStatus(argc, argv);
    }
    if (argc >= 3 && std::string_view(argv[1]) == "trace" &&
        std::string_view(argv[2]) == "selftest") {
        return RunTraceSelfTests(argc, argv);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "info") {
        return RunDiscoveryCommand(false, argc, argv);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "topology") {
        return RunDiscoveryCommand(true, argc, argv);
    }

    PrintUsage();
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Run(argc, argv);
    } catch (const sidecar::database::DatabaseError& error) {
        std::cerr << "SQLite error " << error.sqliteCode() << ": " << error.what() << '\n';
        return 10;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 11;
    }
}
