#include "sidecar/database/database.hpp"
#include "sidecar/database/schema.hpp"
#include "sidecar/core/sha256.hpp"
#include "sidecar/hardware/format.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/identity.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/trace/record.hpp"
#include "sidecar/trace/spsc_ring.hpp"
#include "sidecar/version.hpp"

#include <sqlite3.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using sidecar::trace::EventType;
using sidecar::trace::MemoryTier;
using sidecar::trace::TraceRecord32;
using sidecar::trace::TraceRecord64;

sidecar::hardware::MachineIdentityInput StrongIdentityInput() {
    return sidecar::hardware::MachineIdentityInput{
        std::string("12345678-1234-5678-9abc-def012345678"),
        std::string("Example Systems"),
        std::string("Workstation X"),
        std::string("System-Serial-1"),
        std::string("Example Board Vendor"),
        std::string("Board Z"),
        std::string("Board-Serial-1"),
        std::string("fallback-guid")};
}

sidecar::hardware::RawDiscovery SyntheticDiscovery() {
    sidecar::hardware::RawDiscovery raw;
    raw.identity_input = StrongIdentityInput();
    raw.snapshot.operating_system = {"Windows Test", "10.0.1", "1", "x86_64", "test-host"};
    raw.snapshot.cpu.vendor = "TestVendor";
    raw.snapshot.cpu.model = "Test CPU";
    raw.snapshot.cpu.architecture = "x86_64";
    raw.snapshot.cpu.physical_core_count = 8;
    raw.snapshot.cpu.logical_processor_count = 16;
    raw.snapshot.cpu.processor_group_count = 1;
    raw.snapshot.cpu.numa_node_count = 1;
    raw.snapshot.cpu.package_count = 1;
    raw.snapshot.cpu.numa_nodes.push_back({0, {0}, 32ULL * 1024 * 1024 * 1024});
    raw.snapshot.memory.installed_physical_bytes = 96ULL * 1024 * 1024 * 1024;
    raw.snapshot.memory.visible_physical_bytes = 95ULL * 1024 * 1024 * 1024;
    raw.snapshot.memory.available_physical_bytes = 64ULL * 1024 * 1024 * 1024;
    raw.snapshot.platform = {"Example Systems", "Workstation X", "Example Board Vendor",
                             "Board Z", "Example BIOS", "1.0", "3.6"};
    sidecar::hardware::GpuInfo gpu0;
    gpu0.persistent_id = "GPU-test-0";
    gpu0.model = "Synthetic GPU 0";
    gpu0.cuda_device_index = 0;
    gpu0.vram_bytes = 24ULL * 1024 * 1024 * 1024;
    gpu0.pci_bus = 2;
    gpu0.pci_device = 0;
    gpu0.discovered_by_cuda = true;
    sidecar::hardware::GpuInfo gpu1;
    gpu1.persistent_id = "GPU-test-1";
    gpu1.model = "Synthetic GPU 1";
    gpu1.cuda_device_index = 1;
    gpu1.vram_bytes = 16ULL * 1024 * 1024 * 1024;
    raw.snapshot.gpus = {gpu0, gpu1};
    raw.snapshot.cuda_available = true;
    sidecar::hardware::StorageDeviceInfo disk0;
    disk0.persistent_id = "disk-test-0";
    disk0.model = "Synthetic NVMe 0";
    disk0.bus_type = "NVMe";
    disk0.capacity_bytes = 2ULL * 1024 * 1024 * 1024 * 1024;
    disk0.topology_confidence = sidecar::hardware::DiscoveryConfidence::DirectlyReported;
    disk0.topology_source = "synthetic";
    sidecar::hardware::StorageDeviceInfo disk1;
    disk1.persistent_id = "disk-test-1";
    disk1.model = "Synthetic NVMe 1";
    disk1.bus_type = "NVMe";
    disk1.capacity_bytes = 1ULL * 1024 * 1024 * 1024 * 1024;
    disk1.topology_confidence = sidecar::hardware::DiscoveryConfidence::Unknown;
    disk1.topology_source = "synthetic unknown";
    raw.snapshot.storage_devices = {disk0, disk1};
    raw.snapshot.storage_discovery_available = true;
    raw.snapshot.topology.push_back({"machine", std::nullopt, "MACHINE", "Test machine",
                                     sidecar::hardware::DiscoveryConfidence::DirectlyReported,
                                     "synthetic", {}, {}});
    return raw;
}

class SyntheticProvider final : public sidecar::hardware::IHardwareDiscoveryProvider {
public:
    explicit SyntheticProvider(sidecar::hardware::RawDiscovery discovery)
        : discovery_(std::move(discovery)) {}
    sidecar::hardware::RawDiscovery Discover() override { return discovery_; }
private:
    sidecar::hardware::RawDiscovery discovery_;
};

void Expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Record>
Record MakeRecord(std::uint64_t index) {
    Record record{};
    record.host_timestamp_ns = index * 101 + 7;
    record.operation_id = static_cast<decltype(record.operation_id)>(index);
    record.parent_operation_id = static_cast<decltype(record.parent_operation_id)>(
        index == 0 ? 0 : index - 1);
    record.subject_id = static_cast<decltype(record.subject_id)>(index ^ 0x11223344ULL);
    record.payload_bytes = static_cast<decltype(record.payload_bytes)>(index * 4096);
    record.event_type = EventType::GpuTransferStart;
    record.source_tier = MemoryTier::PinnedDmaRing;
    record.destination_tier = MemoryTier::VramL1;
    record.producer_id = static_cast<decltype(record.producer_id)>(13);
    if constexpr (std::is_same_v<Record, TraceRecord32>) {
        record.auxiliary = static_cast<std::uint16_t>((index ^ 0xBEEF) & 0xFFFF);
    } else {
        record.auxiliary_0 = index ^ 0xDEADBEEFCAFEBABEULL;
        record.auxiliary_1 = ~record.auxiliary_0;
    }
    return record;
}

template <typename Record>
bool RecordMatches(const Record& record, std::uint64_t index) {
    const Record expected = MakeRecord<Record>(index);
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
void TestBasicAndPublication() {
    sidecar::trace::SpscRing<Record, 8> ring;
    Record output{};
    Expect(!ring.tryPop(output), "new ring must be empty");
    Expect(ring.tryPush(MakeRecord<Record>(0)), "first record must publish");
    Expect(ring.publishedSequence() == 1, "published sequence must advance");
    Expect(ring.consumedSequence() == 0, "consumed sequence must not advance before pop");
    Expect(ring.tryPop(output), "published record must be consumable");
    Expect(RecordMatches(output, 0), "consumed record must match published bytes");
    Expect(ring.consumedSequence() == 1, "consumed sequence must advance after pop");
    Expect(ring.producerSequenceQuiescent() == 1, "producer-private sequence must advance");
    Expect(ring.consumerSequenceQuiescent() == 1, "consumer-private sequence must advance");
}

template <typename Record>
void TestWraparoundNoOverrun() {
    sidecar::trace::SpscRing<Record, 4> ring;
    std::uint64_t next_produced = 0;
    std::uint64_t next_consumed = 0;
    Record output{};

    for (int cycle = 0; cycle < 1000; ++cycle) {
        for (int index = 0; index < 4; ++index) {
            Expect(ring.tryPush(MakeRecord<Record>(next_produced)),
                   "ring must accept a free slot across wraparound");
            ++next_produced;
        }
        for (int index = 0; index < 4; ++index) {
            Expect(ring.tryPop(output), "ring must return every record across wraparound");
            Expect(RecordMatches(output, next_consumed), "wraparound changed record data");
            ++next_consumed;
        }
    }

    Expect(next_consumed == 4000, "wraparound test must consume all records");
    Expect(ring.droppedEvents() == 0, "non-overrun test must not report drops");
    Expect(ring.highWaterMarkQuiescent() == 4, "high-water mark must reach capacity");
}

template <typename Record>
void TestFullAndDroppedAccounting() {
    sidecar::trace::SpscRing<Record, 4> ring;
    for (std::uint64_t index = 0; index < 4; ++index) {
        Expect(ring.tryPush(MakeRecord<Record>(index)), "ring must fill to advertised capacity");
    }
    Expect(!ring.tryPush(MakeRecord<Record>(4)), "full ring must reject the new event");
    Expect(!ring.tryPush(MakeRecord<Record>(5)), "full ring must remain safe on repeated overflow");
    Expect(ring.droppedEvents() == 2, "every rejected event must be counted");

    Record output{};
    for (std::uint64_t index = 0; index < 4; ++index) {
        Expect(ring.tryPop(output), "pre-overrun records must remain readable");
        Expect(RecordMatches(output, index), "overflow must never overwrite unread records");
    }
    Expect(!ring.tryPop(output), "ring must be empty after consuming retained records");
}

template <typename Record>
void TestConcurrentStressNoTornRecords() {
    constexpr std::uint64_t kRecordCount = 200000;
    // The concurrent integrity test deliberately has enough storage for every
    // record. It therefore never uses diagnostic occupancy as backpressure.
    // Small-capacity wraparound is covered independently above.
    constexpr std::uint64_t kCapacity = 262144;
    sidecar::trace::SpscRing<Record, kCapacity> ring;
    std::atomic<bool> producer_done{false};
    std::atomic<bool> producer_ok{true};

    std::thread producer([&] {
        for (std::uint64_t index = 0; index < kRecordCount; ++index) {
            if (!ring.tryPush(MakeRecord<Record>(index))) {
                producer_ok.store(false, std::memory_order_relaxed);
                break;
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    Record output{};
    std::uint64_t consumed = 0;
    bool consumer_ok = true;
    while (consumed < kRecordCount) {
        if (ring.tryPop(output)) {
            if (!RecordMatches(output, consumed)) {
                consumer_ok = false;
                break;
            }
            ++consumed;
        } else if (producer_done.load(std::memory_order_acquire)) {
            if (!producer_ok.load(std::memory_order_relaxed)) {
                break;
            }
            continue;
        } else {
            std::this_thread::yield();
        }
    }
    producer.join();

    Expect(producer_ok.load(std::memory_order_relaxed), "stress producer unexpectedly overflowed");
    Expect(consumer_ok, "stress test observed a torn or out-of-order record");
    Expect(consumed == kRecordCount, "stress test did not consume every record");
    Expect(ring.droppedEvents() == 0, "backpressured stress test must not drop records");
    Expect(ring.publishedSequence() == kRecordCount, "stress published sequence mismatch");
    Expect(ring.consumedSequence() == kRecordCount, "stress consumed sequence mismatch");
}

std::filesystem::path MakeTemporaryDatabasePath() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("sidecar-test-" + std::to_string(stamp) + ".db");
}

void TestDatabaseBootstrapAndForeignKeys() {
    const auto path = MakeTemporaryDatabasePath();
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
    database.Initialize();

    const auto status = database.Status();
    Expect(status.initialized, "database must report initialized");
    Expect(status.application_id == sidecar::database::kApplicationId,
           "database application ID mismatch");
    Expect(status.schema_version == sidecar::database::kCurrentSchemaVersion,
           "database schema version mismatch");
    Expect(status.latest_migration == sidecar::database::kCurrentSchemaVersion,
           "latest migration mismatch");
    Expect(status.foreign_keys_enabled, "foreign keys must be enabled per connection");
    Expect(status.foreign_key_violations == 0,
           "fresh database has foreign-key violations");
    Expect(status.user_table_count >= 14, "required forensic tables are missing");

    bool foreign_key_rejected = false;
    try {
        (void)database.StartBenchmarkSession(sidecar::database::BenchmarkSessionInput{
            "missing-machine", "M0-FROZEN-1", "UNCOMMITTED", std::nullopt,
            "TRACE_NONE", std::nullopt});
    } catch (const sidecar::database::DatabaseError&) {
        foreign_key_rejected = true;
    }
    Expect(foreign_key_rejected, "benchmark session must reject an unknown hardware profile");

    database.UpsertHardwareProfile(sidecar::database::HardwareProfileInput{
        "test-machine-hash", "test-host", "Test OS", "1.0", "1", "Test CPU",
        8, 16, 1, 96LL * 1024 * 1024 * 1024, 64LL * 1024 * 1024 * 1024, "{}"});
    const auto session_id = database.StartBenchmarkSession(
        sidecar::database::BenchmarkSessionInput{
            "test-machine-hash", "M0-FROZEN-1", "UNCOMMITTED", std::nullopt,
            "TRACE_LIGHT_SPSC", std::string("unit test")});
    Expect(session_id > 0, "valid benchmark session must receive an ID");
    database.CompleteBenchmarkSession(session_id, "COMPLETE");
}

void TestDatabaseRejectsNewerSchema() {
    const auto path = MakeTemporaryDatabasePath();
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } cleanup{path};

    sqlite3* raw_database = nullptr;
    const std::string path_utf8 = path.generic_string();
    Expect(sqlite3_open(path_utf8.c_str(), &raw_database) == SQLITE_OK,
           "test database setup failed");
    const auto newer = std::string("PRAGMA user_version = ") +
                       std::to_string(sidecar::database::kCurrentSchemaVersion + 1) + ";";
    Expect(sqlite3_exec(raw_database, newer.c_str(), nullptr, nullptr, nullptr) ==
               SQLITE_OK,
           "test schema version setup failed");
    sqlite3_close(raw_database);

    auto database = sidecar::database::Database::Open(
        path, sidecar::database::OpenMode::ExistingReadWrite);
    bool rejected = false;
    try {
        database.Initialize();
    } catch (const sidecar::database::DatabaseError&) {
        rejected = true;
    }
    Expect(rejected, "an older Sidecar build must reject a newer database schema");
}

void TestDatabaseMigratesV1() {
    const auto path = MakeTemporaryDatabasePath();
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } cleanup{path};
    sqlite3* raw_database = nullptr;
    const std::string path_utf8 = path.generic_string();
    Expect(sqlite3_open(path_utf8.c_str(), &raw_database) == SQLITE_OK,
           "migration database setup failed");
    const std::string schema(sidecar::database::kSchemaSql);
    Expect(sqlite3_exec(raw_database, schema.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK,
           "v1 schema setup failed");
    sqlite3_close(raw_database);

    auto database = sidecar::database::Database::Open(
        path, sidecar::database::OpenMode::ExistingReadWrite);
    database.Initialize();
    const auto status = database.Status();
    Expect(status.schema_version == sidecar::database::kCurrentSchemaVersion,
           "v1 database did not migrate to current schema");
    Expect(status.latest_migration == sidecar::database::kCurrentSchemaVersion,
           "current migration history was not recorded");
}

void TestSha256KnownVector() {
    Expect(sidecar::core::Sha256Hex("abc") ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "SHA-256 implementation failed the standard abc vector");
    const auto path = std::filesystem::temp_directory_path() /
        ("sidecar-sha-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    {
        std::ofstream output(path, std::ios::binary);
        output << "abc";
    }
    Expect(sidecar::core::Sha256File(path) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "streaming file SHA-256 failed the standard abc vector");
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

void TestMachineIdentityExcludesSnapshotState() {
    SyntheticProvider provider_a(SyntheticDiscovery());
    sidecar::hardware::HardwareDiscoveryService service_a(provider_a);
    const auto report_a = service_a.Discover();

    auto changed = SyntheticDiscovery();
    changed.snapshot.memory.installed_physical_bytes = 256ULL * 1024 * 1024 * 1024;
    changed.snapshot.memory.available_physical_bytes = 200ULL * 1024 * 1024 * 1024;
    changed.snapshot.gpus[0].persistent_id = "GPU-replacement";
    changed.snapshot.gpus[0].model = "Different GPU";
    changed.snapshot.storage_devices[0].persistent_id = "replacement-nvme";
    changed.snapshot.nvidia_driver_version = "999.1";
    changed.snapshot.cuda_runtime_version = 99999;
    changed.snapshot.operating_system.build = "99999";
    SyntheticProvider provider_b(std::move(changed));
    sidecar::hardware::HardwareDiscoveryService service_b(provider_b);
    const auto report_b = service_b.Discover();
    Expect(report_a.identity.machine_hash == report_b.identity.machine_hash,
           "snapshot RAM/GPU/storage/software changes altered machine_hash");
    Expect(report_a.identity.canonical_material == report_b.identity.canonical_material,
           "snapshot state leaked into canonical identity material");
}

void TestMachineIdentityDifferentSystemUuid() {
    auto first = StrongIdentityInput();
    auto second = first;
    second.system_uuid = "87654321-4321-8765-abcd-0123456789ab";
    Expect(sidecar::hardware::BuildMachineIdentity(first).machine_hash !=
               sidecar::hardware::BuildMachineIdentity(second).machine_hash,
           "different valid SMBIOS UUIDs must produce different hashes");
}

void TestMachineIdentityNormalizationAndRepeatability() {
    auto first = StrongIdentityInput();
    auto second = first;
    second.system_uuid = "{12345678-1234-5678-9ABC-DEF012345678}";
    second.system_manufacturer = "  EXAMPLE   SYSTEMS  ";
    second.system_product = "WORKSTATION x";
    second.board_manufacturer = "Example Board Vendor\t";
    const auto identity_a = sidecar::hardware::BuildMachineIdentity(first);
    const auto identity_b = sidecar::hardware::BuildMachineIdentity(second);
    const auto identity_repeat = sidecar::hardware::BuildMachineIdentity(first);
    Expect(identity_a.machine_hash == identity_b.machine_hash,
           "defined case/whitespace normalization changed the hash");
    Expect(identity_a.canonical_material == identity_b.canonical_material,
           "normalized canonical bytes differ");
    Expect(identity_a.machine_hash == identity_repeat.machine_hash &&
               identity_a.canonical_material == identity_repeat.canonical_material,
           "identity serialization is not byte-stable");
    Expect(identity_a.canonical_material.find("System-Serial-1") == std::string::npos &&
               identity_a.canonical_material.find("Board-Serial-1") == std::string::npos,
           "raw platform serial leaked into canonical identity material");
}

void TestMachineIdentityPlaceholdersDowngradeQuality() {
    sidecar::hardware::MachineIdentityInput input;
    input.system_uuid = "00000000-0000-0000-0000-000000000000";
    input.system_manufacturer = "To Be Filled By O.E.M.";
    input.system_product = "Default string";
    input.board_manufacturer = "System Manufacturer";
    input.board_product = "Unknown";
    input.board_serial = "0123456789";
    input.fallback_installation_id = "real-installation-guid";
    const auto identity = sidecar::hardware::BuildMachineIdentity(input);
    Expect(identity.quality == sidecar::hardware::IdentityQuality::Fallback,
           "placeholder SMBIOS data must not receive strong/moderate quality");
    Expect(identity.canonical_material.find("00000000-0000") == std::string::npos,
           "invalid UUID leaked into canonical identity");
    Expect(identity.basis.size() == 1 &&
               identity.basis[0] == "hashed_windows_installation_id",
           "fallback identity source was not documented");
}

void TestSyntheticDiscoveryAndSerialization() {
    auto raw = SyntheticDiscovery();
    raw.snapshot.cuda_available = false;
    raw.snapshot.nvml_available = false;
    raw.snapshot.gpus[1].vram_bytes.reset();
    raw.snapshot.topology[0].confidence = sidecar::hardware::DiscoveryConfidence::Unknown;
    SyntheticProvider provider(std::move(raw));
    sidecar::hardware::HardwareDiscoveryService service(provider);
    const auto report = service.Discover();
    Expect(report.snapshot.gpus.size() == 2, "synthetic multi-GPU discovery lost devices");
    Expect(report.snapshot.storage_devices.size() == 2,
           "synthetic multi-storage discovery lost devices");
    const std::string json_a = sidecar::hardware::DiscoveryReportToJson(report);
    const std::string json_b = sidecar::hardware::DiscoveryReportToJson(report);
    Expect(json_a == json_b, "hardware JSON serialization is not stable");
    Expect(json_a.find("\"available\":false") != std::string::npos,
           "no-CUDA/no-NVML state was not serialized");
    Expect(json_a.find("\"vram_bytes\":null") != std::string::npos,
           "missing optional field was not serialized as null");
    Expect(json_a.find("System-Serial-1") == std::string::npos &&
               json_a.find("Board-Serial-1") == std::string::npos,
           "sensitive platform serial leaked into ordinary JSON output");
    const std::string human = sidecar::hardware::FormatSystemInformation(report);
    Expect(human.find("SIDECAR SYSTEM INFORMATION") != std::string::npos &&
               human.find("Synthetic GPU 1") != std::string::npos,
           "human-readable discovery output smoke test failed");
    const std::string topology_json = sidecar::hardware::TopologyToJson(report);
    Expect(topology_json == sidecar::hardware::TopologyToJson(report),
           "topology JSON serialization is not stable");
}

void TestPersistenceRefreshIsIdempotent() {
    const auto path = MakeTemporaryDatabasePath();
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

    SyntheticProvider provider(SyntheticDiscovery());
    sidecar::hardware::HardwareDiscoveryService service(provider);
    auto report = service.Discover();
    sidecar::hardware::PersistDiscovery(database, report);
    auto counts = database.InventoryCounts(report.identity.machine_hash);
    Expect(counts.active_gpus == 2 && counts.total_gpu_rows == 2,
           "initial GPU inventory persistence failed");
    Expect(counts.active_storage_devices == 2 && counts.total_storage_rows == 2,
           "initial storage inventory persistence failed");

    report.snapshot.gpus.resize(1);
    report.snapshot.gpus[0].model = "Updated GPU model";
    report.snapshot.storage_devices.resize(1);
    sidecar::hardware::PersistDiscovery(database, report);
    sidecar::hardware::PersistDiscovery(database, report);
    counts = database.InventoryCounts(report.identity.machine_hash);
    Expect(counts.active_gpus == 1 && counts.total_gpu_rows == 2,
           "GPU refresh duplicated or deleted historical inventory rows");
    Expect(counts.active_storage_devices == 1 && counts.total_storage_rows == 2,
           "storage refresh duplicated or deleted historical inventory rows");

    auto unavailable_report = report;
    unavailable_report.snapshot.cuda_available = false;
    unavailable_report.snapshot.nvml_available = false;
    unavailable_report.snapshot.storage_discovery_available = false;
    unavailable_report.snapshot.gpus.clear();
    unavailable_report.snapshot.storage_devices.clear();
    sidecar::hardware::PersistDiscovery(database, unavailable_report);
    counts = database.InventoryCounts(report.identity.machine_hash);
    Expect(counts.active_gpus == 1 && counts.active_storage_devices == 1,
           "unavailable optional providers incorrectly marked inventory absent");
}

void TestVersionMetadata() {
    const auto version = sidecar::CurrentVersionInfo();
    Expect(!version.version.empty(), "Sidecar version must not be empty");
    Expect(version.spec_version == "M0-FROZEN-1", "frozen spec version mismatch");
    Expect(!version.git_commit.empty(), "git revision marker must not be empty");
}

struct TestCase {
    std::string_view name;
    std::function<void()> run;
};

}  // namespace

int main() {
    const std::vector<TestCase> tests{
        {"record_sizes", [] {
             Expect(sizeof(TraceRecord32) == 32, "TraceRecord32 size changed");
             Expect(sizeof(TraceRecord64) == 64, "TraceRecord64 size changed");
         }},
        {"spsc_basic_32", TestBasicAndPublication<TraceRecord32>},
        {"spsc_basic_64", TestBasicAndPublication<TraceRecord64>},
        {"spsc_wraparound_32", TestWraparoundNoOverrun<TraceRecord32>},
        {"spsc_wraparound_64", TestWraparoundNoOverrun<TraceRecord64>},
        {"spsc_overflow_32", TestFullAndDroppedAccounting<TraceRecord32>},
        {"spsc_overflow_64", TestFullAndDroppedAccounting<TraceRecord64>},
        {"spsc_concurrent_stress_32", TestConcurrentStressNoTornRecords<TraceRecord32>},
        {"spsc_concurrent_stress_64", TestConcurrentStressNoTornRecords<TraceRecord64>},
        {"database_bootstrap_foreign_keys", TestDatabaseBootstrapAndForeignKeys},
        {"database_rejects_newer_schema", TestDatabaseRejectsNewerSchema},
        {"database_migrates_v1", TestDatabaseMigratesV1},
        {"sha256_known_vector", TestSha256KnownVector},
        {"machine_identity_excludes_snapshot_state", TestMachineIdentityExcludesSnapshotState},
        {"machine_identity_different_uuid", TestMachineIdentityDifferentSystemUuid},
        {"machine_identity_normalization_repeatability",
         TestMachineIdentityNormalizationAndRepeatability},
        {"machine_identity_placeholders", TestMachineIdentityPlaceholdersDowngradeQuality},
        {"synthetic_discovery_serialization", TestSyntheticDiscoveryAndSerialization},
        {"persistence_refresh_idempotent", TestPersistenceRefreshIsIdempotent},
        {"version_metadata", TestVersionMetadata},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }

    std::cout << "tests=" << tests.size() << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
