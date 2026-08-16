#include "sidecar/database/database.hpp"
#include "sidecar/trace/benchmark.hpp"
#include "sidecar/trace/format.hpp"
#include "sidecar/trace/recorder.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace sidecar::trace;

void Expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

std::filesystem::path TempPath(std::string_view suffix) {
    return std::filesystem::temp_directory_path() /
           ("sidecar-trace-test-" + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()) + std::string(suffix));
}

struct Cleanup {
    std::vector<std::filesystem::path> paths;
    ~Cleanup() {
        for (const auto& path : paths) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            std::filesystem::remove(path.string() + "-wal", ignored);
            std::filesystem::remove(path.string() + "-shm", ignored);
        }
    }
};

TraceHeader Header(RecordFormat format = RecordFormat::Compact32) {
    TraceHeader header;
    header.sidecar_version = "test-version";
    header.spec_version = "M0-FROZEN-1";
    header.git_commit = "0123456789012345678901234567890123456789";
    header.machine_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    header.session_id = "trace-test";
    header.trace_mode = "TEST";
    header.record_format = format;
    header.record_size = format == RecordFormat::Compact32 ? 32 : 64;
    header.producer_count = 2;
    header.wall_clock_created_ns = 11;
    header.monotonic_start_ns = 22;
    return header;
}

TraceRecord32 Record32(std::uint16_t producer, std::uint32_t operation) {
    TraceRecord32 record{};
    record.host_timestamp_ns = 1000 + operation;
    record.operation_id = operation;
    record.parent_operation_id = operation ? operation - 1 : 0;
    record.subject_id = producer + 10;
    record.payload_bytes = operation * 4096;
    record.event_type = EventType::HostIoComplete;
    record.source_tier = MemoryTier::NvmeRaw;
    record.destination_tier = MemoryTier::WarmManagedRam;
    record.producer_id = producer;
    record.auxiliary = 0xAA55;
    return record;
}

TraceRecord64 Record64(std::uint32_t producer, std::uint64_t operation) {
    TraceRecord64 record{};
    record.host_timestamp_ns = 2000 + operation;
    record.operation_id = operation;
    record.parent_operation_id = operation ? operation - 1 : 0;
    record.subject_id = 0x100000000ULL + producer;
    record.payload_bytes = 0x200000000ULL + operation;
    record.producer_id = producer;
    record.event_type = EventType::GpuKernelEnd;
    record.source_tier = MemoryTier::VramL1;
    record.destination_tier = MemoryTier::PinnedDmaRing;
    record.auxiliary_0 = 0xABCDEF0123456789ULL;
    record.auxiliary_1 = ~record.auxiliary_0;
    return record;
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void WriteBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

void Put32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index)
        bytes[offset + index] = static_cast<std::uint8_t>(value >> (index * 8));
}

void Put64(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index)
        bytes[offset + index] = static_cast<std::uint8_t>(value >> (index * 8));
}

void RefreshHeaderCrc(std::vector<std::uint8_t>& bytes) {
    Put32(bytes, 400, Crc32c(std::span<const std::uint8_t>(bytes).first(400)));
}

std::filesystem::path MakeValidTrace(Cleanup& cleanup) {
    const auto path = TempPath(".sidecartrace");
    cleanup.paths.push_back(path);
    TraceWriter writer(path, Header(), {{MetadataKind::Producer, 0, "p0", "test"},
                                        {MetadataKind::Producer, 1, "p1", "test"}});
    const std::vector<TraceRecord32> p0{Record32(0, 1), Record32(0, 2), Record32(0, 3)};
    const std::vector<TraceRecord32> p1{Record32(1, 1), Record32(1, 2)};
    writer.Append(0, 0, p0);
    writer.Append(1, 0, p1);
    writer.Finalize({5, 0, 3, 3, 1, writer.bytesWritten(), 100, 999});
    return path;
}

void TestHeaderRoundTrip() {
    auto header = Header();
    header.metadata_offset = 512;
    header.event_data_offset = 5ULL * 1024 * 1024 * 1024;
    header.footer_offset = header.event_data_offset + 4096;
    const auto bytes = SerializeHeader(header);
    Expect(bytes.size() == 512, "header must be exactly 512 bytes");
    TraceHeader parsed;
    TraceValidationStatus status{};
    std::string error;
    Expect(ParseHeader(bytes, parsed, status, error), "header did not parse");
    Expect(parsed.machine_hash == header.machine_hash, "64-byte machine hash was truncated");
    Expect(parsed.footer_offset == header.footer_offset, "64-bit file offset did not round trip");
}

void TestRecordRoundTrip() {
    const auto compact = Record32(7, 99);
    TraceRecord32 compact_parsed;
    const auto compact_bytes = SerializeRecord(compact);
    Expect(compact_bytes.size() == 32 && ParseRecord(compact_bytes, compact_parsed),
           "compact record parse failed");
    Expect(compact_parsed.operation_id == compact.operation_id &&
               compact_parsed.auxiliary == compact.auxiliary,
           "compact record fields changed");
    const auto forensic = Record64(8, 0x100000001ULL);
    TraceRecord64 forensic_parsed;
    const auto forensic_bytes = SerializeRecord(forensic);
    Expect(forensic_bytes.size() == 64 && ParseRecord(forensic_bytes, forensic_parsed),
           "forensic record parse failed");
    Expect(forensic_parsed.operation_id == forensic.operation_id &&
               forensic_parsed.auxiliary_1 == forensic.auxiliary_1,
           "forensic record fields changed");
}

void TestMetadataDeterminism() {
    const std::vector<MetadataEntry> entries{{MetadataKind::Subject, 9, "tensor", "resource"},
                                             {MetadataKind::Producer, 1, "worker", "host"}};
    const auto a = SerializeMetadata(entries);
    const auto b = SerializeMetadata(entries);
    Expect(a == b, "metadata encoding is not deterministic");
    std::vector<MetadataEntry> parsed;
    Expect(ParseMetadata(a, parsed) && parsed.size() == 2, "metadata parse failed");
    Expect(parsed[0].kind == MetadataKind::Producer, "metadata canonical ordering failed");
}

void TestWriterReaderAndOrdering() {
    Cleanup cleanup;
    const auto path = MakeValidTrace(cleanup);
    const auto trace = TraceReader::Read(path, true);
    Expect(trace.status == TraceValidationStatus::Valid, "valid trace did not validate");
    Expect(trace.valid_event_records == 5 && trace.records32.size() == 5,
           "reader event count mismatch");
    Expect(trace.footer && trace.footer->records_written == 5, "footer summary mismatch");
    std::uint32_t last0 = 0, last1 = 0;
    for (const auto& record : trace.records32) {
        auto& last = record.producer_id == 0 ? last0 : last1;
        Expect(record.operation_id > last, "per-producer ordering was not preserved");
        last = record.operation_id;
    }
    Expect(TraceSummaryToJson(trace) == TraceSummaryToJson(trace),
           "trace summary JSON is not stable");
}

void TestCorruptionAndRecovery() {
    Cleanup cleanup;
    const auto valid = MakeValidTrace(cleanup);
    const auto original = ReadBytes(valid);
    const auto mutate = [&](std::string_view suffix, auto fn) {
        auto bytes = original;
        fn(bytes);
        const auto path = TempPath(suffix);
        cleanup.paths.push_back(path);
        WriteBytes(path, bytes);
        return TraceReader::Read(path, false);
    };
    auto truncated_header = mutate("-short.sidecartrace", [](auto& bytes) { bytes.resize(100); });
    Expect(truncated_header.status == TraceValidationStatus::Truncated,
           "truncated header was not detected");
    auto bad_magic = mutate("-magic.sidecartrace", [](auto& bytes) { bytes[0] ^= 0xFF; });
    Expect(bad_magic.status == TraceValidationStatus::Corrupt, "corrupt magic was not rejected");
    auto newer = mutate("-version.sidecartrace", [](auto& bytes) {
        bytes[16] = 2; bytes[17] = 0; RefreshHeaderCrc(bytes);
    });
    Expect(newer.status == TraceValidationStatus::UnsupportedVersion,
           "unsupported version was not distinguished");
    auto bad_record_size = mutate("-record.sidecartrace", [](auto& bytes) {
        bytes[26] = 31; bytes[27] = 0; RefreshHeaderCrc(bytes);
    });
    Expect(bad_record_size.status == TraceValidationStatus::Corrupt,
           "invalid record size was not rejected");
    auto checksum = mutate("-crc.sidecartrace", [](auto& bytes) { bytes[512 + 96 + 8] ^= 1; });
    Expect(checksum.status == TraceValidationStatus::Corrupt,
           "checksum mismatch was not detected");
    auto chunk_version = mutate("-chunk-version.sidecartrace", [](auto& bytes) {
        bytes[512 + 10] = 2;
    });
    Expect(chunk_version.status == TraceValidationStatus::Corrupt,
           "unsupported chunk version was not rejected");
    auto bad_length = mutate("-chunk-length.sidecartrace", [](auto& bytes) {
        Put64(bytes, 512 + 32, static_cast<std::uint64_t>(bytes.size()));
    });
    Expect(bad_length.status == TraceValidationStatus::Truncated,
           "invalid first chunk length was not bounded by file size");

    const auto parsed = TraceReader::Read(valid, false);
    Expect(parsed.header.footer_offset > 0, "test trace has no footer offset");
    auto missing_footer = mutate("-no-footer.sidecartrace", [&](auto& bytes) {
        bytes.resize(static_cast<std::size_t>(parsed.header.footer_offset));
    });
    Expect(missing_footer.status == TraceValidationStatus::IncompleteButRecoverable &&
               missing_footer.valid_event_records == 5,
           "missing footer did not recover completed event chunks");
    auto partial_tail = mutate("-tail.sidecartrace", [](auto& bytes) { bytes.resize(bytes.size() - 20); });
    Expect(partial_tail.status == TraceValidationStatus::IncompleteButRecoverable &&
               partial_tail.valid_event_records == 5,
           "partial tail did not recover completed chunks");
}

template <typename Record>
void RunMillionRecordStress(const std::filesystem::path& path) {
    constexpr std::uint32_t kProducers = 4;
    constexpr std::uint64_t kPerProducer = 250000;
    FlightRecorder<Record, 262144> recorder;
    using Handle = typename FlightRecorder<Record, 262144>::ProducerHandle;
    std::vector<Handle> handles;
    for (std::uint32_t producer = 0; producer < kProducers; ++producer)
        handles.push_back(recorder.RegisterProducer("stress-" + std::to_string(producer)));
    auto header = Header(std::is_same_v<Record, TraceRecord32>
                             ? RecordFormat::Compact32 : RecordFormat::Forensic64);
    recorder.Start(path, header);
    std::vector<std::thread> threads;
    for (std::uint32_t producer = 0; producer < kProducers; ++producer) {
        threads.emplace_back([&, producer] {
            for (std::uint64_t index = 1; index <= kPerProducer; ++index) {
                if constexpr (std::is_same_v<Record, TraceRecord32>)
                    (void)handles[producer].tryPush(Record32(static_cast<std::uint16_t>(producer),
                                                             static_cast<std::uint32_t>(index)));
                else
                    (void)handles[producer].tryPush(Record64(producer, index));
            }
        });
    }
    for (auto& thread : threads) thread.join();
    const auto metrics = recorder.Stop();
    Expect(metrics.records_written + metrics.records_dropped == kProducers * kPerProducer,
           "million-record stress accounting mismatch");
    const auto trace = TraceReader::Read(path, false);
    Expect(trace.status == TraceValidationStatus::Valid &&
               trace.valid_event_records == metrics.records_written,
           "million-record writer/reader reconciliation failed");
}

void TestMillionRecordStress32And64() {
    Cleanup cleanup;
    const auto path32 = TempPath("-million32.sidecartrace");
    const auto path64 = TempPath("-million64.sidecartrace");
    cleanup.paths = {path32, path64};
    RunMillionRecordStress<TraceRecord32>(path32);
    RunMillionRecordStress<TraceRecord64>(path64);
}

void TestInterruptedWriterRecovery() {
    Cleanup cleanup;
    const auto path = TempPath("-interrupted.sidecartrace");
    cleanup.paths.push_back(path);
    {
        TraceWriter writer(path, Header(), {{MetadataKind::Producer, 0, "p0", "test"}});
        const std::vector<TraceRecord32> records{Record32(0, 1), Record32(0, 2)};
        writer.Append(0, 0, records);
        writer.Flush();
    }
    const auto trace = TraceReader::Read(path, false);
    Expect(trace.status == TraceValidationStatus::IncompleteButRecoverable &&
               trace.valid_event_records == 2,
           "interrupted writer trace was not recoverable");
}

void TestMultiRingAndDropAccounting() {
    Cleanup cleanup;
    const auto path = TempPath("-multiring.sidecartrace");
    cleanup.paths.push_back(path);
    FlightRecorder<TraceRecord32, 4> recorder({2, 1, 1, std::chrono::microseconds(1), 1});
    auto p0 = recorder.RegisterProducer("p0");
    auto p1 = recorder.RegisterProducer("p1");
    for (std::uint32_t index = 1; index <= 10; ++index) {
        (void)p0.tryPush(Record32(0, index));
        (void)p1.tryPush(Record32(1, index));
    }
    recorder.Start(path, Header());
    const auto metrics = recorder.Stop();
    Expect(metrics.records_written + metrics.records_dropped == 20,
           "produced != persisted + dropped");
    Expect(metrics.records_written == 8 && metrics.records_dropped == 12,
           "intentional overflow accounting mismatch");
    Expect(metrics.high_water_mark == 4, "high-water tracking mismatch");
    const auto trace = TraceReader::Read(path, true);
    Expect(trace.status == TraceValidationStatus::Valid && trace.valid_event_records == 8,
           "multi-ring trace did not finalize cleanly");
}

void TestOperationIdsAndCalculations() {
    OperationIdAllocator allocator;
    Expect(allocator.AllocateCompact() == 1 && allocator.Allocate() == 2,
           "operation IDs were reused or did not start at one");
    Expect(ObserverOverheadRaw(100, 105) == 0.05, "observer overhead formula changed");
    Expect(ObserverOverheadRaw(100, 95) == -0.05, "negative measurement noise was clamped");
    const auto distribution = CalculateDistribution({1, 2, 3, 4, 100});
    Expect(distribution.count == 5 && distribution.median == 3 &&
               distribution.p95 == 100 && distribution.maximum == 100 &&
               !distribution.p999,
           "benchmark distribution calculation failed");
}

void TestMigration3AndBenchmarkPersistence() {
    Cleanup cleanup;
    const auto path = TempPath(".db");
    cleanup.paths.push_back(path);
    auto database = sidecar::database::Database::Open(
        path, sidecar::database::OpenMode::CreateOrOpen);
    database.Initialize();
    const auto status = database.Status();
    Expect(status.schema_version == sidecar::database::kCurrentSchemaVersion &&
               status.latest_migration == sidecar::database::kCurrentSchemaVersion,
           "database did not retain trace tables through current migration");
    database.UpsertHardwareProfile({"trace-machine", "host", "Windows", "11", "1", "CPU",
                                    1, 1, 1, 1024, 512, "{}"});
    const auto session = database.StartBenchmarkSession(
        {"trace-machine", "M0-FROZEN-1", "commit", std::nullopt, "LIGHT_SPSC", std::nullopt});
    const auto configuration = database.InsertTraceConfiguration(
        {session, 32, 65536, 1, 4096, "spin_yield_sleep", "steady_clock_ns"});
    Expect(database.InsertTraceBenchmark(
        {session, configuration, "LIGHT_COMPUTE", 0, 100, 101, 10, 10, 0, 1000,
         90, 10, 0.01, 2, 1000, 10000.0, 10.0, 20.0, 30.0, "[10,20,30]"}) > 0,
        "trace benchmark row was not persisted");
    Expect(database.InsertTraceFile(
        {session, "trace.sidecartrace", 1, 32, "LIGHT_SPSC", 1000, 10, 0, 2, true,
         std::nullopt}) > 0, "trace file row was not persisted");
    database.CompleteBenchmarkSession(session, "COMPLETE");
}

struct Test { std::string_view name; std::function<void()> run; };

}  // namespace

int main() {
    const std::vector<Test> tests{{"header_round_trip", TestHeaderRoundTrip},
                                  {"record_round_trip", TestRecordRoundTrip},
                                  {"metadata_determinism", TestMetadataDeterminism},
                                  {"writer_reader_ordering", TestWriterReaderAndOrdering},
                                  {"corruption_recovery", TestCorruptionAndRecovery},
                                  {"interrupted_writer", TestInterruptedWriterRecovery},
                                  {"multi_ring_drop_accounting", TestMultiRingAndDropAccounting},
                                  {"million_record_stress_32_64", TestMillionRecordStress32And64},
                                  {"operation_ids_calculations", TestOperationIdsAndCalculations},
                                  {"migration3_persistence", TestMigration3AndBenchmarkPersistence}};
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
