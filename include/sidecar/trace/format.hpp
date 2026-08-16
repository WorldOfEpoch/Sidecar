#pragma once

#include "sidecar/trace/record.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sidecar::trace {

inline constexpr std::uint16_t kTraceFormatVersion = 1;
inline constexpr std::uint32_t kTraceHeaderSize = 512;
inline constexpr std::uint16_t kTraceChunkHeaderSize = 96;
inline constexpr std::uint32_t kTraceEndianMarker = 0x01020304U;
inline constexpr std::array<std::uint8_t, 16> kTraceMagic{
    'S', 'I', 'D', 'E', 'C', 'A', 'R', '_', 'T', 'R', 'A', 'C', 'E', '_', 'V', '1'};
inline constexpr std::array<std::uint8_t, 8> kChunkMagic{'S', 'C', 'C', 'H', 'U', 'N', 'K', '1'};

enum class RecordFormat : std::uint16_t { Compact32 = 1, Forensic64 = 2 };
enum class ChunkType : std::uint16_t { Metadata = 1, Events = 2, Footer = 3 };
enum class MetadataKind : std::uint16_t {
    Producer = 1,
    Subject = 2,
    Operation = 3,
    EventType = 4,
    MemoryTier = 5,
};
enum class TraceValidationStatus {
    Valid,
    Truncated,
    Corrupt,
    UnsupportedVersion,
    IncompleteButRecoverable,
};

struct TraceHeader {
    std::string sidecar_version;
    std::string spec_version;
    std::string git_commit;
    std::string machine_hash;
    std::string session_id;
    std::string trace_mode;
    std::string timestamp_method{"steady_clock_ns"};
    std::uint16_t machine_identity_version{1};
    RecordFormat record_format{RecordFormat::Compact32};
    std::uint16_t record_size{32};
    std::uint64_t wall_clock_created_ns{0};
    std::uint64_t monotonic_start_ns{0};
    std::uint32_t producer_count{0};
    std::uint32_t flags{0};
    std::uint64_t metadata_offset{kTraceHeaderSize};
    std::uint64_t event_data_offset{0};
    std::uint64_t footer_offset{0};
};

struct MetadataEntry {
    MetadataKind kind{MetadataKind::Producer};
    std::uint32_t id{0};
    std::string name;
    std::string type;
};

struct ChunkInfo {
    ChunkType type{ChunkType::Events};
    std::uint16_t version{1};
    std::uint32_t producer_id{0};
    RecordFormat record_format{RecordFormat::Compact32};
    std::uint32_t record_count{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t first_timestamp_ns{0};
    std::uint64_t last_timestamp_ns{0};
    std::uint64_t first_sequence{0};
    std::uint64_t last_sequence{0};
    std::uint32_t flags{0};
    std::uint32_t crc32c{0};
    std::uint64_t file_offset{0};
    bool checksum_valid{false};
};

struct TraceFooter {
    std::uint64_t records_written{0};
    std::uint64_t records_dropped{0};
    std::uint64_t high_water_mark{0};
    std::uint64_t chunks_written{0};
    std::uint64_t flush_count{0};
    std::uint64_t bytes_written{0};
    std::uint64_t collector_cpu_ns{0};
    std::uint64_t monotonic_end_ns{0};
};

struct TraceReadResult {
    TraceValidationStatus status{TraceValidationStatus::Corrupt};
    std::string message;
    TraceHeader header;
    std::vector<MetadataEntry> metadata;
    std::vector<ChunkInfo> chunks;
    std::optional<TraceFooter> footer;
    std::vector<TraceRecord32> records32;
    std::vector<TraceRecord64> records64;
    std::uint64_t valid_event_records{0};
    std::uint64_t valid_bytes{0};
    bool header_valid{false};
    bool footer_valid{false};
};

[[nodiscard]] std::uint32_t Crc32c(std::span<const std::uint8_t> bytes) noexcept;
[[nodiscard]] std::vector<std::uint8_t> SerializeHeader(const TraceHeader& header);
[[nodiscard]] bool ParseHeader(std::span<const std::uint8_t> bytes,
                               TraceHeader& header,
                               TraceValidationStatus& status,
                               std::string& error);
[[nodiscard]] std::vector<std::uint8_t> SerializeMetadata(
    const std::vector<MetadataEntry>& entries);
[[nodiscard]] bool ParseMetadata(std::span<const std::uint8_t> bytes,
                                 std::vector<MetadataEntry>& entries);
[[nodiscard]] std::vector<std::uint8_t> SerializeRecord(const TraceRecord32& record);
[[nodiscard]] std::vector<std::uint8_t> SerializeRecord(const TraceRecord64& record);
[[nodiscard]] bool ParseRecord(std::span<const std::uint8_t> bytes, TraceRecord32& record);
[[nodiscard]] bool ParseRecord(std::span<const std::uint8_t> bytes, TraceRecord64& record);
[[nodiscard]] const char* ToString(TraceValidationStatus status) noexcept;
[[nodiscard]] const char* ToString(RecordFormat format) noexcept;

class TraceWriter final {
public:
    class Impl;
    TraceWriter(const std::filesystem::path& path,
                TraceHeader header,
                std::vector<MetadataEntry> metadata);
    ~TraceWriter();
    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;

    void Append(std::uint32_t producer_id,
                std::uint64_t first_sequence,
                std::span<const TraceRecord32> records);
    void Append(std::uint32_t producer_id,
                std::uint64_t first_sequence,
                std::span<const TraceRecord64> records);
    void Finalize(const TraceFooter& footer);
    void Flush();
    [[nodiscard]] std::uint64_t bytesWritten() const noexcept;
    [[nodiscard]] std::uint64_t chunksWritten() const noexcept;
    [[nodiscard]] const TraceHeader& header() const noexcept;

private:
    Impl* impl_{nullptr};
};

class TraceReader final {
public:
    [[nodiscard]] static TraceReadResult Read(const std::filesystem::path& path,
                                              bool load_records = true);
};

[[nodiscard]] std::string TraceSummaryToJson(const TraceReadResult& trace);
[[nodiscard]] std::string FormatTraceSummary(const TraceReadResult& trace);

}  // namespace sidecar::trace
