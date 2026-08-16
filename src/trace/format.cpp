#include "sidecar/trace/format.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace sidecar::trace {
namespace {

template <typename T>
void Put(std::vector<std::uint8_t>& bytes, std::size_t offset, T value) {
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        bytes.at(offset + index) = static_cast<std::uint8_t>(value >> (index * 8));
    }
}

template <typename T>
bool Get(std::span<const std::uint8_t> bytes, std::size_t offset, T& value) {
    static_assert(std::is_unsigned_v<T>);
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T)) {
        return false;
    }
    value = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<T>(bytes[offset + index]) << (index * 8);
    }
    return true;
}

void PutString(std::vector<std::uint8_t>& bytes,
               std::size_t offset,
               std::size_t width,
               const std::string& value) {
    const auto length = std::min(width, value.size());
    std::copy_n(value.begin(), length, bytes.begin() + static_cast<std::ptrdiff_t>(offset));
}

std::string GetString(std::span<const std::uint8_t> bytes,
                      std::size_t offset,
                      std::size_t width) {
    if (offset > bytes.size() || bytes.size() - offset < width) {
        return {};
    }
    std::size_t length = 0;
    while (length < width && bytes[offset + length] != 0) {
        ++length;
    }
    return std::string(reinterpret_cast<const char*>(bytes.data() + offset), length);
}

std::string JsonEscape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::vector<std::uint8_t> SerializeChunkHeader(const ChunkInfo& chunk) {
    std::vector<std::uint8_t> bytes(kTraceChunkHeaderSize, 0);
    std::copy(kChunkMagic.begin(), kChunkMagic.end(), bytes.begin());
    Put<std::uint16_t>(bytes, 8, static_cast<std::uint16_t>(chunk.type));
    Put<std::uint16_t>(bytes, 10, chunk.version);
    Put<std::uint16_t>(bytes, 12, kTraceChunkHeaderSize);
    Put<std::uint16_t>(bytes, 14, static_cast<std::uint16_t>(chunk.record_format));
    Put<std::uint32_t>(bytes, 16, chunk.producer_id);
    Put<std::uint32_t>(bytes, 20, chunk.record_count);
    Put<std::uint32_t>(bytes, 24, chunk.flags);
    Put<std::uint32_t>(bytes, 28, chunk.crc32c);
    Put<std::uint64_t>(bytes, 32, chunk.payload_bytes);
    Put<std::uint64_t>(bytes, 40, chunk.payload_bytes);
    Put<std::uint64_t>(bytes, 48, chunk.first_timestamp_ns);
    Put<std::uint64_t>(bytes, 56, chunk.last_timestamp_ns);
    Put<std::uint64_t>(bytes, 64, chunk.first_sequence);
    Put<std::uint64_t>(bytes, 72, chunk.last_sequence);
    return bytes;
}

bool ParseChunkHeader(std::span<const std::uint8_t> bytes,
                      ChunkInfo& chunk,
                      std::string& error) {
    if (bytes.size() < kTraceChunkHeaderSize) {
        error = "truncated chunk header";
        return false;
    }
    if (!std::equal(kChunkMagic.begin(), kChunkMagic.end(), bytes.begin())) {
        error = "invalid chunk magic";
        return false;
    }
    std::uint16_t type = 0;
    std::uint16_t header_size = 0;
    std::uint16_t format = 0;
    if (!Get(bytes, 8, type) || !Get(bytes, 10, chunk.version) ||
        !Get(bytes, 12, header_size) || !Get(bytes, 14, format) ||
        !Get(bytes, 16, chunk.producer_id) || !Get(bytes, 20, chunk.record_count) ||
        !Get(bytes, 24, chunk.flags) || !Get(bytes, 28, chunk.crc32c) ||
        !Get(bytes, 32, chunk.payload_bytes) || !Get(bytes, 48, chunk.first_timestamp_ns) ||
        !Get(bytes, 56, chunk.last_timestamp_ns) || !Get(bytes, 64, chunk.first_sequence) ||
        !Get(bytes, 72, chunk.last_sequence)) {
        error = "invalid chunk header";
        return false;
    }
    if (header_size != kTraceChunkHeaderSize || chunk.version != 1 ||
        type < static_cast<std::uint16_t>(ChunkType::Metadata) ||
        type > static_cast<std::uint16_t>(ChunkType::Footer)) {
        error = "unsupported or corrupt chunk header";
        return false;
    }
    chunk.type = static_cast<ChunkType>(type);
    chunk.record_format = static_cast<RecordFormat>(format);
    return true;
}

std::vector<std::uint8_t> SerializeFooter(const TraceFooter& footer) {
    std::vector<std::uint8_t> bytes(64, 0);
    Put(bytes, 0, footer.records_written);
    Put(bytes, 8, footer.records_dropped);
    Put(bytes, 16, footer.high_water_mark);
    Put(bytes, 24, footer.chunks_written);
    Put(bytes, 32, footer.flush_count);
    Put(bytes, 40, footer.bytes_written);
    Put(bytes, 48, footer.collector_cpu_ns);
    Put(bytes, 56, footer.monotonic_end_ns);
    return bytes;
}

void EncodeRecordAt(std::vector<std::uint8_t>& bytes,
                    std::size_t offset,
                    const TraceRecord32& record) {
    Put(bytes, offset + 0, record.host_timestamp_ns);
    Put(bytes, offset + 8, record.operation_id);
    Put(bytes, offset + 12, record.parent_operation_id);
    Put(bytes, offset + 16, record.subject_id);
    Put(bytes, offset + 20, record.payload_bytes);
    Put(bytes, offset + 24, static_cast<std::uint16_t>(record.event_type));
    bytes[offset + 26] = static_cast<std::uint8_t>(record.source_tier);
    bytes[offset + 27] = static_cast<std::uint8_t>(record.destination_tier);
    Put(bytes, offset + 28, record.producer_id);
    Put(bytes, offset + 30, record.auxiliary);
}

void EncodeRecordAt(std::vector<std::uint8_t>& bytes,
                    std::size_t offset,
                    const TraceRecord64& record) {
    Put(bytes, offset + 0, record.host_timestamp_ns);
    Put(bytes, offset + 8, record.operation_id);
    Put(bytes, offset + 16, record.parent_operation_id);
    Put(bytes, offset + 24, record.subject_id);
    Put(bytes, offset + 32, record.payload_bytes);
    Put(bytes, offset + 40, record.producer_id);
    Put(bytes, offset + 44, static_cast<std::uint16_t>(record.event_type));
    bytes[offset + 46] = static_cast<std::uint8_t>(record.source_tier);
    bytes[offset + 47] = static_cast<std::uint8_t>(record.destination_tier);
    Put(bytes, offset + 48, record.auxiliary_0);
    Put(bytes, offset + 56, record.auxiliary_1);
}

bool ParseFooter(std::span<const std::uint8_t> bytes, TraceFooter& footer) {
    return bytes.size() == 64 && Get(bytes, 0, footer.records_written) &&
           Get(bytes, 8, footer.records_dropped) && Get(bytes, 16, footer.high_water_mark) &&
           Get(bytes, 24, footer.chunks_written) && Get(bytes, 32, footer.flush_count) &&
           Get(bytes, 40, footer.bytes_written) && Get(bytes, 48, footer.collector_cpu_ns) &&
           Get(bytes, 56, footer.monotonic_end_ns);
}

void WriteAll(std::fstream& stream, std::span<const std::uint8_t> bytes) {
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream) {
        throw std::runtime_error("trace write failed");
    }
}

std::uint64_t Tell(std::fstream& stream) {
    const auto position = stream.tellp();
    if (position < 0) {
        throw std::runtime_error("trace offset query failed");
    }
    return static_cast<std::uint64_t>(position);
}

}  // namespace

std::uint32_t Crc32c(std::span<const std::uint8_t> bytes) noexcept {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const auto byte : bytes) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0x82F63B78U & (0U - (crc & 1U)));
        }
    }
    return ~crc;
}

std::vector<std::uint8_t> SerializeHeader(const TraceHeader& header) {
    std::vector<std::uint8_t> bytes(kTraceHeaderSize, 0);
    std::copy(kTraceMagic.begin(), kTraceMagic.end(), bytes.begin());
    Put<std::uint16_t>(bytes, 16, kTraceFormatVersion);
    Put<std::uint16_t>(bytes, 18, static_cast<std::uint16_t>(kTraceHeaderSize));
    Put<std::uint32_t>(bytes, 20, kTraceEndianMarker);
    Put<std::uint16_t>(bytes, 24, static_cast<std::uint16_t>(header.record_format));
    Put<std::uint16_t>(bytes, 26, header.record_size);
    Put<std::uint16_t>(bytes, 28, header.machine_identity_version);
    Put<std::uint32_t>(bytes, 32, header.flags);
    Put<std::uint32_t>(bytes, 36, header.producer_count);
    Put<std::uint64_t>(bytes, 40, header.wall_clock_created_ns);
    Put<std::uint64_t>(bytes, 48, header.monotonic_start_ns);
    Put<std::uint64_t>(bytes, 56, header.metadata_offset);
    Put<std::uint64_t>(bytes, 64, header.event_data_offset);
    Put<std::uint64_t>(bytes, 72, header.footer_offset);
    PutString(bytes, 80, 32, header.sidecar_version);
    PutString(bytes, 112, 32, header.spec_version);
    PutString(bytes, 144, 64, header.git_commit);
    PutString(bytes, 208, 64, header.machine_hash);
    PutString(bytes, 272, 64, header.session_id);
    PutString(bytes, 336, 32, header.trace_mode);
    PutString(bytes, 368, 32, header.timestamp_method);
    Put<std::uint32_t>(bytes, 400, Crc32c(std::span<const std::uint8_t>(bytes).first(400)));
    return bytes;
}

bool ParseHeader(std::span<const std::uint8_t> bytes,
                 TraceHeader& header,
                 TraceValidationStatus& status,
                 std::string& error) {
    if (bytes.size() < kTraceHeaderSize) {
        status = TraceValidationStatus::Truncated;
        error = "truncated trace header";
        return false;
    }
    if (!std::equal(kTraceMagic.begin(), kTraceMagic.end(), bytes.begin())) {
        status = TraceValidationStatus::Corrupt;
        error = "invalid trace magic";
        return false;
    }
    std::uint16_t version = 0;
    std::uint16_t header_size = 0;
    std::uint32_t endian = 0;
    std::uint16_t format = 0;
    std::uint32_t header_crc = 0;
    if (!Get(bytes, 16, version) || !Get(bytes, 18, header_size) ||
        !Get(bytes, 20, endian) || !Get(bytes, 24, format) ||
        !Get(bytes, 26, header.record_size) || !Get(bytes, 28, header.machine_identity_version) ||
        !Get(bytes, 32, header.flags) || !Get(bytes, 36, header.producer_count) ||
        !Get(bytes, 40, header.wall_clock_created_ns) ||
        !Get(bytes, 48, header.monotonic_start_ns) || !Get(bytes, 56, header.metadata_offset) ||
        !Get(bytes, 64, header.event_data_offset) || !Get(bytes, 72, header.footer_offset) ||
        !Get(bytes, 400, header_crc)) {
        status = TraceValidationStatus::Corrupt;
        error = "malformed trace header";
        return false;
    }
    if (version != kTraceFormatVersion) {
        status = TraceValidationStatus::UnsupportedVersion;
        error = "unsupported trace format version";
        return false;
    }
    if (header_size != kTraceHeaderSize || endian != kTraceEndianMarker ||
        Crc32c(bytes.first(400)) != header_crc) {
        status = TraceValidationStatus::Corrupt;
        error = "trace header size, endianness, or checksum is invalid";
        return false;
    }
    if ((format == static_cast<std::uint16_t>(RecordFormat::Compact32) &&
         header.record_size != 32) ||
        (format == static_cast<std::uint16_t>(RecordFormat::Forensic64) &&
         header.record_size != 64) ||
        (format != static_cast<std::uint16_t>(RecordFormat::Compact32) &&
         format != static_cast<std::uint16_t>(RecordFormat::Forensic64))) {
        status = TraceValidationStatus::Corrupt;
        error = "impossible trace record format or size";
        return false;
    }
    if (header.metadata_offset < kTraceHeaderSize ||
        (header.event_data_offset != 0 && header.event_data_offset < header.metadata_offset) ||
        (header.footer_offset != 0 && header.event_data_offset != 0 &&
         header.footer_offset < header.event_data_offset)) {
        status = TraceValidationStatus::Corrupt;
        error = "trace header contains invalid offsets";
        return false;
    }
    header.record_format = static_cast<RecordFormat>(format);
    header.sidecar_version = GetString(bytes, 80, 32);
    header.spec_version = GetString(bytes, 112, 32);
    header.git_commit = GetString(bytes, 144, 64);
    header.machine_hash = GetString(bytes, 208, 64);
    header.session_id = GetString(bytes, 272, 64);
    header.trace_mode = GetString(bytes, 336, 32);
    header.timestamp_method = GetString(bytes, 368, 32);
    status = TraceValidationStatus::Valid;
    return true;
}

std::vector<std::uint8_t> SerializeMetadata(const std::vector<MetadataEntry>& entries) {
    auto ordered = entries;
    std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        return std::tie(left.kind, left.id, left.name, left.type) <
               std::tie(right.kind, right.id, right.name, right.type);
    });
    std::vector<std::uint8_t> bytes(8, 0);
    Put<std::uint32_t>(bytes, 0, static_cast<std::uint32_t>(ordered.size()));
    for (const auto& entry : ordered) {
        if (entry.name.size() > std::numeric_limits<std::uint16_t>::max() ||
            entry.type.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::invalid_argument("metadata string too large");
        }
        const auto offset = bytes.size();
        bytes.resize(offset + 12 + entry.name.size() + entry.type.size());
        Put<std::uint16_t>(bytes, offset, static_cast<std::uint16_t>(entry.kind));
        Put<std::uint16_t>(bytes, offset + 2, static_cast<std::uint16_t>(entry.name.size()));
        Put<std::uint16_t>(bytes, offset + 4, static_cast<std::uint16_t>(entry.type.size()));
        Put<std::uint32_t>(bytes, offset + 8, entry.id);
        std::copy(entry.name.begin(), entry.name.end(), bytes.begin() + offset + 12);
        std::copy(entry.type.begin(), entry.type.end(),
                  bytes.begin() + offset + 12 + entry.name.size());
    }
    return bytes;
}

bool ParseMetadata(std::span<const std::uint8_t> bytes, std::vector<MetadataEntry>& entries) {
    std::uint32_t count = 0;
    if (!Get(bytes, 0, count) || bytes.size() < 8) {
        return false;
    }
    std::size_t offset = 8;
    entries.clear();
    for (std::uint32_t index = 0; index < count; ++index) {
        std::uint16_t kind = 0, name_size = 0, type_size = 0;
        std::uint32_t id = 0;
        if (!Get(bytes, offset, kind) || !Get(bytes, offset + 2, name_size) ||
            !Get(bytes, offset + 4, type_size) || !Get(bytes, offset + 8, id)) {
            return false;
        }
        const std::size_t content = offset + 12;
        if (content > bytes.size() || bytes.size() - content < name_size + type_size) {
            return false;
        }
        entries.push_back({static_cast<MetadataKind>(kind), id,
                           std::string(reinterpret_cast<const char*>(bytes.data() + content),
                                       name_size),
                           std::string(reinterpret_cast<const char*>(
                                           bytes.data() + content + name_size), type_size)});
        offset = content + name_size + type_size;
    }
    return offset == bytes.size();
}

std::vector<std::uint8_t> SerializeRecord(const TraceRecord32& record) {
    std::vector<std::uint8_t> bytes(32, 0);
    EncodeRecordAt(bytes, 0, record);
    return bytes;
}

std::vector<std::uint8_t> SerializeRecord(const TraceRecord64& record) {
    std::vector<std::uint8_t> bytes(64, 0);
    EncodeRecordAt(bytes, 0, record);
    return bytes;
}

bool ParseRecord(std::span<const std::uint8_t> bytes, TraceRecord32& record) {
    std::uint16_t event = 0;
    if (bytes.size() != 32 || !Get(bytes, 0, record.host_timestamp_ns) ||
        !Get(bytes, 8, record.operation_id) || !Get(bytes, 12, record.parent_operation_id) ||
        !Get(bytes, 16, record.subject_id) || !Get(bytes, 20, record.payload_bytes) ||
        !Get(bytes, 24, event) || !Get(bytes, 28, record.producer_id) ||
        !Get(bytes, 30, record.auxiliary)) {
        return false;
    }
    record.event_type = static_cast<EventType>(event);
    record.source_tier = static_cast<MemoryTier>(bytes[26]);
    record.destination_tier = static_cast<MemoryTier>(bytes[27]);
    return true;
}

bool ParseRecord(std::span<const std::uint8_t> bytes, TraceRecord64& record) {
    std::uint16_t event = 0;
    if (bytes.size() != 64 || !Get(bytes, 0, record.host_timestamp_ns) ||
        !Get(bytes, 8, record.operation_id) || !Get(bytes, 16, record.parent_operation_id) ||
        !Get(bytes, 24, record.subject_id) || !Get(bytes, 32, record.payload_bytes) ||
        !Get(bytes, 40, record.producer_id) || !Get(bytes, 44, event) ||
        !Get(bytes, 48, record.auxiliary_0) || !Get(bytes, 56, record.auxiliary_1)) {
        return false;
    }
    record.event_type = static_cast<EventType>(event);
    record.source_tier = static_cast<MemoryTier>(bytes[46]);
    record.destination_tier = static_cast<MemoryTier>(bytes[47]);
    return true;
}

class TraceWriter::Impl final {
public:
    Impl(const std::filesystem::path& path, TraceHeader value, std::vector<MetadataEntry> metadata)
        : header(std::move(value)) {
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        stream.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot create trace file: " + path.string());
        }
        header.metadata_offset = kTraceHeaderSize;
        WriteAll(stream, SerializeHeader(header));
        const auto payload = SerializeMetadata(metadata);
        WriteChunk(ChunkType::Metadata, 0, 0, {}, payload);
        header.event_data_offset = Tell(stream);
        RewriteHeader();
    }

    ~Impl() {
        if (stream.is_open()) {
            stream.flush();
        }
    }

    void WriteChunk(ChunkType type,
                    std::uint32_t producer,
                    std::uint64_t first_sequence,
                    std::span<const std::uint64_t> timestamps,
                    std::span<const std::uint8_t> payload) {
        if (payload.size() > std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("trace chunk is too large");
        }
        ChunkInfo chunk;
        chunk.type = type;
        chunk.producer_id = producer;
        chunk.record_format = header.record_format;
        chunk.record_count = type == ChunkType::Events
                                 ? static_cast<std::uint32_t>(timestamps.size()) : 0;
        chunk.payload_bytes = payload.size();
        chunk.first_sequence = first_sequence;
        chunk.last_sequence = timestamps.empty() ? first_sequence
                                                  : first_sequence + timestamps.size() - 1;
        chunk.first_timestamp_ns = timestamps.empty() ? 0 : timestamps.front();
        chunk.last_timestamp_ns = timestamps.empty() ? 0 : timestamps.back();
        chunk.crc32c = Crc32c(payload);
        WriteAll(stream, SerializeChunkHeader(chunk));
        WriteAll(stream, payload);
        ++chunks;
        bytes = Tell(stream);
    }

    void RewriteHeader() {
        const auto end = stream.tellp();
        stream.seekp(0);
        WriteAll(stream, SerializeHeader(header));
        stream.seekp(end);
        if (!stream) {
            throw std::runtime_error("trace header rewrite failed");
        }
    }

    std::fstream stream;
    TraceHeader header;
    std::uint64_t chunks{0};
    std::uint64_t bytes{0};
    bool finalized{false};
    std::vector<std::uint8_t> event_payload;
    std::vector<std::uint64_t> event_timestamps;
};

TraceWriter::TraceWriter(const std::filesystem::path& path,
                         TraceHeader header,
                         std::vector<MetadataEntry> metadata)
    : impl_(new Impl(path, std::move(header), std::move(metadata))) {}

TraceWriter::~TraceWriter() { delete impl_; }

template <typename Record>
void AppendRecords(TraceWriter::Impl& impl,
                   std::uint32_t producer_id,
                   std::uint64_t first_sequence,
                   std::span<const Record> records) {
    if (records.empty()) return;
    if (records.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("too many records in trace chunk");
    }
    impl.event_payload.resize(records.size() * sizeof(Record));
    impl.event_timestamps.resize(records.size());
    for (std::size_t index = 0; index < records.size(); ++index) {
        EncodeRecordAt(impl.event_payload, index * sizeof(Record), records[index]);
        impl.event_timestamps[index] = records[index].host_timestamp_ns;
    }
    impl.WriteChunk(ChunkType::Events, producer_id, first_sequence,
                    impl.event_timestamps, impl.event_payload);
}

void TraceWriter::Append(std::uint32_t producer_id,
                         std::uint64_t first_sequence,
                         std::span<const TraceRecord32> records) {
    AppendRecords(*impl_, producer_id, first_sequence, records);
}

void TraceWriter::Append(std::uint32_t producer_id,
                         std::uint64_t first_sequence,
                         std::span<const TraceRecord64> records) {
    AppendRecords(*impl_, producer_id, first_sequence, records);
}

void TraceWriter::Finalize(const TraceFooter& footer) {
    if (impl_->finalized) return;
    impl_->header.footer_offset = Tell(impl_->stream);
    auto completed_footer = footer;
    completed_footer.bytes_written = impl_->bytes + kTraceChunkHeaderSize + 64;
    const auto payload = SerializeFooter(completed_footer);
    impl_->WriteChunk(ChunkType::Footer, 0, 0, {}, payload);
    impl_->RewriteHeader();
    Flush();
    impl_->finalized = true;
}

void TraceWriter::Flush() {
    impl_->stream.flush();
    if (!impl_->stream) throw std::runtime_error("trace flush failed");
}

std::uint64_t TraceWriter::bytesWritten() const noexcept { return impl_->bytes; }
std::uint64_t TraceWriter::chunksWritten() const noexcept { return impl_->chunks; }
const TraceHeader& TraceWriter::header() const noexcept { return impl_->header; }

TraceReadResult TraceReader::Read(const std::filesystem::path& path, bool load_records) {
    TraceReadResult result;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        result.status = TraceValidationStatus::Truncated;
        result.message = "trace file cannot be opened";
        return result;
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0) {
        result.message = "trace size query failed";
        return result;
    }
    const auto file_size = static_cast<std::uint64_t>(end);
    stream.seekg(0);
    std::vector<std::uint8_t> header_bytes(kTraceHeaderSize);
    stream.read(reinterpret_cast<char*>(header_bytes.data()),
                static_cast<std::streamsize>(header_bytes.size()));
    if (static_cast<std::size_t>(stream.gcount()) < header_bytes.size()) {
        result.status = TraceValidationStatus::Truncated;
        result.message = "truncated trace header";
        return result;
    }
    if (!ParseHeader(header_bytes, result.header, result.status, result.message)) {
        return result;
    }
    result.header_valid = true;
    if (result.header.metadata_offset >= file_size) {
        result.status = TraceValidationStatus::Truncated;
        result.message = "metadata offset is outside trace file";
        return result;
    }
    std::uint64_t offset = result.header.metadata_offset;
    bool saw_completed_chunk = false;
    while (offset < file_size) {
        if (file_size - offset < kTraceChunkHeaderSize) {
            result.status = saw_completed_chunk ? TraceValidationStatus::IncompleteButRecoverable
                                                : TraceValidationStatus::Truncated;
            result.message = "incomplete trailing chunk header";
            break;
        }
        stream.seekg(static_cast<std::streamoff>(offset));
        std::vector<std::uint8_t> chunk_header(kTraceChunkHeaderSize);
        stream.read(reinterpret_cast<char*>(chunk_header.data()),
                    static_cast<std::streamsize>(chunk_header.size()));
        ChunkInfo chunk;
        chunk.file_offset = offset;
        std::string chunk_error;
        if (!ParseChunkHeader(chunk_header, chunk, chunk_error)) {
            result.status = TraceValidationStatus::Corrupt;
            result.message = chunk_error;
            break;
        }
        if (chunk.payload_bytes > file_size - offset - kTraceChunkHeaderSize ||
            chunk.payload_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            result.status = saw_completed_chunk ? TraceValidationStatus::IncompleteButRecoverable
                                                : TraceValidationStatus::Truncated;
            result.message = "incomplete trailing chunk payload";
            break;
        }
        std::vector<std::uint8_t> payload(static_cast<std::size_t>(chunk.payload_bytes));
        stream.read(reinterpret_cast<char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size()));
        if (static_cast<std::size_t>(stream.gcount()) != payload.size()) {
            result.status = TraceValidationStatus::IncompleteButRecoverable;
            result.message = "short trace chunk read";
            break;
        }
        chunk.checksum_valid = Crc32c(payload) == chunk.crc32c;
        if (!chunk.checksum_valid) {
            result.status = TraceValidationStatus::Corrupt;
            result.message = "chunk checksum mismatch";
            break;
        }
        if (chunk.type == ChunkType::Metadata) {
            if (!ParseMetadata(payload, result.metadata)) {
                result.status = TraceValidationStatus::Corrupt;
                result.message = "invalid metadata dictionary";
                break;
            }
        } else if (chunk.type == ChunkType::Events) {
            if (chunk.record_format != result.header.record_format ||
                chunk.record_count > chunk.payload_bytes / result.header.record_size ||
                static_cast<std::uint64_t>(chunk.record_count) * result.header.record_size !=
                    chunk.payload_bytes) {
                result.status = TraceValidationStatus::Corrupt;
                result.message = "invalid event chunk record layout";
                break;
            }
            result.valid_event_records += chunk.record_count;
            if (load_records) {
                for (std::uint32_t index = 0; index < chunk.record_count; ++index) {
                    const auto record_bytes = std::span<const std::uint8_t>(payload).subspan(
                        static_cast<std::size_t>(index) * result.header.record_size,
                        result.header.record_size);
                    if (result.header.record_format == RecordFormat::Compact32) {
                        TraceRecord32 record;
                        if (!ParseRecord(record_bytes, record)) {
                            result.status = TraceValidationStatus::Corrupt;
                            result.message = "invalid compact record";
                            return result;
                        }
                        result.records32.push_back(record);
                    } else {
                        TraceRecord64 record;
                        if (!ParseRecord(record_bytes, record)) {
                            result.status = TraceValidationStatus::Corrupt;
                            result.message = "invalid forensic record";
                            return result;
                        }
                        result.records64.push_back(record);
                    }
                }
            }
        } else if (chunk.type == ChunkType::Footer) {
            if (offset != result.header.footer_offset || result.footer_valid) {
                result.status = TraceValidationStatus::Corrupt;
                result.message = "footer offset or multiplicity is invalid";
                break;
            }
            TraceFooter footer;
            if (!ParseFooter(payload, footer)) {
                result.status = TraceValidationStatus::Corrupt;
                result.message = "invalid trace footer";
                break;
            }
            result.footer = footer;
            result.footer_valid = true;
        }
        result.chunks.push_back(chunk);
        offset += kTraceChunkHeaderSize + chunk.payload_bytes;
        result.valid_bytes = offset;
        saw_completed_chunk = true;
    }
    if (offset == file_size) {
        if (result.footer_valid) {
            result.status = TraceValidationStatus::Valid;
            result.message = "trace is valid";
        } else {
            result.status = TraceValidationStatus::IncompleteButRecoverable;
            result.message = "trace has valid chunks but no footer";
        }
    }
    return result;
}

const char* ToString(TraceValidationStatus status) noexcept {
    switch (status) {
        case TraceValidationStatus::Valid: return "VALID";
        case TraceValidationStatus::Truncated: return "TRUNCATED";
        case TraceValidationStatus::Corrupt: return "CORRUPT";
        case TraceValidationStatus::UnsupportedVersion: return "UNSUPPORTED_VERSION";
        case TraceValidationStatus::IncompleteButRecoverable:
            return "INCOMPLETE_BUT_RECOVERABLE";
    }
    return "CORRUPT";
}

const char* ToString(RecordFormat format) noexcept {
    return format == RecordFormat::Compact32 ? "COMPACT_32" : "FORENSIC_64";
}

std::string TraceSummaryToJson(const TraceReadResult& trace) {
    std::ostringstream out;
    out << "{\"format_version\":" << kTraceFormatVersion
        << ",\"record_format\":\"" << ToString(trace.header.record_format)
        << "\",\"record_size_bytes\":" << trace.header.record_size
        << ",\"trace_mode\":\"" << JsonEscape(trace.header.trace_mode)
        << "\",\"provenance\":{\"sidecar_version\":\""
        << JsonEscape(trace.header.sidecar_version) << "\",\"spec_version\":\""
        << JsonEscape(trace.header.spec_version) << "\",\"git_commit\":\""
        << JsonEscape(trace.header.git_commit) << "\",\"machine_hash\":\""
        << JsonEscape(trace.header.machine_hash) << "\",\"session_id\":\""
        << JsonEscape(trace.header.session_id) << "\"},\"events\":{\"producer_count\":"
        << trace.header.producer_count << ",\"records\":" << trace.valid_event_records
        << ",\"dropped\":" << (trace.footer ? trace.footer->records_dropped : 0)
        << ",\"bytes\":" << (trace.footer ? trace.footer->bytes_written : trace.valid_bytes)
        << "},\"collector\":{\"high_water\":"
        << (trace.footer ? trace.footer->high_water_mark : 0)
        << ",\"chunks\":" << trace.chunks.size() << ",\"flushes\":"
        << (trace.footer ? trace.footer->flush_count : 0) << ",\"cpu_ns\":"
        << (trace.footer ? trace.footer->collector_cpu_ns : 0)
        << "},\"integrity\":{\"status\":\"" << ToString(trace.status)
        << "\",\"header_valid\":" << (trace.header_valid ? "true" : "false")
        << ",\"footer_valid\":" << (trace.footer_valid ? "true" : "false")
        << ",\"recoverable\":"
        << (trace.status == TraceValidationStatus::Valid ||
                    trace.status == TraceValidationStatus::IncompleteButRecoverable
                ? "true" : "false")
        << "}}";
    return out.str();
}

std::string FormatTraceSummary(const TraceReadResult& trace) {
    std::ostringstream out;
    out << "SIDECAR TRACE\n\nFormat\n"
        << "  Version: " << kTraceFormatVersion << '\n'
        << "  Record: " << ToString(trace.header.record_format) << " ("
        << trace.header.record_size << " bytes)\n"
        << "  Trace mode: " << trace.header.trace_mode << "\n\nProvenance\n"
        << "  Sidecar version: " << trace.header.sidecar_version << '\n'
        << "  Spec: " << trace.header.spec_version << '\n'
        << "  Git commit: " << trace.header.git_commit << '\n'
        << "  Machine hash: " << trace.header.machine_hash << "\n\nEvents\n"
        << "  Producers: " << trace.header.producer_count << '\n'
        << "  Records: " << trace.valid_event_records << '\n'
        << "  Dropped: " << (trace.footer ? trace.footer->records_dropped : 0) << '\n'
        << "  Bytes: " << (trace.footer ? trace.footer->bytes_written : trace.valid_bytes)
        << "\n\nCollector\n"
        << "  High-water: " << (trace.footer ? trace.footer->high_water_mark : 0) << '\n'
        << "  Chunks: " << trace.chunks.size() << '\n'
        << "  Flushes: " << (trace.footer ? trace.footer->flush_count : 0) << '\n'
        << "  CPU ns: " << (trace.footer ? trace.footer->collector_cpu_ns : 0)
        << "\n\nIntegrity\n"
        << "  Status: " << ToString(trace.status) << '\n'
        << "  Header: " << (trace.header_valid ? "valid" : "invalid") << '\n'
        << "  Chunks: " << (trace.status == TraceValidationStatus::Corrupt ? "invalid" : "valid")
        << '\n' << "  Footer: " << (trace.footer_valid ? "valid" : "missing/invalid") << '\n'
        << "  Message: " << trace.message << '\n';
    return out.str();
}

}  // namespace sidecar::trace
