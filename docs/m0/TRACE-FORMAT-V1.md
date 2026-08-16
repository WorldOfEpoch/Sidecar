# Sidecar Trace Format V1

Status: M0 Work Unit 3. The conventional extension is `.sidecartrace`.

## Encoding and compatibility

Every integer is an unsigned fixed-width little-endian value. Text fields are UTF-8 byte
fields; fixed header text is zero padded when shorter than its field. Readers must use the
declared offsets and sizes and must not map these bytes onto native C++ structures. File
offsets and counters are 64-bit. V1 is independent of the Windows C++ ABI.

## File header (512 bytes)

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 16 | exact magic `SIDECAR_TRACE_V1` |
| 16 | 2 | format version (`1`) |
| 18 | 2 | header size (`512`) |
| 20 | 4 | endian marker (`0x01020304`) |
| 24 | 2 | record format: `1` compact, `2` forensic |
| 26 | 2 | record size: `32` or `64` |
| 28 | 2 | machine identity version |
| 30 | 2 | reserved |
| 32 | 4 | flags |
| 36 | 4 | producer count |
| 40 | 8 | UTC creation time, Unix epoch nanoseconds |
| 48 | 8 | monotonic start reference, nanoseconds |
| 56 | 8 | metadata chunk offset |
| 64 | 8 | first event-data offset |
| 72 | 8 | footer offset, zero until finalized |
| 80 | 32 | Sidecar version |
| 112 | 32 | Sidecar specification version |
| 144 | 64 | git revision |
| 208 | 64 | Machine Identity V1 SHA-256 |
| 272 | 64 | session/benchmark ID |
| 336 | 32 | trace mode |
| 368 | 32 | timestamp method |
| 400 | 4 | CRC32C of header bytes 0 through 399 |
| 404 | 108 | reserved, zero in V1 |

The reader rejects bad magic, unsupported versions, wrong endian markers, bad header CRC,
record-format/size mismatches, and impossible ordered offsets.

## Chunk header (96 bytes)

Every chunk begins with the exact eight-byte magic `SCCHUNK1`.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | chunk magic |
| 8 | 2 | type: metadata `1`, events `2`, footer `3` |
| 10 | 2 | chunk version (`1`) |
| 12 | 2 | chunk header size (`96`) |
| 14 | 2 | record format |
| 16 | 4 | producer ID; zero for non-event chunks |
| 20 | 4 | record count |
| 24 | 4 | flags |
| 28 | 4 | CRC32C of stored payload bytes |
| 32 | 8 | stored payload bytes |
| 40 | 8 | uncompressed bytes; equal to stored bytes in V1 |
| 48 | 8 | first host timestamp |
| 56 | 8 | last host timestamp |
| 64 | 8 | first producer sequence |
| 72 | 8 | last producer sequence |
| 80 | 16 | reserved, zero in V1 |

V1 event payloads are raw explicitly encoded records and are not compressed. CRC32C uses
the reflected Castagnoli polynomial. Checksumming happens in the collector/writer, never in
the producer.

## Metadata dictionary

The metadata payload starts with a 32-bit entry count and four reserved zero bytes. Each
entry contains kind (16-bit), name length (16-bit), type length (16-bit), two reserved bytes,
ID (32-bit), then the name and type UTF-8 bytes. Entries are serialized deterministically by
kind, ID, name, and type. V1 implements producer entries and reserves kinds for subjects,
operations, event descriptions, and memory tiers. Registration happens before tracing.

## Event records

Compact record format `1` is exactly 32 bytes:

`timestamp:u64, operation:u32, parent:u32, subject:u32, payload:u32, event:u16,
source-tier:u8, destination-tier:u8, producer:u16, auxiliary:u16`.

Compact operation IDs are nonzero, session-local 32-bit values. Sidecar's allocator is
64-bit and never reuses an ID in a session; a compact producer must reject or switch format
before truncating an ID that exceeds `UINT32_MAX`.

Forensic record format `2` is exactly 64 bytes:

`timestamp:u64, operation:u64, parent:u64, subject:u64, payload:u64, producer:u32,
event:u16, source-tier:u8, destination-tier:u8, auxiliary-0:u64, auxiliary-1:u64`.

Per-producer sequence and record order are canonical. Drain order across rings is not a
global event order. Readers may offer a best-effort merged view sorted by host timestamp,
then producer and operation ID, but must label that reconstruction.

## Footer and recovery

The 64-byte footer payload contains eight 64-bit values: written records, dropped records,
maximum ring high-water mark, chunks before the footer, flush count, final file bytes,
collector-thread CPU nanoseconds, and monotonic end time.

Chunks are append-oriented. A clean file has a checksummed footer and header footer offset.
A crash can leave a missing footer or partial final chunk. The reader retains every earlier
complete checksummed chunk and reports `INCOMPLETE_BUT_RECOVERABLE`. Other statuses are
`VALID`, `TRUNCATED`, `CORRUPT`, and `UNSUPPORTED_VERSION`. It never treats a corrupt CRC as
valid data or reads beyond the physical file.
