# M0 Work Unit 7: NVMe Storage Physics

## Scope and architecture

WU7 measures one path: a resolved local NVMe device into host memory. It does
not measure GPU transfer or contention, scheduling, prefetching, or Guardian
policy.

The authoritative Windows backend is overlapped `ReadFile` using an I/O
Completion Port (IOCP) on a file opened with
`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`. Buffered and memory-mapped
reads are controls labeled `OS_CACHE_INFLUENCED`. `IStorageProvider` separates
resolution, dataset management, health collection, and execution from the
platform-neutral planner and metrics. Synthetic providers let tests run
without the development NVMe or CUDA.

| Backend | Mechanism | Classification |
|---|---|---|
| `WINDOWS_OVERLAPPED_UNBUFFERED` | `ReadFile` + IOCP + no buffering | authoritative |
| `WINDOWS_BUFFERED` | overlapped `ReadFile` + IOCP | cache-influenced control |
| `WINDOWS_MEMORY_MAPPED` | read-only mapping and bounded copy | first/warm cache control |
| `DIRECTSTORAGE` | optional provider slot | `SKIPPED_UNSUPPORTED` in this build |

The IOCP queue depth is actual simultaneous in-flight IO. Timeout uses the
completion-port wait; cancellation uses `CancelIoEx` and drains completions
before releasing handles or buffers. Partial reads and Windows errors are
retained.

## Target resolution

`ResolveTarget` consumes the existing Windows hardware inventory. A target must
report NVMe, have a physical-disk number, and map to a mounted volume.
`--device` matches model or persistent ID. With no selector and multiple NVMe
devices, an unambiguous 990 PRO model match is preferred on this development
host. The physical-disk number, volume GUID, and mount are still discovered at
runtime; no drive letter or disk number is hardcoded. USB storage is ineligible.

An explicit `--dataset` must be below the selected mount. Plans show persistent
ID, physical disk, volume, mount, device-location evidence, mapping confidence,
and alignment before IO.

## Dataset and verification

The default is a normal, non-sparse 8 GiB file:

`<resolved-volume>/SidecarData/sidecar-storage-dataset-v1-8gib.bin`

It is sequentially written and flushed once. Existing content is never
silently overwritten. Generator `SIDECAR-SPLITMIX64-BYTE-V1` with seed
`0x5349444543415237` makes every byte a pure function of its absolute offset.
Dataset identity is SHA-256 over a canonical manifest containing format,
generator, seed, and size. This identifies the recipe; sampled or full byte
verification establishes integrity. `dataset verify --full` checks every byte.
All verification is outside timing.

## Alignment and safety

Logical and physical sector sizes come from
`IOCTL_STORAGE_QUERY_PROPERTY(StorageAccessAlignmentProperty)`, with
`GetDiskFreeSpaceW` as a fallback. Plans expose logical/physical sector,
file-offset, length, and destination-address alignment. Unknown or violated
unbuffered alignment is rejected.

The outstanding-byte cap is the smaller of 1 GiB and one eighth of available
physical RAM, with a 128 MiB planning floor. Overflow or over-cap combinations
are `SKIPPED_SAFETY_LIMIT`.

Primary block sizes are 64 KiB, 256 KiB, 1, 2, 4, 8, 16, 32, 64, and 128 MiB.
Primary queue depths are 1, 2, 4, 8, 16, 32, 64, and 128. Sequential and
deterministic full-dataset pseudorandom patterns form the primary matrix. A
limited windowed/strided cross-check uses 1 and 16 MiB at QD 4 and 16.

The explicit 256 MiB submission shapes are 2×128, 4×64, 8×32, and 16×16 MiB.
Normal coarse points target about 512 MiB of reads. P99.9 is null when fewer
than 1,000 samples exist; Sidecar does not extrapolate it.

## Destination memory

The primary destination is reusable committed, pre-touched pageable memory.
Address alignment is verified after allocation. Limited `cudaHostAlloc`
cross-checks use 1, 4, 16, 64, and 128 MiB with QD 1, 4, and 16, subject to the
same cap. CUDA-disabled builds retain pageable storage and return
`SKIPPED_UNSUPPORTED` for pinned memory. Final buffer content is verified after
the timing window.

## Ordering, timing, and CPU accounting

Sequential requests wrap at the dataset boundary. Pseudorandom and windowed
ordering use a deterministic coprime-stride permutation. All offsets are
aligned and bounded.

Each request records request/batch index, offset, requested/completed bytes,
submission reference time, submission-call cost, submission-to-completion
latency, completion-processing time, status, and native error. A window also
records wall time, aggregate bytes/s, IOPS, process user/kernel CPU, and
completion-thread user/kernel CPU. CPU seconds/GiB and completion CPU
ns/request are derived.

Samples are held in pre-reserved RAM. SQLite, JSON, console/file logging,
Flight Recorder, and verification do not occur in timed windows. Distributions
include count, mean, min, max, standard deviation, P50, P90, P95, P99, and
P99.9 when sample count is adequate.

## Deadline analysis

Every measured configuration is evaluated at 0.5, 1, 2, 4, 8, 16, 32, 64,
and 128 ms. Sidecar persists hits, misses, success rate, and
deadline-compliant bandwidth:

`sum(bytes from successful requests meeting deadline) / window wall time`

This differs from average throughput. These rows support later 95%, 99%, and
99.9% safe envelopes; WU7 creates no scheduler policy.

## Health and stop policy

The provider issues a read-only Windows NVMe protocol SMART/health log query
against the resolved physical disk. Unsupported values remain SQL `NULL` and
JSON `null`. It can obtain critical warnings, composite temperature, available
spare/threshold, percentage used, data units read/written, host commands,
controller busy minutes, power cycles/hours, unsafe shutdowns, media/data
errors, and error-log count. Raw evidence identifies the IOCTL, log page,
physical disk, and response lengths.

Health is sampled before/after a run and at low rate between matrix groups.
Measurement stops for nonzero critical warnings or temperature at the
configured cap (75 °C default). A media/data-error increase fails the session.
These are objective future Guardian inputs; WU7 implements no score or warning.

## Persistence

Migration 007 advances `PRAGMA user_version` to 7 without recreating the
database. It adds `storage_datasets`, `storage_health_snapshots`,
`storage_benchmark_configurations`, `storage_benchmarks`, `storage_samples`,
`storage_deadline_profiles`, and `storage_backend_profiles`. Dataset upsert is
idempotent. Request/deadline data is inserted in one transaction after
measurement. WU1–WU6 rows and newer-schema rejection are preserved; foreign
keys remain enabled.

## CLI

    sidecar-lab storage info|health [--json]
    sidecar-lab storage dataset create|verify [--full] [--json]
    sidecar-lab storage plan|run|matrix|deadline [filters] [--json]
    sidecar-lab storage report [--database path] [--json]
    sidecar-lab storage validate [--json]

Filters cover backend, pattern, block, QD, destination, request count, device,
dataset, timeout, temperature cap, database, and a descriptive `--phase` used
to retain refinement-selection provenance.

## Limitations and interpretation

`FILE_FLAG_NO_BUFFERING` requests filesystem cache bypass; it does not disable
controller DRAM, firmware/SLC behavior, or platform caching below Windows.
Buffered/mapped controls can be dominated by the standby list. Mapped “first”
means first in Sidecar's ordered control pair, not guaranteed cold media.
Discovered topology does not prove CPU-direct versus chipset routing.

WU6 measured RAM-to-VRAM behavior under compute. WU7 measures the separate
NVMe-to-RAM distribution. Adding their times can provide conservative lead-time
evidence, but cannot prove overlap or composition. WU8 must measure the actual
NVMe → RAM → pinned → VRAM plus compute pipeline.

## Authoritative results

The broad quiet matrix is session 42 from clean implementation revision
`5407f591f1e0222b8c367f9410711c3ec7fcba94`. Phase-B sessions 43–60 use clean
refinement-provenance revision
`a120838927b82caf15242a55b043e40372476dd2`. Revision
`57cd50d67dcf9169b523c1a9a98b2ad5b05945d2` makes dataset verification evidence
monotonic; it changes persistence only, not timing.

The target was runtime-resolved as Samsung SSD 990 PRO 2TB, firmware
`4B2QJXD7`, PhysicalDrive 0, volume
the resolved `C:` volume (unique volume GUID redacted), PCI location
`PCIROOT(0)#PCI(0100)#PCI(0000)`. Windows reported 512-byte logical and
4,096-byte physical/destination alignment. Mapping confidence is `DERIVED`
from the volume extents plus Configuration Manager parent/location data; this
does not establish CPU-direct routing.

Dataset 1 is the full-verified 8 GiB generator image at
`C:/SidecarData/sidecar-storage-dataset-v1-8gib.bin`. Its identity is
`da0f22a709d6d4f6d2424fac8335a2a65d0e9d358190b40b695a8ceba1774c97`.
The database retains `full_verification=1` and 8,589,934,592 verified bytes.

Session 42 persisted 187 configurations: 164 successful, 23 deliberate
unsupported/safety skips, zero failures, and 182,598 samples. Sessions 42–60
together contain 182 successful benchmark rows, 23 skips, zero failures,
zero successful-but-unverified rows, and 242,054 request samples. Health
snapshots 1–45 cover this authoritative set.

### Important latency populations

| Configuration | Throughput | P50 / P95 / P99 / P99.9 |
|---|---:|---:|
| 256 KiB, QD8, sequential | 7.119 GB/s | 0.294 / 0.305 / 0.335 / 0.405 ms |
| 1 MiB, QD4, sequential broad cell | 7.125 GB/s | 0.588 / 0.623 / 0.721 / insufficient |
| 1 MiB, QD4, sequential 8 GiB repeats | 5.247–6.996 GB/s | P99 0.973–1.136; P99.9 1.181–2.399 ms |
| 1 MiB, QD4, pseudorandom 8 GiB | 5.600 GB/s | 0.595 / 1.076 / 1.161 / 1.780 ms |
| 1 MiB, QD4, `cudaHostAlloc` 8 GiB repeats | 5.995–6.386 GB/s | P99 1.049–1.082; P99.9 1.128–1.940 ms |
| 1 MiB, QD1, buffered 4 GiB | 2.020 GB/s | 0.521 / 0.668 / 0.784 / 1.648 ms |
| 1 MiB, QD1, mmap warm 4 GiB | 3.394 GB/s | 0.303 / 0.345 / 0.436 / 0.512 ms |
| 32 MiB, QD8, sequential 8 GiB peak | 7.135 GB/s | 37.594 / 37.632 / 37.663 / insufficient |

The 512 MiB coarse cells characterize burst/plateau behavior. The 8 GiB cells
show sustained behavior and exposed material run-to-run variation at 1 MiB
QD4. Both are retained rather than substituting one for the other.

### Completion questions

1. **Peak unbuffered sequential throughput:** 7.135 GB/s (6.645 GiB/s) in
   sustained Phase B. The broad-matrix burst peak was 7.125 GB/s.

2. **Peak block/QD:** 32 MiB at QD8, session 55. The much lower-latency broad
   plateau point was 1 MiB QD4 at 7.125 GB/s.

3. **90%/95% saturation knee:** relative to 7.135 GB/s, no QD1 point reached
   90%; 2 MiB QD2 reached both 90% and 95%. Holding block size at 1 MiB, QD2
   reached 90% and QD4 reached 95%.

4. **Important P50/P95/P99/P99.9:** the table above is the authoritative
   summary. P99.9 is explicitly insufficient for large-block populations with
   fewer than 1,000 requests.

5. **Best throughput/tail tradeoff:** 256 KiB QD8 sequential. It delivered
   99.77% of the final peak with 0.335 ms P99 and 0.405 ms P99.9.

6. **Latency versus QD:** after saturation, latency rises approximately with
   outstanding bytes while throughput stays flat. For 1 MiB sequential,
   P99 rose from 0.721 ms at QD4 to 4.820 ms at QD32 and 18.940 ms at QD128
   without a throughput gain.

7. **Random reads:** full-dataset pseudorandom reads reached the same broad
   plateau (7.125 GB/s at 1 MiB QD4). At 64 KiB QD1 random was about 20% below
   sequential; at high QD that gap narrowed to roughly 2.5%. The long 1 MiB
   random population was 5.600 GB/s with 1.780 ms P99.9, inside the observed
   sustained sequential run-to-run range.

8. **Buffered difference:** buffered QD1 was slower, not faster: 37% lower at
   the 1 MiB coarse control and 45% lower in the long 1 MiB control; the 64 MiB
   control was 36% lower. These are OS-cache-influenced observations.

9. **mmap first versus warm:** at 1 MiB, warm improved throughput by 0.55%
   (3.133 to 3.150 GiB/s); at 64 MiB it improved 4.93% (2.733 to
   2.868 GiB/s). The long warm 1 MiB control reached 3.161 GiB/s. “First” is
   ordered first access, not guaranteed globally cold media.

10. **DirectStorage availability:** no; both planned cells are
    `SKIPPED_UNSUPPORTED`.

11. **DirectStorage comparison:** not applicable because no provider was
    available. No IOCP-relative throughput, latency, tail, or CPU claim is made.

12. **Primary IOCP CPU:** long cells consumed approximately
    0.0098–0.0215 process CPU seconds/GiB. The repeated 1 MiB QD4 populations
    correspond to about 9.5–21 microseconds of completion-thread CPU per
    request. `GetProcessTimes` has 15.625 ms quantization here, so zeros in
    short coarse cells are below timer resolution, not zero work.

13. **`cudaHostAlloc` material effect:** no stable material change was
    established. The broad paired point differed by only -0.17%; long pinned
    and pageable ranges overlapped the larger run-to-run storage variation.
    Pinned reads did verify and had comparable tails.

14. **Reliable supply times:** dedicated populations measured P99 of 12.424 ms
    for 64 MiB and 21.713 ms for 128 MiB. Repeated 2×128 MiB batches supplied
    256 MiB with 39.236–40.568 ms P99.

15. **Best 256 MiB decomposition:** 2×128 MiB is the best reliable
    throughput/tail choice (about 7.074 GB/s median across two 8 GiB passes and
    about 39.9 ms mean P99). 4×64 had the highest median throughput but a
    variable 37.7–51.7 ms P99; 8×32 had 37.7–43.0 ms; 16×16 was consistently
    worse at about 53.8 ms P99.

16. **Deadline envelopes:** with at least 100 samples, 64 MiB outstanding
    (for example 2 MiB QD32) met 16 ms at both 95% and 99%; 128 MiB outstanding
    (2 MiB QD64) met 32 ms; and the refined 256 MiB 8×32/4×64 shapes met 64 ms
    at 99%. With at least 1,000 samples for the 99.9% claim, 256 KiB QD128
    (32 MiB outstanding) met all three deadlines at 100% observed success.

17. **Best deadline-compliant bandwidth:** 7.125 GB/s at 16/32 ms for the
    95%/99% criteria, and 7.135 GB/s at 64 ms. The strongest 99.9%-sampled
    result was 7.121 GB/s, 256 KiB QD32, with 100% success at 16/32/64 ms.

18. **Severe tails behind good averages:** yes. Random 4 MiB QD128 sustained
    7.117 GB/s but P99 was 66.371 ms versus 33.738 ms P50; sequential 8 MiB
    QD128 sustained 7.113 GB/s with 134.036 ms P99. Their P99.9 is
    statistically insufficient. This is why high QD is not recommended.

19. **Temperature:** 44.85–49.85 °C, well below the 75 °C cap.

20. **Critical warnings/counter anomalies:** critical warnings remained zero.
    Spare stayed 100%, percentage used stayed 6%, power cycles stayed 275, and
    unsafe shutdowns stayed 35. Expected read/command counters increased;
    small write-command changes correspond to out-of-window SQLite activity.

21. **Media/data integrity errors:** unchanged at zero; error-log entries also
    stayed zero.

22. **Verification/IO errors:** none. Every successful benchmark verified;
    there were no partial-read, IO, timeout, cancellation, or data-mismatch
    failures.

23. **WU8 primary storage path:** Windows overlapped unbuffered `ReadFile` with
    IOCP.

24. **WU8 block/QD choices:** use 256 KiB QD8 for the low-tail continuous
    supply baseline; use 2×128 MiB for the one-shot 256 MiB baseline; retain
    8×32 and 4×64 as pipeline-granularity controls. Avoid increasing QD once
    the approximately 7.1 GB/s plateau is reached.

25. **Future urgent and normal paths:** **YES.** WU7 verified both direct
    NVMe→`cudaHostAlloc` staging and NVMe→pre-touched pageable RAM. A performance
    advantage for pinned destination alone is inconclusive, but capability is
    established.

26. **WU6+WU7 lead time:** the measured lower bounds are about 15 ms for
    64 MiB, 27 ms for 128 MiB, and 50 ms for 256 MiB after adding WU5's
    approximately 25.3 GB/s pinned H2D time to WU7 P99 storage time. Candidate
    future lead windows are therefore at least 16/32/64 ms respectively, with
    64 ms especially consistent with WU6's workload-dependent 256 MiB,
    P99≤5% result. These are experiment inputs, not scheduler policy.

27. **Exact WU8 experiment:** run randomized paired baselines and combined
    trials for (a) IOCP NVMe→pre-touched pageable→persistent pinned copy→H2D and
    (b) IOCP NVMe→persistent `cudaHostAlloc`→H2D. Test 2×128, 4×64, and 8×32 MiB
    staging, persistent nonblocking CUDA streams, and WU6's ALU, memory-bound,
    and FP16 GEMM compute at 32/64 ms. Timestamp IO submission/completion,
    pageable-to-pinned copy, H2D CUDA events, compute events, and final
    makespan in one correlated record; verify every stage outside timing.
    Compare isolated storage, copy, H2D, and compute baselines against the
    combined pipeline at P50/P95/P99/P99.9 and 16/32/64 ms success. Do not add
    scheduler policy in that experiment.

### Guardian storage signals available

For the Samsung 990 PRO Sidecar now obtains critical-warning bits, composite
temperature, available spare and threshold, percentage used, data units
read/written, host read/write commands, controller busy minutes, power cycles,
power-on hours, unsafe shutdowns, media/data errors, and error-log entry count,
plus raw Windows/NVMe query evidence. WU7 assigns no score and emits no
consumer-facing Guardian warning.
