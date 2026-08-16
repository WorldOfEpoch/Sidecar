# CUDA Host-Device Transfer Physics

Status: M0 Work Unit 5 implementation. `M0-FROZEN-1` remains unchanged. This
laboratory measures copy physics without GPU compute and cannot establish compute
overlap, scheduler policy, model block size, or final DMA-arena capacity.

## Buffer lifetime and memory classes

Every timed configuration uses buffers created before the sample loop:

- `PAGEABLE_PRETOUCHED`: Windows `VirtualAlloc(MEM_RESERVE | MEM_COMMIT)`, with
  every byte initialized before transfer timing.
- `PINNED_HOSTALLOC`: one persistent `cudaHostAllocDefault` allocation.
- `PINNED_REGISTERED`: one pre-touched `VirtualAlloc` allocation registered once
  with `cudaHostRegisterDefault`.
- Device storage is allocated once with `cudaMalloc` and reused.

Allocation, registration, initialization, verification, telemetry, persistence,
JSON, and cleanup are outside transfer timing. Isolated tests use one host/device
buffer; bidirectional tests use two of each. Registered buffers are unregistered
before their Windows backing is released. WU4 owns the lifecycle-cost measurements.

The isolated maximum is 1 GiB, already within WU4's safely tested 4 GiB host range.
The device planner preserves the larger of 2 GiB or 20% of total VRAM. The existing
WU4 host policy preserves the dynamic RAM reserve and applies the pageable/pinned
method caps. Bidirectional planning accounts for two host and two device buffers.

## Initialization, warmup, and validation

CUDA device selection, context creation, stream/event construction, deterministic
host initialization, initial device population, and three default warmup transfers
precede measurement. CUDA errors are checked and cleared at phase boundaries.

The source pattern is a deterministic function of byte offset and buffer index.
After each configuration, 257 evenly distributed locations—including the last byte—
are checked outside timing. H2D verification performs untimed one-byte D2H reads;
D2H verification directly checks host destinations. Results remain observable and
impossibly short or corrupt samples are treated as invalid rather than optimized wins.

## Copy modes and three timing domains

Synchronous mode calls `cudaMemcpy`. Asynchronous mode calls `cudaMemcpyAsync` on an
explicit `cudaStreamNonBlocking` stream. CUDA events are created once, reused, and
recorded around only the copy work.

Every raw sample preserves:

1. Host API time: calibrated monotonic host time spent in the copy API call(s).
2. Device/stream time: reused CUDA-event elapsed time around the stream copy work.
3. Host end-to-end time: host time from submission boundary through explicit event
   completion.

The event duration is the primary payload-throughput denominator. Raw bytes and raw
nanoseconds remain authoritative; reports derive decimal GB/s and binary GiB/s with
explicit labels. Async behavior is classified from the observed median API/end-to-end
ratio as quick-return, staging block observed, effectively synchronous, or inconclusive.
The vocabulary describes observations and does not claim undocumented CUDA internals.

## Sweep and repetition policy

The default sweep is:

```text
4, 16, 64, 256 KiB
1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512 MiB
1 GiB
```

Default repetitions are 1,000 through 64 KiB; 500 at 256 KiB; 300 at 1 MiB;
200 at 2 MiB; 150 at 4 MiB; 100 at 8 MiB; 80 at 16 MiB; 60 at 32 MiB;
40 at 64 MiB; 30 at 96 MiB; 25 at 128 MiB; 20 at 192 MiB; 15 at
256 MiB; 10 at 384 MiB; 8 at 512 MiB; and 5 at 1 GiB. This supplies useful
small-copy distributions while bounding total large-copy traffic.

Memory-class order is deterministically rotated by recorded seed
`0x5349444543415235`. API passes reverse the size order on alternating passes and
directions alternate by size index. This is reproducible and reduces simple
temperature/power ordering bias without reallocating inside timed regions.

For each direction, class, and API, peak median device-time bandwidth is found first.
The 80%, 90%, and 95% knees are the smallest tested sizes whose median reaches the
corresponding fraction of that measured peak. These are calculated profile facts, not
automatic scheduler block choices. P99 is considered meaningful only at 100+ samples.

## Supplemental experiments

Small-copy batching measures 64 KiB, 256 KiB, 1 MiB, and 4 MiB with counts
2, 4, 8, 16, and 32. One mode synchronizes after every async copy; the other enqueues
the complete batch in one stream and synchronizes once. Host API, event, and end-to-end
domains remain separate.

Contiguous chunking holds total payload at 128 MiB and compares 1x128, 2x64, 4x32,
8x16, and 16x8 MiB copies. It measures submission and completion overhead without
making a compute-overlap claim.

Bidirectional tests use two non-blocking copy streams, buffers, and event pairs. A third
non-blocking stream runs a deferred host callback and records the common gate only after
the host releases it; both copy streams wait on that event. The host releases the gate
after both copies and stop events have been queued. The experiment does not claim
physically identical starts. It records isolated H0/D0, concurrent Hc/Dc, host makespan,
aggregate payload rate, slowdown, and `(H0 + D0) / makespan` concurrency benefit. No
GPU compute is present.

Sustained mode repeatedly transfers a selected persistent buffer for a bounded 8 GiB
payload target. A separate low-rate 20 ms NVML observer samples temperature, clocks,
power, and PCIe generation/width. This observer is used only for sustained link-state
characterization, never inside isolated microsecond samples. Idle/before, maximum during,
and after observations remain distinct from measured payload bandwidth and nominal link
capability. Before and after readings each follow a one-second settling interval; the
reported state is exactly what NVML supplies and is not assumed to have downshifted.

## Persistence and commands

Migration 005 incrementally adds transfer configurations, benchmark aggregates, raw
samples, bidirectional results, link-state observations, and machine transfer profiles.
Configuration identity uses a fixed `SIDECAR-CUDA-TRANSFER-CONFIG-V1` serializer and
SHA-256. Sessions retain machine hash, exact Sidecar revision, spec, CUDA runtime/driver,
OS, and NVIDIA driver context.

```text
sidecar-lab cuda transfer info [--json]
sidecar-lab cuda transfer plan --dry-run [filters]
sidecar-lab cuda transfer run [--direction h2d|d2h] [--memory ...] [--api ...]
sidecar-lab cuda transfer sustained [filters]
sidecar-lab cuda transfer link-state [filters]
sidecar-lab cuda transfer bidirectional [filters]
sidecar-lab cuda transfer batching [filters]
sidecar-lab cuda transfer chunking [filters]
sidecar-lab cuda transfer report [--json] [--database path]
```

`plan --dry-run` reports buffer sizes, configuration count, estimated payload bytes,
and device reserve status without allocating transfer buffers or moving payload.
CUDA-disabled builds retain planning/math and all previous Sidecar features; transfer
execution reports unsupported.

## Authoritative real-machine results

The authoritative WU5 measurement set was produced by clean Release revision
`1414628a1950482a5b1cdd210fcbd92961bf6cea` on machine
`56ca8ca732b5006b04908e253be5ab77e689c67692e957509c230243f5b00961`.
Database session 7 contains the isolated sweep; sessions 8-9 contain HostAlloc and
registered bidirectional tests; sessions 10-11 contain H2D and D2H batching; sessions
12-13 contain H2D and D2H chunking; and sessions 14-19 contain the six sustained/link
state configurations. Schema version and migration history are 5, and prior WU1-WU4
records remain intact.

The primary asynchronous isolated profiles were:

| Direction | Host memory | Peak GB/s (GiB/s) | Peak size | 80% / 90% / 95% knee |
|---|---|---:|---:|---:|
| H2D | Pageable pretouched | 19.581 (18.236) | 4 MiB | 2 / 2 / 4 MiB |
| H2D | Pinned HostAlloc | 25.316 (23.577) | 512 MiB | 1 / 2 / 2 MiB |
| H2D | Pinned registered | 25.307 (23.569) | 96 MiB | 1 / 1 / 2 MiB |
| D2H | Pageable pretouched | 14.439 (13.448) | 384 MiB | 4 / 16 / 16 MiB |
| D2H | Pinned HostAlloc | 26.341 (24.532) | 128 MiB | 1 / 1 / 2 MiB |
| D2H | Pinned registered | 26.350 (24.540) | 96 MiB | 1 / 1 / 2 MiB |

Synchronous pinned peaks were similarly bounded at 25.281-25.293 GB/s H2D and
26.279-26.314 GB/s D2H, but their 95% knee was 8 MiB because fixed synchronous
overhead remained in the event-bracketed path. At 64 MiB, HostAlloc versus registered
async rates differed by only 0.07% H2D (25.282 versus 25.300 GB/s) and 0.27% D2H
(26.277 versus 26.347 GB/s). Transfer throughput therefore does not materially
distinguish the two persistent pinned backends on this machine. Pinned D2H was about
4.1% faster than pinned H2D; pageable D2H's peak was about 26% below pageable H2D.

For 4 KiB pinned async copies, median device time was 4.1-7.4 microseconds, host API
time 2.5-4.1 microseconds, and end-to-end time 10.1-12.7 microseconds. At 1 MiB the
pinned ranges were 43.4-46.7, 2.4-4.3, and 49.2-62.8 microseconds respectively; at
4 MiB they were 162.6-169.5, 2.5-2.8, and 174.0-189.1 microseconds. Small-copy tails
were nontrivial despite low medians and are preserved in the raw samples.

Pageable async behavior was not equivalent to pinned async behavior. At 4 MiB, H2D
spent 158.1 microseconds in `cudaMemcpyAsync`, 214.0 microseconds on the event timeline,
and 220.0 microseconds end to end, classified `HOST_STAGING_BLOCK_OBSERVED`. Pageable
D2H spent 355.0 / 362.1 / 367.7 microseconds and was classified
`EFFECTIVELY_SYNCHRONOUS`. At 64 MiB, both pageable directions were effectively
synchronous, with 4.82-5.28 ms inside the API, while pinned API medians remained
3.3-5.9 microseconds.

The best matched bidirectional case was registered 512 MiB at 25.816 GB/s aggregate.
Its concurrency benefit was 0.9994x relative to simple serialization; H2D and D2H
durations slowed by approximately 1.96x and 2.04x. HostAlloc produced the same result
within noise. Thus this device's reported `asyncEngineCount=1` coincides with observed
copy/copy serialization, but the metadata was not used to infer that outcome.

Batching 32 small copies before one final synchronization reduced end-to-end time versus
synchronizing each copy. At 64 KiB it changed H2D 385.7 to 306.4 microseconds and D2H
370.0 to 191.7 microseconds; at 256 KiB it changed 777.6 to 489.6 and 598.9 to 406.8
microseconds; at 1 MiB it changed 2.095 to 1.421 ms and 1.556 to 1.343 ms. Benefits
narrowed at 4 MiB. Submission time sometimes increased for a queued small-copy batch,
so API, device, and completion domains must remain separate.

Splitting one 128 MiB transfer into 16 contiguous chunks did not improve the clean copy
baseline. H2D device time changed 5.298 to 5.338 ms, host submission 2.6 to 67.3
microseconds, and end-to-end 5.304 to 5.358 ms. D2H changed 5.091 to 5.121 ms,
3.8 to 56.4 microseconds, and 5.108 to 5.140 ms. Chunking may still improve availability
to compute, but only WU6 can measure that tradeoff.

Each sustained case moved 8 GiB using a persistent 64 MiB buffer:

| Direction | Host memory | Sustained GB/s |
|---|---|---:|
| H2D | Pageable pretouched | 14.113 |
| D2H | Pageable pretouched | 14.137 |
| H2D | Pinned HostAlloc | 25.319 |
| D2H | Pinned HostAlloc | **26.363** |
| H2D | Pinned registered | 25.317 |
| D2H | Pinned registered | 26.354 |

NVML reported active PCIe Gen4 x16 in every before, during, and after observation.
Temperatures across the sequential sustained set ranged from 44 to 47 C, clocks were
1860-1995 MHz graphics and 9501 MHz memory, and no result met the noisy threshold.
No CUDA errors, verification failures, failed samples, or foreign-key violations were
observed. The measured WU6 candidate list is 1, 2, 96, 128, 512, and 1024 MiB; it is a
short experimental list, not a block-size or arena-size decision.

Combined with WU4, WU5 continues to support a large pageable warm pool plus small
persistent pinned DMA staging. Pageable memory remains viable for capacity, while
persistent pinned memory avoids lifecycle cost and provides materially higher bulk rate
and fast-returning async submission. HostAlloc versus registered remains a later runtime
engineering choice rather than a transfer-throughput decision.

## Limitations

- CUDA events have finite resolution and represent stream chronology, not PCIe analyzer
  traces.
- Sampled verification is strong corruption detection, not a cryptographic full-buffer
  proof after every repetition.
- NVML polling can miss very short state transitions; sustained mode exists to make the
  load interval observable.
- `asyncEngineCount` is metadata, not proof of concurrency; bidirectional results decide
  observed behavior.
- No transfer/compute overlap, GEMM, kernel contention, CUPTI, NVMe, model inference,
  scheduler, prefetcher, or final arena-size conclusion belongs to WU5.
- WU3 LIGHT observer overhead remains above its `<2%` target. That M0 gate remains open,
  and real-inference observer overhead remains not yet measured.
