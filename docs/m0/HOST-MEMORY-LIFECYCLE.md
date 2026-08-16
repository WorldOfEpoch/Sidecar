# Host Memory Lifecycle Laboratory

Status: M0 Work Unit 4 implementation. This document does not modify
`M0-FROZEN-1` and makes no host/device transfer-performance claim.

## Scope

The laboratory measures host-memory lifecycle phases independently:

- Windows pageable backing: `VirtualAlloc(MEM_RESERVE | MEM_COMMIT)`, deterministic
  first touch, warm traversal, and `VirtualFree(MEM_RELEASE)`.
- CUDA host allocation: `cudaHostAlloc(..., cudaHostAllocDefault)`, first touch,
  warm traversal, and `cudaFreeHost`.
- CUDA host registration: pageable allocation, `cudaHostRegisterDefault`,
  traversal, `cudaHostUnregister`, and pageable release.

`REGISTER_COLD` calls `cudaHostRegister` before the full first-touch traversal.
`REGISTER_PRETOUCHED` touches every OS page before registration. Backing allocation,
first touch, registration, unregistration, and backing release retain separate raw
timings; no combined value is called registration cost.

The provider interface isolates native Windows/CUDA calls from the benchmark engine
and is replaceable by synthetic providers in unit tests. CUDA-disabled builds retain
pageable functionality and classify CUDA methods as `SKIPPED_UNSUPPORTED`.

## Touch and warmup policy

Every traversal accesses one byte per native OS page in deterministic ascending page
order and consumes the values through a checksum so the compiler cannot remove it.
The first traversal exposes page population/fault behavior; the second is the warm
control. CUDA is explicitly initialized with `cudaFree(nullptr)` before CUDA-host
measurements, and initialization time is reported separately.

The first measured repetition is labelled cold setup and subsequent repetitions warm
setup. Default measured repetitions are size-aware: 9 through 256 MiB, 7 through
512 MiB, 5 through 1 GiB, and 3 above 1 GiB. The CLI can override this for controlled
experiments. After each sample the engine checks memory recovery up to five times at
100 ms intervals before allowing escalation.

## Safety policy

Every attempted allocation is replanned from a fresh native memory snapshot. The
default required free-RAM reserve is:

```text
max(4 GiB, 15% of installed physical RAM)
```

The default per-allocation policy cap is 25% of installed RAM for pageable memory and
10% for CUDA-pinned allocation/registration. `--reserve` and `--max-fraction` expose
controlled overrides. Decisions distinguish safe, insufficient headroom, method
policy limit, previous pressure signal, and unknown memory state.

Escalation stops monotonically for allocation/registration/resource failures, cleanup
failure, or retained pressure. Missing cleanup recovery above the larger of 512 MiB or
10% of the region marks a sample noisy. Above the larger of 2 GiB or 25% it triggers a
hard pressure stop. A skipped size is not reported as a failed allocation.

On the 96 GiB development host the dynamic reserve is 14.4 GiB and the default pinned
cap is 9.6 GiB. Consequently 12 GiB and 16 GiB pinned candidates are skipped by policy;
the laboratory does not probe them merely to discover a failure point.

## Timing and statistics

The direct phase timer is `std::chrono::steady_clock`. Each run records a 10,000-sample
empty-bracket calibration, its median overhead, and observed non-zero resolution.
Raw nanoseconds are always retained. Corrected values subtract the median bracket only
when the raw value is larger, without changing the raw observation.

For every phase the laboratory computes count, mean, median/P50, P90, P95, minimum,
maximum, and population standard deviation. P99 is emitted only with at least 100
samples; otherwise output explicitly marks it insufficient. Lifecycle setup is also
divided, as a calculation, by conceptual reuse counts 1, 10, 100, 1,000, and 10,000.
This is an amortization illustration, not measured inference or transfer overhead.

## Bounded churn and persistent arena

`memory stress` uses a deterministic sequence (64 MiB, 256 MiB, 1 GiB, 512 MiB,
2 GiB by default) through the same per-sample safety and cleanup checks. It reports
first-to-last allocation and cleanup latency drift, recovery, failures, and phase tails.
It is intentionally bounded rather than an allocator torture test.

`PersistentPinnedArena` is a laboratory-only fixed-capacity RAII object. It supports
`cudaHostAlloc` and pre-touched registered-pageable backing, bounds-checked spans,
one-time setup, repeated deterministic read/write verification without reallocation or
reregistration, and one-time shutdown cleanup. It is not the future Sidecar allocator:
there are no free lists, scheduling, eviction, tensor placement, or transfer tests.

## Persistence and commands

Migration 004 adds `host_memory_benchmarks`, `host_memory_samples`,
`memory_pressure_events`, and `persistent_arena_tests`. Rows belong to an existing
benchmark session and therefore inherit machine identity, Sidecar revision, spec, and
runtime context. Raw phase samples and before/setup/cleanup snapshots are preserved.
Migration is incremental; existing history is not rebuilt.

```text
sidecar-lab memory info [--json]
sidecar-lab memory plan [--method pageable|hostalloc|hostregister] [--json]
sidecar-lab memory lifecycle [--method ...] [--sizes ...] [--repetitions N]
sidecar-lab memory lifecycle --dry-run
sidecar-lab memory arena [--backend hostalloc|hostregister] [--size ...]
sidecar-lab memory stress [--method ...] [--sizes ...]
sidecar-lab memory report [--json] [--database path]
```

Lifecycle, plan, arena, and report support deterministic structured JSON. `--no-persist`
is intended for preliminary probes; authoritative measurements use session persistence.

## Real-machine validation

Pre-commit validation on the discovered RTX 3090 / CUDA 12.8 host safely exercised
4 MiB and 64 MiB `cudaHostAlloc`, both cold and pre-touched `cudaHostRegister`, and
both 4 MiB persistent-arena backends. All allocations, registrations, verification,
unregistration, and cleanup succeeded. At 64 MiB, the short two-sample validation saw
approximately 7.19 ms median `cudaHostAlloc`, 4.54 ms `cudaFreeHost`, 7.75 ms cold
registration, 0.302 ms pre-touched registration, and roughly 2.75--3.13 ms unregister.
These smoke values verify phase separation and are not the authoritative size sweep.

## Limitations and preserved gates

- Windows memory snapshots are process/system observations, not proof of exact physical
  page ownership; recovery is therefore tolerance-based.
- OS/runtime caching and unrelated activity can introduce noise. Such samples remain
  stored and labelled rather than silently removed.
- No `VirtualLock` experiment is included. The architectural hypothesis remains a large
  pageable warm pool plus a small pinned DMA arena.
- No H2D/D2H, copy-engine, PCIe-bandwidth, overlap, NVMe, scheduler, or inference claim
  can be derived from WU4.
- WU3's LIGHT event-every-iteration synthetic observer-overhead result remains above the
  `<2%` target and is still an open M0 performance gate. WU4 neither weakens nor closes it.
