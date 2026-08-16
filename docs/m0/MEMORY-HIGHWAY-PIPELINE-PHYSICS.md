# M0 Work Unit 8: Memory-Highway Pipeline Physics

## Status and scope

WU8 connects the storage, host-memory, PCIe-transfer, and GPU-compute physics
measured in WU4-WU7. It is an experimental pipeline controller and measurement
laboratory. It is not a production scheduler, prefetcher, residency manager,
llama.cpp integration, Guardian policy engine, or M1 implementation.

The authoritative machine is identified by machine hash
`56ca8ca732b5006b04908e253be5ab77e689c67692e957509c230243f5b00961`.
Runtime discovery reported Windows 11, Intel Core Ultra 9 285K, approximately
96 GiB RAM, an NVIDIA GeForce RTX 3090 with 24 GiB VRAM, and a Samsung SSD 990
PRO 2TB. The storage dataset is the existing 8,589,934,592-byte
`C:/SidecarData/sidecar-storage-dataset-v1-8gib.bin`, generator
`SIDECAR-SPLITMIX64-BYTE-V1`, identity
`da0f22a709d6d4f6d2424fac8335a2a65d0e9d358190b40b695a8ceba1774c97`.
Full deterministic dataset verification passed before authoritative work. The
dataset was not rewritten.

## Pipeline architecture

WU8 compares two paths:

| Path | Stages | Intended role under test |
|---|---|---|
| `NVME_PAGEABLE_PINNED_H2D` (A) | NVMe -> pageable -> pinned -> VRAM | normal warm-reservoir path |
| `NVME_PINNED_H2D` (B) | NVMe -> pinned -> VRAM | urgent cold-block path |

Both paths reuse the WU7 Windows backend: unbuffered overlapped `ReadFile` with
an I/O Completion Port. WU8 adds a reusable asynchronous-reader interface to
that storage layer instead of implementing a second engine. CUDA allocations,
host allocations, storage handles, streams, events, and destination regions are
created before sample timing and reused.

Each staging slot follows an explicit state machine:

`FREE -> NVME_READING -> [HOST_COPY_PENDING -> HOST_COPY_ACTIVE] -> H2D_READY
-> H2D_ACTIVE -> GPU_READY -> VERIFY_PENDING -> FREE`

The bracketed states apply only to Pipeline A. A transition validates current
state and ownership. Storage cannot write a pinned slot while CUDA reads it;
pageable, pinned, and device regions cannot be reused until the owning operation
completes. Cancellation and native errors drain outstanding work before release.
The tests exercise the state machine, invalid ownership, rotation, cancellation,
and injected storage, copy, H2D, and verification failures.

Pipeline A overlaps NVMe block N+1, pageable-to-pinned copy of block N, and H2D
of block N-1 when depth permits. Pipeline B overlaps NVMe block N+1 and H2D of
block N. Persistent depths 1-4 are supported. Slots rotate by sample and block;
their persisted arena allocation ID plus offset and size are session-local
identities. Virtual addresses are not treated as stable physical-RAM identity.

## Timing model and cross-clock limitations

The persisted variables are:

- `C0`: matched compute-alone duration.
- `P0`: matched pipeline-alone duration.
- `Cc`: compute branch duration while the pipeline is active.
- `Pc`: pipeline duration while compute is active.
- `M`: completion of the later measured branch on a common host timeline.
- `Nc`, `Rc`, and `Tc`: concurrent NVMe, host-copy, and H2D stage evidence.

Derived values preserve raw inputs:

```
compute_path_delta       = M - C0
compute_path_added       = max(0, compute_path_delta)
compute_path_added_pct   = compute_path_added / C0
pipeline_slowdown        = (Pc - P0) / P0
compute_slowdown         = (Cc - C0) / C0
pipeline_overlap_raw     = (C0 + P0 - M) / min(C0, P0)
pipeline_overlap_display = clamp(pipeline_overlap_raw, 0, 1)
```

Windows monotonic timestamps define storage, host-copy, state-transition,
bubble, deadline, and outer pipeline intervals. CUDA events define GPU branch
durations. WU6's gate-release callback publishes an explicit host atomic only
after the CUDA common gate is released; WU8 waits for that handshake before
setting the common host origin. `M` is the maximum of the compute branch on
that common origin and pipeline `Pc` plus its host-origin offset. A runtime
invariant rejects an `M` that ends materially before either measured branch.

No attempt is made to subtract a host/CUDA clock offset inferred from unrelated
events. Cross-domain relationships therefore use the explicit gate handshake
and host-observable completion, while within-GPU durations use CUDA events.
Event timestamps and Windows timestamps should not be numerically compared as
if they shared a clock epoch. Full D2H verification of every GPU destination is
performed after the timed interval and before a sample can be accepted.

`READY_AHEAD_US` is `deadline - GPU_READY`. Positive is early, zero is just in
time, and negative is late. For positive-margin reporting, P1 and P5 describe
the dangerous low-margin tail; calling P99 of positive margin the dangerous
tail would reverse the interpretation. Raw values remain stored.

## Methodology corrections and provenance

WU8 entered from clean WU7 revision
`96c952b9e5f1dd59e09e4da78747b78d61adfce5`. The final authoritative timing
implementation is revision `6e078fcbd8f4fa0781e1b6039c75978a83a06570`.

Two timing defects were discovered during validation. Evidence was preserved,
annotated, and superseded rather than deleted:

1. Early compute sessions used a fixed host sleep before releasing pipeline
   work. Compute setup could exceed the sleep. These rows are annotated
   `SUPERSEDED_COMPUTE_TIMELINE_GATE`; their P0 and stage evidence remains
   useful, but they are excluded from final contention recommendations.
2. The first explicit-gate revision included post-timing D2H verification in
   the outer makespan. These rows are annotated
   `SUPERSEDED_MAKESPAN_INCLUDED_POST_TIMING_VERIFICATION`; P0, Pc, stages, and
   verification evidence remain useful, but their makespan is excluded.

Revision `963e3bd10e415431bdc40774012148bab00910ef` introduced the explicit gate
handshake. Revision `6e078fc` then excluded post-timing verification from `M`
without weakening verification. Final recommendations use only timing rows
from `6e078fc`. The corrected campaign contains 493 successful benchmark
configurations and 7,113 successful, GPU-verified pipeline samples: 40 baseline
configurations/800 samples, 420 coarse configurations/2,100 samples, 20
refinement configurations/2,000 samples, 11 independent-repeat
configurations/2,000 samples, and 2 streaming configurations/213 samples.
There are 59,564 successful final-revision stage rows. All final-revision
sessions completed; no native, CUDA, storage, or verification error occurred.

P99.9 is emitted only with at least 1,000 samples. JSON uses explicit `null`
when unsupported. Manual refinement and repeat runs persist a nonempty
selection reason.

## Isolated pageable-to-pinned host copy

Sources were preallocated and pre-touched pageable memory; destinations were
persistent pinned memory. Allocation and first touch were outside timing. The
primary operation was the optimized CRT copy. The full size sweep included
1, 2, 16, 32, 64, 128, and 256 MiB; bounded worker comparisons used 1, 2, and
4 workers at 64, 128, and 256 MiB.

| Size | Workers | P50 / P95 / P99 wall ms | Median GB/s |
|---:|---:|---:|---:|
| 64 MiB | 1 | 2.542 / 2.800 / 2.893 | 26.38 |
| 64 MiB | 2 | 2.178 / 2.420 / 2.484 | 30.76 |
| 64 MiB | 4 | 1.941 / 2.122 / 2.161 | 34.57 |
| 128 MiB | 1 | 5.461 / 5.769 / 6.168 | 24.58 |
| 128 MiB | 2 | 4.765 / 5.125 / 5.448 | 28.12 |
| 128 MiB | 4 | 4.280 / 4.561 / 6.527 | 31.36 |
| 256 MiB | 1 | 11.449 / 12.308 / 12.839 | 23.44 |
| 256 MiB | 2 | 9.842 / 10.461 / 11.029 | 27.27 |
| 256 MiB | 4 | 8.813 / 9.229 / 13.108 | 30.45 |

Four workers improve median wall time by about 21-24%, but 128/256 MiB P99 is
worse than with two workers. Process CPU timing from `GetProcessTimes` is
quantized in 15.625 ms increments on this host: the P95 observations scale from
roughly 15.625 ms at one worker to 31.25 ms at two and 62.5 ms at four. These
values establish direction, not microsecond CPU precision. More copy workers
therefore buy median bandwidth with proportionally greater CPU consumption and
unfavorable large-copy tails; WU8 does not select a production worker policy.

With compute active, single-copy medians rose by 2-20%, with the largest effect
at 64 MiB. Observed P99 degradation reached 35.6%. Host-copy pressure is real
even without an error, and the full Pipeline A stage rows retain its behavior
while NVMe and H2D are also active.

## Isolated pipeline baselines

The high-count isolated depth-1 results show the direct-path advantage before
deep staging overlap:

| Aggregate/path | P50 / P95 / P99 ms | B median saving |
|---|---:|---:|
| 64 MiB A | 16.636 / 17.643 / 17.892 | - |
| 64 MiB B | 13.419 / 13.659 / 13.850 | 3.217 ms (19.3%) |
| 128 MiB A | 33.193 / 35.263 / 36.783 | - |
| 128 MiB B | 26.733 / 27.071 / 27.387 | 6.460 ms (19.5%) |
| 256 MiB, 8x32, A depth 3 | 41.424 / 41.898 / 42.297 | - |
| 256 MiB, 8x32, B depth 3 | 39.821 / 40.037 / 42.199 | 1.603 ms (3.9%) |

Deep overlap hides much of Pipeline A's copy cost at 256 MiB, but cannot remove
the extra stage from tight single-block deadlines.

### 256 MiB decomposition and depth

The table below is the final isolated baseline distribution. Values are
P50/P99 milliseconds.

| Path/decomposition | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|
| A, 8x32 MiB | 68.443/72.724 | 55.926/58.639 | 41.491/42.610 | 41.735/46.242 |
| A, 4x64 MiB | 66.798/77.904 | 54.496/59.777 | 44.761/46.003 | 44.936/45.774 |
| A, 2x128 MiB | 67.023/70.559 | 52.509/53.878 | 52.104/54.979 | 52.690/54.891 |
| B, 8x32 MiB | 53.666/54.943 | 44.201/45.325 | 39.663/43.019 | 39.717/42.008 |
| B, 4x64 MiB | 53.040/54.245 | 44.502/46.269 | 41.604/41.746 | 41.642/42.495 |
| B, 2x128 MiB | 53.148/53.560 | 45.438/45.722 | 45.452/45.803 | 45.434/45.739 |

For 8x32 MiB, depth 2 reduces the median by 18.3% for A and 17.6% for B.
Depth 3 then improves it by another 25.8% for A and 10.3% for B. Depth 4
changes the median by less than 1%, does not improve steady cadence, reserves
another slot, and increases drain time. It remains a possible tail-control
experiment rather than a default. The integrated winner is 8x32 MiB at depth
3 for both paths; 32-64 MiB is the useful later tuning region, not a declaration
of one final block size.

At 256 MiB depth 3, fill latency was about 11.9-12.0 ms for A and
10.21-10.28 ms for B. Steady-state block cadence was about 4.20-4.23 ms.
Drain latency was about 10.56-10.72 ms. Depth 4 retained similar fill and
cadence but raised drain to about 15.3-15.4 ms.

## Stage timing and bubbles

Representative high-count per-block stage distributions are:

| Cell/stage | P50 / P95 / P99 ms |
|---|---:|
| A 64 MiB NVMe | 10.889 / 11.077 / 11.129 |
| A 64 MiB copy | 2.989 / 3.870 / 4.059 |
| A 64 MiB H2D | 2.667 / 2.703 / 2.722 |
| B 64 MiB NVMe | 10.730 / 10.915 / 10.977 |
| B 64 MiB H2D | 2.675 / 2.706 / 2.743 |
| A 128 MiB NVMe | 21.683 / 21.943 / 22.102 |
| A 128 MiB copy | 6.099 / 7.800 / 8.800 |
| A 128 MiB H2D | 5.322 / 5.357 / 5.373 |
| B 128 MiB NVMe | 21.376 / 21.676 / 21.812 |
| B 128 MiB H2D | 5.337 / 5.371 / 5.393 |
| A 256 MiB 8x32, NVMe per request | 12.368 / 18.270 / 18.337 |
| A 256 MiB 8x32, copy per block | 1.660 / 2.207 / 2.701 |
| A 256 MiB 8x32, H2D per block | 1.341 / 1.411 / 1.445 |
| B 256 MiB 8x32, NVMe per request | 14.028 / 18.281 / 18.343 |
| B 256 MiB 8x32, H2D per block | 1.343 / 1.406 / 1.428 |

The rows are stage populations and must not be added as if all stages were on
one serial critical path. Overlap is directly observed in the state and stage
timestamps. NVMe is the largest absolute stage at the measured deadline
boundaries. The pageable copy is the marginal stage that separates A from B at
16 ms/64 MiB and 32 ms/128 MiB.

Persisted bubbles distinguish GPU waiting for data, H2D waiting for source,
pinned/pageable slot waits, and storage queue starvation. GPU-waiting time was
nonzero at the deliberately missed boundaries: Pipeline A at 64 MiB/16 ms had
about 0.62 ms median lateness and A at 128 MiB/32 ms about 1.25 ms. It was zero
in the selected positive-margin cells.

## A-J contention decomposition

The controlled 256 MiB, 2x128 MiB, depth-2, 64 ms matrix is:

| ID | Active components |
|---|---|
| A | compute only |
| B | NVMe -> pageable |
| C | pageable -> pinned copy |
| D | pinned -> VRAM H2D |
| E | NVMe -> pinned |
| F | NVMe -> pageable + host copy |
| G | host copy + H2D |
| H | NVMe -> pinned + H2D |
| I | full Pipeline A + compute |
| J | full Pipeline B + compute |

Representative final-revision P50/P99 elapsed milliseconds are:

| Cell | ALU | MEMORY_BOUND | FP16_GEMM |
|---|---:|---:|---:|
| B | 40.381/41.504 | 40.338/40.611 | 40.367/40.628 |
| C | 11.754/19.216 | 12.612/14.145 | 11.980/12.152 |
| D | 10.767/10.806 | 10.671/11.052 | 10.659/10.765 |
| E | 40.052/40.130 | 40.118/40.156 | 40.149/40.422 |
| F | 46.473/48.348 | 46.403/46.874 | 46.294/46.553 |
| G | 19.299/19.687 | 19.750/22.453 | 19.462/30.345 |
| H | 45.489/46.356 | 45.518/45.627 | 45.441/45.841 |
| I | 52.121/53.566 | 52.010/53.602 | 51.896/53.026 |
| J | 45.566/45.961 | 45.727/45.806 | 45.591/45.854 |

A-H isolate component combinations and are not all compute-concurrent; their
compute-added fields must not be interpreted as contention results. I/J are the
complete concurrent comparisons. J saves roughly 6-8 ms here by eliminating
the host-copy stage.

## Compute contention

At 256 MiB, 8x32 MiB, depth 3, and approximately 64 ms compute windows, the
independent repeat P99 compute-path additions were:

| Workload | Pipeline A | Pipeline B |
|---|---:|---:|
| SYNTHETIC_ALU | 3.16% | 2.74% |
| MEMORY_BOUND | 3.63% | 3.04% |
| FP16_GEMM | 2.79% | 2.90% |

Depth-4 ALU repeats were 3.08% for A and 2.43% for B. These repeats meet WU6's
5% band. They do not establish a 1% or 2% band.

The preceding independent refinement pass was materially worse in its tails:
A reached 21.87%, 7.95%, and 12.83% for ALU, memory-bound, and GEMM; B reached
4.57%, 7.63%, and 5.27%. Median compute-path addition remained below 2% in
these cells, but P99 was pass-sensitive. All samples were valid and verified,
so both passes are retained. WU8 therefore cannot claim a reproducible <=5%
complete-pipeline P99 envelope across passes. Pipeline B is the most favorable
path overall; A produced the worst first-pass compute tails.

Adding NVMe did not reproducibly break WU6's 5% H2D-plus-compute result in the
repeat pass, but the first pass demonstrates new full-system tail instability.
Adding host memcpy materially changes tight-path latency and first-pass compute
tails even though its repeat-pass compute penalty is smaller. The result is not
evidence that asynchronous H2D, storage, or copy is free.

## Deadline and ready-ahead results

The table describes refined selected cells. `100/100` supports observed 99%
success, not 99.9%. Only the 1,000-sample B 64 MiB cell supports an observed
99.9% statement.

| Deadline | Pipeline A | Pipeline B |
|---:|---|---|
| 16 ms | 64 MiB: 0/100; larger miss | 64 MiB: 1,000/1,000; larger miss |
| 32 ms | 64 MiB: 100/100; 128 MiB: 0/100; 256 MiB miss | 64 and 128 MiB: 100/100; 256 MiB miss |
| 64 ms | selected 64/128/256 MiB cells: 100/100 | selected 64/128/256 MiB cells: 100/100 |
| 96 ms | selected 64/128/256 MiB cells: 100/100 | selected 64/128/256 MiB cells: 100/100 |
| 128 ms | selected 64/128/256 MiB cells: 100/100 | selected 64/128/256 MiB cells: 100/100 |

At 64 MiB/16 ms, B ALU's 1,000-sample Pc distribution was
13.447/13.693/13.951/14.247 ms at P50/P95/P99/P99.9. Its ready-ahead lower
tail was 2.039 ms at P1, 2.300 ms at P5, and 2.551 ms at P50. It therefore
supports observed 99.9% deadline success at 16 ms, while its P99.9 compute-path
addition was 11.91%; deadline reliability and compute-interference bands are
separate properties.

At 128 MiB/32 ms, repeated B GEMM Pc was 26.672/26.996/27.075 ms at
P50/P95/P99. Its ready-ahead was 4.857 ms at P1, 4.998 ms at P5, and 5.327 ms
at P50. A's median lateness at the same boundary was about 1.25-1.41 ms.

At 256 MiB/64 ms, depth-4 streaming A Pc was
41.358/41.976/44.579 ms at P50/P95/P99, with ready-ahead
19.421/22.024/22.642 ms at P1/P5/P50. B was
39.742/39.928/39.989 ms, with ready-ahead 24.011/24.072/24.254 ms. The
selected depth-3 cells also had substantial positive margins, but their lower
tail varied more between passes because of isolated NVMe supply tails.

The smallest practical lead among the tested standard deadlines is:

| Aggregate | Pipeline A | Pipeline B |
|---:|---:|---:|
| 64 MiB | 32 ms | 16 ms |
| 128 MiB | 64 ms | 32 ms |
| 256 MiB | 64 ms | 64 ms |

These are empirical planning inputs on this machine, not scheduler policy.

## Streaming, telemetry, health, and slot observations

The bounded 256 MiB, 8x32 MiB, depth-4 stream produced 107 A and 106 B
verified samples. Its fill, cadence, drain, ready-ahead, occupancy, bubbles,
and per-slot totals are persisted. The run did not perform prolonged thermal
stress.

Across the final revision, all 493 before/after GPU PCIe observations reported
generation 4 x16. GPU temperature ranged from 41-68 C before configurations
and 43-67 C afterward. The NVMe composite temperature evidence ranged from
49.85-56.85 C. NVMe critical warnings, media/data errors, and error-log count
did not increase. No CUDA, storage, native-I/O, or transfer-verification error
occurred, and no major anomaly was associated with thermal or power state.

There are 1,135 final-revision slot profiles covering 23,904 transfers.
Rotation found no repeatable performance difference, verification failure, or
unusual deadline-failure rate by slot. The observation is
`NO_REGION_VARIATION_OBSERVED`. This is not proof that all physical RAM is
identical, and it is not a permanent mapping from an arena offset to a DIMM.
WU8 supplies a future profiling data model but no present evidence justifies
per-region policy, ranking, blacklisting, quarantine, or a RAM-failure claim.

## Persistence and CLI

Migration 008 advances `PRAGMA user_version` to 8 without recreating the
database. It adds host-copy configuration/benchmark/sample tables and pipeline
configuration, benchmark, sample, stage, deadline, slot, and health tables.
Foreign keys remain enabled, newer-schema rejection remains intact, and all
WU1-WU7 history is preserved.

After the two final human/JSON `pipeline validate` smoke checks, the cumulative
WU8 database contains 31 host-copy benchmarks/2,600 samples, 1,439 pipeline
benchmarks/16,131 pipeline samples, 120,472 stage samples, 8,634 deadline
profiles, 3,309 slot profiles, and 2,878 health observations. The two one-sample
validation rows are correctness smoke evidence, not part of the 493-configuration
authoritative campaign. Superseded rows remain queryable with their annotations.

The command family is:

```
sidecar-lab pipeline info [--json]
sidecar-lab pipeline host-copy [filters] [--json]
sidecar-lab pipeline plan --dry-run [filters] [--json]
sidecar-lab pipeline baseline [filters] [--json]
sidecar-lab pipeline run [filters] [--json]
sidecar-lab pipeline matrix [filters] [--json]
sidecar-lab pipeline stream [filters] [--json]
sidecar-lab pipeline report [--json]
sidecar-lab pipeline validate [--json]
```

Dry-run planning reports paths, dataset identity, aggregates, decompositions,
depths, pageable/pinned/VRAM and outstanding-I/O requirements, workloads,
windows, repetitions, telemetry and health policy, slot rotation, and estimated
bytes read. It performs no benchmark activity. CUDA-disabled builds retain the
host-copy, storage, database, hardware, trace, and WU7 surface and return
`SKIPPED_UNSUPPORTED` for GPU-pipeline work.

## Required question answers

1. **Pageable-to-pinned bandwidth:** at one worker the 64/128/256 MiB medians
   are 26.38/24.58/23.44 GB/s; two workers reach 30.76/28.12/27.27 GB/s and
   four reach 34.57/31.36/30.45 GB/s.
2. **Copy tails:** the exact P50/P95/P99 values are in the host-copy table.
   Single-worker P99 is 2.893, 6.168, and 12.839 ms at 64/128/256 MiB.
3. **Do 2/4 workers improve latency?** Yes at the median; four-worker large-copy
   P99 regresses relative to two workers.
4. **CPU cost:** it scales approximately with worker count; the available
   process accounting is quantized to 15.625 ms and shows roughly 1x/2x/4x
   P95 CPU increments.
5. **Does copy change under contention?** Yes. Compute raised copy medians by
   2-20% and observed P99 by as much as 35.6%; full-pipeline stage evidence
   also shows copy pressure.
6. **Isolated A latency:** P50/P95/P99 is 16.636/17.643/17.892 ms at 64 MiB,
   33.193/35.263/36.783 ms at 128 MiB, and 41.424/41.898/42.297 ms for the
   selected 256 MiB 8x32 depth-3 cell.
7. **Isolated B latency:** 13.419/13.659/13.850, 26.733/27.071/27.387, and
   39.821/40.037/42.199 ms for the corresponding cells.
8. **B saving:** 3.217 ms/19.3% at 64 MiB, 6.460 ms/19.5% at 128 MiB, and
   1.603 ms/3.9% in the deeply overlapped 256 MiB cell.
9. **Best 256 MiB decomposition for A:** 8x32 MiB at depth 3 in the integrated
   baseline.
10. **Best for B:** 8x32 MiB at depth 3; depth 4 is only a tail-control option.
11. **Best depth balance:** depth 3 for 32/64 MiB chunks; depth 2 is the minimum
   practical pipeline and depth 4 adds little steady benefit.
12. **Depth-2 benefit:** for 8x32, median improves 18.3% A and 17.6% B over
   depth 1.
13. **Depth 3:** yes; it adds 25.8% A and 10.3% B median improvement for 8x32.
14. **Depth 4:** median/cadence gains are negligible; it consumes memory and
   increases drain, though some runs show a modest tail benefit.
15. **Fill latency:** about 12.0 ms A and 10.2 ms B for 256 MiB 8x32 at depth 3.
16. **Steady cadence:** about 4.20-4.23 ms per 32 MiB GPU-ready block.
17. **Drain:** about 10.6-10.7 ms at depth 3 and 15.3-15.4 ms at depth 4.
18. **ALU compute P99 extension:** independent repeat A/B is 3.16%/2.74%; the
   preceding pass was 21.87%/4.57%.
19. **MEMORY_BOUND:** repeat A/B is 3.63%/3.04%; preceding pass 7.95%/7.63%.
20. **FP16 GEMM:** repeat A/B is 2.79%/2.90%; preceding pass 12.83%/5.27%.
21. **Does NVMe worsen WU6 contention?** It did not reproducibly break 5% in
   the repeat pass, but it introduced valid pass-to-pass P99 excursions, so the
   complete-system <=5% band is not reproducible.
22. **Does host copy worsen it?** It materially worsens tight-path latency and
   the first-pass compute tails; the repeat compute penalty is smaller.
23. **Worst complete path:** Pipeline A, especially its first-pass ALU tail.
24. **Most favorable:** Pipeline B.
25. **Deadline reliability:** B64 is the only measured 1,000-sample cell and
   achieved observed >=99.9% at 16 ms. Selected 100-sample winners establish
   observed >=99% at 32 ms for A64/B64/B128 and at 64/96/128 ms for both paths
   at 64/128/256 MiB. A64 misses 16 ms and A128 misses 32 ms.
26. **Ready-ahead distributions:** representative P1/P5/P50 margins are
   2.039/2.300/2.551 ms for B64 at 16 ms, 4.857/4.998/5.327 ms for B128 at
   32 ms, 19.421/22.024/22.642 ms for streaming A256 at 64 ms, and
   24.011/24.072/24.254 ms for B256.
27. **GPU waiting:** observed at the intended A64/16 ms and A128/32 ms miss
   boundaries; zero in the selected positive cells.
28. **Miss cause:** NVMe is the largest absolute stage; pageable copy is the
   marginal stage that makes A miss boundaries B meets.
29. **Smallest practical lead:** A/B is 32/16 ms for 64 MiB, 64/32 ms for
   128 MiB, and 64/64 ms for 256 MiB.
30. **Normal path:** **YES**, at >=32 ms for 64 MiB and >=64 ms for 128/256 MiB.
31. **Urgent path:** **YES**, including 16 ms/64 MiB and 32 ms/128 MiB.
32. **Maintain both paths:** **YES**. Direct pinned changes deadline class at
   64/128 MiB; pageable staging remains a viable warm-reservoir path.
33. **Double buffering minimum:** **YES** for multi-block runtime experiments.
34. **Triple buffering:** **YES**, useful for 32/64 MiB chunks; a fourth slot is
   not a default.
35. **Later chunk region:** 32-64 MiB, while retaining 64/128/256 MiB aggregate
   controls. No final block size is selected.
36. **Slot variation:** none repeatable was observed.
37. **Slot failures/anomalies:** no verification failure, repeated latency
   anomaly, or unusual deadline-failure rate by slot.
38. **Evidence for per-region performance profiling:** **NO** at present. The
   schema is a future measurement seed, not a policy justification.
39. **RAM hardware failure:** **NO** evidence.
40. **NVMe health:** unchanged; critical warning/media-error/error-log deltas
   were zero.
41. **GPU PCIe state:** generation 4 x16 before and after every final cell.
42. **CUDA errors:** none.
43. **Transfer verification errors:** none.
44. **Storage errors:** none.
45. **Thermal/power association:** no major anomaly was associated with the
   observed thermal or power state.

## Primary WU8 decision

**PIPELINE DEPENDENT**

The complete pipeline can keep data ready on this machine when its measured
lead-time and buffering requirements are respected. Pipeline B changes the
deadline class at the tight boundaries: it delivered 64 MiB 1,000/1,000 times
inside 16 ms and 128 MiB 100/100 inside 32 ms, where Pipeline A missed every
refined sample. Both paths delivered the selected 256 MiB configurations inside
64 ms. Complete-pipeline compute medians were modest, but P99 interference was
pass-sensitive and did not sustain one universal <=5% envelope. Deadline
success therefore depends primarily on path and lead time, with workload and
run-to-run compute tails as secondary constraints.

## WU9 input profile

The future real-workload observation unit should carry both measured paths:

- Use Pipeline A as the normal pageable warm-reservoir experiment.
- Use Pipeline B as the urgent path for tight leads.
- Focus chunk experiments on 32-64 MiB without selecting a final size.
- Retain aggregate controls at 64, 128, and 256 MiB.
- Use depth 2 as the minimum and depth 3 as the primary multi-block candidate;
  keep depth 4 only as a measured tail-control option.
- Seed leads at A/B 32/16 ms for 64 MiB, 64/32 ms for 128 MiB, and 64/64 ms
  for 256 MiB.
- Expect isolated single-thread pageable-copy P50 costs of about 2.54, 5.46,
  and 11.45 ms at 64/128/256 MiB, plus contention tails.
- Expect H2D around 2.67 ms per 64 MiB, 5.33 ms per 128 MiB, and 1.34-1.45 ms
  per 32 MiB block on this machine.
- Preserve NVMe tails as the dominant absolute supply stage and do not assume
  the WU6 5% compute band remains stable under the full system.

WU9 should observe real llama.cpp/GGUF block demand and compute windows against
this profile. WU8 does not implement that observation or any scheduler policy.

## Limitations

Results apply to this machine, dataset, device state, Windows storage stack,
CUDA/NVIDIA stack, and synthetic WU6 workloads. Unbuffered IO bypasses the
filesystem cache but not controller DRAM, firmware behavior, or other platform
caching. Discovered PCI topology does not prove CPU-direct versus chipset
routing. The host CPU-time clock is coarse. Low-rate telemetry cannot attribute
individual microsecond tails to instantaneous clocks, power, or temperature.
The 100-sample refined cells support observed 99%, not population guarantees or
99.9%; only the 1,000-sample B64 cell supports an observed P99.9 statistic.

The independent passes establish real tail variability, not a single universal
compute-interference envelope. Synthetic ALU, memory-bound, and FP16 GEMM are
controls rather than inference. The exact production lead, chunk size, depth,
and path must be decided only after real-workload observation. WU8 stops here.
