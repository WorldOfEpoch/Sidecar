# CUDA Compute + Transfer Overlap Physics

## Status and scope

This document defines Sidecar M0 Work Unit 6 (WU6). WU6 measures how much
host-to-VRAM or VRAM-to-host traffic can execute beside useful GPU work without
materially extending the compute critical path. It is a physics laboratory, not
a scheduler or a claim about inference.

WU6 does not implement residency, prefetch, eviction policy, NVMe measurement,
CUPTI tracing, llama.cpp hooks, model inference, or M1 behavior. Flight Recorder
per-sample tracing is disabled. The WU3 LIGHT observer gate remains unchanged;
observer cost in real llama.cpp inference is still not measured.

## Workloads

`SYNTHETIC_ALU` is a deterministic dependent FMA chain. The calibrated iteration
count changes duration, the recurring loop performs no global-memory traffic,
and each thread writes one observable result. This is a favorable compute-heavy
control; it is not inference.

`MEMORY_BOUND` performs coalesced reads and writes over two disjoint 256 MiB
device buffers. Its working set is larger than the RTX 3090 caches. Repeated
passes alternate source and destination, duration changes through pass count,
and the output is checked against the deterministic pass count. Its buffers do
not overlap the DMA buffers.

`FP16_GEMM` uses `cublasGemmEx` with a stable 2048 x 2048 x 2048 shape. A and B
are `CUDA_R_16F`; C is `CUDA_R_32F`; accumulation is
`CUBLAS_COMPUTE_32F`. The handle uses `CUBLAS_DEFAULT_MATH`, the algorithm is
`CUBLAS_GEMM_DEFAULT`, and duration changes through back-to-back repetition
count. Sidecar does not force deprecated tensor math mode or deprecated numbered
algorithms. The handle, matrices, stream binding, initialization, and warmup are
outside measured C0/Cc intervals. Every stored profile includes shape, types,
math mode, algorithm, repetition count, and cuBLAS version. This is representative
mixed-precision GEMM, not LLM inference.

## Calibration

Each workload is calibrated independently for nominal windows of 0.10, 0.25,
0.50, 1, 2, 4, 8, 16, 32, and 64 ms. Iteration, pass, or GEMM repetition counts
are integer controls, so nominal targets are not treated as exact. The measured
C0 distribution is authoritative. Profiles store mean, median, standard
deviation, minimum, maximum, P50, P90, P95, and P99 together with validation and
status. A workload that cannot reach a short target retains its actual measured
duration rather than inventing precision.

## Memory backends and transfer matrix

Persistent `cudaHostAlloc` memory is the primary DMA backend. Allocation and
release occur outside timing. H2D primary sizes are 1, 2, 16, 64, 96, 128, 256,
and 512 MiB. A 1024 MiB extreme control is restricted to a deliberately short
negative control and the 32/64 ms windows. D2H uses the reduced 2, 64, 128, and
512 MiB set.

Registered pinned memory is a selected-cell cross-check at approximately 2, 64,
128, and 512 MiB. Pageable memory is a separate small control, normally 4 and
64 MiB. Neither control is folded into the primary pinned Safe Transfer
Envelope.

## Streams, events, and pending common gate

The native provider creates four nonblocking streams before measurement:

- `GATE`
- `COMPUTE`
- `TRANSFER`
- `JOIN`

The default stream is not used inside a measured topology. `GATE_RELEASE` is
created with `cudaEventDisableTiming`; it exists only as a dependency. Branch
and overall timing markers use timing-enabled events only where a timestamp is
required.

For each matched sample, `GATE` runs a timed host callback that blocks only that
dedicated stream, records the overall timing start immediately after the
callback, then records `GATE_RELEASE`. The event has therefore already been
recorded and remains pending when `COMPUTE` and `TRANSFER` register
`cudaStreamWaitEvent`. Both workloads are submitted behind those waits. The gate
delay is excluded from the overall device interval. The callback design avoids
using an SM merely to hold the gate, which an observer-control trial showed can
change GPU clock state and bias a short compute workload.

The delay is adaptive: at least 2 ms and otherwise four times the observed P99
host submission duration. A monotonic host timestamp measures the actual gate
delay on every sample; target and actual delay are stored. A 250 us margin is required.
If host submission plus margin is not complete before the measured release, the sample is
`INVALID_GATE`, preserved, and excluded from aggregates.

## Matched topology and timing definitions

C0, T0, and concurrent samples use the same gate, two branch streams, event
placement, and join topology. Inactive branches still wait on the same gate and
record start/end markers; only their workload is absent. This matched design is
preferred to subtracting a convenient instrumentation constant.

Definitions:

- `C0`: isolated compute branch duration, `C_START` to `C_END`.
- `T0`: isolated transfer branch duration, `T_START` to `T_END`.
- `Cc`: compute branch duration during concurrency.
- `Tc`: transfer branch duration during concurrency.
- `M_device_primary`: overall start to `JOIN_END` after JOIN waits for both
  branch-end events.
- `M_device_crosscheck`: maximum of overall-start-to-compute-end and
  overall-start-to-transfer-end.
- `M_host`: independent host-side bracket after the overall start becomes
  observable and until JOIN completes.

Primary/cross-check disagreement is stored; it is not silently replaced. All
allocation, initialization, warmup, deterministic verification, and telemetry
reads are outside the measured interval.

## Instrumentation-control experiment

Before an authoritative matrix, the `instrumentation` command compares:

1. minimal same-stream workload timing;
2. pending common gate plus dependency topology without JOIN;
3. the complete WU6 gate, branch timing, JOIN, device-cross-check, and host
   bracket topology.

Representative nominal windows are 0.1, 1, 4, and 16 ms and representative
transfers are 2, 64, 128, and 512 MiB for selected workloads. Raw distributions
are stored. Thirty-one control blocks rotate the A/B/C submission order and
calculate median paired A-to-B and B-to-C bias so clock drift is not confused
with topology cost. A-to-B explicitly reports the effect of introducing the
pending common gate and dependency boundary. B-to-C is the qualification gate
for the added branch events, JOIN, cross-check, and host bracket. This is the
structurally valid authority comparison because C0, T0, and concurrent samples
all use the complete C topology; A is retained as a diagnostic control and is
never used as an unmatched performance baseline.

Sidecar currently classifies a B-to-C difference as material when its
absolute relative change exceeds 2% and its absolute median change exceeds
5 us. A material delta must also have a direction resolved at p < 0.05 by an
exact two-sided paired sign test before it is attributed to observer bias.
Material but statistically unresolved deltas are retained as
`MEASUREMENT_SENSITIVE`. This compound gate avoids calling noise or
sub-event-resolution changes material while still protecting the 1/2/5% fit
bands. A resolved material result is
`INSTRUMENTATION_BIAS` and stops the matrix when the affected isolated interval
is at least 0.5 ms; it is not corrected by subtraction. A material difference
on a shorter interval is preserved as `MEASUREMENT_SENSITIVE` and cannot
produce a trusted envelope. Material A-to-B changes remain visible in their raw
distributions and bias fields but do not reject the matched experiment: the
gate is mandatory and identically present in every authoritative arm.

CUDA events have finite resolution. Nominal windows at or below 0.5 ms are reported
as `MEASUREMENT_SENSITIVE`, even when otherwise valid.

## Pairing and drift control

Each configuration uses an interleaved C0, T0, concurrent block. Every raw
sample stores a monotonically increasing `baseline_block_id`, deterministic
ordering seed, target/actual gate data, and its local C0/T0 references. Long
experiments therefore do not reuse a single beginning-of-run baseline.

## Pageable negative control

Pageable memory is not admitted to the pending-gate WU6 topology. On this
Windows/CUDA stack, the attempted limited 4/64 MiB control demonstrated that
`cudaMemcpyAsync` from pageable memory can block the submitting host while the
copy is queued behind the still-pending release gate. The two-cell control did
not complete after more than six minutes and was interrupted without committed
database rows. Sidecar now returns `SKIPPED_UNSUPPORTED` before allocation for
such a request instead of risking that submission deadlock. WU5's isolated
pageable measurements remain the evidence for the capacity-tier/staging
behavior. Pageable data never enters a pinned Safe Transfer Envelope.

## Metrics

Raw metrics use the following definitions:

```text
critical_path_delta = M - max(C0, T0)
critical_path_added = max(0, critical_path_delta)
compute_slowdown = (Cc - C0) / C0
transfer_slowdown = (Tc - T0) / T0
overlap_efficiency_raw = (C0 + T0 - M) / min(C0, T0)
compute_path_delta = M - C0
compute_path_added = max(0, compute_path_delta)
compute_path_added_percent = 100 * compute_path_added / C0
hidden_fraction_raw = 1 - ((M - C0) / T0)
compute_retention = C0 / Cc
```

Normalized overlap efficiency and hidden fraction are clamped to [0, 1], but
raw values below zero or above one remain stored. Every valid sample records
whether compute-path extension is at most 1%, 2%, and 5%. Refined cells store
fit rates and distributions of absolute/percentage added time, compute and
transfer slowdown, makespan, hidden fraction, and primary/cross-check delta.
P99 is authoritative only with at least 100 valid samples.

## Coarse coverage, pruning, and refinement

The coarse phase uses moderate repetitions over broad workload/window/size
coverage. If measured T0 exceeds C0 by more than four times, three negative
controls establish the relationship and the cell becomes
`OUTSIDE_FULL_HIDE_REGION`; this is analytical pruning, not failure.

Automated refinement selects measured sizes where neighboring 2% fit
classification changes, P95 extension approaches the 1/2/5% bands, compute
slowdown rises, or transfer slowdown rises. Selected cells are rerun with 101
samples and a stored refinement reason. No unmeasured size is interpolated.

## Safe Transfer Envelope

For a machine, workload, measured nominal window, direction, host backend, and
tolerance, the Safe Transfer Envelope is the largest measured size whose P99
compute-path extension is within 1%, 2%, or 5%. Only cells with at least 100
valid samples can produce an envelope; the final H2D authority profile below
uses the explicitly selected `REFINE` phase. Confidence is
`MEASUREMENT_SENSITIVE` for nominal windows below 0.5 ms and `TRUSTED`
otherwise. This is an empirical profile, not scheduler policy, block-size
policy, or pinned-arena sizing policy.

## Telemetry and causal limits

NVML telemetry is read before and after a cell, never per micro-sample. Stored
state includes temperature, graphics/memory clocks, power, power limit, and PCIe
generation/width when available. WU6 can correlate large state changes with
results but does not claim causality from coarse telemetry. No CUPTI, profiler,
high-rate polling, or Flight Recorder event stream runs in authoritative loops.

## Persistence and CLI

Migration 006 raises `user_version` to 6 without recreating the database. It
adds `compute_workload_profiles`, `cuda_overlap_configurations`,
`cuda_overlap_benchmarks`, `cuda_overlap_samples`,
`cuda_overlap_fit_profiles`, and `instrumentation_control_benchmarks`. Every
sample retains local baselines, branch and makespan times, raw/normalized
metrics, gate validity, fit flags, status, native error, and message. Foreign
keys remain enabled and WU1-WU5 rows are preserved.

Commands:

```text
sidecar-lab cuda overlap info
sidecar-lab cuda overlap calibrate
sidecar-lab cuda overlap instrumentation
sidecar-lab cuda overlap plan --dry-run
sidecar-lab cuda overlap run
sidecar-lab cuda overlap matrix [--coarse|--refine]
sidecar-lab cuda overlap report
sidecar-lab cuda overlap validate
```

Filters include `--compute`, `--direction`, `--transfer-size`/`--sizes`,
`--compute-window`, `--backend`, `--repetitions`, `--device`, `--database`,
`--no-persist`, and `--json`. CUDA-disabled executable commands return the
structured status `SKIPPED_UNSUPPORTED`; unrelated Sidecar commands remain
available.

## Authoritative result status

Measurements were performed on 2026-08-15 on machine
`56ca8ca732b5006b04908e253be5ab77e689c67692e957509c230243f5b00961`, an
RTX 3090 24 GiB host using CUDA Toolkit 12.8 and NVIDIA driver 591.86. No
subagents, profiler, CUPTI, Flight Recorder sample tracing, compilation, tests,
source scans, or unrelated benchmarks ran concurrently with authoritative
commands.

Implementation and methodology revisions are preserved in git. The primary
implementation commit is `44c677cdb08fa7c126b6a62c13d1a8d3fda7ae72`;
observer-control hardening culminated in
`811b1436c15b77af3cf4b6d26a100465f3f1d0b6`; direction-only and pageable
safety fixes are `1db99e65e4952930bbadbb8b04262476512a8454` and
`d2f7e611a36bd112adfbfb18d9807d3122c96f31`. Every authoritative session names
an exact committed revision. Failed/superseded sessions 20-23 and 32 remain in
the database and are not used as authority.

### Observer qualification and calibration

Session 24 passed 144 rotated, paired A/B/C controls with zero rejected B-to-C
cells: 127 controls were `SUCCESS` and 17 were
`MEASUREMENT_SENSITIVE`. The complete topology's median path overhead across
controls was 99.632 us (71.552-265.696 us control-median range). Its maximum
absolute transfer-duration bias was 0.287%. The largest compute delta was
3.855%, but its paired direction was statistically unresolved and is retained
as measurement-sensitive. No correction or subtraction was applied.

Session 25 validated all 30 compute profiles. Hardware granularity prevented
some nominal short windows: the minimum measured GEMM was approximately
259 us, and the 256 MiB memory-bound workload's minimum was approximately
700 us. Those nominal targets are not presented as achieved durations.

WU6 isolated H2D T0 medians were 23.49, 25.30, 25.33, and 25.34 GB/s for
2/64/128/512 MiB. The 64-512 MiB results agree with WU5 asynchronous HostAlloc
within 0.1%. The 2 MiB value was 2.9% lower than WU5 and reproduced at
88.960 us over 101 samples (P99 90.368 us); it remains a saturation-knee and
instrumentation-sensitive point. WU6 always uses its own local matched T0 and
does not mix the WU5 baseline into overlap metrics.

Primary and cross-check makespans were close but not identical across 9,696
selected high-repetition samples: absolute relative disagreement was 0.131%
median, 0.693% P95, 1.976% P99, and 4.402% maximum. The corresponding absolute
P99 was 504.829 us. Both values remain stored; the conservative primary JOIN
makespan is not replaced. All 19,282 persisted WU6 samples from complete
sessions had valid gates; there were zero `INVALID_GATE` samples.

### Coverage and authoritative H2D envelope

Session 28 completed the broad 369-cell coarse matrix. It produced 190
`SUCCESS`, 23 `MEASUREMENT_SENSITIVE`, 89
`OUTSIDE_FULL_HIDE_REGION`, 46 `NOISY`, and 21 `BASELINE_DRIFT` cells. The last
three categories cannot produce an authority envelope. Sessions 29-35 supplied
101-sample HostAlloc boundary populations in both directions; sessions 36-37
were registered-memory controls. Sessions 38-40 independently repeated key H2D
boundaries. Session 41 then persisted the automated H2D selection as 36 coarse
plus 36 true `REFINE` cells, all successful, with 3,960/3,960 valid gates.

The final H2D profile uses session 41's 101-sample `REFINE` cells. At nominal
32 ms, no tested workload/size qualified at P99 <=1%, <=2%, or <=5%. At nominal
64 ms, no tested cell qualified at <=1% or <=2%. The measured <=5% envelopes
were:

| Workload | Direction | Window | P99 <=1% | P99 <=2% | P99 <=5% |
|---|---:|---:|---:|---:|---:|
| SYNTHETIC_ALU | H2D | 64 ms | none | none | 256 MiB at 4.976% |
| MEMORY_BOUND | H2D | 64 ms | none | none | 256 MiB at 4.787% |
| FP16_GEMM | H2D | 64 ms | none | none | 256 MiB at 4.653% |

These are largest satisfying measured points, not interpolation or a monotonic
claim. Some smaller points have worse tails. All 16 ms high-repetition cells
missed 5%; 512 MiB was a decisive negative control, with P99 added cost near
48-57% at 16 ms and 6.5-11.1% in session 41 at 64 ms. Interference generally
began rising in the 128-256 MiB region and increased sharply by 512 MiB.

The independent repeats explain why earlier session-local 2% envelopes are not
the final profile. ALU 64/128/256 MiB changed from 1.448/1.585/1.795% P99 in
session 29 to 2.276/2.627/5.208% in session 38. GEMM 96/128 MiB changed from
1.893/1.792% in session 31 to 8.247/4.350% in session 40. These rows remain
preserved, but strict 2% H2D fit was not cross-session reproducible.

At 64 ms in the final refinement, median compute slowdowns remained small, but
tails were workload- and size-dependent. At 256/512 MiB respectively:

| Workload | Median compute slowdown | P99 compute slowdown |
|---|---:|---:|
| SYNTHETIC_ALU | 0.30% / 0.89% | 4.66% / 6.36% |
| MEMORY_BOUND | 1.40% / 3.11% | 4.65% / 7.44% |
| FP16_GEMM | 1.15% / 2.52% | 4.24% / 10.60% |

Transfer P99 slowdown in the same 64 ms refinement reached 4.55% for ALU,
7.32% for memory-bound compute, and 3.15% for GEMM across measured sizes.
Median raw hidden fraction was approximately 97-135% for ALU, 65-97% for
memory-bound, and 78-92% for GEMM; values above 100% are retained raw noise or
speedup observations while normalized values clamp to 100%.

### Direction and memory-backend controls

The reduced D2H populations show that direction matters rather than providing
a universal advantage. At 64 ms, ALU's 2% envelope was 64 MiB and memory-bound
compute's was 128 MiB; GEMM had no 2% D2H envelope. At 5%, ALU and memory-bound
each reached the measured 512 MiB point, and GEMM reached 512 MiB, but GEMM's
64/128 MiB P99 costs were anomalously worse (7.83/8.61%) than its 512 MiB point
(2.43%). The raw non-monotonic observations are preserved.

Registered-memory T0 was effectively identical to HostAlloc at selected 32 ms
H2D cells: 64-512 MiB medians differed by less than about 0.2%. Overlap tails
were mixed rather than consistently better: registered GEMM was similar at
64 MiB and worse at 128/512 MiB, while registered memory-bound results varied
in both directions. This does not justify replacing persistent HostAlloc as the
primary staging backend.

The attempted 4/64 MiB pageable GEMM control did not complete after more than
six minutes because pageable submission blocked behind the pending gate. It was
interrupted with no committed partial rows. The command is now a structured
`SKIPPED_UNSUPPORTED`; together with WU5 it reinforces the architecture:
large pageable warm capacity plus persistent pinned DMA staging.

### Telemetry, database, and decision

Selected authority and repeat cells observed 57-70 C, 1860-1995 MHz graphics
clock, 9501 MHz memory clock, approximately 302-388 W, and stable PCIe Gen4 x16.
Repeat classification shifts coincided with different C0 calibration and power
states (for example GEMM C0 approximately 64.4 to 61.5 ms), but low-rate
before/after telemetry cannot assign causality. Thermal, clock, and power state
remain material interpretation limits.

Migration 006 is active (`user_version=6`) with zero foreign-key violations.
At result capture the WU6 tables contained 244 compute profiles, 569
configurations/benchmarks, 19,282 raw samples, 20 fit profiles, and 720
instrumentation-control rows; all 312 WU5 configurations/benchmarks and 57,222
WU5 raw samples remained present.

WU6 hypothesis decision: **WORKLOAD DEPENDENT**. A meaningful 256 MiB measured
H2D point can fit within a 5% P99 compute-path extension behind approximately
64 ms of ALU, memory-bound, or GEMM work on this machine. Median interference
is often much smaller, but no tested H2D point produced a reproducible <=2%
P99 envelope across the final refinement and independent repeats. WU6 therefore
supports useful overlap as a bounded empirical capability, not as a universal
or strict-tail guarantee.
