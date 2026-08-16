# SIDECAR RUNTIME

## Master Codex Project Specification

You are creating a new software project named:

# **SIDECAR**

Sidecar is an experimental high-performance AI inference memory runtime for consumer and workstation PCs.

Its long-term purpose is to make GPU VRAM behave as the fastest tier of a much larger intelligent model-memory hierarchy consisting of:

```text
GPU VRAM
    ↑
Pinned DMA staging memory
    ↑
System RAM
    ↑
NVMe
    ↑
Model storage
```

Sidecar must intelligently measure, manage, stage, cache, prefetch, and eventually predict model-memory movement so models larger than GPU VRAM can run as efficiently as physically possible.

The project must be built scientifically.

Do not assume an optimization works.

Measure it.

Benchmark it.

Compare it against a baseline.

Keep the optimization only if empirical results justify it.

---

# 0. INITIAL DEVELOPMENT MACHINE

Initial target machine:

```text
GPU:
NVIDIA GeForce RTX 3090
24 GB VRAM

System RAM:
96 GB currently

Future RAM:
up to 256 GB

Storage:
NVMe SSD

Primary OS:
Windows 11

Secondary OS:
Linux

Initial AI backend:
llama.cpp / ggml / GGUF

Initial GPU backend:
NVIDIA CUDA
```

Do not hardcode behavior specifically for this machine.

Detect capabilities at runtime.

This computer is the first development and validation platform.

---

# 1. PROJECT VISION

A normal local-AI runtime tends to think in terms of:

```text
Model fits GPU VRAM
or
Model must be offloaded
```

Sidecar should eventually think instead in terms of:

```text
WHAT DATA IS NEEDED?

WHEN IS IT NEEDED?

WHERE SHOULD IT LIVE RIGHT NOW?

WHEN MUST ITS TRANSFER BEGIN?

WHAT CAN BE EVICTED?

WHAT SHOULD REMAIN HOT?

WHAT CAN BE PREFETCHED WHILE THE GPU IS COMPUTING?
```

The ultimate Sidecar runtime will treat VRAM as an extremely fast active workspace rather than the only place model data may live.

Conceptual hierarchy:

```text
TIER 0
GPU VRAM
HOT / ACTIVE

TIER 1
Pinned DMA staging buffers
READY / IN TRANSIT

TIER 2
System RAM
WARM MODEL MEMORY

TIER 3
NVMe optimized cache
COLD MODEL MEMORY

TIER 4
Original model storage
ARCHIVAL SOURCE
```

Future physical hardware may introduce:

```text
Dedicated Sidecar Memory Device
128 GB
256 GB
512 GB
1 TB+
```

The software architecture created now must make that possible later without rewriting the entire scheduler.

---

# 2. CRITICAL DEVELOPMENT RULE

## DO NOT BUILD THE SIDECAR SCHEDULER YET.

The first project milestone is:

# SIDECAR M0: THE LABORATORY

M0 has been formally frozen.

Specification identifier:

```text
SIDECAR_SPEC_VERSION = M0-FROZEN-1
```

Record this specification version in every benchmark session.

M0 exists to discover the actual physical behavior of the computer before scheduling algorithms are written.

The purpose of M0 is:

> Measure the machine accurately enough that future Sidecar schedulers can operate from empirical physics instead of assumptions.

Do not allow feature creep into M0.

Do not implement predictive scheduling in M0.

Do not implement Oracle in M0.

Do not replace llama.cpp memory management in M0.

Do not build a GUI before the laboratory works.

---

# 3. TECHNOLOGY STACK

Use:

```text
C++20
CUDA C++
CMake
SQLite3
CUPTI where supported
llama.cpp / ggml integration
```

Optional small support utilities may use Python if genuinely useful for:

```text
report generation
plotting
offline analysis
```

However:

> No Python code may sit in a high-frequency inference or memory-transfer hot path.

Performance-critical runtime functionality belongs in native code.

---

# 4. REPOSITORY STRUCTURE

Create a clean repository approximately following:

```text
sidecar/
│
├── CMakeLists.txt
├── README.md
├── LICENSE
├── VERSION
│
├── docs/
│   ├── architecture/
│   ├── m0/
│   ├── benchmarks/
│   └── roadmap/
│
├── cmake/
│
├── third_party/
│
├── include/
│   └── sidecar/
│
├── src/
│   ├── core/
│   ├── hardware/
│   ├── database/
│   ├── telemetry/
│   ├── trace/
│   ├── cuda/
│   ├── storage/
│   ├── model/
│   └── llama/
│
├── tools/
│   ├── sidecar_lab/
│   ├── topology/
│   ├── trace_reader/
│   └── report/
│
├── benchmarks/
│   ├── memory/
│   ├── storage/
│   ├── cuda/
│   ├── contention/
│   └── inference/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── stress/
│
├── schemas/
│   └── sidecar.sql
│
├── traces/
│
├── reports/
│
└── data/
```

Adjust where technically useful, but preserve strong separation of responsibilities.

---

# 5. VERSION CONTROL AND REPRODUCIBILITY

Every benchmark session must record:

```text
Sidecar git commit
Sidecar spec version

llama.cpp git commit

CUDA runtime version
CUDA driver version
NVIDIA driver version

OS version/build

machine hardware profile
```

Never silently benchmark against a changing llama.cpp checkout.

Use a known commit and record it.

If llama.cpp must be patched for observation hooks, keep Sidecar-specific modifications minimal, clearly documented, and easy to rebase.

---

# 6. SIDECAR M0 PRIMARY COMPONENTS

M0 consists of:

```text
1. Hardware / topology discovery

2. Forensic SQLite database

3. Lock-free Flight Recorder

4. CUDA timing system

5. CUPTI forensic integration

6. RAM and pinned-memory laboratory

7. NVMe laboratory

8. CUDA transfer laboratory

9. Compute/transfer overlap laboratory

10. Full contention laboratory

11. llama.cpp graph observer

12. GGUF/model inventory scanner

13. Standardized inference benchmark

14. Reporting

15. M0 validation gate
```

Implement these carefully.

---

# 7. FORENSIC DATABASE

Use SQLite.

All measurements must be associated with a benchmark session.

Create at minimum:

```text
hardware_profiles
gpu_devices
storage_devices
benchmark_sessions
models
memory_benchmarks
storage_microbenchmarks
transfer_benchmarks
contention_benchmarks
transfer_compute_grid
inference_runs
trace_files
software_versions
```

The design should support future schema migrations.

Enable foreign keys.

Use transactions for benchmark result insertion.

Do not perform SQLite writes from performance-critical hot paths.

Collect results first, persist afterward.

---

# 8. BENCHMARK SESSION

Create:

```sql
CREATE TABLE benchmark_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_hash TEXT NOT NULL,
    sidecar_spec_version TEXT NOT NULL,
    sidecar_git_commit TEXT NOT NULL,
    llama_cpp_git_commit TEXT,
    cuda_runtime_version INTEGER,
    nvidia_driver_version TEXT,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    trace_mode TEXT NOT NULL,
    trace_overhead_pct REAL,
    gpu_temperature_start_c REAL,
    gpu_power_limit_watts REAL,
    notes TEXT
);
```

Expand where appropriate.

Every benchmark result references:

```text
session_id
```

---

# 9. HARDWARE PROFILE

Record:

## CPU

```text
model
physical cores
logical cores
NUMA topology
```

## RAM

```text
installed bytes
available bytes
speed if discoverable
channel configuration if discoverable
```

## Motherboard

```text
manufacturer
model
BIOS/UEFI revision
```

## GPU

```text
GPU UUID
model
VRAM
PCI domain
PCI bus
PCI device
negotiated PCIe generation
maximum PCIe generation
negotiated lane width
maximum lane width
CUDA compute capability
asyncEngineCount
relevant device attributes
```

## Storage

```text
device model
firmware
capacity
filesystem
bus
PCI path
PCI generation
lane width
NUMA relationship
```

Never fail the entire profiler because a particular field cannot be discovered.

Unknown fields should be recorded as unknown/null.

---

# 10. HARDWARE TOPOLOGY

Attempt to map:

```text
CPU
 │
 ├── GPU
 │
 ├── NVMe 0
 │
 ├── NVMe 1
 │
 └── chipset / bridges
```

On Windows capture available:

```text
ACPI paths
PCI location paths
device relationships
```

On Linux use appropriate:

```text
sysfs
lspci data where available
NUMA information
```

Do not assume software topology discovery perfectly identifies CPU-direct versus chipset-routed devices.

Therefore Sidecar must maintain two concepts:

```text
DISCOVERED TOPOLOGY

and

EMPIRICAL CONTENTION TOPOLOGY
```

Empirical benchmark behavior is authoritative when topology information is ambiguous.

---

# 11. FLIGHT RECORDER

Build a highly efficient Sidecar Flight Recorder.

Purpose:

Record performance-critical events without materially altering them.

Primary architecture:

```text
Producer Thread 0 → SPSC Ring 0
Producer Thread 1 → SPSC Ring 1
Producer Thread 2 → SPSC Ring 2
Producer Thread 3 → SPSC Ring 3
                       │
                       ▼
              Background Collector
                       │
                       ▼
                  Binary Trace
```

No shared multi-producer queue is required for the hot path.

---

# 12. SPSC REQUIREMENTS

Use:

```text
Single Producer
Single Consumer
64-bit sequence counters
power-of-two capacity
release/acquire publication
```

Strictly separate the following control states onto independent 64-byte cache-line-aligned structures:

```text
Producer-private write sequence

Producer → Consumer published sequence

Consumer-private read sequence

Consumer → Producer consumed sequence

Dropped-event counter
```

Do not place producer-written and consumer-written synchronization state on the same cache line.

Requirements:

```text
No mutex

No blocking

No filesystem access

No SQLite

No heap allocation

No string formatting

No logging framework

No JSON
```

inside:

```text
record()
```

---

# 13. TRACE RECORD

Support empirical comparison between:

```text
32-byte records

and

64-byte records
```

Do not assume one is superior.

Benchmark observer overhead.

A record must include enough information to correlate:

```text
host timestamp

operation_id

parent_operation_id

model/tensor/node identifier

payload size

event type

source memory tier

destination memory tier

producer/thread ID

auxiliary data
```

Possible events:

```text
HOST_SCHED_DECISION
HOST_PREFETCH_SUBMIT

HOST_IO_SUBMIT
HOST_IO_COMPLETE

HOST_CUDA_LAUNCH

GPU_TRANSFER_START
GPU_TRANSFER_END

GPU_KERNEL_START
GPU_KERNEL_END

STALL_BEGIN
STALL_RESOLVED

DEADLINE_EXCEEDED

CACHE_HIT
CACHE_MISS
```

M0 will not necessarily generate all scheduler-related event classes, but reserve extensibility.

---

# 14. TRACE OVERRUN HANDLING

Never silently overwrite unread records.

If a producer reaches capacity:

```text
increment dropped-event counter
return failure
continue safely
```

Do not block inference.

At the end of the benchmark report:

```text
dropped_events
ring_high_water_mark
trace_bytes
```

Any benchmark containing dropped trace events must be visibly marked as having incomplete tracing.

---

# 15. OBSERVER EFFECT TESTING

M0 must benchmark itself.

Compare:

```text
TRACE_NONE

TRACE_LIGHT_SPSC

TRACE_CUDA_EVENTS

TRACE_FORENSIC_CUPTI

TRACE_CROSSCHECK
```

Calculate observer overhead.

Initial target for Light SPSC mode:

```text
<1% preferred

<2% acceptable
```

This is an engineering target, not a claim.

Forensic CUPTI mode may exceed this and is not required to be suitable for normal production execution.

Its purpose is forensic validation.

---

# 16. HOST TIMELINE

Use a stable high-resolution monotonic host clock.

Record host activities such as:

```text
scheduler decision
I/O submission
I/O completion
CUDA submission
model graph observation
```

Do not use wall-clock time for performance intervals.

Wall time may still be recorded separately for human-readable session metadata.

---

# 17. CUDA LIGHT TIMING

Use:

```text
cudaEvent_t
non-blocking streams
```

for lightweight GPU timing.

Do not infer exact device-side execution start from CPU launch timestamps.

CUDA events provide device-stream timing relationships and durations.

Use them accordingly.

---

# 18. CUPTI FORENSIC MODE

Integrate CUPTI Activity API.

Purpose:

Correlate:

```text
Sidecar host operation

CUDA API call

GPU transfer

GPU kernel

GPU timestamp

host timeline
```

Use CUPTI external correlation support where available.

Bind:

```text
Sidecar operation_id
```

to CUPTI's external correlation mechanism.

This makes it possible to reconstruct:

```text
Sidecar Operation #81244
        │
        ├── Host I/O
        ├── CUDA API submission
        ├── H2D transfer
        ├── GPU activity
        └── kernel execution
```

Do not make CUPTI mandatory for normal low-overhead execution.

---

# 19. MEMORY TIERS

Define these conceptual identifiers from the beginning:

```text
NVME_RAW

COLD_PAGEABLE_RAM

WARM_MANAGED_RAM

LOCKED_HOT_RAM

PINNED_DMA_RING

VRAM_L1
```

M0 measures these.

M1+ will control them.

---

# 20. PINNED MEMORY LABORATORY

Benchmark:

```text
cudaHostAlloc

cudaFreeHost

cudaHostRegister

cudaHostUnregister

pageable memory allocations
```

Sweep allocation sizes approximately:

```text
64 MB
256 MB
512 MB
1 GB
2 GB
4 GB
8 GB
12 GB
16 GB
```

Gracefully stop if the system cannot safely allocate a requested size.

Never cause OS memory starvation for a benchmark.

Measure:

```text
allocation time
deallocation time
registration time
unregistration time

H2D throughput
D2H throughput

reuse performance

system impact
```

The key engineering question is:

> Is a persistent pinned DMA arena better than dynamically registering model memory?

Measure it rather than assuming.

---

# 21. NVME LABORATORY

Test storage request/block sizes approximately:

```text
64 KB
256 KB
1 MB
2 MB
4 MB
8 MB
16 MB
32 MB
64 MB
128 MB
```

Queue depths:

```text
QD1
QD2
QD4
QD8
QD16
QD32
QD64
QD128
```

Where supported.

Measure:

```text
throughput

IOPS

P50 latency
P90 latency
P95 latency
P99 latency
P99.9 latency

standard deviation / jitter

worst observed latency

CPU utilization
```

---

# 22. DEADLINE-COMPLIANT BANDWIDTH

Sidecar does not care only about maximum storage throughput.

Introduce the metric:

# DEADLINE-COMPLIANT BANDWIDTH

For selected deadline windows calculate:

```text
percentage of requests completing before deadline
usable bandwidth while satisfying deadline
```

Example concept:

```text
4 MB QD8
6.6 GB/s
99.5% under deadline

32 MB QD32
7.1 GB/s
91.8% under deadline
```

Future schedulers may prefer the first despite lower headline throughput.

---

# 23. TRANSFER BENCHMARKS

Measure:

```text
Pageable H2D

Pinned H2D

D2H

simultaneous H2D + D2H

multiple CUDA streams

persistent pinned-buffer reuse
```

Record:

```text
mean
median
min
max
stddev
P95
P99
```

Do not record only one timing sample.

Warm up before measuring.

---

# 24. COMPUTE WORKLOADS

Do not benchmark overlap using only one artificial kernel.

Implement at least:

## A. SYNTHETIC_ALU

Compute-heavy artificial workload.

## B. MEMORY_BOUND

GPU memory-intensive workload.

## C. FP16_GEMM

Representative tensor-core matrix multiplication workload.

## D. LLAMA_REPRESENTATIVE

A representative llama.cpp/ggml workload or trace when integration reaches that stage.

---

# 25. COMPUTE WINDOW CALIBRATION

Create compute windows around:

```text
0.1 ms
0.25 ms
0.5 ms
1 ms
2 ms
4 ms
8 ms
16 ms
32 ms
```

Calibration need not hit those durations perfectly.

Record actual measured duration.

---

# 26. COMMON LAUNCH GATE

When testing compute/transfer overlap, do not simply enqueue work sequentially and call it simultaneous.

Create a common CUDA dependency boundary using:

```text
cudaEvent_t
cudaStreamWaitEvent()
```

Separate streams:

```text
CONTROL

COMPUTE

TRANSFER
```

Conceptually:

```text
CONTROL
   │
 GATE
   │
 ┌─┴───────────────┐
 │                 │
 ▼                 ▼
COMPUTE          TRANSFER
STREAM            STREAM
```

Interpret this correctly:

> The gate makes both workloads eligible after the same dependency boundary.

Do not claim physical nanosecond-identical hardware start times.

Use forensic CUPTI runs to investigate real hardware scheduling.

---

# 27. OVERLAP PHYSICS

For every transfer-size / compute-window pair measure:

```text
C0 = isolated compute duration

T0 = isolated transfer duration

Cc = compute duration while concurrent

Tc = transfer duration while concurrent

M  = concurrent makespan
```

Calculate:

```text
critical_path_delta =
M - max(C0, T0)
```

Also:

```text
critical_path_added =
max(0, critical_path_delta)
```

Compute slowdown:

```text
Sc =
(Cc - C0) / C0
```

Transfer slowdown:

```text
St =
(Tc - T0) / T0
```

Raw overlap efficiency:

```text
Eraw =
(C0 + T0 - M) / min(C0, T0)
```

Preserve the raw result even if noise causes values outside 0..1.

For human-facing presentation only:

```text
Enormalized =
clamp(Eraw, 0, 1)
```

Never delete raw data.

---

# 28. DO NOT MISINTERPRET OVERLAP FAILURE

Poor overlap does not automatically prove:

```text
"memory controller saturation"
```

Potential causes include:

```text
PCIe bandwidth
VRAM bandwidth
copy engine contention
GPU memory subsystem
power limits
GPU clock changes
stream scheduling
kernel scheduling
storage contention
shared root complex
other resources
```

M0 reports measurements.

Future analysis determines causes.

---

# 29. TRANSFER/COMPUTE GRID

Store fields including:

```text
session_id
gpu_id
compute_type

transfer_size

repetitions

compute_solo_us
transfer_solo_us

compute_concurrent_us
transfer_concurrent_us

concurrent_makespan_us

compute_slowdown_pct
transfer_slowdown_pct

overlap_efficiency_raw
overlap_efficiency_normalized

critical_path_delta_us
critical_path_added_us

critical_path_added_p95_us
critical_path_added_p99_us

fits_without_delay
```

`fits_without_delay` must use a documented tolerance.

Do not imply literal mathematical zero.

Example:

```text
acceptable critical path increase <= configured threshold
```

---

# 30. CONTENTION MATRIX

Benchmark at minimum:

```text
A. COMPUTE ONLY

B. H2D ONLY

C. D2H ONLY

D. H2D + COMPUTE

E. D2H + COMPUTE

F. H2D + D2H + COMPUTE

G. NVMe → RAM

H. NVMe → RAM + H2D

I. NVMe → RAM + H2D + COMPUTE

J. NVMe → RAM + H2D + D2H + COMPUTE
```

Add other combinations if needed for scientific completeness.

Record:

```text
compute slowdown
H2D slowdown
D2H slowdown
NVMe slowdown

GPU clocks
GPU power
GPU temperature

CPU load
```

Storage must be nullable for tests not involving storage.

---

# 31. GGUF MODEL INVENTORY

Create a model scanner.

Scan configurable directories.

Identify:

```text
filename
path
size
hash
format
architecture where detectable
quantization
parameter estimate where detectable
layer count
MoE status where detectable
expert count where detectable
```

Model identification must be persistent.

A file hash should prevent benchmark data from being incorrectly applied to a changed model with the same filename.

---

# 32. LLAMA.CPP M0 INTEGRATION

M0 is observation only.

Do not replace:

```text
mmap
ggml allocator
CUDA allocator
```

unless absolutely required for observation, and document any unavoidable changes.

Use:

```text
ggml_backend_sched_set_eval_callback
```

as an observation point where practical.

M0 objective:

```text
observe graph execution nodes

assign stable Sidecar IDs

associate observed nodes with model metadata

emit Flight Recorder events

correlate host observation with CUDA activity
```

Do not assume the eval callback provides the entire tensor inventory.

Use model/GGUF parsing for metadata.

Use graph hooks for execution observation.

Combine them.

---

# 33. LLAMA.CPP BASELINES

Benchmark at least relevant existing modes such as:

```text
default mmap

no mmap

mlock where supported/useful

normal GPU offload configuration
```

Do not claim Sidecar improvement in M0.

M0 establishes baseline truth.

---

# 34. STANDARDIZED INFERENCE BENCHMARK

Create repeatable inference tests.

Record:

```text
model hash

prompt

prompt token count

generation token count

sampling configuration

context size
```

Measure:

```text
model load time

time to first token

prompt tokens/sec

generation tokens/sec

peak VRAM

peak system RAM

GPU utilization

CPU utilization

power

temperature
```

Use deterministic/reproducible settings where practical.

---

# 35. REPETITION AND STATISTICS

Each benchmark type must define:

```text
warmup runs

measured repetitions

outlier policy
```

Store:

```text
raw samples where practical

mean

median

minimum

maximum

standard deviation

P95

P99
```

Tail latency matters to Sidecar.

A rare late transfer can stall the GPU.

---

# 36. REPORTING

M0 should produce:

```text
SQLite database

JSON export

CSV summaries

human-readable report
```

Optional HTML reporting is encouraged.

Reports should clearly distinguish:

```text
MEASURED

CALCULATED

ESTIMATED
```

Never present estimates as measurements.

---

# 37. COMMAND-LINE APPLICATION

Create:

```text
sidecar-lab
```

Suggested commands:

```text
sidecar-lab info

sidecar-lab topology

sidecar-lab calibrate

sidecar-lab memory

sidecar-lab storage

sidecar-lab cuda

sidecar-lab contention

sidecar-lab models scan

sidecar-lab llama benchmark

sidecar-lab trace inspect

sidecar-lab report

sidecar-lab m0 validate

sidecar-lab m0 all
```

Exact syntax may evolve.

---

# 38. SAFETY

Never intentionally make the workstation unusable.

Detect available RAM before large allocations.

Maintain a configurable minimum system-memory reserve.

Example configuration concept:

```text
minimum_free_system_ram_gb = 16
```

Do not allocate pinned memory until the OS becomes unstable.

Do not disable the Windows page file.

Do not require unsafe driver modifications.

Do not alter GPU firmware.

Do not modify BIOS.

Do not require kernel patches for M0.

---

# 39. ERROR HANDLING

All CUDA APIs must be checked.

All CUPTI APIs must be checked.

All SQLite operations must be checked.

Hardware discovery must tolerate unavailable fields.

Benchmark failures should produce structured errors, not silent missing results.

Never let one unsupported test invalidate every other M0 benchmark.

---

# 40. TESTING

Implement unit and integration tests.

At minimum test:

```text
SPSC no-overrun

SPSC wraparound

SPSC overflow detection

SPSC producer/consumer stress

trace serialization

trace parsing

SQLite migration/setup

foreign keys

hardware profile hashing

model hashing

statistical calculations

overlap calculations

clamping while preserving raw values

CLI argument parsing

CUDA capability detection
```

Run sanitizers where appropriate on CPU code.

---

# 41. PERFORMANCE TEST OF THE FLIGHT RECORDER

Measure trace cost.

For record size:

```text
32 byte

64 byte
```

and various event rates.

Record:

```text
events/sec

CPU overhead

cache impact where measurable

ring occupancy

dropped events

benchmark slowdown
```

Use whichever payload configuration has the lowest acceptable observer effect.

Do not assume 32 bytes wins.

---

# 42. M0 COMPLETION GATE

M0 is complete only when all of the following pass.

## TOPOLOGY

```text
[ ] CPU topology captured
[ ] GPU topology captured
[ ] storage topology captured
[ ] NUMA relationships captured where available
```

## DATABASE

```text
[ ] session-oriented SQLite schema operational
[ ] hardware profiles persistent
[ ] storage profiles persistent
[ ] GPU profiles persistent
[ ] software revisions captured
```

## MEMORY

```text
[ ] cudaHostAlloc sweep complete
[ ] cudaHostRegister sweep complete
[ ] allocation/free cost measured
[ ] register/unregister cost measured
[ ] pinned reuse measured
```

## STORAGE

```text
[ ] NVMe block-size sweep complete
[ ] queue-depth sweep complete
[ ] P50/P95/P99/P99.9 available
[ ] deadline-compliance measurement available
```

## CUDA

```text
[ ] H2D measured
[ ] D2H measured
[ ] bidirectional transfer measured
[ ] copy/compute overlap measured
[ ] multiple compute types measured
```

## CONTENTION

```text
[ ] tests A-J implemented
[ ] contention matrix populated
[ ] thermal and clock effects captured
```

## FLIGHT RECORDER

```text
[ ] SPSC operational
[ ] 64-bit sequence counters
[ ] cache-isolated control state
[ ] zero hot-path allocations
[ ] zero hot-path locks
[ ] dropped-event reporting
[ ] 32/64-byte payload benchmark
```

## TIMELINES

```text
[ ] CUDA light timing operational
[ ] CUPTI forensic mode operational where supported
[ ] Sidecar operation correlation operational
```

## LLAMA.CPP

```text
[ ] GGUF model inventory operational
[ ] graph observation operational
[ ] stable Sidecar node/tensor IDs
[ ] inference baseline operational
```

## OBSERVER EFFECT

```text
[ ] tracing overhead measured
[ ] Light SPSC target <2% or limitation documented
```

## REPORTING

```text
[ ] SQLite results
[ ] JSON export
[ ] CSV export
[ ] human-readable M0 report
```

Finally:

```text
sidecar-lab m0 validate
```

must produce:

```text
M0 PASS
```

or an explicit list of failed requirements.

---

# 43. WHAT HAPPENS AFTER M0

Do not implement these phases yet, but architect M0 so they remain possible.

---

# M1: STATIC SIDECAR MEMORY

Future objective:

Reserve a managed system-RAM pool.

Create explicit model block representation.

Control:

```text
RAM residency

Pinned staging

VRAM residency
```

Prove a model larger than VRAM can run under explicit Sidecar control.

No prediction.

---

# M2: ASYNCHRONOUS SIDECAR

Add:

```text
double buffering

persistent pinned rings

asynchronous H2D

compute/transfer overlap

explicit CUDA streams
```

Goal:

Reduce GPU waiting relative to M1.

---

# M3: BACKWARD DEADLINE SCHEDULER

For every required block calculate:

```text
T_start =
T_need
-
T_transfer
-
T_preparation
-
T_margin
```

For multi-stage storage:

```text
NVMe → RAM
RAM → pinned
pinned → VRAM
```

calculate deadlines backward through the entire pipeline.

Physics limits performance.

The scheduler minimizes avoidable stalls.

Never promise zero stalls.

---

# M4: DEEP STORAGE STAGING

Enable:

```text
MODEL SIZE > AVAILABLE RAM
```

Pipeline:

```text
NVMe
 ↓
RAM window
 ↓
Pinned buffer
 ↓
VRAM
 ↓
GPU
```

Maintain a moving model window.

This is "staging for the staging."

---

# M5: MODEL LAB AUTOTUNER

Automatically test:

```text
block size

prefetch depth

pinned-window size

RAM-window size

transfer batching

transfer splitting

buffer counts
```

Record best configuration by:

```text
model hash
+
machine profile
```

---

# M6: ADVANCED MEMORY MANAGEMENT

Future features:

```text
KV cache tiering

multi-model warm residency

elastic model allocation

MoE expert residency

shared immutable weights

compression awareness

quantization-aware staging
```

---

# M7: SIDECAR ORACLE

Oracle is a separate optional predictor.

It predicts future demand.

It does not move memory directly.

It produces recommendations such as:

```text
KEEP HOT

PREFETCH

LIKELY EXPERT

LIKELY MODEL

DEMOTE
```

Begin in:

```text
SHADOW MODE
```

Record what Oracle predicted and what actually occurred.

Only permit active influence after empirical evidence demonstrates benefit.

---

# 44. FUTURE MULTI-MODEL ARCHITECTURE

Long term Sidecar should allow:

```text
General model

Coding model

Vision model

Voice model

Embedding model
```

to remain warm in a large RAM pool.

Only active working sets enter VRAM.

Goal:

Avoid repeated:

```text
NVMe → RAM → VRAM
```

model reloads.

Prefer:

```text
warm RAM → VRAM
```

when changing models.

---

# 45. FUTURE LARGE MODEL ARCHITECTURE

Example:

```text
180 GB model

24 GB VRAM

64 GB usable Sidecar RAM

remaining model on NVMe
```

Future pipeline:

```text
GPU:
active blocks

Pinned:
next-ready blocks

RAM:
future working window

NVMe:
later model blocks
```

All tiers operate concurrently where possible.

---

# 46. FUTURE PROVIDER ABSTRACTION

Eventually define:

```text
IMemoryProvider
```

Potential providers:

```text
VRAMProvider

PinnedRAMProvider

SystemRAMProvider

NVMeProvider

PhysicalSidecarProvider
```

M0 should not implement the future hardware provider.

But avoid design decisions that make it impossible.

---

# 47. FUTURE PHYSICAL SIDECAR

Potential future hardware:

```text
128 GB
256 GB
512 GB
1 TB+
```

with:

```text
custom FPGA/ASIC

multiple memory channels

DMA

cache/prefetch assist

compression/decompression

accelerator fabric adapters
```

Software measurements should eventually be able to simulate hypothetical physical Sidecar properties:

```text
capacity

bandwidth

latency
```

so hardware requirements are derived from evidence.

---

# 48. FUTURE MULTI-GPU

Eventually support:

```text
GPU 0 NVIDIA

GPU 1 NVIDIA

GPU 2 AMD

future accelerators
```

with independent memory domains.

Do not implement heterogeneous GPU scheduling in M0.

Do not let assumptions about one CUDA device infect generic core interfaces unnecessarily.

---

# 49. CORE SIDECAR ENGINEERING PHILOSOPHY

Every optimization must answer:

> Could this data have been moved while the GPU was doing something else?

If yes:

attempt to hide the transfer.

If no:

measure the unavoidable stall.

Another core metric is:

# COMPUTE PER BYTE MOVED

Maximize useful computation obtained for every byte that must traverse slower memory links.

---

# 50. NO-MAGIC RULE

Sidecar cannot make:

```text
DDR5 = GDDR6X

NVMe = RAM

PCIe = local VRAM bandwidth
```

Do not make false performance claims.

Sidecar seeks to make slower tiers less painful through:

```text
residency

caching

prefetch

deadline scheduling

reuse

overlap

batching

prediction
```

---

# 51. SUCCESS CRITERIA FOR THE OVERALL PROJECT

Future Sidecar success means:

```text
Run models larger than GPU VRAM

Run models larger than available RAM when needed

Reduce avoidable GPU stalls

Beat conventional/static offload on appropriate workloads

Keep multiple models warm

Switch models rapidly

Tier KV cache

Optimize MoE expert residency

Automatically discover best settings per machine/model

Remain stable under memory pressure
```

But none of these performance improvements should be claimed until benchmarked.

---

# 52. CODING STANDARDS

Use:

```text
RAII

strong ownership rules

minimal global state

explicit error propagation

modern C++20

clear interfaces

unit-testable components

thread-safe design
```

Performance-critical code must avoid:

```text
unnecessary allocation

locks in hot paths

string manipulation in hot paths

unbounded queues

hidden blocking

silent CUDA synchronization
```

Document every intentional synchronization point.

---

# 53. COMMENTS AND DOCUMENTATION

Document:

```text
why a synchronization exists

why an atomic ordering was selected

which thread owns each state variable

which API calls may block

which operations allocate memory

which operations are hot-path safe
```

For subtle concurrency code, comments should explain the invariant, not merely repeat the line of code.

---

# 54. DO NOT OVERENGINEER M0

Do not create:

```text
GUI dashboards

Oracle AI

complex plugin systems

multi-GPU scheduler

ROCm runtime

physical Sidecar drivers

production model server

network API
```

during M0.

Create clean seams/interfaces where useful.

Do not implement future functionality prematurely.

---

# 55. FIRST CODING TASK

Start now.

First:

1. Inspect the local development environment.

2. Confirm available:

```text
compiler
CMake
CUDA toolkit
CUDA-capable RTX 3090
SQLite
Git
```

3. Create the Sidecar repository.

4. Create:

```text
VERSION
README.md
docs/m0/M0-FROZEN-1.md
schemas/sidecar.sql
```

5. Implement the SQLite schema and migration/bootstrap layer.

6. Implement the SPSC Flight Recorder.

7. Add unit tests for the recorder before proceeding.

The recorder tests must verify:

```text
single producer/single consumer correctness

wraparound

queue full behavior

dropped-event accounting

sequence publication

concurrent stress

no torn records

32-byte configuration

64-byte configuration
```

8. Build and run tests.

9. Report actual build/test status.

Do not merely output proposed code.

Create the project files and get the initial project compiling.

---

# 56. INITIAL DELIVERABLE

The first completed Codex response/work unit should leave the repository with:

```text
Compiling CMake project

SQLite schema initialized

Sidecar versioning

hardware-profile scaffolding

benchmark-session scaffolding

working SPSC Flight Recorder

Flight Recorder unit tests

initial CLI executable

documentation for M0
```

The CLI should at minimum support:

```text
sidecar-lab version

sidecar-lab db init

sidecar-lab db status

sidecar-lab trace selftest
```

---

# 57. DEVELOPMENT LOOP

For every future M0 subsystem:

```text
IMPLEMENT
   ↓
UNIT TEST
   ↓
BENCHMARK
   ↓
STORE RAW RESULTS
   ↓
VERIFY
   ↓
DOCUMENT
   ↓
COMMIT-READY STATE
```

Do not advance because something "looks correct."

Advance because it is tested.

---

# 58. IMPORTANT FINAL DIRECTIVE

Sidecar is intended to become a serious systems-engineering project.

Treat benchmark integrity as more important than exciting-looking results.

If an assumption in this specification is wrong:

```text
measure it
document it
correct the implementation
preserve the evidence
```

Do not manipulate benchmark methodology to produce better numbers.

Negative results are useful.

If an optimization makes performance worse, record it.

If existing software already performs an operation optimally, record that too.

If hardware physics prevents an intended improvement, report it clearly.

The objective is not to prove Sidecar is fast.

The objective is to discover **how to make Sidecar as fast as the hardware physically permits**.

---

# BEGIN

Create the new **Sidecar** project now.

Implement only **SIDECAR M0: THE LABORATORY**.

Start with:

```text
repository/bootstrap
SQLite schema
benchmark-session architecture
SPSC Flight Recorder
Flight Recorder tests
initial CLI
```

Build everything locally.

Run tests.

Fix compile/runtime errors encountered.

Keep M0 scope frozen.

When this initial work unit is complete, provide:

```text
files created
architecture implemented
tests executed
test results
build results
known limitations
next M0 task
```

Do not begin M1 until the complete M0 execution gate passes.

