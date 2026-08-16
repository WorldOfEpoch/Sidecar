# Sidecar M0 overview: WU1–WU9

M0 is the measurement laboratory for Sidecar. It establishes the real behavior
of one Windows/CUDA/NVMe system before a scheduler or production model-memory
runtime is built.

## WU1 — Project and forensic database

- **Purpose:** establish reproducible versioning, persistence, and test/build
  foundations.
- **Major implementation:** C++20/CMake project, `sidecar-lab`, pinned SQLite,
  migrations, sessions, raw-sample-oriented schema.
- **Major result:** a durable evidence substrate with newer-schema rejection.
- **Architecture change:** measurements became revision- and machine-addressable.
- **Known limitation:** infrastructure, not performance physics.

## WU2 — Hardware identity and topology

- **Purpose:** identify the physical host without hashing upgradeable state.
- **Major implementation:** Machine Identity V1, Windows CPU/RAM/SMBIOS/GPU/
  storage discovery, optional CUDA/NVML, JSON, idempotent persistence.
- **Major result:** stable host identity and explicitly qualified discovered
  topology.
- **Architecture change:** later evidence can be scoped to a canonical machine.
- **Known limitation:** Windows device topology cannot always prove CPU-direct
  versus chipset routing. See [HARDWARE-DISCOVERY.md](HARDWARE-DISCOVERY.md).

## WU3 — Flight Recorder

- **Purpose:** capture causal events with bounded observer cost.
- **Major implementation:** versioned binary format, cache-isolated SPSC rings,
  collector, recovery/validation, and overhead tests.
- **Major result:** a forensic trace path whose drops and timing effects are
  explicit.
- **Architecture change:** low-overhead observation is available but never
  assumed free.
- **Known limitation:** instrumentation must be requalified per workload. See
  [FLIGHT-RECORDER.md](FLIGHT-RECORDER.md) and
  [TRACE-FORMAT-V1.md](TRACE-FORMAT-V1.md).

## WU4 — Host-memory lifecycle

- **Purpose:** measure pageable, allocated-pinned, and registered-pinned memory.
- **Major implementation:** guarded planning, allocation/touch/register timing,
  churn, persistent arena prototype, and raw persistence.
- **Major result:** repeated large pinned lifecycles are expensive; a large
  pageable warm pool plus a small persistent pinned arena is supported.
- **Architecture change:** pinned memory became a bounded staging resource, not
  the whole warm reservoir.
- **Known limitation:** channel topology and OS pressure remain machine/state
  dependent. See [HOST-MEMORY-LIFECYCLE.md](HOST-MEMORY-LIFECYCLE.md).

## WU5 — CUDA transfer physics

- **Purpose:** measure H2D/D2H by size, host backend, batching, and direction.
- **Major implementation:** isolated host/API/device timing, validation, warmup,
  sustained/link-state and bidirectional controls.
- **Major result:** pinned transfers reached the 25–26 GB/s class; H2D reached
  about 95% peak near 2 MiB; bidirectional copies did not scale aggregate
  throughput usefully on this setup.
- **Architecture change:** transfer planning gained measured size/backend
  envelopes.
- **Known limitation:** values apply first to the RTX 3090 platform. See
  [CUDA-TRANSFER-PHYSICS.md](CUDA-TRANSFER-PHYSICS.md).

## WU6 — Compute-transfer overlap

- **Purpose:** test whether transfers hide behind representative compute.
- **Major implementation:** calibrated ALU, memory-bound, FP16 GEMM, matched
  gates, event timing, instrumentation control, and safe-envelope analysis.
- **Major result:** `WORKLOAD DEPENDENT`; a 256 MiB H2D transfer met an
  approximately 5% P99 extension condition at about 64 ms compute windows, but
  no reproducible 1% or 2% envelope was established.
- **Architecture change:** overlap became a workload-specific budget.
- **Known limitation:** synthetic workloads are not causal model graphs. See
  [CUDA-COMPUTE-TRANSFER-OVERLAP.md](CUDA-COMPUTE-TRANSFER-OVERLAP.md).

## WU7 — NVMe storage physics

- **Purpose:** measure unbuffered local-storage throughput, tails, queues, and
  deadlines.
- **Major implementation:** physical-target provenance, verified dataset,
  aligned I/O, QD/block sweeps, health gates, and latency populations.
- **Major result:** Samsung 990 PRO peaked at 7.135 GB/s; 256 KiB/QD8 delivered
  7.119 GB/s with 0.294/0.305/0.335/0.405 ms P50/P95/P99/P99.9. Excess queueing
  worsened tails without useful gain.
- **Architecture change:** storage staging gained a tail-aware operating region.
- **Known limitation:** one drive, filesystem, dataset, and host state. See
  [NVME-STORAGE-PHYSICS.md](NVME-STORAGE-PHYSICS.md).

## WU8 — Integrated memory highway

- **Purpose:** measure full NVMe→host→VRAM deadlines and contention.
- **Major implementation:** normal Pipeline A, urgent Pipeline B, common
  timelines, isolated baselines, depth/chunk sweeps, health and verification.
- **Major result:** `PIPELINE DEPENDENT`; practical measured A/B leads were
  32/16 ms (64 MiB), 64/32 ms (128 MiB), and 64/64 ms (256 MiB). Pipeline B
  achieved 1000/1000 at 64 MiB/16 ms. The useful experimental region was
  32–64 MiB at depth 2–3.
- **Architecture change:** two staging paths and empirical deadlines replaced
  isolated-bandwidth assumptions.
- **Known limitation:** constants are experimental, not production defaults.
  See [MEMORY-HIGHWAY-PIPELINE-PHYSICS.md](MEMORY-HIGHWAY-PIPELINE-PHYSICS.md).

## WU9 — llama.cpp/GGUF observation

- **Purpose:** connect the laboratory to real model structure and execution.
- **Major implementation:** pinned llama.cpp integration, GGUF tensors/offsets,
  model/storage provenance, deterministic baseline, callback observers, demand
  projection, overhead gate, and WU8 shadow analysis.
- **Major result:** `MODEL DEPENDENT`. The 3.21B Q5_K_M dense control fit fully
  in VRAM, reused all 255 persistent tensors per token, and exposed no new dense
  blocks after load. Mean decode was about 4.600 ms and observer modes exceeded
  the authority threshold (3.26–5.57% median overhead).
- **Architecture change:** WU10 was redirected toward deliberately oversized
  dense and MoE workloads.
- **Known limitation:** the authoritative model was fully resident, so WU9 did
  not test the >VRAM hypothesis. See
  [LLAMA-GGUF-REAL-WORKLOAD-OBSERVATION.md](LLAMA-GGUF-REAL-WORKLOAD-OBSERVATION.md).

## Next boundary

WU10 is **Oversized Real-Model Physics** and has not started. See the repository
[roadmap](../../ROADMAP.md). No scheduler, prefetcher, or production paging
engine is claimed by M0.

