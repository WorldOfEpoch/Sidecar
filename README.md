# Sidecar

Sidecar is experimental systems research into deadline-driven AI model-memory
staging. It asks whether models larger than GPU VRAM can use VRAM as the fastest
tier of a hierarchy spanning NVMe, pageable RAM, persistent pinned staging, and
GPU memory—without pretending that software can defeat storage or PCIe physics.

The repository currently contains **SIDECAR M0: The Laboratory**, through Work
Unit 9 under specification `M0-FROZEN-1`. It is measurement infrastructure, not
a production inference runtime.

## Why Sidecar?

Conventional execution often treats available VRAM as the practical model-size
boundary or uses coarse CPU/GPU offload. Sidecar is investigating a different
question: can known or predicted model blocks be staged early enough to arrive
in VRAM before their execution deadlines?

```text
model storage / NVMe
        ↓
pageable system RAM
        ↓
persistent pinned DMA staging
        ↓
GPU VRAM
        ↓
compute
```

The intended future runtime would overlap storage I/O, host-memory movement,
PCIe transfers, and GPU compute where the hardware allows it. M0 exists to
measure those limits before building the scheduler.

## Research philosophy

Sidecar preserves raw measurements, tail percentiles, negative results,
superseded sessions, observer conditions, machine identity, and revision
provenance. Asynchronous APIs are not treated as proof of physical overlap, and
unmeasured interpolation is not treated as truth. See [RESEARCH.md](RESEARCH.md).

## Current status

| Work Unit | Focus | Status |
|---|---|---|
| WU1 | Project / forensic DB | Complete |
| WU2 | Hardware identity / topology | Complete |
| WU3 | Flight Recorder | Complete |
| WU4 | Host memory | Complete |
| WU5 | CUDA transfer physics | Complete |
| WU6 | Compute-transfer overlap | Complete |
| WU7 | NVMe storage | Complete |
| WU8 | Full memory highway | Complete |
| WU9 | llama.cpp / GGUF observation | Complete |
| WU10 | Oversized real-model physics | **Next—not started** |
| WU11 | Targeted CUPTI / final M0 gate | Planned |

Schema/user version is 9. The public snapshot was prepared from private WU9
HEAD `d425bd102b570ef3009898e405e662c826e553b5`; the private development history
was intentionally not published because it contained machine-local paths.

## Major findings

All figures below apply first to the measured machine and methodology. Detailed
conditions and limitations are in [docs/m0/M0-OVERVIEW.md](docs/m0/M0-OVERVIEW.md).

- **Host memory (WU4):** repeated large pinned allocation lifecycles are
  expensive. The measurements support a large pageable warm pool plus a small,
  persistent pinned DMA arena that amortizes pinning cost.
- **PCIe (WU5):** representative pinned peaks were 25.316 GB/s H2D with
  `cudaHostAlloc`, 25.307 GB/s H2D with registered memory, 26.341 GB/s D2H with
  `cudaHostAlloc`, and 26.350 GB/s D2H with registered memory. H2D reached about
  95% of peak around 2 MiB. Simultaneous H2D+D2H did not yield useful aggregate
  concurrency scaling on this RTX 3090 system.
- **Compute + transfer (WU6):** decision `WORKLOAD DEPENDENT`. At roughly 64 ms
  compute windows, a 256 MiB H2D transfer met an approximately 5% P99
  compute-path-extension condition for synthetic ALU, memory-bound, and FP16
  GEMM workloads. No reproducible 1% or 2% final envelope was established.
- **NVMe (WU7):** the Samsung 990 PRO reached 7.135 GB/s sequential peak at
  32 MiB/QD8. The best measured throughput/tail balance was 256 KiB/QD8 at
  7.119 GB/s, with 0.294/0.305/0.335/0.405 ms P50/P95/P99/P99.9. Large queues
  could worsen tails sharply without useful bandwidth gain.
- **Memory highway (WU8):** decision `PIPELINE DEPENDENT`. Smallest practical
  measured lead times for normal Pipeline A versus urgent Pipeline B were
  32/16 ms at 64 MiB, 64/32 ms at 128 MiB, and 64/64 ms at 256 MiB. Pipeline B
  at 64 MiB/16 ms achieved 1000/1000 observed deadline successes. The useful
  experimental region was 32–64 MiB chunks at depth 2–3—not production constants.
- **Real llama.cpp/GGUF (WU9):** decision `MODEL DEPENDENT`. A Llama 3.2 3B
  Instruct Q5_K_M control (3.21B parameters, 2.314 GB persistent weights,
  255 tensors) fit fully in RTX 3090 VRAM. All 255 persistent tensors were reused
  per token and no new dense weight blocks appeared after residency. Mean decode
  was about 4.600 ms (P50 4.651, P95 5.208, P99 5.878 ms), or about 217.44 tok/s.
  The real observer gate failed: median decode overhead was about 3.26% for
  `LIGHT_LAYER`, 3.68% with recorder, and 5.57% for `LIGHT_NODE`.

## What WU9 changed

The 3B model was a useful fully resident control, but it did not exercise the
core >VRAM hypothesis. WU10 therefore changed from a primarily CUPTI-focused
step to **oversized real-model physics**: one oversized dense model and one
oversized MoE model. Targeted CUPTI validation is now likely WU11. No WU10 code
or performance claim is included here.

## Test platform

- Intel Core Ultra 9 285K
- 96 GiB RAM
- NVIDIA GeForce RTX 3090, 24 GiB VRAM
- Samsung SSD 990 PRO 2TB NVMe
- CUDA 12.8 measurement environment
- Machine identity hash:
  `56ca8ca732b5006b04908e253be5ab77e689c67692e957509c230243f5b00961`

Raw hardware serials and unique device identifiers are not published. Results
are machine-specific empirical envelopes, not universal hardware constants.

## Building

Requirements: Windows 11, CMake 3.25+, Visual Studio 2022 with C++20 support,
and—when enabled—the CUDA toolkit. The initial configure downloads the pinned
official SQLite amalgamation and verifies its published SHA3-256 digest.

CUDA-enabled Release:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DSIDECAR_ENABLE_CUDA=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA-enabled Debug uses the same configured tree:

```powershell
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

CUDA-disabled Release:

```powershell
cmake -S . -B build-nocuda -G "Visual Studio 17 2022" -A x64 -DSIDECAR_ENABLE_CUDA=OFF
cmake --build build-nocuda --config Release
ctest --test-dir build-nocuda -C Release --output-on-failure
```

WU9 execution additionally uses official `ggml-org/llama.cpp` pinned at commit
`0d9ceae1e38291035605613ab41a8f5e693d6fcd`. Clone and build that dependency
outside this repository, then configure Sidecar with
`-DSIDECAR_ENABLE_LLAMA=ON`, `-DSIDECAR_LLAMA_CPP_ROOT=<checkout>`, and
`-DSIDECAR_LLAMA_CPP_BUILD_DIR=<build-directory>`. Model weights are never
included here.

## Running `sidecar-lab`

`sidecar-lab` is the M0 laboratory executable. These read-only or low-impact
information commands are implemented and covered by CLI tests:

```powershell
.\build\Release\sidecar-lab.exe version
.\build\Release\sidecar-lab.exe info --no-persist
.\build\Release\sidecar-lab.exe topology --no-persist
.\build\Release\sidecar-lab.exe memory info
.\build\Release\sidecar-lab.exe storage info --json
.\build\Release\sidecar-lab.exe pipeline info --json
.\build\Release\sidecar-lab.exe llama info --json
```

Benchmark commands can allocate large buffers, stress storage, or exercise the
GPU. Read the relevant methodology document and use its dry-run/planning mode
before launching an experiment.

## Model provenance without model publication

The authoritative dense WU9 control was Llama 3.2 3B Instruct Q5_K_M, SHA-256
`05fc42664a9311c427413f9bf2077bd5ee7d59d6a5a034d54fc738f93976d065`, executed
from the Samsung 990 PRO. The structural-only MoE candidate was Qwen3-Coder-Next
Q4_K_M, approximately 48.19 GiB, SHA-256
`30e51a7cb1cf1333b9e298b90b4c7790fe2572d8736b002482a0ac96328a2ffb`.
Neither model is in Git.

## Repository map

- `include/sidecar`, `src`: C++20 laboratory implementation
- `tools/sidecar_lab`: command-line application
- `tests`: unit, integration, CLI, and stress-oriented tests
- `schemas`: SQLite schema and migrations through version 9
- `docs/m0`: experimental methods and results
- `cmake`: generated-version/schema configuration

Start with [STATUS.md](STATUS.md), [ROADMAP.md](ROADMAP.md),
[ARCHITECTURE.md](ARCHITECTURE.md), [RESEARCH.md](RESEARCH.md), and the
[M0 overview](docs/m0/M0-OVERVIEW.md).

## What Sidecar is not yet

Sidecar is not yet a production inference backend, a drop-in llama.cpp paging
engine, a transparent “100B on 24 GB” runtime, a finished predictive scheduler,
a replacement for VRAM, or a guarantee that any oversized model will execute
efficiently. It has not solved VRAM limits.

## Experimental status and license

This software performs low-level memory, CUDA, and storage experiments. Review
plans before running stress commands and preserve health/error evidence.

No open-source license has been selected. Public visibility does not grant
permission to use, copy, modify, or redistribute the software; see [LICENSE](LICENSE).

