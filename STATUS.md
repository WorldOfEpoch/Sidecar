# Sidecar status

## Snapshot

- Specification: `M0-FROZEN-1`
- Database schema/user version: 9
- Completed: M0 Work Units 1–9
- Next: WU10, **Oversized Real-Model Physics**
- Not started: WU10 implementation or authoritative measurements
- Likely following step: WU11, targeted CUPTI forensic validation and the M0
  final gate

This public repository is a sanitized, fresh-history snapshot derived from
private WU9 HEAD `d425bd102b570ef3009898e405e662c826e553b5`. The source, tests, migrations,
and safe research documents are included. Private development history, raw
SQLite evidence, traces, generated reports, models, datasets, dependency trees,
and compiled output are excluded.

## M0 completion matrix

| Work unit | Laboratory | Result state |
|---|---|---|
| WU1 | Project/database foundation | Complete |
| WU2 | Canonical hardware identity and discovered topology | Complete |
| WU3 | Versioned multi-ring Flight Recorder | Complete |
| WU4 | Host-memory lifecycle | Complete |
| WU5 | CUDA host-device transfer physics | Complete |
| WU6 | Compute-transfer overlap | Complete; workload-dependent |
| WU7 | NVMe storage physics | Complete |
| WU8 | Integrated memory-highway pipeline | Complete; pipeline-dependent |
| WU9 | llama.cpp/GGUF real-workload observation | Complete; model-dependent |

## Implemented now

- Versioned SQLite evidence schema and migrations
- Stable machine identity, Windows hardware discovery, and topology reporting
- Versioned binary trace format and lock-free SPSC recorder rings
- Host allocation/registration lifecycle and persistent-arena experiments
- CUDA transfer and compute-overlap laboratories
- Windows unbuffered NVMe and deadline experiments
- Integrated NVMe→RAM/pinned→VRAM pipeline experiments
- GGUF inspection, pinned llama.cpp observation, storage provenance, observer
  qualification, demand projection, and WU8 shadow analysis
- Deterministic JSON/reporting paths and automated tests

## Not implemented

- Production model paging or offload backend
- Predictive/deadline scheduler and prefetcher
- Automatic VRAM residency manager or expert cache
- Oversized dense/MoE execution campaign
- Guardian health diagnosis or hardware self-tuning runtime
- M1 or any later runtime milestone

## Known scientific boundary

WU9's dense model fit entirely in VRAM. It validated the observation machinery
and exposed measurable observer overhead, but it could not validate Sidecar's
central >VRAM hypothesis. That is why WU10 changed direction.

