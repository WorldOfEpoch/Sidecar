# Sidecar roadmap

The roadmap follows evidence. Work-unit names describe research questions, not
promised product features.

## Completed M0 laboratories

WU1–WU3 established the repository, forensic database, canonical machine
identity, topology discovery, and low-overhead Flight Recorder. WU4–WU8 measured
host memory, PCIe transfer, compute overlap, NVMe, and the integrated memory
highway. WU9 connected those measurements to real GGUF/llama.cpp execution and
qualified the observer effect.

## Next: WU10 — Oversized Real-Model Physics

### Why WU10 changed

A fully resident 3B model does not require Sidecar streaming. Sidecar is aimed
primarily at models that exceed VRAM, so the next experiment must exercise that
condition directly rather than treating more tracing as the primary next step.

WU10 will investigate two deliberately oversized workloads. It has not started.

### A. Oversized dense model

Every dense layer generally participates in every token. WU10 must measure:

- total weight bytes, usable VRAM, and minimum nonresident bytes;
- bytes that must cross PCIe per token;
- compute time available to hide those transfers;
- the theoretical bandwidth floor versus the empirical WU8 pipeline limit.

Possible evidence-based conclusions include `VIABLE`, `SLIGHTLY_OVERSIZED_ONLY`,
`BANDWIDTH_LIMITED`, or `NOT_PRACTICAL`. No result is predetermined.

### B. Oversized MoE model

Sparse expert activation may offer a more favorable residency problem. The
current structural candidate is Qwen3-Coder-Next Q4_K_M: about 79.67B parameters,
48.19 GiB, and 512 possible experts. WU9 used it only for structural inspection
from a nonauthoritative storage location—not performance execution.

Future research must determine selected experts, when routing becomes causally
known, tensor sizes and offsets, hot/cold reuse, resident-cache opportunities,
required block staging, router-to-first-use time, and whether WU8 Pipeline B can
meet that interval. There is no performance claim yet.

## Likely WU11 — Targeted CUPTI and M0 final gate

Use CUPTI only where WU10 identifies unresolved timing or causality questions,
then close the M0 evidence gate. Scope may change in response to WU10 data.

## Later runtime milestones

- **M1:** static, explicitly configured Sidecar memory placement
- **M2:** asynchronous staging and residency mechanisms
- **M3:** backward deadline scheduling and prefetch decisions
- **M4+:** deeper storage staging, machine adaptation, multi-model/multi-GPU
  research, and reliability/Guardian concepts

These are planned directions, not implemented capabilities.

