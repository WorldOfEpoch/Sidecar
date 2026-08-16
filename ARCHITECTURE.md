# Sidecar architecture

This document separates the laboratory that exists today from the runtime
architecture the research may justify later.

## Implemented laboratory infrastructure

M0 provides machine identity/topology discovery, a forensic SQLite schema,
versioned trace recording, host-memory experiments, CUDA transfer and overlap
experiments, NVMe experiments, integrated pipeline measurements, GGUF
inspection, and llama.cpp observer qualification. It can measure and project
candidate staging paths; it does not schedule production inference memory.

## Planned runtime architecture

### Memory hierarchy

```text
GPU VRAM          fastest, smallest execution tier
    ↑
pinned staging    persistent DMA-capable transfer tier
    ↑
system RAM        pageable warm reservoir
    ↑
NVMe              fast local backing tier
    ↑
model storage     canonical model bytes
```

### Normal path

`NVMe → pageable warm reservoir → pinned staging → VRAM`

This path favors reuse and a bounded pinned footprint.

### Urgent path

`NVMe → pinned staging → VRAM`

The shorter path may reduce host copies when a deadline is tight, but its
storage and pinning behavior must remain safe and empirically justified.

### Deadline scheduling

A block has a required-use time. A future scheduler would work backward from
that deadline through measured P95/P99 stage times, queue state, and safety
margins to decide when storage read and transfer must begin.

### Residency

Hot blocks should remain in VRAM when their reuse value exceeds the opportunity
cost. Streaming must not reload blocks unnecessarily. Residency decisions need
real demand observations, not only static file layout.

### Dense models

Substantially oversized dense models may be limited by recurring weight bytes
per token because most layers execute every token. WU10 will test the boundary.

### MoE models

Sparse expert activation may permit a resident expert cache plus staged cold
experts, but only if routing becomes known early enough for the measured memory
highway to meet first-use deadlines.

### Hardware adaptation

A future adaptation layer may learn H2D/D2H latency, tail distributions,
jitter, deadline success, contention sensitivity, and machine-specific
placement performance. Persistent arena regions may have session-local
performance profiles.

Planned performance states might include `UNTESTED`, `LEARNING`, `PREFERRED`,
`NORMAL`, `DEGRADED`, `RETEST`, and `QUARANTINED`. A slow region is not thereby
defective: performance and health are separate dimensions.

### Guardian

A future Sidecar Guardian may correlate Windows WHEA events, NVMe health,
CUDA/NVML errors, PCIe state, verification failures, storage errors, and
performance drift, then recommend external diagnostics. Active diagnosis is
not implemented.

