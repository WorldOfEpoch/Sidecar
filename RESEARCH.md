# Research method

Sidecar M0 is an empirical systems laboratory. Its primary output is trustworthy
evidence about this machine's storage, host-memory, PCIe, CUDA, and workload
behavior—not a favorable benchmark headline.

## Evidence rules

- Preserve raw samples where practical and retain machine identity, source
  revision, configuration, warmup state, observer mode, and timing domain.
- Report distributions including P50, P95, P99, and P99.9 where sample counts
  support them; a mean alone is insufficient for deadline work.
- Preserve negative results and failed gates. They constrain architecture.
- Retain superseded methodology sessions with their qualification status rather
  than rewriting them as authoritative evidence.
- Measure observer effect explicitly and keep authority separate from forensic
  modes when instrumentation changes timing materially.
- Treat asynchronous API completion as a software event, not proof of physical
  overlap. Use device timing, common launch gates, and matched controls.
- Treat each machine's results as an empirical envelope. Do not promote an
  interpolation or hardware capability bit into a measured guarantee.
- Monitor hardware health and PCIe state during stress campaigns.
- Run authoritative campaigns under documented quiet-mode constraints and keep
  load/model timing separate from warmed prompt/decode timing.

## Corrections increase confidence

M0 intentionally found measurement defects: mixed timing domains, observer
overhead, synchronization mistakes, load-time contamination, storage-location
confounding, and pipeline verification work inside measured intervals. Corrected
and superseded sessions are part of the scientific record. Finding a defect and
re-running the experiment is stronger evidence than hiding it.

## Interpretation discipline

Measured, calculated, estimated, and projected values remain distinct. A fast
isolated stage does not prove a complete pipeline meets a deadline. A successful
fully resident model does not prove an oversized model can stream. A capability
is claimed only at the layer the evidence actually tested.

