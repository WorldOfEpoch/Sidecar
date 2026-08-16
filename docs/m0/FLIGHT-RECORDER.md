# M0 Flight Recorder

The Work Unit 3 recorder assigns one existing lock-free SPSC ring to each registered
producer. Registration, names, metadata, ring allocation, and file creation happen before
the producer hot path. The hot path is record construction plus `tryPush()`; it has no lock,
blocking wait, allocation, filesystem access, SQLite, JSON, or checksum work.

The default ring capacity is 65,536 records per producer and the default collector batch is
4,096 records. Both are configurable and are benchmarked. The collector uses a bounded spin,
then yield, then 100-microsecond sleep when idle. It drains using `tryPop()` only;
`sizeApproximate()` remains diagnostic and is never synchronization or backpressure.

The collector reuses a preallocated drain vector, writes one producer-specific event chunk
per drain batch, periodically flushes completed chunks, and adds a footer on clean shutdown.
Producer overflow is nonblocking and increments the owning ring's drop counter. Accounting
uses `attempted = persisted + dropped` after producers quiesce.

`OperationIdAllocator` supplies monotonically increasing 64-bit session-local IDs. Forensic
records store these directly. Compact records deliberately carry 32-bit session-local IDs;
callers must select forensic format or end the compact session before overflow rather than
silently truncate or reuse IDs.

The observer laboratory compares no-trace baseline loops with 32- and 64-byte active
collectors across saturation, light compute, medium compute, and 1/2/4/8 producers. It stores
the raw signed overhead ratio, throughput, producer and collector CPU time, records, drops,
high-water, bytes, sampled `tryPush` P50/P95/P99 (one sample per 1,024 events), timestamp
cost, copy cost, capacity sweep, and batch sweep. SQLite writes occur only after each trace
has completed.

Copy cost is measured as repeated 4,096-record block copies with a compiler barrier and a
data-dependent checksum; setup allocation is excluded. Batch-sweep throughput and disk rate
use the full start-through-finalization wall time, while observer overhead uses producer-loop
wall time so collector shutdown is not charged to the instrumented producer.

CLI commands are:

```text
sidecar-lab trace selftest [--record-size 32|64|both]
sidecar-lab trace generate-test <file> [--record-size 32|64]
sidecar-lab trace inspect <file> [--json]
sidecar-lab trace summary <file> [--json]
sidecar-lab trace validate <file>
sidecar-lab trace benchmark [--json] [--quick] [--events N]
                            [--repetitions N] [--database path] [--output path]
```

Generated traces default below `traces/` and are excluded from Git. The binary format does
not claim a global multi-producer order, CUDA correlation timeline, or empirical storage/GPU
topology; those belong to later work units.
