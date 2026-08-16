# Contributing to Sidecar

Sidecar is research-heavy C++20/CMake/CUDA systems work. Contributions should
make the evidence easier to reproduce, audit, or falsify.

Performance changes require measurements. Do not remove negative results,
replace empirical thresholds with assumptions, or silently change benchmark
methodology. A methodology change must explain its timing domain, synchronization,
warmup, validation, observer mode, and comparability with earlier sessions.

New experiments should preserve:

- canonical machine identity and relevant topology;
- Sidecar and external dependency revisions;
- build/runtime configuration and device state;
- raw samples and percentile definitions;
- observer/instrumentation conditions;
- failed, invalidated, or superseded evidence.

Keep model weights, generated databases, traces, datasets, dependency trees, and
compiled artifacts out of Git. Use dry-run/planning modes before hardware stress.
Run the applicable Debug and Release builds and CTest suites, and document any
test that requires CUDA hardware, external llama.cpp, or a local GGUF.

Public visibility is not an open-source license grant. Contributions cannot be
accepted under an implied license; consult the repository owner before investing
substantial work.

