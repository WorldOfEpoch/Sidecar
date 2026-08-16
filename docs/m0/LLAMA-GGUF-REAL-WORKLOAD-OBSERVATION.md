# Llama.cpp / GGUF Real-Workload Observation (M0 WU9)

## Scope and scientific boundary

WU9 observes real llama.cpp/GGUF execution and performs offline shadow
feasibility analysis against WU8 pipeline evidence. It does not move model
weights, implement a cache, prefetch live inference, patch llama.cpp, use
CUPTI, or implement a scheduler. Native llama.cpp partial offload is labelled
`LLAMA_NATIVE_PARTIAL_OFFLOAD_CONTROL`; it is not Sidecar streaming.

Hot callback data is accumulated in preallocated RAM. SQLite writes occur only
after measured inference. `LIGHT_LAYER_FLIGHT_RECORDER` is the deliberately
separate WU3 compact-ring comparison mode and may run its collector while the
inference region is measured.

## Pinned dependency

- Upstream: `https://github.com/ggml-org/llama.cpp.git`
- Checkout: `C:\SidecarDeps\llama.cpp`
- Commit: `0d9ceae1e38291035605613ab41a8f5e693d6fcd`
- Source patches: none
- Dependency checkout requirement: clean
- Build: MSVC 19.38, CUDA Toolkit 12.8.93, Release, GGML CUDA, sm_86
- Standalone build: `C:\SidecarDeps\llama.cpp\build-sidecar`
- CPU-only verification build: `C:\SidecarDeps\llama.cpp\build-sidecar-cpu`
- Baseline tool: the pinned `llama-bench.exe`

Sidecar links the pinned prebuilt `llama`, `ggml`, and `ggml-base` import
libraries. The required runtime DLLs are copied next to Sidecar executables.
`SIDECAR_ENABLE_LLAMA` is independent of `SIDECAR_ENABLE_CUDA`: a CUDA-disabled
Sidecar build retains pinned GGUF metadata/tensor inspection while GPU inference
commands return `SKIPPED_UNSUPPORTED`.

## Local model inventory and acquisition decision

Model acquisition was stopped before WU9 authority and is not benchmark
evidence. The Hugging Face gpt-oss transfer was cancelled cleanly. Final-path
resolution found that `%USERPROFILE%\.ollama` resolves to a relocated model
store on a USB disk. The original blobs remain untouched.

### Dense reference (selected)

- Source: `OLLAMA_LOCAL_BLOB`
- Ollama tag: `llama3.2:3b-instruct-q5_K_M`
- Ollama digest and full-file SHA-256:
  `05fc42664a9311c427413f9bf2077bd5ee7d59d6a5a034d54fc738f93976d065`
- Requested original path:
  `%USERPROFILE%\.ollama\models\blobs\sha256-05fc42664a9311c427413f9bf2077bd5ee7d59d6a5a034d54fc738f93976d065`
- Resolved original path: a redacted user directory on the same external `D:`
  model-store volume
- Original backing: `D:\`, `\\.\PhysicalDrive1`, WD easystore 2647, USB,
  external, not a Samsung 990 PRO
- Authoritative execution copy:
  `C:\SidecarModels\Llama-3.2-3B-Instruct-Q5_K_M-ollama-05fc4266.gguf`
- Execution backing: `C:\`, `\\.\PhysicalDrive0`, Samsung SSD 990 PRO 2TB,
  NVMe, not USB/external
- Copy relationship:
  `ONE_TIME_VERIFIED_COPY_FROM_OLLAMA_USB_BLOB_TO_SAMSUNG_990_PRO`
- Original/copy byte counts: 2,322,153,696 / 2,322,153,696
- Original/copy SHA-256: identical
- Bytes: 2,322,153,696
- GGUF: version 3, `llama`, Q5_K Medium
- Parameters: 3,212,749,888
- Layers/tensors: 28 / 255
- Pinned compatibility: model load, tensor validation, tokenization, prompt
  decode, deterministic generation, CUDA offload, and public callback PASS

This exact local model is not represented as the suggested SmolLM3 Hugging
Face revision. Reproducibility is by exact bytes and Ollama provenance, not by
name substitution.

### MoE structural candidate (not a primary inference reference)

- Source: `OLLAMA_LOCAL_BLOB`
- Ollama tag: `qwen3-coder-next:latest`
- Ollama digest and full-file SHA-256:
  `30e51a7cb1cf1333b9e298b90b4c7790fe2572d8736b002482a0ac96328a2ffb`
- Original path:
  `%USERPROFILE%\.ollama\models\blobs\sha256-30e51a7cb1cf1333b9e298b90b4c7790fe2572d8736b002482a0ac96328a2ffb`
- Bytes: 51,741,599,936
- Resolved backing: `D:\`, `\\.\PhysicalDrive1`, WD easystore 2647, USB
- Storage label: `STRUCTURAL_INSPECTION_ONLY_NONAUTHORITATIVE_STORAGE_LOCATION`
- Copy: none (the 48.19 GiB blob is not duplicated)
- GGUF: version 3, `qwen3next`, Q4_K Medium
- Parameters: 79,674,391,296
- Layers/experts/tensors: 48 / 512 / 843
- Pinned compatibility: metadata, tensor types, offsets, spans, and all file
  ranges accepted by the pinned ggml GGUF API
- Execution classification: structurally useful but unsuitable as a bounded
  fully GPU-resident reference on a 24 GiB RTX 3090; no authoritative MoE
  inference is claimed

No installed `gpt-oss:20b` was found, so no additional 12+ GB MoE artifact was
downloaded. A SmolLM3 GGUF completed before the reuse instruction and remains
outside source control under `C:\SidecarModels`; it is not the selected WU9
model and is not measured.

## Physical-storage provenance gate

Before any inference load, Sidecar opens the selected file read-only, resolves
the final kernel path through reparse points with `GetFinalPathNameByHandleW`,
resolves its volume, maps the volume with `IOCTL_STORAGE_GET_DEVICE_NUMBER`, and
queries the backing physical device with `IOCTL_STORAGE_QUERY_PROPERTY`. It
records volume/mount identity, physical-drive number/path, product model, bus
type, Samsung 990 PRO classification, USB/external classification, storage role,
and copy relationship.

Commands marked `--authoritative` fail closed unless the resolved execution
path is backed by the Samsung 990 PRO and is not USB. Model-load provenance is
included in every inference result and persisted with the benchmark. Model load
is recorded separately and never mixed into warmed prompt/decode timing.

## GGUF index and dictionaries

`InspectGguf` uses the pinned `gguf_init_from_file(..., no_alloc=true)` API.
For every tensor it records the stable integer ID, name, ggml type, dimensions,
payload bytes, absolute file offset, aligned file span, alignment, derived layer
and expert identifiers, role, and persistent-weight classification. All ranges
must remain within the file. WU9 currently supports one normal GGUF file;
split-GGUF range indexing is explicitly unsupported.

The observer builds a sorted tensor-name dictionary before context creation.
Callbacks use compact integer tensor/op IDs. There is no string construction,
container growth, mutex acquisition, SQLite, model-file I/O, or tensor-payload
copy in the light callback.

Block projection takes the union of exact tensor file spans at 32, 64, and
128 MiB. It reports useful tensor bytes, exact block IDs, projected bytes,
overfetch bytes, and overfetch ratio. Duplicate tensor demands are removed
before projection.

## Public callback semantics

WU9 uses only `llama_context_params.cb_eval` and the public
`ggml_backend_sched_eval_callback` contract.

- `ask == true`: the scheduler asks whether the node output must be
  materialized for observation.
- `CALLBACK_NOOP`, `LIGHT_LAYER`, and `LIGHT_NODE` return false. They do not
  request node contents or add a node-specific materialization barrier. The
  public scheduler callback path still synchronizes each computed backend
  split, so its measured overhead must not be described as free/asynchronous.
- `FORENSIC_SELECTED` returns true for one selected matrix-multiply node per
  evaluation and records the corresponding `ask == false` materialized call.
  That delivery call returns true so graph evaluation continues; false would
  cancel the rest of the graph.

A callback timestamp is host observation order, not an exact CUDA-kernel
timestamp. Public API access does not currently expose authoritative internal
backend split boundaries; output is labelled
`BACKEND_DETAIL_UNAVAILABLE_PUBLIC_API`. Selected regions are deferred to WU10
CUPTI validation.

## Observer modes

- `OBSERVER_NONE`: no callback installed.
- `CALLBACK_NOOP`: callback dispatch and counters only.
- `LIGHT_LAYER`: preallocated compact event at layer transitions.
- `LIGHT_LAYER_FLIGHT_RECORDER`: `LIGHT_LAYER` plus the existing 32-byte SPSC
  Flight Recorder path.
- `LIGHT_NODE`: preallocated compact structural event for every asked node.
- `FORENSIC_SELECTED`: one selected materialization request per evaluation;
  validation only, never the production gate candidate.

`ObserverEvent32` is exactly 32 bytes. It carries timestamp, evaluation,
sequence, two tensor IDs, op ID, layer, event class, and flags. Strings live in
the cold dictionary.

## Deterministic harness

The harness uses `WU9-DETERMINISTIC-V1-N` fixtures, where `N` is the resolved
target token count. Greedy sampling produces deterministic output token IDs.
Tokenization is outside prompt-processing timing. Model load, tokenization,
prompt decode, every single-token decode, sampling, and total request time are
recorded separately. Each token preserves input/output ID, context depth,
decode interval, sampling duration, graph-node count, demand IDs, byte totals,
and projected blocks.

Machine-readable inference output also includes the exact prompt token IDs and
the compact observer-event fields. This permits offline layer-order and
lookahead analysis without reading from SQLite in the measured region.

Authority uses full GPU offload (`n_gpu_layers=-1`) where possible. Any bounded
alternative placement is explicitly a native llama.cpp control. Measured data
is held in RAM and persisted after the run.

## CLI

```text
sidecar-lab llama info [--json]
sidecar-lab llama model inspect <path> [--tensors] [--json]
sidecar-lab llama model verify <path> [--json]
sidecar-lab llama baseline --model <path> [configuration]
sidecar-lab llama observe --model <path> --observer <mode> [configuration]
sidecar-lab llama overhead --model <path> --observer <mode> [configuration]
sidecar-lab llama demand --model <path> [configuration]
sidecar-lab llama shadow [--database path] [lead/profile options]
sidecar-lab llama report --database <path> [--json]
sidecar-lab llama validate --model <path> [--json]
```

`--sha256` supplies a digest already computed and verified outside timing. When
persisting an Ollama model, also record `--source OLLAMA_LOCAL_BLOB`,
`--ollama-tag`, `--ollama-digest`, and `--original-blob`. Supplying a digest
does not claim equivalence to a Hugging Face artifact.

Observer overhead uses interleaved AB/BA pairs. Prompt and complete decode
sequences are compared separately. The implementation records paired median,
empirical paired-percentile uncertainty, and the 2% gate. A high-overhead
forensic mode remains valid for interpretation but cannot be the production
candidate.

## Offline WU8 shadow analysis

`llama shadow --database` looks up exact-size, repeated WU8 Pipeline A/B P99
profiles; it does not invent interpolation. It preserves causal and
perfect-oracle knowledge as separate records with lead time, measured WU8
session reference, predicted deadline, hit/miss, margin, and extrapolation
flag. Caller-supplied lead values are labelled as such. The command never moves
a model byte.

Public callback timestamps measure host scheduler enumeration, not CUDA layer
execution. They are therefore not authoritative layer-use deadlines. Automatic
shadow persistence records causal device lead as unavailable/zero and uses the
complete observed token-decode duration only as an optimistic perfect-oracle
upper bound. Exact layer deadlines and CUDA cadence remain WU10 CUPTI targets.

## Persistence

Migration 009 advances both `PRAGMA user_version` and migration history to 9.
It preserves WU1-WU8 tables and adds dependency, model, tensor, fixture,
configuration, benchmark, sample, token, observer, demand/projection, overhead,
MoE, physical-storage provenance, and shadow-result entities. Model records include local Ollama provenance
without asserting an unproved upstream artifact identity. Foreign keys remain
enabled and migration is non-destructive.

## Known limitations and WU10 targets

- Split GGUF indexing is not implemented.
- Public callback metadata establishes host graph order, not CUDA kernel start
  and completion time.
- Internal backend split count/detail is unavailable through the selected
  public context API and remains explicitly unknown.
- Static MoE expert tensors are indexable, but actual routed experts are not
  claimed without a safe executed MoE reference and causal router evidence.
- Low-rate telemetry remains run-level rather than per-node.

Bounded WU10 targets are dense prompt evaluation, one single-token decode, one
representative layer region, and the selected callback interval; MoE routing is
included only if a safe executed model later supplies real expert selection.

## Authoritative campaign and provenance

The implementation commit is
`d6049e86012c54f69188ef59358a0fd4bd81dc79`. Two issues discovered by the
campaign were fixed and committed before affected evidence was repeated:

- `dc6e62ca0eb39a6ae395e915fcb24ddad91d023b`: an `ask == false` forensic
  delivery returned false and cancelled the graph. Tests now require identical
  deterministic tokens and complete callback traversal.
- `c3b6efb46106e736ed49ab48fd2ec8d478875eee` and
  `89d3385d8ff94c08d9c54bc47730857151f10aca`: host callback-enumeration timing
  is explicitly separated from unavailable CUDA device deadlines in JSON and
  shadow persistence.

Authoritative Sidecar sessions are 1048-1055, 1057-1060, 1062-1076, and 1078.
Sessions 1056 and 1061 preserve the forensic cancellation defect and are
`EXCLUDED_METHODOLOGY_DEFECT`. Session 1077 is
`SUPERSEDED_OUTPUT_QUALIFICATION`; its corrected repeat is 1078. Development
sessions 1045-1047 are non-authoritative. The independent upstream baselines
are JSONL artifacts from the clean `d6049e8` state; no Sidecar callback code is
used by `llama-bench`.

All selected dense execution resolves to `C:`, `\\.\PhysicalDrive0`, Samsung
SSD 990 PRO 2TB, NVMe, non-USB. The measured model-load mode is llama.cpp
`auto`/default mmap behavior. Full offload reports 29/29 GPU layers, 2 graph
splits, a 2,207.1 MiB CUDA model buffer, a 308.23 MiB CPU-mapped model buffer,
a 448 MiB CUDA KV buffer, and Flash Attention enabled. Model load is separate
from timing; the combined 512+128 fixture loaded in 1,162.01 ms, tokenized in
1.531 ms, then warmed its measured context before each repetition.

## Measured performance

Pinned upstream `llama-bench`, ten repetitions per fixture:

| Fixture | Mean duration | Mean throughput | Throughput stddev |
|---|---:|---:|---:|
| PP128 | 20.956 ms | 6,129.10 tok/s | 361.97 tok/s |
| PP512 | 52.739 ms | 9,712.49 tok/s | 210.26 tok/s |
| PP2048 | 223.574 ms | 9,160.80 tok/s | 73.19 tok/s |
| TG128 | 577.019 ms | 221.83 tok/s | 0.86 tok/s |
| PG512,128 | 652.310 ms | 981.42 aggregate tok/s | 17.68 tok/s |

Sidecar `OBSERVER_NONE` independently measured prompt processing as follows.
Tokenization, sampling, model load, and generation are excluded from these
prompt intervals.

| Prompt | Mean | P50 | P95/P99 (10 reps) | Mean throughput |
|---|---:|---:|---:|---:|
| 128 | 23.235 ms | 22.228 ms | 32.498 ms | 5,508.99 tok/s |
| 512 | 52.595 ms | 52.346 ms | 53.569 ms | 9,734.69 tok/s |
| 2048 | 218.561 ms | 218.047 ms | 223.780 ms | 9,370.38 tok/s |

For the 512-token prompt plus 128 generated tokens, ten repetitions produced
1,280 individual token samples. The prompt mean was 52.456 ms, complete decode
mean was 588.779 ms, generation throughput was 217.44 tok/s, initial sampling
mean was 189.29 us, subsequent sampling mean was 143.25 us/token, and mean
total request time was 659.804 ms.

| Decode statistic | Duration |
|---|---:|
| Count | 1,280 tokens |
| Mean / stddev | 4.600 / 0.484 ms |
| Min / max | 4.082 / 13.962 ms |
| P50 | 4.651 ms |
| P90 | 5.022 ms |
| P95 | 5.208 ms |
| P99 | 5.878 ms |

Bounded context-depth results use five independently warmed repetitions and
640 token samples per row:

| Prompt/context depth | Mean | P50 | P95 | P99 | Throughput |
|---|---:|---:|---:|---:|---:|
| 128 | 4.598 ms | 4.655 ms | 5.203 ms | 6.484 ms | 217.48 tok/s |
| 2,048 | 4.871 ms | 4.970 ms | 5.350 ms | 6.606 ms | 205.28 tok/s |
| 4,096 | 5.109 ms | 5.207 ms | 5.545 ms | 6.472 ms | 195.73 tok/s |
| 8,192 | 5.755 ms | 5.812 ms | 6.314 ms | 7.622 ms | 173.75 tok/s |

Real decode windows are much shorter than WU6's 16/32/64 ms synthetic
regions. Prompt PP128 lies between 16 and 32 ms, PP512 is closest to 64 ms,
and PP2048 is longer than the WU6 set. No WU6 window is a faithful single-token
decode match.

## Observer qualification

The production gate uses ten interleaved A/B-B/A pairs of complete PP512 and
TG128 sequences. Negative medians are retained as noise, not interpreted as a
speedup.

| Mode | Prompt median / P95 overhead | Prompt status | Decode median / P95 overhead | Decode status |
|---|---:|---|---:|---|
| CALLBACK_NOOP | -0.18% / 2.33% | MARGINAL | 1.67% / 3.05% | MARGINAL |
| LIGHT_LAYER | 0.29% / 2.12% | MARGINAL | 3.26% / 4.75% | FAIL |
| LIGHT_LAYER + Flight Recorder | 0.00% / 2.90% | MARGINAL | 3.68% / 5.63% | FAIL |
| LIGHT_NODE | -0.27% / 5.97% | MARGINAL | 5.57% / 7.54% | FAIL |
| FORENSIC_SELECTED (tiny validation) | -14.75% / 5.16% | MARGINAL/noisy | -6.19% / 7.23% | MARGINAL/noisy |

The corrected forensic validation traversed 4,370 callback asks, materialized
five selected nodes, dropped no events, and preserved deterministic output.
The real-inference observer gate is **FAIL**: LIGHT_LAYER is the lowest useful
production candidate, but its median complete-decode overhead is 3.26%, above
the 2% target. Prompt processing alone is MARGINAL. The historical WU3
synthetic gate remains a separate result.

## Dense GGUF demand and block projection

The model contains 255 persistent weight tensors totaling 2,314,316,032 bytes.
Its 28 transformer layers account for 1,991,098,368 bytes: mean 71,110,656,
minimum 69,230,592, maximum 72,990,720 bytes per layer. Global embedding,
output, and normalization weights account for the remaining 323,217,664 bytes.

| Block size | Full-model blocks | Projected bytes | Overfetch | Overfetch ratio |
|---|---:|---:|---:|---:|
| 32 MiB | 70 | 2,348,810,240 | 34,494,208 | 1.49% |
| 64 MiB | 35 | 2,348,810,240 | 34,494,208 | 1.49% |
| 128 MiB | 18 | 2,415,919,104 | 101,603,072 | 4.39% |

A layer spans on average 3.18, 2.11, and 1.57 blocks at 32, 64, and 128 MiB.
Twenty-seven of 28 layer tensor ranges are individually contiguous in GGUF
file order. Layer 20 is the exception: 72,572,928 useful bytes span
565,125,120 bytes because two ranges are separated by a 492,552,192-byte gap.

Across 128 generated-token evaluations, all 70/35/18 candidate blocks are in
the union, all are permanently repeated, mean repeated fraction is 1.0, and
zero blocks become newly demanded after the first token. The dense graph is
static and every layer is known structurally in advance, but public callback
timestamps do not establish CUDA use deadlines. The observed host enumeration
cadence (roughly microseconds between callback layer records) is explicitly not
used as device lead-time evidence.

## WU8 shadow feasibility

Exact, non-interpolated WU8 P99 profiles were selected:

| Block | Pipeline B P99/session | Pipeline A P99/session |
|---|---:|---:|
| 32 MiB | 39.990 ms / 1042 | 44.579 ms / 1041 |
| 64 MiB | 13.951 ms / 1040 | 17.687 ms / 1038 |
| 128 MiB | 27.075 ms / 1039 | 35.574 ms / 1026 |

Causal device-layer lead is `UNAVAILABLE_PUBLIC_CALLBACK_API`, so automatic
causal rows use zero rather than a fabricated deadline. As an optimistic
perfect-oracle upper bound, each full observed token duration was compared with
the pipeline P99 latency. Pipeline B at 64 MiB meets only 1 of 1,280 windows
(0.078%); Pipeline A at 64 MiB and both pipelines at 32/128 MiB meet 0 of 1,280.
Because a target layer is needed before token completion, real feasibility can
only be lower. No 1-4-layer lookahead produces strong deadline reliability for
32, 64, or 128 MiB. Exact per-layer fractions require WU10 device timing.

For this fully resident dense model the relevant steady-state policy is to keep
all blocks hot: there is no genuinely new token-dependent weight demand after
the first token. Per-token restaging would require repeatedly supplying the
entire 2.314 GB working set and is not supported by the measured deadlines.

## MoE structural result

The 48.19 GiB Qwen3-Coder-Next candidate was not executed from its USB backing
and was not copied. Structural inspection finds 843 persistent tensors totaling
51,735,004,160 bytes, 512 possible experts, 201,523,200 router/always-hot bytes,
50,130,321,408 packed expert bytes, and an estimated 97,910,784 bytes per
expert. Actual selected experts, reuse, causal router cue timing, urgent
Pipeline B feasibility, and comparative MoE staging promise are all
`SKIPPED_OPTIONAL_MODEL`; static possible experts are not execution evidence.

## Native partial-offload control

| Placement | TG128 throughput | Change vs 217.44 tok/s full GPU | Graph splits | CUDA/CPU model buffers |
|---|---:|---:|---:|---:|
| 29/29 GPU | 217.44 tok/s | baseline | 2 | 2,207.1 / 308.23 MiB |
| 21/29 GPU | 44.11 tok/s | -79.7% | 99 | 1,664.57 / 1,574.84 MiB |
| 14/29 GPU | 26.42 tok/s | -87.9% | 183 | 1,194.84 / 1,707.29 MiB |

The public path exposes the placement and split count but not authoritative
CPU/GPU transition timestamps. Native CPU-layer offload does **not** resemble
Sidecar deadline-driven staging; it is ordinary multi-backend layer execution.

## Completion answers

1. llama.cpp revision: `0d9ceae1e38291035605613ab41a8f5e693d6fcd`,
   clean upstream master, no patch.
2. Executed model: Llama 3.2 3B Instruct Q5_K Medium, SHA-256
   `05fc42664a9311c427413f9bf2077bd5ee7d59d6a5a034d54fc738f93976d065`.
   MoE structural-only hash is
   `30e51a7cb1cf1333b9e298b90b4c7790fe2572d8736b002482a0ac96328a2ffb`.
3. Upstream performance: PP128 6,129.10; PP512 9,712.49; PP2048 9,160.80;
   TG128 221.83; PG512,128 981.42 aggregate tok/s.
4. Sidecar independently measures model load, tokenization, prompt, each token,
   sampling, and total request; key values are in the tables above.
5. Real P50/P95/P99 token durations: 4.651/5.208/5.878 ms at the normal
   512-token fixture.
6. Mean decode rises from 4.598 ms at 128 context to 5.755 ms at 8K.
7. No WU6 synthetic window matches decode; PP128 is 16-32 ms, PP512 is closest
   to 64 ms, and PP2048 is beyond 64 ms.
8. CALLBACK_NOOP: -0.18% prompt median and 1.67% decode median; MARGINAL tails.
9. LIGHT_LAYER: 0.29% prompt, 3.26% decode median.
10. LIGHT_LAYER + recorder: 0.00% prompt, 3.68% decode median.
11. LIGHT_NODE: -0.27% prompt, 5.57% decode median.
12. FORENSIC_SELECTED: tiny/noisy validation only; corrected graph complete,
    P95 5.16% prompt and 7.23% decode.
13. Production LIGHT prompt target: MARGINAL, not a confident pass.
14. Production LIGHT token-generation target: FAIL.
15. Real inference observer gate: **FAIL**.
16. Persistent weights: 255 tensors, 2,314,316,032 bytes.
17. Layer working set: mean 71,110,656 bytes across 28 layers.
18. Full model maps to 70/35/18 blocks at 32/64/128 MiB.
19. Overfetch: 34,494,208/34,494,208/101,603,072 bytes.
20. GGUF layer demand: 27/28 layers individually contiguous; layer 20 is split.
21. Cross-token block reuse: 100% repeated in the 128-token trace.
22. Permanently hot: all 70/35/18 projected blocks.
23. Genuinely new token-dependent dense weight demand: none after token one.
24. Exact consecutive device-layer time: unavailable from the public callback;
    host enumeration cadence is not substituted.
25. Exact 1-4-layer device lead: unavailable; even a full-token oracle bound is
    normally shorter than WU8 supply P99.
26. Perfect-oracle Pipeline A hit upper bound: 0/1,280 for every block size.
27. Perfect-oracle Pipeline B upper bound: 1/1,280 at 64 MiB; 0 at 32/128 MiB.
28. Causal dense-layer fractions: exact fractions require device timing; the
    conservative qualified shadow is 0% and never invents callback lead.
29. Smallest reliable 32 MiB lookahead: none in 1-4 layers/full-token bound.
30. Smallest reliable 64 MiB lookahead: none; one full-token outlier is 0.078%.
31. Smallest reliable 128 MiB lookahead: none.
32. MoE selected experts: `SKIPPED_OPTIONAL_MODEL`; static possibilities only.
33. MoE executed expert working set: `SKIPPED_OPTIONAL_MODEL`; static estimate
    is 97,910,784 bytes/expert.
34. MoE expert reuse: `SKIPPED_OPTIONAL_MODEL`.
35. MoE causal expert cue: `SKIPPED_OPTIONAL_MODEL`.
36. MoE urgent Pipeline B reaction: `SKIPPED_OPTIONAL_MODEL`.
37. MoE predictive advantage: `SKIPPED_OPTIONAL_MODEL`; no claim.
38. Native partial offload reduces TG128 to 44.11 and 26.42 tok/s.
39. Graph splits increase from 2 to 99 and 183.
40. Placement/buffer changes are visible; exact transition timing is
    `BACKEND_DETAIL_UNAVAILABLE_PUBLIC_API`.
41. Native llama.cpp offload resembles Sidecar staging: **NO**.

The primary WU9 decision is **MODEL DEPENDENT**. For the measured dense model,
all weights fit in VRAM, every block is reused every token, and WU8 supply P99
does not fit real decode deadlines, so active per-token staging is not useful.
Oversized or sparse models may differ, but WU9 deliberately does not promote
the structural-only MoE artifact to execution evidence.

## Database and verification status

Schema/user version and latest migration are 9; foreign keys are enabled with
zero violations and WU1-WU8 data remains present. Final WU9 counts are three
dependency build records, two models, 1,098 tensors, four storage-provenance
records, 28 configurations/benchmarks, 6,156 token samples, 498,488 observer
events, 150,918 tensor-demand rows, 18,726 block projections, 38,824 shadow
rows, 12 overhead rows, and one static MoE profile.

CUDA Release and Debug each pass 44/44 CTest registrations. Genuine CPU-only
Release passes 41/41 without loading `ggml-cuda.dll`; this is also the fix for
the earlier `sidecar_llama_tests.exe` missing-DLL dialog. The repeated unit and
Flight Recorder concurrency subset passes ten consecutive iterations. Quiet
mode used no Ollama, local LLM, build/test worker, profiler, download, hashing,
or concurrent benchmark process.
