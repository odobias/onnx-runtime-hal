# Benchmark results

The result directory separates suite-of-record evidence from derived research
reports and ephemeral local outputs:

| Path | Role |
|------|------|
| `accuracy-runs/`, `run-attempts/`, `host-snapshots/`, `model-compilation/`, `ledgers/` | Published evidence (commit selectively) |
| `reports/` | Historical human reports (not suite-of-record) |
| `local/` | Ephemeral research/sweep outputs (gitignored) |

See [`LAYOUT.md`](../LAYOUT.md) for the repo-wide source vs artifact contract.

## Authoritative ledgers

`accuracy-runs/<environment-snapshot-id>.jsonl` contains immutable,
invocation-scoped accuracy evidence. The default suite mode is
`accuracy-quick`; it runs every valid execution profile once and writes a new
campaign file instead of appending to a shared tracked ledger.

`ledgers/accuracy.jsonl` is the frozen pre-migration history. New campaigns must
not append to it. `benchmark/validate-accuracy.ps1` reads that history together
with every run-scoped accuracy file by default.

`ledgers/asr.csv` and `ledgers/classifiers.csv` contain latency measurements
created only by explicit `-Mode latency` runs.

`ledgers/attempts.jsonl` is ignored rolling local state. At the end of each
completed suite invocation, the runner extracts that invocation's records into
`run-attempts/<environment-snapshot-id>.jsonl`. Those immutable, run-scoped
files are the published source of truth for requested workload/device/provider
attempts, including:

- `ok`
- `unsupported`
- `assets-missing`
- `executor-failed`
- `accuracy-valid`
- `accuracy-invalid`
- provider initialization and inference errors

A missing CSV measurement is therefore distinguishable from a combination that
was never requested.

The published file makes unsupported and failed combinations auditable beside
successful accuracy rows instead of relying on an uncommitted developer
workspace. Validate either a published or rolling attempt ledger with
`benchmark/validate-attempts.ps1 -Ledger <path>`.

## Benchmarking approach

The portable suite separates three contracts:

1. `accuracy-quick` is the default benchmark-of-record. It evaluates the full
   reference fixture with one measured inference per clip or fixture, applies
   provider and numerical trust gates, and publishes immutable accuracy and
   attempt files for that invocation.
2. `latency` is explicit. Its defaults are 10 measured Whisper transcriptions
   per clip and 20 measured classifier inferences per fixture. It writes only
   the performance CSVs; an accuracy-quick mean is orientation data, not a
   latency leaderboard.
3. Direct CLI invocations are smoke or executor diagnostics. They do not become
   comparable benchmark evidence merely because inference succeeded.

Both contracts keep untimed warm-up separate from measured repetitions.
`-Runs <N>` overrides Whisper repetitions and `-ClassifierRuns <N>` overrides
the repetitions performed for each classifier fixture in either mode.

Comparisons require matching workload/profile, model and fixture hashes,
reference contract, runtime target, requested device, and measurement purpose.
Host architecture, runtime version, power source, thermal conditions, and
provider assignment remain visible caveats rather than silently normalized.

Each accuracy record carries the execution profile, graph role, model and
fixture hashes, runtime environment snapshot ID, compilation-provenance IDs,
requested/resolved provider, fallback status, workload metrics, and a
`trust_gate`. A record is valid only when outputs are finite, CPU-reference
decisions do not flip, the intended provider resolves, and the profile is
accuracy-eligible. New NPU records must additionally contain a trustworthy
CPU/NPU operation assignment. Invalid and diagnostic payloads remain published
but must not enter valid summaries.

Use selectors to narrow the default matrix:

```powershell
.\benchmark\run-suite.ps1 -Runtime bundled -Only fakeaudio -Device npu
.\benchmark\run-suite.ps1 -Profile npu-split-generic
.\benchmark\run-suite.ps1 -Mode latency -Runtime winml -Device gpu
```

Static and dynamic-KV Whisper have different workload/profile IDs and must
remain separate cohorts.

## Executor diagnostics

Successful rows include:

- `requested_provider`
- `resolved_provider`
- `fallback_occurred`
- `provider_attempts`
- `ep_nodes`
- `cpu_nodes`
- `cpu_offload_pct`
- `cpu_offload_ops`
- `assigned_ops_cpu`
- `assigned_ops_npu`
- `operation_assignment_source`
- `error`

`provider_attempts` is JSON stored inside the CSV field. It preserves each
provider tried, whether it succeeded, and the provider-specific error.

CPU offload is based on distinct ORT profiler nodes, not compute share.
`ep_nodes` and `cpu_nodes` expose the underlying counts; one accelerator event
may represent hundreds of fused ONNX operations while cheap shape/control
operations remain individual CPU nodes. The same profiler parser is used for
OpenVINO, VitisAI, QNN, and DirectML, so the audit is vendor-neutral. Only the
hot session that performs inference is profiled.

`assigned_ops_cpu` and `assigned_ops_npu` are a separate, optional measurement
family populated only from provider/compiler assignment artifacts or ORT's
retained EP graph-assignment API. They are left empty when neither source
exposes a trustworthy operation mapping; ORT profiler node counts are never
promoted into these fields. For VitisAI,
`operation_assignment_source=vitisai-cache-context-gops` means the total graph
operation rows came from `gops.csv` and NPU partition membership came from
`context.json`. On ORT API v24 or newer,
`operation_assignment_source=ort-ep-graph-assignment` counts the original,
unfused graph nodes retained for each EP subgraph. Non-CPU assignments are
recorded as NPU operations only for a valid, non-fallback NPU request. The suite
marks a new NPU accuracy row invalid when these fields are absent, malformed, or
report no NPU operations.

Classifier startup is measured twice against a dedicated cache:

- `cold_load_seconds`: first session creation after the suite clears the cache.
- `hot_load_seconds`: second session creation from the artifact cold produced.
- `load_seconds`: compatibility alias for the cold value.

Whisper follows the same policy. `warm_load_seconds` remains only as a migration
alias for historical ASR rows.

## Immutable context

`host-snapshots/<id>.json` captures the executable and runtime DLL hashes and
versions, hardware identities and driver metadata, Windows build/UBR,
BIOS/firmware, memory, architecture, AC state, active power plan, and available
SDK package manifests.

`model-compilation/<id>.json` and `model_compilation_provenance_ids` retain
their names for schema compatibility. They identify the exact graph presented
to the provider before runtime compilation, plus source hashes when known,
target architecture, transformation metadata, and compiler/runtime version
context. They do **not** identify a compiled provider-cache artifact.

New records make this explicit with `provenance_scope`,
`artifact_stage=provider-input`, and
`compiled_cache_artifact.status=not-captured`. Older records without those
fields have the same input-artifact semantics; absence must never be interpreted
as a compiled-cache hash. Unknown versions and options are recorded with a
reason rather than guessed. Split FakeAudio records link separate CPU-frontend
and NPU-backbone input artifacts.

## Schema authority

Machine-readable schemas live in:

- `../benchmark/schemas/asr-results.columns.json`
- `../benchmark/schemas/classifier-results.columns.json`
- `../benchmark/schemas/accuracy-record.schema.json`
- `../benchmark/schemas/attempt-record.schema.json`

The C++ writer and PowerShell harness follow those column orders.
Run `benchmark/validate-schemas.ps1` to verify both writers against the JSON
contracts.

`model_sha256` identifies the exact inference artifacts used by a row. Each
ONNX, OpenVINO XML/BIN, or external data artifact is hashed by content; the
sorted artifact digests are then hashed together. Package paths and filenames
therefore do not affect identity, while any graph or weight change does.
Historical rows created before this field remain blank.

`inference_precision` records the provider-neutral precision policy. CPU and
GPU runs default to `f32`; NPU runs default to `preferred`. Use
`run-suite.ps1 -Precision preferred` (or set
`NPU_INFERENCE_BENCH_PRECISION=preferred`) to let the executor choose. OpenVINO
maps explicit policies to `INFERENCE_PRECISION_HINT`. ORT CPU and DirectML can
guarantee FP32 for FP32 model tensors but cannot perform a provider-wide
FP16/BF16 conversion. QNN and VitisAI precision is compiler/model-defined, so
non-`preferred` requests are rejected rather than silently mislabeled.

## Historical reports

Snapshots and derived research reports live under `reports/`. They are useful
context, but they are not substitutes for the portable suite ledger and may
describe older model exports or runtime versions.

Current hardware snapshots:

- `reports/amd-directml-2026-07-13.md`: audited Radeon 890M DirectML results for
  both Whisper variants, TSC, and FakeAudio, including CPU-offload node counts.

Quantization sweeps under `benchmark/research/` are explicitly research-only and
write ephemeral outputs to `local/` (gitignored). Neutral ONNX portability
findings remain in `reports/onnx-portability.md` and
`reports/cpp-onnx-npu-findings.md`.

### Classifier snapshot curation

The pre-schema-v1 classifier snapshot is split by evidence quality:

- `ledgers/classifiers.csv` retains nine rows that include latency, accuracy,
  probability agreement, and provider-placement metrics: the matched AMD
  CPU/DirectML/VitisAI sweep and the complete Qualcomm QNN sweep.
- `ledgers/classifiers.legacy.csv` preserves 18 superseded or incomplete rows.
  The sole Intel OpenVINO EP result is retained there pending a current-contract
  rerun; it lacks provider-placement metrics.

Do not mix the legacy rows into current latency or offload comparisons. These
snapshots also predate the extended schema-v1 fields for cache state, provider
attempts, fallback, and errors, so future suite runs should write a fresh
schema-v1 ledger rather than append incompatible rows.

The snapshots record architecture, OS build, and AC/battery state, but not full
hardware identity or Windows power-plan/Energy Saver state. `battery` alone does
not establish equivalent power conditions.
