# Benchmark results

The result directory separates raw benchmark ledgers from derived research
reports.

## Authoritative ledgers

`ledgers/accuracy.jsonl` contains accuracy-first evidence. The default suite mode
is `accuracy-quick`; it runs every valid execution profile once and never appends
to either performance CSV.

`ledgers/asr.csv` and `ledgers/classifiers.csv` contain latency measurements
created only by explicit `-Mode latency` runs.

`ledgers/attempts.jsonl` is the suite-level source of truth. It contains one JSON
record for every requested workload/device/provider attempt, including:

- `ok`
- `unsupported`
- `assets-missing`
- `executor-failed`
- `accuracy-valid`
- `accuracy-invalid`
- provider initialization and inference errors

A missing CSV measurement is therefore distinguishable from a combination that
was never requested.

Each accuracy record carries the execution profile, graph role, model and
fixture hashes, runtime environment snapshot ID, compilation-provenance IDs,
requested/resolved provider, fallback status, workload metrics, and a
`trust_gate`. A record is valid only when outputs are finite, CPU-reference
decisions do not flip, the intended provider resolves, and the profile is
accuracy-eligible. Invalid and diagnostic payloads remain in the ledgers but
must not enter valid summaries.

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
- `error`

`provider_attempts` is JSON stored inside the CSV field. It preserves each
provider tried, whether it succeeded, and the provider-specific error.

CPU offload is based on distinct ORT profiler nodes, not compute share.
`ep_nodes` and `cpu_nodes` expose the underlying counts; one accelerator event
may represent hundreds of fused ONNX operations while cheap shape/control
operations remain individual CPU nodes. The same profiler parser is used for
OpenVINO, VitisAI, QNN, and DirectML, so the audit is vendor-neutral. Only the
hot session that performs inference is profiled.

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

`model-compilation/<id>.json` captures transformed/compiled artifact hashes,
source hashes when known, target architecture, transformation metadata, and an
authoritative vendor SDK/compiler version source. Unknown versions are recorded
as unknown with a reason; they are never guessed from a vendor name. Split
FakeAudio records link separate CPU-frontend and NPU-backbone provenance and
hashes.

## Schema authority

Machine-readable schemas live in:

- `../benchmark/schemas/asr-results.columns.json`
- `../benchmark/schemas/classifier-results.columns.json`
- `../benchmark/schemas/accuracy-record.schema.json`

The C++ writer and PowerShell harness follow those column orders.

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

Other Markdown, CSV, and Canvas files in this directory are snapshots or derived
research reports. They are useful evidence, but they are not substitutes for the
portable suite ledger and may describe older model exports or runtime versions.

Current hardware snapshots:

- `amd-directml-2026-07-13.md`: audited Radeon 890M DirectML results for both
  Whisper variants, TSC, and FakeAudio, including CPU-offload node counts.

Quantization sweeps under `benchmark/research/` are explicitly research-only.
Neutral ONNX portability findings remain in `onnx-portability.md` and
`cpp-onnx-npu-findings.md`.

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
