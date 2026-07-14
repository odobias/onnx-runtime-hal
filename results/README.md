# Benchmark results

The result directory separates raw benchmark ledgers from derived research
reports.

## Authoritative ledgers

`ledgers/asr.csv` contains successful Whisper measurements.

`ledgers/classifiers.csv` contains successful TSC and FakeAudio measurements.

`ledgers/attempts.jsonl` is the suite-level source of truth. It contains one JSON
record for every requested workload/device/provider attempt, including:

- `ok`
- `unsupported`
- `assets-missing`
- `executor-failed`
- provider initialization and inference errors

A missing CSV measurement is therefore distinguishable from a combination that
was never requested.

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
OpenVINO, VitisAI, QNN, and DirectML, so the audit is vendor-neutral. Whisper
profiling ends after the first transcription (normally warmup), before measured
latency runs.

Classifier startup is measured twice against a dedicated cache:

- `cold_load_seconds`: first session creation after the suite clears the cache.
- `hot_load_seconds`: second session creation from the artifact cold produced.
- `load_seconds`: compatibility alias for the cold value.

## Schema authority

Machine-readable schemas live in:

- `../benchmark/schemas/asr-results.columns.json`
- `../benchmark/schemas/classifier-results.columns.json`

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
