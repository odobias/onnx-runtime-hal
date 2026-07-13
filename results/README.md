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
- `cpu_offload_pct`
- `cpu_offload_ops`
- `error`

`provider_attempts` is JSON stored inside the CSV field. It preserves each
provider tried, whether it succeeded, and the provider-specific error.

CPU offload is based on distinct ORT profiler nodes, not compute share. One
accelerator event may represent hundreds of fused ONNX operations while cheap
shape/control operations remain individual CPU nodes.

Classifier startup is measured twice against a dedicated cache:

- `cold_load_seconds`: first session creation after the suite clears the cache.
- `hot_load_seconds`: second session creation from the artifact cold produced.
- `load_seconds`: compatibility alias for the cold value.

## Schema authority

Machine-readable schemas live in:

- `../benchmark/schemas/asr-results.columns.json`
- `../benchmark/schemas/classifier-results.columns.json`

The C++ writer and PowerShell harness follow those column orders.

## Historical reports

Other Markdown, CSV, and Canvas files in this directory are snapshots or derived
research reports. They are useful evidence, but they are not substitutes for the
portable suite ledger and may describe older model exports or runtime versions.

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
