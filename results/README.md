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

CPU offload is based on distinct ORT profiler nodes, not compute share. It is
evidence that operations executed on CPU, but it must not be interpreted as a
percentage of runtime spent on CPU.

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
