# Benchmark Results

Benchmark runs append rows to `benchmark-results.csv` by default:

```powershell
.\scripts\run.ps1 -Backend amd -Device npu
.\scripts\run.ps1 -Backend intel -Device npu
```

Each platform should execute only the tests it can run locally. The shared CLI
records the same CSV schema for every backend, so result files from different
machines can be concatenated or imported later.

Use `-Results <path>` to write somewhere else, or `-NoResults` for scratch runs.

## Schema

- `timestamp_utc`: UTC timestamp when the row was written.
- `requested_backend`: CLI backend selector, e.g. `AmdRyzenAI`, `IntelOpenVINO`, or `Auto`.
- `resolved_backend`: concrete engine that actually ran.
- `device`: logical device selector, e.g. `NPU`, `CPU`, `GPU`.
- `device_name`: backend-specific device string.
- `model_dir`: model directory used by the backend.
- `audio_path`: WAV file used for the benchmark.
- `audio_seconds`: input audio duration.
- `runs`: measured inference iterations.
- `warmup`: warmup iterations before measurement.
- `cache_dir`: compiled-model cache directory, if enabled.
- `cold_load_seconds`: first engine load time.
- `warm_load_seconds`: second engine load time, or `-1` when no cache/warm reload was used.
- `mean_infer_seconds`: mean measured transcription latency.
- `rtf`: real-time factor, `mean_infer_seconds / audio_seconds`.
- `realtime_factor`: inverse RTF.
- `label`: free-form tag for the row, e.g. the quantization variant (`wten-ov-int8`). Empty if unset.
- `model_size_mb`: on-disk size of `model_dir`. Empty if unavailable.
- `avg_logprob`: mean per-token log-prob the model self-reports (confidence proxy). Empty when the backend can't provide token metrics.
- `ttft_ms` / `tpot_ms` / `throughput_tps`: time-to-first-token, time-per-output-token, tokens/sec. Empty when unavailable.
- `wer` / `cer`: word/char error rate vs. `--ref` (fraction, 0..1). Empty when no reference was given.
- `transcription`: final transcription text from the last measured run.

The trailing metric columns are backend-neutral and optional: each backend fills
only what it can measure (e.g. the Intel OpenVINO backend reports confidence and
token perf; the AMD scaffold currently leaves them empty). This keeps one schema
across machines so files concatenate cleanly.

## Quantization sweep

`scripts\benchmark.ps1` compares quantization variants from `models\manifest.json`
across `variant × device × clip` and writes an aggregate report to
`results\quantization-benchmark.{md,csv}` (micro-averaged WER/CER, confidence, load
and latency). It also appends one canonical per-variant row to
`benchmark-results.csv` via the shared CLI, so single-run and swept results live in
the same schema.
