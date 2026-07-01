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
- `transcription`: final transcription text from the last measured run.
