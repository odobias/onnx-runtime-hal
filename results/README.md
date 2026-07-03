# Benchmark Results

Benchmark runs append rows to `benchmark-results.csv` by default:

```powershell
.\scripts\run.ps1 -Backend amd -Device npu
.\scripts\run.ps1 -Backend intel -Device npu
.\scripts\benchmark.ps1 -Runs 3
```

Each platform should execute only the manifest entries it can run locally. The shared CLI
and manifest benchmark record the same CSV schema for every backend, so result files from
different machines can be concatenated or imported later.

Use `-Results <path>` to write somewhere else, or `-NoResults` for scratch runs.

## Schema

The **authoritative column order** is defined once in `scripts/benchmark.lib.ps1`
(`Get-BenchmarkResultColumns`). The PowerShell harness and the C++ executor
(`app/main.cpp`) are the two writers of this ledger and both follow that list — keep
all three in sync when adding or reordering columns.

- `timestamp_utc`: UTC timestamp when the row was written.
- `requested_backend`: CLI backend selector, e.g. `AmdRyzenAI`, `IntelOpenVINO`, or `Auto`.
- `resolved_backend`: concrete engine that actually ran.
- `device`: logical device selector, e.g. `NPU`, `CPU`, `GPU`.
- `device_name`: backend-specific device string (`NPU`/`GPU`/`CPU` for Intel; an EP name like `CPUExecutionProvider` for AMD).
- `device_full_name`: the actual hardware identity when the backend can query it, e.g. `Intel(R) AI Boost` or `13th Gen Intel(R) Core(TM) i7-1370P` on Intel (via `ov::device::full_name`). Empty when the backend doesn't expose one (AMD/Qualcomm currently don't) or for rows written before this column existed.
- `model_package`: basename of `model_dir`, e.g. `whisper-tiny-en-static-onnx`. Always populated.
- `variant_id`: manifest `id` or `--label`; falls back to `model_package`. Always populated.
- `base_model`: Hugging Face model id from `models/manifest.json` (default `openai/whisper-tiny.en`).
- `precision`: `fp32`, `fp16`, `int8`, `int4`, `fp32-static`, etc. From manifest when available, else inferred from the package name.
- `quant_method`: human-readable compression/export method from the manifest `method` field.
- `execution_provider`: backend-specific runtime device string (ORT EP name, OpenVINO device, QNN backend). Mirrors `device_name` but kept explicit for filtering alongside stack metadata.
- `model_dir`: model directory used by the backend.
- `audio_path`: WAV file used for the benchmark.
- `audio_seconds`: input audio duration.
- `runs`: measured inference iterations.
- `warmup`: inference warmup iterations before latency measurement; it is not a startup metric.
- `cache_dir`: compiled-model cache directory, if enabled.
- `cold_load_seconds`: legacy name for first engine load time.
- `warm_load_seconds`: legacy name for second engine load time, or `-1` when no cache/hot reload was used.
- `mean_infer_seconds`: mean measured transcription latency.
- `rtf`: real-time factor, `mean_infer_seconds / audio_seconds`.
- `realtime_factor`: inverse RTF.
- `label`: free-form tag for the row; defaults to `variant_id` when unset.
- `model_size_mb`: on-disk size of `model_dir`. Empty if unavailable.
- `avg_logprob`: mean per-token log-prob the model self-reports (confidence proxy). Empty when the backend can't provide token metrics.
- `ttft_ms` / `tpot_ms` / `throughput_tps`: time-to-first-token, time-per-output-token, tokens/sec. Empty when unavailable.
- `wer` / `cer`: word/char error rate vs. `--ref` (fraction, 0..1). Empty when no reference was given.
- `transcription`: final transcription text from the last measured run.
- `cold_start_seconds`: first engine creation after the harness deletes that variant/device cache.
- `hot_start_seconds`: second engine creation in the same process after the compiled cache has been populated.
- `eval_clips`: number of clips aggregated into this row. C++ single-clip runs = `1`; `benchmark.ps1` aggregate rows use the number of eval clips that completed.
- `status`: `ok`, or a short failure reason for unsupported/missing configurations.
- `runtime`: execution stack that actually ran the model, e.g. `openvino-genai`, `onnxruntime`, `onnxruntime-qnn`, `onnxruntime-vitisai`. Always populated (from the backend or manifest lookup).
- `model_format`: `onnx` (vendor-neutral) or `ov-ir` (Intel OpenVINO IR). Always populated when inferable.
- `decode_strategy`: how the autoregressive loop runs — `genai-bounded-kv` (OV GenAI), `static-no-kv` (fixed-context recompute, NPU-compilable), etc. Always populated for ONNX static backends.
- `max_context`: static decoder context length for `static-no-kv`; blank when the shape is dynamic or N/A.
- `power_source`: whether the machine was on wall power or battery when the run was recorded — `ac`, `battery`, or `unknown`. Read from `GetSystemPowerStatus` (C++) / `SystemInformation.PowerStatus` (PowerShell) at run time. Battery means throttled CPU/NPU clocks, so **do not compare an `ac` row against a `battery` row and pretend the delta is architectural.** Rows written before this column existed are blank (state unknown).

The `cold_load_seconds`/`warm_load_seconds` columns are retained for compatibility, but new
analysis should use `cold_start_seconds`/`hot_start_seconds`. Do not mix startup time with
inference warmup; those are different measurements and pretending otherwise is how bogus
benchmarks are born.

These trailing columns are appended after `transcription`, so tools that read the original
schema by position are unaffected. `benchmark.ps1` upgrades an older local CSV header before
appending aggregate rows.

The trailing metric columns are backend-neutral and optional: each backend fills only what
it can measure. This keeps one schema across machines so files concatenate cleanly.

## Host hardware

Each benchmark run emits a detailed inventory of the chips it ran on to
`host-info.json` (next to the report: `results/` for `benchmark.ps1`,
`build/reports/` for `compare-devices.ps1`) and a **Host hardware** section in the
Markdown report. It captures CPU (name, vendor, cores/threads, clock), every display
adapter (name, driver, approximate VRAM), and the NPU (name, manufacturer, driver
version, PnP instance id), plus RAM and OS. Caveats: `Win32_Processor.MaxClockSpeed`
is the nominal/base clock, not turbo; `AdapterRAM` is a uint32 that saturates around
4 GB, so `vram_mb_approx` is a lower bound for large GPUs.

## Manifest sweep

`scripts\benchmark.ps1` compares entries from `models\manifest.json` across
`variant × device × clip` and writes an aggregate report to
`results\quantization-benchmark.{md,csv}` (micro-averaged WER/CER, confidence, startup,
and latency). It also appends one canonical aggregate row per `variant × device` to
`benchmark-results.csv`, so single-run and swept results live in the same schema.

`models\manifest.json` is the platform contract. Each variant entry must include:
`id`, `backend`, `precision`, `method`, `model_dir`, `devices`, and `size_mb`.
Intel export scripts and AMD model download scripts merge their own entries into this file
instead of overwriting other platforms.

**Quantized OV-IR variants (fp16/int8/int4) are research-only** and live in
`models\manifest.research.json`, not the product `manifest.json`. The default benchmark
never touches them. To sweep them for research, point the harness at that manifest:
`benchmark.ps1 -Manifest models\manifest.research.json`.

## Neutral ONNX portability

`results\onnx-portability.{md,csv}` captures a separate experiment: running one
**vendor-neutral ONNX** (not the Intel-specific OV-IR) across CPU/GPU/NPU to test
whether a single model artifact is viable across runtimes. It uses a hand-rolled
decode loop (`scripts\experiments\onnx_ov_decode.py`, `scripts\experiments\onnx_npu_static.py`) so quality
is identical across runtimes and only load/latency vary. These are 12-clip
aggregates, so they live in their own file rather than the per-clip
`benchmark-results.csv` schema.

`results\cpp-onnx-npu-findings.md` distills the **key lessons** from turning that
experiment into the C++ HAL backend (`Backend::IntelOnnx`): which decode strategy
actually compiles on the NPU, why `maxlen` isn't a latency lever, the mel-FFT win
(~2× on NPU/GPU), and the GenAI-parity verdict. Read this first before revisiting
NPU latency.
