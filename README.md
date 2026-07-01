# whisper-npu-hal

A C++ **hardware-abstraction layer** for running `whisper-tiny(.en)` speech-to-text on
different vendor NPUs behind one stable API. Proof-of-concept for a
hardware-independent runner with swappable per-platform backends.

- **Intel** — OpenVINO GenAI (NPU / GPU / CPU). **Working reference implementation.**
- **AMD** — Ryzen AI / XDNA via ONNX Runtime + VitisAI EP. **Working ONNX implementation**
  for AMD's `whisper-tiny-onnx-npu` model.
- **Qualcomm** — Snapdragon Hexagon via ONNX Runtime + QNN EP. **Prepared scaffold.**

Build system: **MSBuild / Visual Studio 2026** (`WhisperNpuHal.sln`).
`PlatformToolset=$(DefaultPlatformToolset)`, so it also builds on older VS if needed.

## Design

```
include/whisper_npu/whisper_engine.hpp   Public API: IWhisperEngine, Backend, factory
include/whisper_npu/audio.hpp            Dependency-free 16 kHz mono WAV loader
src/factory.cpp                          create_engine() + backend availability
src/backends/intel/                      OpenVINO GenAI backend (real)
src/backends/amd/                        Ryzen AI / VitisAI backend
src/backends/qualcomm/                   QNN backend (scaffold)
include/whisper_npu/metrics.hpp          Backend-neutral WER/CER + text normalization
app/main.cpp                             CLI runner: cold/warm load, inference bench,
                                         confidence, WER/CER, --json for the harness
msbuild/*.props                          Shared + per-backend build settings
projects/*/*.vcxproj, WhisperNpuHal.sln  MSBuild projects (Core static lib + App exe)
scripts/                                 setup / build / run / export / benchmark helpers
```

The application depends only on `whisper_npu/whisper_engine.hpp`. Backends are selected
at runtime via `create_engine(Backend, EngineOptions)`; which backends exist depends on
the compile-time toggles `EnableIntel` / `EnableAmd` / `EnableQualcomm`. Backends not
compiled with their SDK still link (as throwing stubs) so the repo always builds.

## Model caching (the fast-load story)

The first NPU load compiles the model to a device blob (slow, ~seconds). Set a cache
directory (`EngineOptions::cache_dir`, CLI `--cache <dir>`) and OpenVINO persists that
blob, so subsequent **warm** loads import it and are near-instant. The CLI loads the
engine twice (cold then warm) and prints both times plus the speedup. AMD/Qualcomm
backends have matching cache hooks stubbed (VitisAI EP context cache / QNN context
binary) for when they're implemented.

## Build & run (PowerShell 7)

The repo is **self-installing** — a fresh clone pulls everything it needs (toolchain,
SDK, model, audio); none of it is committed.

```powershell
git clone https://github.com/odobias/whisper-npu-hal
cd whisper-npu-hal
.\scripts\bootstrap.ps1       # installs prereqs + SDK + model + audio, then builds
.\scripts\run.ps1             # NPU, exported model, cold/warm cache demo
```

`bootstrap.ps1` runs these steps (also usable individually):

| Script | Fetches / does | Output (gitignored) |
|---|---|---|
| `setup-intel.ps1` | OpenVINO GenAI C++ SDK (download or link) | `third_party/` |
| `get-model.ps1` | export `whisper-tiny.en` to OpenVINO IR (Python venv + optimum-cli) | `models/whisper-tiny-en-ov/` |
| `get-amd-model.ps1` | download AMD ONNX Tiny + OpenAI tokenizer/config sidecars | `models/whisper-tiny-amd/` |
| `get-audio.ps1` | public-domain 16 kHz sample | `models/jfk.wav` |
| `build.ps1` | MSBuild Release\|x64 (prefers VS 2026) | `build/` |

If VS Build Tools is installed for the first time, reboot and re-run `bootstrap.ps1`.

Run directly if you prefer:

```powershell
.\build\x64\Release\WhisperNpuHal.App.exe <model_dir> <audio.wav> intel npu 5 --cache .\build\cache\npu
```

## Quantization benchmark (confidence + performance + accuracy)

The runner reports three axes per run, all backend-neutral (any backend fills what it
can; missing values are simply omitted):

- **Performance** — cold/warm load, mean/median/p90 latency, RTF, TTFT/TPOT, tok/s, model size.
- **Confidence** — mean per-token log-prob the model self-reports (`WhisperDecodedResults.scores`).
  Honest caveat: for whisper-tiny.en this barely moves across precisions, so treat it as
  weakly informative — **WER is the real discriminator.**
- **Accuracy** — WER/CER (`--ref "<text>"`) via the common `metrics.hpp`, computed identically
  regardless of backend.

Compare quantization *methods* across `variant × device × clip`:

```powershell
.\scripts\export-variants.ps1     # fp32, fp16, int8, int4 OpenVINO variants + models/manifest.json
.\scripts\get-eval-set.ps1        # small labeled LibriSpeech sample -> models/eval/eval.jsonl
.\scripts\benchmark.ps1 -Devices NPU -Runs 3   # -> build/reports/benchmark.{md,csv}
```

`manifest.json` is the backend-neutral contract: each entry has `backend`, `precision`,
`method`, `model_dir`, `devices`. Adding AMD/Qualcomm means **appending entries with
`backend=amd|qualcomm`** (and their own model dirs/devices) — the harness, metrics, and
report need no changes. Example NPU result (whisper-tiny.en, 12 clips):

| Variant | Prec | Size MB | Cold s | Warm s | Mean ms | xRT | WER % |
|---|---|---|---|---|---|---|---|
| fp32 | fp32 | 150.7 | 12.8 | 0.7 | 141.0 | 70.5 | 9.57 |
| fp16 | fp16 | 78.8 | 10.1 | 0.8 | 141.7 | 69.1 | 9.57 |
| int8 | int8 | 44.9 | 11.2 | 0.7 | 143.0 | 67.9 | 10.23 |
| int4 | int4 | 37.6 | 19.3 | 0.8 | 154.9 | 63.9 | 14.85 |

Takeaways: **fp16 is a free win** (half the size, identical WER); **int8** trades +0.7% WER
for 3.3× smaller; **int4** is both less accurate *and slower* on this NPU (dequant overhead),
so it's a poor fit here. Full W+A INT8 and the INT8-encoder/FP32-decoder hybrid aren't
auto-exported yet (full stateful INT8 doesn't run on the NPU — see the findings docs); the
manifest is ready to hold them as extra methods once wired.

## AMD Ryzen AI

AMD Ryzen AI / VitisAI:

```powershell
.\scripts\get-amd-model.ps1
.\scripts\get-audio.ps1
.\scripts\build.ps1 -EnableAmd -DisableIntel
.\scripts\run.ps1 -Backend amd -Device npu
```

The AMD backend expects `model_dir` to contain `tiny_encoder.onnx`,
`tiny_decoder.onnx`, `preprocessor_config.json`, `vocab.json`, and the VitisAI
config JSON files generated by `get-amd-model.ps1`. Runtime does not require
Python; the app copies the RyzenAI ONNX Runtime / VitisAI DLLs next to the exe.

## Benchmark Results

`run.ps1` appends benchmark rows to `results\benchmark-results.csv` by default:

```powershell
.\scripts\run.ps1 -Backend amd -Device npu
.\scripts\run.ps1 -Backend intel -Device npu
```

Each platform should run only its own backend/device tests locally. The CSV schema
is documented in `results\README.md`; use `-Results <path>` to write to another
CSV or `-NoResults` for scratch runs.

## Adding a backend

1. Implement `create()` / `available()` in `src/backends/<vendor>/...cpp` behind your
   `WHISPER_HAL_<VENDOR>` macro (stub otherwise).
2. Add include/lib/define wiring in `msbuild/backend.<vendor>.props`.
3. Toggle with `/p:Enable<Vendor>=true` (see `Directory.Build.props`).

## Status / caveats

- **Intel** and **AMD** backends are implemented. Qualcomm remains a structural
  scaffold with clearly-marked TODOs and compiles as a throwing stub by default.
- AMD currently supports CPU and NPU. GPU is not wired.
- 16 kHz mono WAV only (no resampler).
- See `../NPU-FINDINGS.md` and `../NPU-ECOSYSTEM-STATUS.md` for the underlying research.
