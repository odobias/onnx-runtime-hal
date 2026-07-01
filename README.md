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
scripts/compare-devices.ps1              NPU vs GPU vs CPU comparison for one variant
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
report need no changes.

Reports land in `results/` (tracked): `quantization-benchmark.{md,csv}` (aggregate sweep)
plus one canonical per-variant row appended to `results/benchmark-results.csv` (the shared
cross-backend file — AMD/Intel rows share the same schema). Example NPU result
(whisper-tiny.en, 12 LibriSpeech clips):

| Variant | Prec | Size MB | Cold s | Warm s | Mean ms | xRT | WER % |
|---|---|---|---|---|---|---|---|
| fp32 | fp32 | 150.7 | 10.0 | 0.70 | 137.3 | 71.3 | 9.57 |
| fp16 | fp16 | 78.8 | 9.4 | 0.62 | 139.3 | 71.1 | 9.57 |
| int8 | int8 | 44.9 | 10.5 | 0.70 | 141.5 | 60.1 | 10.23 |
| int4 | int4 | 37.6 | 21.7 | 0.70 | 119.9 | 83.1 | 14.85 |

Takeaways (this NPU): **fp16 is a free win** — half the size, identical WER. **int8** trades
+0.7% WER for 3.3× smaller. **int4** is the smallest and, on these kernels, not slower at
inference — but it costs ~5 pts of WER and the longest cold compile (~22 s), so it's an
accuracy-poor trade for a tiny model. Warm cache makes every NPU load ~0.7 s regardless of
precision. (GPU/CPU land at slightly lower WER on fp32/fp16 — 8.6% vs the NPU's 9.6% —
purely from kernel numerics; see the full table in `results/`.)

Full W+A INT8 and the INT8-encoder/FP32-decoder hybrid aren't auto-exported yet (full
stateful INT8 doesn't run on the NPU — see the findings docs); the manifest is ready to hold
them as extra methods once wired.

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

## Device comparison (NPU vs GPU vs CPU)

CPU is not a separate backend or code path — it's the same Intel OpenVINO backend, just
targeting `Device::CPU` instead of `NPU`/`GPU` (`ov_device()` in
`intel_openvino_engine.cpp` maps all three to plain OpenVINO device strings). It's also
the one device guaranteed to exist on *any* x86_64 machine, making it the natural
baseline: "is the NPU actually worth it, or is CPU good enough?"

```powershell
.\scripts\run.ps1 -Device cpu                    # single run on CPU
.\scripts\run.ps1 -Device cpu -Threads 8         # pin OpenVINO to 8 inference threads
.\scripts\compare-devices.ps1                    # fp16 variant across NPU, GPU, CPU
.\scripts\compare-devices.ps1 -Devices CPU -Threads 16 -Runs 5
```

`--threads N` (CLI) / `-Threads N` (`run.ps1`/`compare-devices.ps1`) maps to OpenVINO's
`ov::inference_num_threads` and only applies to the CPU device; NPU/GPU use their own
scheduler and ignore it. It matters: on a 32-thread workstation, whisper-tiny.en (fp16)
went from **249 ms/clip at 1 thread to 93 ms/clip at 32 threads** (~2.7x) — CPU thread
count is not a cosmetic knob, it's the main lever you have.

`compare-devices.ps1` runs a single variant (fp16 by default) across every requested
device and pivots the report on *device* instead of *quantization method* (that's
`benchmark.ps1`'s job). Unsupported combos — no NPU present, an Intel-only `GPU` plugin
pointed at non-Intel silicon, etc. — are recorded as failures, not crashes, per the same
backend-neutral contract as the quantization benchmark.

Measured on this dev machine (AMD Threadripper PRO 7955WX, 16C/32T, NVIDIA T1000 — i.e.
**zero Intel NPU/GPU hardware**), whisper-tiny.en fp16, 12 LibriSpeech clips:

| Device | Status | Threads | Cold s | Warm s | Mean ms | xRT | tok/s | WER % |
|---|---|---|---|---|---|---|---|---|
| NPU | unsupported | - | - | - | - | - | - | - |
| GPU | unsupported | - | - | - | - | - | - | - |
| CPU | ok | 8/32 | 0.44 | 0.32 | 106.3 | 91.3 | 472.7 | 8.91 |

NPU fails because there's no Intel NPU on this box (`NPU_VCL` can't find a device to
compile for). GPU fails because OpenVINO's `GPU` plugin only targets **Intel**
GPUs (Level Zero / oneAPI) — it doesn't drive the NVIDIA card, so kernel selection fails
partway through compiling the model graph. Both are genuine `ok:false` results with a
real error string, not a script bug — that's the graceful-degradation contract working
as designed. **CPU's own number is real and unremarkable-in-a-good-way**: 91x real-time
on a 32-thread workstation chip is plenty fast for whisper-tiny.en; it just isn't the
point of this repo (an idle high-core-count CPU also isn't a fair power/latency
comparison against a purpose-built NPU on a laptop's power budget — treat this table as
"does it run and is it correct", not a verdict on NPU vs. CPU efficiency).

To get real NPU/GPU rows, run `compare-devices.ps1` on actual Intel NPU/GPU hardware
(Core Ultra / Meteor Lake or newer for NPU; Intel integrated or Arc/Flex for GPU) and
drop the resulting `build\reports\device-compare.md` numbers in here.

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
- OpenVINO's `GPU` device only drives **Intel** GPUs (integrated or Arc/Flex, via
  Level Zero/oneAPI); it will not use an NVIDIA/AMD GPU even if one is present.
  `CPU` runs on any AVX2 x86_64 chip, Intel or not — it's the universal fallback.
- Exporting variants (`export-variants.ps1` / `get-model.ps1`) needs `optimum-intel`,
  which currently breaks on **Python 3.14+** (`NormalizedConfig.__init__() got multiple
  values for argument 'allow_new'` — a `functools.partial`-as-descriptor change in
  3.14 trips up `optimum`'s `with_args()` config classes; see
  [huggingface/optimum#2409](https://github.com/huggingface/optimum/pull/2409),
  opened Feb 2026, stale-closed without merging). Use Python ≤3.13, or patch
  `.venv\Lib\site-packages\optimum\exporters\base.py`: change
  `self.NORMALIZED_CONFIG_CLASS(self._config)` to
  `self.__class__.NORMALIZED_CONFIG_CLASS(self._config)`. `export-variants.ps1` detects
  this on 3.14+ and prints the same hint on export failure.
- See `../NPU-FINDINGS.md` and `../NPU-ECOSYSTEM-STATUS.md` for the underlying research.
