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
app/main.cpp                             CLI runner: cold/hot start, inference bench,
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
blob, so subsequent **hot** starts import it and are near-instant. The CLI loads the
engine twice (cold then hot) and prints both times plus the speedup. AMD/Qualcomm
backends have matching cache hooks stubbed (VitisAI EP context cache / QNN context
binary) for when they're implemented.

## Build & run (PowerShell 7)

The repo is **self-installing** — a fresh clone pulls everything it needs (toolchain,
SDK, model, audio); none of it is committed.

```powershell
git clone https://github.com/odobias/whisper-npu-hal
cd whisper-npu-hal
.\scripts\bootstrap.ps1       # auto-detects platform, installs its toolchain + model + audio, then builds
.\scripts\run.ps1             # NPU, exported model, cold/hot cache demo
```

`bootstrap.ps1` **detects the platform** (NPU/CPU vendor) and bootstraps *that platform's*
toolchain; override with `-Platform intel|amd|qualcomm`. It runs these steps (also usable
individually):

| Script | Platform | Fetches / does | Output (gitignored) |
|---|---|---|---|
| `setup-intel.ps1` | intel | OpenVINO GenAI C++ SDK (download or link) | `third_party/` |
| `setup-amd.ps1` | amd | Ryzen AI SDK + NPU driver (vendor installers, elevated) | `C:\Program Files\RyzenAI\...`, conda env |
| `setup-qualcomm.ps1` | qualcomm | link an installed QNN SDK (scaffold) | `third_party/qnn` |
| `get-model.ps1` | intel | export `whisper-tiny.en` to OpenVINO IR (Python venv + optimum-cli) | `models/whisper-tiny-en-ov/` |
| `get-amd-model.ps1` | amd | download AMD ONNX Tiny + OpenAI tokenizer/config sidecars | `models/whisper-tiny-amd/` |
| `get-audio.ps1` | all | public-domain 16 kHz sample | `models/jfk.wav` |
| `build.ps1` | all | MSBuild Release\|x64 (prefers VS 2026), right backend enabled | `build/` |

If VS Build Tools is installed for the first time, reboot and re-run `bootstrap.ps1`.

> **AMD note:** the Ryzen AI SDK and NPU driver ship no silent installer and need
> elevation, so `setup-amd.ps1` downloads them, launches the vendor installers
> (approve UAC + the wizard, keeping defaults), and polls for completion. Re-running
> after success is a fast no-op. Pin versions/URLs via its params if AMD moves them.

Run directly if you prefer:

```powershell
.\build\x64\Release\WhisperNpuHal.App.exe <model_dir> <audio.wav> intel npu 5 --cache .\build\cache\npu
```

## Model store (Hugging Face)

`models/` is gitignored (~1.5 GB of ONNX + OpenVINO variants) — committing it to GitHub
LFS would blow past the 1 GB free tier and its bandwidth cap immediately. Instead the
whole tree is snapshotted to the **private Hugging Face model repo
[`odobias/npu-hal-over-9000`](https://huggingface.co/odobias/npu-hal-over-9000)**, which
gives 100 GB free private storage and no LFS bandwidth throttling. Both scripts are wired
to that repo by default (override with `-Repo <user>/<name>`).

```powershell
hf auth login                    # once; token needs Write permission
.\scripts\get-models.ps1         # pull the model snapshot into models/ (after clone)
.\scripts\push-models.ps1        # re-snapshot models/ to HF after re-exporting
```

The repo is **private**, so pulling requires an HF account that's been granted read access
to it (a token in `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, or `hf auth login`); pushing
requires **write** access (owner). This is an alternative to reproducing models locally via
`get-model.ps1` / `get-amd-model.ps1` / `export-variants.ps1` — pull the exact pinned
artifacts instead of re-running the export toolchain.

## Quantization benchmark (confidence + performance + accuracy)

The runner reports three axes per run, all backend-neutral (any backend fills what it
can; missing values are simply omitted):

- **Performance** — cold/hot start, mean/median/p90 latency, RTF, TTFT/TPOT, tok/s, model size.
- **Confidence** — mean per-token log-prob the model self-reports (`WhisperDecodedResults.scores`).
  Honest caveat: for whisper-tiny.en this barely moves across precisions, so treat it as
  weakly informative — **WER is the real discriminator.**
- **Accuracy** — WER/CER (`--ref "<text>"`) via the common `metrics.hpp`, computed identically
  regardless of backend.

Compare every manifest entry across its declared `variant × device × clip` matrix:

```powershell
.\scripts\export-variants.ps1     # fp32, fp16, int8, int4 OpenVINO variants + models/manifest.json
.\scripts\get-amd-model.ps1       # AMD ONNX variant + merged models/manifest.json (on AMD machines)
.\scripts\get-eval-set.ps1        # small labeled LibriSpeech sample -> models/eval/eval.jsonl
.\scripts\benchmark.ps1 -Runs 3   # -> results/quantization-benchmark.{md,csv}
```

`manifest.json` is the backend-neutral contract: each entry has `backend`, `precision`,
`method`, `model_dir`, `devices`, and `size_mb`. Each platform script appends or updates
its own entries (`backend=amd|intel|qualcomm`) — the harness, metrics, and reports do not
need platform-specific changes.

Reports land in `results/` (tracked): `quantization-benchmark.{md,csv}` (aggregate sweep)
plus one canonical aggregate row per `variant × device` appended to
`results/benchmark-results.csv` (the shared cross-backend file — AMD/Intel rows share the
same schema). Example NPU result
(whisper-tiny.en, 12 LibriSpeech clips):

| Variant | Prec | Size MB | Cold s | Hot s | Mean ms | xRT | WER % |
|---|---|---|---|---|---|---|---|
| fp32 | fp32 | 150.7 | 10.0 | 0.70 | 137.3 | 71.3 | 9.57 |
| fp16 | fp16 | 78.8 | 9.4 | 0.62 | 139.3 | 71.1 | 9.57 |
| int8 | int8 | 44.9 | 10.5 | 0.70 | 141.5 | 60.1 | 10.23 |
| int4 | int4 | 37.6 | 21.7 | 0.70 | 119.9 | 83.1 | 14.85 |

Takeaways (this NPU): **fp16 is a free win** — half the size, identical WER. **int8** trades
+0.7% WER for 3.3× smaller. **int4** is the smallest and, on these kernels, not slower at
inference — but it costs ~5 pts of WER and the longest cold compile (~22 s), so it's an
accuracy-poor trade for a tiny model. Hot cache makes every NPU load ~0.7 s regardless of
precision. (GPU/CPU land at slightly lower WER on fp32/fp16 — 8.6% vs the NPU's 9.6% —
purely from kernel numerics; see the full table in `results/`.)

Full W+A INT8 and the INT8-encoder/FP32-decoder hybrid aren't auto-exported yet (full
stateful INT8 doesn't run on the NPU — see the findings docs); the manifest is ready to hold
them as extra methods once wired.

## AMD Ryzen AI

AMD Ryzen AI / VitisAI:

```powershell
.\scripts\setup-amd.ps1                       # Ryzen AI SDK + NPU driver (once)
.\scripts\get-amd-model.ps1                   # also creates/merges models\manifest.json
.\scripts\get-audio.ps1
.\scripts\build.ps1 -EnableAmd -DisableIntel
.\scripts\run.ps1 -Backend amd -Device npu
```

`setup-amd.ps1` installs Ryzen AI (default `C:\Program Files\RyzenAI\1.8.0-beta`,
conda env `ryzen-ai-1.8.0-beta`) and the matching NPU driver (min `32.0.20101.3760`),
then sets `RYZEN_AI_INSTALLATION_PATH` so `msbuild/backend.amd.props` finds the ORT/VitisAI
headers and libs. `.\scripts\bootstrap.ps1 -Platform amd` does all of the above in one shot.

The AMD backend expects `model_dir` to contain `tiny_encoder.onnx`,
`tiny_decoder.onnx`, `preprocessor_config.json`, `vocab.json`, and the VitisAI
config JSON files generated by `get-amd-model.ps1`. Runtime does not require
Python; the app copies the RyzenAI ONNX Runtime / VitisAI DLLs next to the exe.

## Benchmark Results

`run.ps1` appends benchmark rows to `results\benchmark-results.csv` by default:

```powershell
.\scripts\run.ps1 -Backend amd -Device npu
.\scripts\run.ps1 -Backend intel -Device npu
.\scripts\benchmark.ps1 -Runs 3               # manifest-driven all-clip sweep
```

Each platform should run only its own manifest entries locally. The CSV schema is
documented in `results\README.md`; use `-Results <path>` to write benchmark sweeps to
another CSV, or `-NoResults` on `run.ps1` for scratch single-clip runs.

## Device comparison (NPU vs GPU vs CPU)

CPU is not a separate backend — it's the same manifest-selected backend targeting
`Device::CPU` instead of `NPU`/`GPU`. It's also the one device likely to exist on any
x86_64 machine, making it the natural baseline: "is the NPU actually worth it, or is CPU
good enough?"

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
`benchmark.ps1`'s job). It reads `backend` from the same manifest entry as the canonical
benchmark. Unsupported combos — no NPU present, GPU not wired for a backend, etc. — are
recorded as failures, not crashes, per the same backend-neutral contract.

Measured on this dev machine (AMD Threadripper PRO 7955WX, 16C/32T, NVIDIA T1000 — i.e.
**zero Intel NPU/GPU hardware**) with `compare-devices.ps1`, whisper-tiny.en fp16, 12
LibriSpeech clips:

| Device | Chip | Status | Threads | Cold s | Hot s | Mean ms | xRT | tok/s | WER % |
|---|---|---|---|---|---|---|---|---|---|
| NPU | - | unsupported | - | - | - | - | - | - | - |
| GPU | - | unsupported | - | - | - | - | - | - | - |
| CPU | AMD Ryzen Threadripper PRO 7955WX 16-Cores | ok | 8/32 | 0.44 | 0.32 | 106.3 | 91.3 | 472.7 | 8.91 |

NPU fails because there's no Intel NPU on this box (`NPU_VCL` can't find a device to
compile for). GPU fails because OpenVINO's `GPU` plugin only targets **Intel**
GPUs (Level Zero / oneAPI) — it doesn't drive the NVIDIA card, so kernel selection fails
partway through compiling the model graph. Both are genuine `ok:false` results with a
real error string, not a script bug — that's the graceful-degradation contract working
as designed.

### Cross-machine comparison (via the shared results ledger)

`results/benchmark-results.csv` is the cross-machine ledger (see `results/README.md`):
every machine appends its own canonical rows with the same schema, so results from
different hardware concatenate without any code changes. It already had one full
NPU/GPU/CPU × fp32/fp16/int8/int4 sweep from a real Intel NPU/GPU laptop; this machine
added its own CPU-only rows on top (same clip `ls_000`, same `--ref`, `runs=2` — matched
methodology, so the comparison below is apples-to-apples, not vibes):

Every row now also self-identifies the exact chip (`device_full_name`), not just the
logical `NPU`/`GPU`/`CPU` class — the Intel backend queries OpenVINO's
`ov::device::full_name` for whatever device it just ran on (e.g. `Intel(R) AI Boost`,
`13th Gen Intel(R) Core(TM) i7-1370P`), so "what NPU/CPU are we actually comparing
against" is answered by the CSV itself instead of tribal knowledge. That query was
added after the laptop sweep below was recorded, so those older rows predate it (empty
`device_full_name`); new rows from any machine will have it populated.

| Precision | Their CPU | Their GPU | Their NPU | **This CPU (AMD Ryzen Threadripper PRO 7955WX, 32T)** | Speedup vs their CPU |
|---|---|---|---|---|---|
| fp32 | 267.7 ms (21.9x) | 119.3 ms (49.1x) | 115.0 ms (50.9x) | **77.9 ms (75.2x)** | 3.44x |
| fp16 | 227.8 ms (25.7x) | 105.2 ms (55.7x) | 109.6 ms (53.4x) | **68.5 ms (85.5x)** | 3.33x |
| int8 | 194.2 ms (30.2x) | 161.7 ms (36.2x) | 113.0 ms (51.8x) | **66.4 ms (88.2x)** | 2.92x |
| int4 | 187.9 ms (31.2x) | 122.6 ms (47.7x) | 84.2 ms (69.6x) | **73.1 ms (80.1x)** | 2.57x |

(mean latency, xRT in parens; WER/CER were identical to their numbers at every
precision on this clip — same model, same math, just different silicon.)

The uncomfortable-for-NPU-marketing part: **this workstation's CPU alone beats their
dedicated NPU and GPU on raw latency, at every precision.** That is a real, measured
result, not a benchmarking mistake — and it is *not* the flex it looks like. A
32-thread desktop-class CPU pulls an order of magnitude more power than a laptop NPU,
and whisper-tiny is small enough that a NPU's fixed per-call overhead (compile-time
batching, driver dispatch) eats into its efficiency advantage before the model is big
enough to need it. This table answers "which is faster on this specific tiny model,
right now" — it does not answer "which is more efficient" or "which scales better to
larger Whisper checkpoints," and treating a latency win here as a verdict on NPUs in
general would be exactly the kind of unearned generalization worth pushing back on.

To add another machine's rows, run the loop below on it and let it append to (or, once
merged, git-diff-and-commit into) `results/benchmark-results.csv`:

```powershell
.\scripts\export-variants.ps1
.\scripts\get-eval-set.ps1
foreach ($id in (Get-Content .\models\manifest.json | ConvertFrom-Json).variants.id) {
    .\build\x64\Release\WhisperNpuHal.App.exe .\models\variants\$id .\models\eval\ls_000.wav `
        intel cpu 2 --cache ".\cache\$id-CPU" `
        --ref "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL" `
        --results .\results\benchmark-results.csv --label $id
}
```

Swap `cpu` for `npu`/`gpu` on hardware that has them. `benchmark.ps1` is deliberately
**not** used for this — it overwrites (not appends) `results/quantization-benchmark.*`,
which would blow away the other machine's NPU/GPU sweep already committed there;
`results/benchmark-results.csv` is the only file in `results/` designed to be
appended to by multiple machines.

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
- See `../NPU-FINDINGS.md` and `../NPU-ECOSYSTEM-STATUS.md` for the underlying research,
  and `results/cpp-onnx-npu-findings.md` for the C++ ONNX-on-NPU key findings (decode
  strategy, the `maxlen` dud, the mel-FFT win, and GenAI-parity verdict).
