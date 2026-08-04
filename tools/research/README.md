# Deepfake classifier experiments (ONNX Runtime)

> **Benchmark of record: `benchmark\run-suite.ps1` (the C++ app).** All
> serious/comparable performance numbers come from the C++ ONNX Runtime path —
> it is the only one that reaches the real vendor NPU EPs (VitisAI / QNN /
> OpenVINO). Everything in *this* directory is **reference / background
> tooling**: fixture generators (`dump_fixtures.py`, still required by the C++
> benchmark), correctness validators (`validate_*.py`), model exporters, and
> the two wheel-limited Python latency probes (`benchmark_deepfake_onnx.py`,
> `bench_static_kv_ov.py`). The Python probes are handy sanity checks but their
> latency is **not** authoritative — cite the C++ harness instead.

Standalone ONNX Runtime harness for the deepfake pipeline's two classifier
models. These are plain ORT classifiers (not OpenVINO GenAI, not the C++
`IWhisperEngine` HAL contract), so they live here rather than in the app.

| Script | What it does |
|---|---|
| `benchmark_deepfake_onnx.py` | Load/latency **and** correctness benchmark across ORT execution providers. Writes `results/reports/deepfake-benchmark.{csv,md}`. |
| `validate_tsc_onnx.py`       | Correctness-only check for the Text Scam Classifier on its own labeled sanity set. |
| `validate_fakeaudio_onnx.py` | Correctness-only check for the Generated Audio Detector on labeled clips. |

The benchmark drives the same real labeled inputs the two validators define
(single source of truth), so every provider is timed on real data and scored
for accuracy in one pass. Data provenance and caveats live in each validator's
module docstring — the accuracy numbers are **plausibility signals, not
certified accuracy** (small samples, best-effort preprocessing).

## C++ classifier harness (`NpuInferenceBench.App --classify`)

The Python harness above only reaches the EPs a pip wheel ships (CPU + DirectML,
or one vendor's stock plugin). The **real vendor NPU paths** (AMD VitisAI,
Qualcomm QNN) only exist in the C++ app's ONNX Runtime backend. So the same two
classifiers can now be replayed through the C++ app to get true on-NPU numbers,
without re-implementing the (non-trivial, already-validated) preprocessing in C++:

1. `dump_fixtures.py` snapshots the validators' exact feeds **and** the CPU
   reference probability to `artifacts\workloads\classifiers\fixtures\<model>\` (gitignored):
   `model.tsv` / `samples.tsv` / `inputs.tsv` + raw `data\*.bin` tensors.
2. `NpuInferenceBench.App --classify <fixture_dir> [device] [runs]` loads those
   tensors, builds an ORT session on the requested EP (NPU→GPU→CPU fallback, or
   `--provider <EP>` verbatim), times inference, and scores each sample for
   correctness **and** numerical agreement with the CPU reference `p`
   (`max |p − cpu_ref_p|` — a nonzero value means the accelerator silently changed
   the output). Writes `results\ledgers\classifiers.csv` (per-sample p in the
   `eval_detail` column) and stamps `host_arch`/`host_os`/`runtime_version`.

```powershell
# Build the ORT flavour of the app (also fetches the C++ ORT SDK if missing):
.\tools\setup\get-ort-sdk.ps1                       # DirectML SDK -> artifacts\third_party\onnxruntime
.\tools\build\build.ps1 -EnableOrt -DisableIntel

# Snapshot fixtures once (needs the Python venv + models), then replay in C++:
artifacts/venv\Scripts\python.exe tools\fixtures\generate.py
$exe = "build\x64\Release\NpuInferenceBench.exe"
& $exe --classify artifacts\workloads\classifiers\fixtures\tsc       cpu 20 --results results\ledgers\classifiers.csv
& $exe --classify artifacts\workloads\classifiers\fixtures\tsc       gpu 20 --results results\ledgers\classifiers.csv
& $exe --classify artifacts\workloads\classifiers\fixtures\fakeaudio npu 20 --provider VitisAIExecutionProvider --results results\ledgers\classifiers.csv
```

Measured on this x64 box (CPU + DirectML GPU), matching the Python results
(TSC 14/15, FakeAudio 3/5 — the misses are the documented `scam_0` false-negative
and the two production-flagged `real_*` false-positives), with every EP
reproducing the CPU probabilities (`max |Δp|` ≤ 1e-6):

| model | EP | mean ms | accuracy |
|---|---|---|---|
| tsc | CPUExecutionProvider | ~79 | 14/15 |
| tsc | DmlExecutionProvider | ~37 | 14/15 |
| fakeaudio | CPUExecutionProvider | ~66 | 3/5 |
| fakeaudio | DmlExecutionProvider | ~38 | 3/5 |

The VitisAI/QNN/OpenVINO EP branches are wired in `src\workloads\classifiers\ort_classifier.cpp`
but **untested** — they need the matching hardware/SDK (Ryzen AI, ARM64 Snapdragon,
OVEP build) and the corresponding `-EnableAmd` / `-EnableQualcomm` / `-EnableOvep`
build. fp32 graphs also need vendor quantization/compile to actually land on an NPU.

## Quick start (this x64 box: CPU + DirectML GPU)

```powershell
# from repo root, with the project venv
artifacts/venv\Scripts\python.exe -m pip install -r tools\research\requirements.txt
artifacts/venv\Scripts\python.exe -m pip install onnxruntime-directml
artifacts/venv\Scripts\python.exe tools\research\benchmark_deepfake_onnx.py
```

Models come from `tools\fetch\get-classifier-models.ps1` (gitignored under
`artifacts\workloads\classifiers\`). The FakeAudio sample clips are gitignored too; if they're
absent that model falls back to synthetic latency-only and reports blank
accuracy.

## Per-platform install matrix

There is **no single wheel with every execution provider**, and the vendor NPU
runtimes cannot coexist in one process. Install exactly one onnxruntime flavour
per environment, matching the accelerator you want to measure:

| Platform / accelerator | ISA | Install | EP name | Notes |
|---|---|---|---|---|
| CPU + DirectML (any GPU) | x64 | `pip install onnxruntime-directml` | `DmlExecutionProvider` | What this repo uses today on x64. |
| Intel NPU / GPU (OpenVINO) | x64 | `pip install onnxruntime-openvino` | `OpenVINOExecutionProvider` | Stock-ORT plugin, bundles OpenVINO+TBB. |
| AMD NPU (Ryzen AI / VitisAI) | x64 | **Ryzen AI SDK conda env** (`tools\setup\setup-amd.ps1`) | `VitisAIExecutionProvider` | Ships its OWN custom `onnxruntime.dll`; don't pip another ORT into that env. fp32 graph needs AMD quantization/compile to actually land on the NPU (see `quantize_int8_vitis.py`). |
| Qualcomm NPU (QNN) | **ARM64** | `pip install onnxruntime-qnn` (ARM64 Python) | `QNNExecutionProvider` | Inference is ARM64-only; the x64 QNN wheel is AOT-compile-only. This is why an ARM64 build is mandatory, not optional. |

`benchmark_deepfake_onnx.py` self-selects: with no `--providers` it runs CPU
plus whatever accelerator EPs the installed ORT actually exposes. Force a set
with e.g. `--providers cpu qnn` or `--providers cpu vitisai`. Aliases: `cpu`,
`gpu`/`dml`, `qnn`, `vitisai`, `openvino`.

Each run stamps `host_arch`, `host_os`, and `ort_version` into the CSV so rows
gathered on x64/DML, x64/VitisAI, and ARM64/QNN stay distinguishable in the
shared ledger.

### Pointing the loader at a vendor runtime

If a vendor's dependent DLLs (e.g. Ryzen AI's `vaip` / `dyn_dispatch` / xclbin)
live outside the Python package, prepend their directory to the native search
path before ORT loads:

```powershell
$env:NPU_INFERENCE_BENCH_ORT_DLL_DIR = "C:\Program Files\RyzenAI\1.8.0-beta\deployment"
python tools\research\benchmark_deepfake_onnx.py --providers cpu vitisai
```

This only fixes dependency resolution for the ORT core already installed in the
env — it does **not** swap which core the wheel is bound to (see below).

## "Can we rename AMD's custom `onnxruntime.dll` to `onnxruntime-amd.dll`?"

**Short answer: no — not as a plain file rename, and it wouldn't buy what you'd
want it for.** Two independent reasons:

1. **Implicit linking binds by filename.** The consumers of that DLL —
   Python's `onnxruntime_pybind11_state.pyd`, and AMD's own
   `onnxruntime_providers_shared.dll` / `onnxruntime_providers_vitisai.dll` —
   all carry the literal string `onnxruntime.dll` in their PE import tables.
   The Windows loader resolves by that exact name. Rename the file and every
   dependent fails to load ("The specified module could not be found").
   Making the new name work means patching the import table of *every*
   dependent (a binary edit, or a from-source rebuild with a renamed target) —
   and AMD's core is a closed fork, so that's a non-starter.

2. **Even if renamed+patched, it solves nothing.** The only reason to rename
   would be to drop AMD's core next to a stock `onnxruntime.dll` and pick one
   at runtime. But two ORT cores can't co-load in one process regardless of
   filename — duplicate exported symbols and internal singletons collide. AMD
   ships a *custom core* (~1.20–1.22 with VitisAI baked in), not a stock-core
   plugin, so it's the whole seam that blocks a single package. The full
   historical analysis is retained in the private benchmark archive.

**What to do instead — directory separation (what the repo already does):**
keep the canonical `onnxruntime.dll` name *inside each vendor's DLL pack*, in
its own folder, and control which folder is on the loader search path:

```
runtime/
  x64-amd/     onnxruntime.dll (AMD custom) + providers_* + vaip/dyn_*/xclbin + DirectML.dll
  x64-ovep/    onnxruntime.dll (stock)      + OpenVINO + TBB
  arm64-qnn/   onnxruntime.dll (stock ARM64) + onnxruntime_providers_qnn.dll + QNN libs
```

Select per host via `NPU_INFERENCE_BENCH_ORT_DLL_DIR` (Python, above),
`SetDllDirectory` / `AddDllDirectory` (C++ app), or simply a separate
venv/conda env per vendor. Same binary, swapped runtime pack — no rename, no
symbol clash.

> Footnote for the C++ side: a differently-named ORT *is* usable if you load it
> **fully dynamically** — `LoadLibrary("onnxruntime-amd.dll")` +
> `GetProcAddress("OrtGetApiBase")`, never implicitly linking. That's a
> deliberate app-loader design, not something the Python wheel does, and it
> still can't share a process with a second core.

## Outputs

These files are generated locally and ignored by the public repository.
Reviewed campaigns are retained in the private benchmark archive.

- `results/reports/deepfake-benchmark.csv` — one row per (model, EP, run); includes
  per-sample `eval_detail` (label / prediction / p) and host/arch/ORT stamps.
- `results/reports/deepfake-benchmark.md` — latency+accuracy summary plus per-model,
  per-sample probability tables (doubling as a cross-EP numerical-agreement
  check: identical p across EPs means the accelerator isn't silently changing
  outputs).
