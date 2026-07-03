# One Binary Across Vendors — Feasibility Notes

Research parked for later: can we ship **one** benchmark deploy that runs the unified
`whisper-tiny-en-static-onnx` model on Intel / AMD / Qualcomm NPUs? This is the
distilled answer, the evidence, and what to watch. Companion to the "Build once,
self-select everywhere" section of the top-level `README.md`.

_Researched 2026-07-03. Sources: onnxruntime.ai EP docs, the `onnxruntime/onnxruntime-qnn`
repo, AMD `ryzenai.docs.amd.com`, PyPI `onnxruntime-ep-openvino` / `onnxruntime-qnn`._

---

## TL;DR

1. **It's a version-floor + custom-core problem, not an "EPs can't coexist" problem.**
   ORT explicitly supports many EPs in one runtime (`build.bat --use_openvino
   --use_vitisai --use_dml`), and the modern **plugin-EP model**
   (`RegisterExecutionProviderLibrary` + `OrtEpFactory`/`OrtEpDevice` + auto-select
   `SetEpSelectionPolicy(PREFER_NPU)`) lets EPs load into a **stock** ORT core at runtime.
2. **Intel and Qualcomm are already stock-ORT plugins** in the same version neighborhood:
   - Intel `onnxruntime-ep-openvino` (PyPI, v1.5.2 2026-06-19) → stock ORT **>= 1.23**, bundles OpenVINO+TBB.
   - Qualcomm `onnxruntime-qnn` **>= 2.0.0** (plugin since 2026-03-31; v2.3.0 2026-06-22) → stock ORT **>= 1.24.2**, `onnxruntime_providers_qnn.dll`, maintained by Qualcomm.
3. **AMD is the sole holdout.** Ryzen AI ships its **own custom-compiled `onnxruntime.dll`**
   (~1.20–1.22) with VitisAI baked in, plus the `voe` graph passes, `vaip` runtime, and
   an xclbin firmware (`XLNX_VART_FIRMWARE`). No stock-ORT VitisAI plugin exists yet.
   That custom core is the seam that blocks a single x64 package.
4. **Cross-ISA can never be one binary.** Intel/AMD are x64; Qualcomm inference is ARM64
   (the x64 QNN wheel is AOT-compile-only). Two CPU ISAs => minimum **2 binaries**, forever.
   The achievable win is collapsing the *x64* side (Intel + AMD + DirectML) into one.

---

## Why two `onnxruntime.dll`s can't share a process

Each vendor currently pins a different ORT **core**, and two cores exporting the same
symbols under the same DLL name cannot co-load:

| Vendor | Delivery | Core it needs | Plugin against stock ORT? |
|---|---|---|---|
| Intel | `onnxruntime-ep-openvino` plugin (bundles OpenVINO+TBB) | stock **>= 1.23** | **Yes** |
| Qualcomm | `onnxruntime-qnn >= 2.0.0` plugin (`onnxruntime_providers_qnn.dll`) | stock **>= 1.24.2** | **Yes** (ARM64 inference) |
| AMD | Ryzen AI SDK's own `onnxruntime.dll` + `voe` + `vaip` + xclbin | **custom** ~1.20–1.22 | **No** |

AMD deployment DLLs (from AMD's own docs, matches our package exactly):
`onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_providers_vitisai.dll`
(thin ORT-side shim, ~452 KB), `onnxruntime_vitisai_ep.dll` (fat AMD impl, ~143 MB),
`onnxruntime_vitis_ai_custom_ops.dll`, `dyn_dispatch_core.dll`, `aiecompiler_client.dll`,
`vaiml.dll`, `DirectML.dll`, `zlib.dll`/`zstd.dll`. The EP is split: a thin upstream shim
(`providers_vitisai`) + AMD's fat fork (`vitisai_ep` + `vaiml` + `dyn_*`). The ABI seam
between an upstream shim and AMD's runtime is the integration risk below.

---

## Resolution paths (rated)

### Option A — Build ONE x64 ORT from source with both EPs
`build.bat --use_openvino <hw> --use_vitisai --use_dml --build_shared_lib` → one core +
`providers_openvino.dll` + `providers_vitisai.dll` + DML, colocated with the OpenVINO
runtime **and** AMD's `vaiml`/`dyn_*`/xclbin.
- **Feasibility: medium; real engineering, not a flag flip.** Mainline ORT lists
  "AMD64 / Ryzen AI / Windows" as a supported VitisAI target, so the shim is upstream.
- **Make-or-break risk:** the upstream `providers_vitisai.dll` shim at a stock >= 1.23/1.24
  core must be ABI-compatible with AMD's fork `onnxruntime_vitisai_ep.dll` + the Ryzen AI
  1.8 `vaip` runtime. AMD's XDNA2/Strix support may only live in *their* core at *their*
  version. Only an actual build+run on the XDNA NPU settles this.
- **Payoff:** collapses the two x64 DLL packs into one. Does **not** shrink size (~700 MB
  AMD runtime + fp32 weights are mandatory regardless).

### Option B — Stock core + plugins (clean endgame, blocked on AMD)
Stock ORT >= 1.24.2 core + Intel OpenVINO plugin (works today) + DML + a VitisAI plugin.
- **Blocked:** AMD does not publish a stock-core VitisAI plugin. The day they ship an
  `onnxruntime-ep-vitisai`-style package (mirroring Intel and Qualcomm — and their EP is
  already split into `providers_vitisai.dll`, so the plumbing is halfway there), this
  becomes runtime registration with zero from-source pain. **This is the thing to watch.**

### Option C — Status quo
One exe, swap the runtime DLL pack per host (`x64` = AMD/VitisAI, `x64-ovep` = Intel,
`ARM64` = QNN). What the repo does today. Works; not "single."

---

## Corrections / gotchas captured

- **QNN floor is 1.24.1/1.24.2, not 1.27.** PR #18's commit message ("QNN needs the
  plugin-EP API that ships in the ARM64/1.27 ORT") is imprecise vs upstream. If a 1.27
  assumption is baked into the ARM64 build, revisit it.
- **Multi-EP in one build still wants separate `InferenceSession`s per device.** ORT
  issues report you can't always list DML + OpenVINO in one EP list (memcpy errors); make
  a session per EP. Our engine already creates a session per device, so this is a non-issue
  for us.
- **VitisAI is invisible to `GetEpDevices()`** (registered the legacy way), so auto-EP
  hardware detection can't see the AMD NPU — see the `best_available_device()` probe notes
  in `src/backends/ort_static/ort_static_engine.cpp` (`WHISPER_HAL_LOG_PROBE=1`).

---

## Recommendation

- Keep the current 2-binary-per-ISA + per-host DLL-pack model; it's the honest state today.
- If a single x64 package becomes a priority, **spike Option A** as a time-boxed experiment:
  build ORT from source with `--use_openvino --use_vitisai --use_dml` and test whether the
  upstream VitisAI shim actually drives *this* XDNA NPU against the Ryzen AI 1.8 `vaip`
  runtime. That one test resolves the whole question; expect it may fail at the ABI seam.
- **Watch:** AMD publishing a stock-ORT VitisAI plugin. That flips Option B from blocked to
  trivial and makes the x64 unified package a runtime-registration exercise.
