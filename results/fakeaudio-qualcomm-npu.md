# fakeaudio on the Qualcomm NPU (QNN / Hexagon HTP)

Companion to `results/fakeaudio-intel-npu.md`. It replays the same
`export_fakeaudio_variants.py` precision variants on the Qualcomm Hexagon HTP
(the one NPU AMD's and Intel's write-ups could not cover). **Result: the model
runs correctly on the Snapdragon NPU via the FE-fp32(CPU) + backbone(NPU) split,
and — unlike Intel's `vpux` — the HTP needs no attention-bias compile surgery.**

Host: Snapdragon (Windows on ARM), Hexagon HTP, native **ARM64** build, ONNX
Runtime QNN EP (plugin ABI). Numbers are `--classify` agreement vs the FULL fp32
model's CPU probability (`expected_p`), so `max|Δp|` is end-to-end error against
the trusted CPU answer — directly comparable to the 0.93 the unmodified model
shows on the HTP.

Reproduce (after `get-deepfake-models.ps1`, from a repo with the `.venv`):

```
.venv\Scripts\python.exe scripts\experiments\export_fakeaudio_variants.py
.venv\Scripts\python.exe scripts\experiments\build_fakeaudio_variant_fixtures.py
build\ARM64\Release\WhisperNpuHal.App.exe --classify models\deepfake\fixtures\fakeaudio-bb-fp32 npu 20
```

## TL;DR (QNN HTP, vs full-fp32 CPU)

| path | input | device(s) | max\|Δp\| | cold load | infer (5 samples) | verdict |
|---|---|---|---|---|---|---|
| **backbone-fp32 split** | mel (CPU front-end) | **CPU + NPU** | **0.0008** | ~12–15 s | **113 ms** (p90 117) | ✅ **correct** |
| fp16-safe (whole graph) | raw PCM | NPU | 0.933 | ~17 s | 187 ms | ❌ front-end downcast |
| backbone-int8 split | mel (CPU front-end) | NPU | 0.9999 | ~9 s | 216 ms | ❌ int8 calibration inverts it |
| backbone-fp32 (sanity) | mel (CPU front-end) | CPU | 0.000000 | 0.34 s | — | ✅ fixture matches full model |

Untouched full model on the HTP: **0.93** divergence. `acc=3/5` on the correct
paths is the model's own score on the tiny borderline fixture set (it exactly
matches full-fp32 CPU), not an NPU regression — same caveat as the Intel doc.

## Cross-vendor summary

| Vendor / NPU | Working on-device path | max\|Δp\| | Extra fix vs precision split | Verdict |
|---|---|---|---|---|
| **Qualcomm** (Hexagon HTP) | FE-fp32 (CPU) + backbone-fp32 (NPU) | 0.0008 | **none** — HTP compiles the backbone as-is | ✅ correct, ~113 ms |
| **Intel** (Lunar Lake, OpenVINO) | FE-fp32 (CPU) + backbone (NPU) | 0.0001 | HTSAT bias-Add `[1,H,L,L]→[nW,H,L,L]` expand (else `vpux` LLVM abort) | ✅ correct, ~30 ms |
| **AMD** (Ryzen AI, VAIML) | none found | ~0.99 | — | ❌ CPU/GPU-only |

## Why the split is the only correct path (and why int8 fails)

1. **Precision.** Whole-graph fp16/quantized flips a real clip to fake: the
   mel-energy MatMul underflows (~2e-17, below fp16's 6.1e-5 floor) → `Log(0)` →
   the classifier inverts. Keeping the log-mel front-end in fp32 on the CPU and
   sending only the transformer backbone to the HTP fixes it (`max|Δp|` 0.93 →
   0.0008). This is the same root cause the AMD toolkit and Intel doc identify.
2. **fp16-safe is not a whole-graph drop-in on the NPU.** As a single graph on
   the HTP the fp32 front-end nodes still get executed in the NPU's reduced
   precision, so it reproduces the original 0.93 to the decimal. It is only a
   drop-in on CPU (and on GPU with an f32 precision hint) — exactly as on Intel.
3. **INT8 is worse than useless here.** The calibrated INT8 backbone is
   numerically dead (0.9999) *and* the slowest of the three on the HTP (216 ms).
   For a borderline binary classifier the calibration budget isn't enough; do not
   ship it.

## HTP vs vpux

The Hexagon HTP compiled and ran the fp32 backbone (HTSAT windowed attention and
all) with no graph surgery — the ~12–15 s cost is a one-time context compile, and
QNN context caching amortizes it on subsequent loads. Intel's `vpux` needed the
bias-Add broadcast expansion (`fakeaudio_npu_intel.py`) or it aborted with an
`LLVM ERROR`. So the Qualcomm on-device path is actually the simpler of the two.

## Verdict

- **fakeaudio → Qualcomm NPU** via **FE-fp32 (CPU) + backbone-fp32 (NPU)**:
  correct (`max|Δp|` 0.0008), ~113 ms backbone + a few ms CPU front-end, no
  compile surgery. Ship this split; do not ship the whole-graph fp16 or the int8
  backbone.
