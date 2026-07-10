# fakeaudio on the Intel NPU (OpenVINO)

Answers the "UNVERIFIED on Intel OpenVINO" note in
`scripts/experiments/export_fakeaudio_variants.py`. On AMD's VAIML every fakeaudio
variant diverged (~0.99) and the model was declared CPU/GPU-only. **On Intel it is
different: the model runs correctly on the NPU** — but only after two independent
fixes, one from AMD's precision toolkit and one compile fix in this repo.

Host: Intel Core Ultra (Lunar Lake), NPU driver 10.0.26100.x, OpenVINO 2026.2.1,
`onnxruntime-openvino` 1.24.1 / OpenVINO 2025.4.1 for the ORT path. On AC power.
Reproduce: `.venv\Scripts\python.exe scripts\experiments\fakeaudio_npu_intel.py --bf16 --full-npu`
(after `export_fakeaudio_variants.py`).

## TL;DR

| path | device(s) | max\|dp\| vs fp32 CPU | flips | latency |
|---|---|---|---|---|
| **split: FE-fp32 + BB (surgery)** | **CPU + NPU** | **0.0001** | **0** | **~30 ms (≈15–22% CPU / 78–85% NPU)** |
| fp16-safe | CPU | 0.0001 | 0 | — |
| fp16-safe (`INFERENCE_PRECISION_HINT=f32`) | GPU | 0.0001 | 0 | — |
| fp16-safe (default / `f16` hint) | GPU | NaN | 4 | — (front-end re-downcast → underflow) |
| whole graph (surgery), f16 | NPU | 0.9924 | 1 | ~28 ms — **runs but mispredicts** |

`acc=3/5` on every correct path — that is the model's own score on the tiny
borderline fixture set (it exactly matches full-fp32 CPU), not an NPU regression.

## Why it needs two fixes

1. **Precision (AMD toolkit).** Whole-graph fp16 flips a *real* clip to *fake*.
   The front-end is a `power_to_db`: `mel = spectrogram @ melW` → `Clip(min=amin)`
   → `Log` → `/ln10` → `*10`, with `amin = 1e-10` (a −100 dB silence floor). The
   mel energies of silent bins are literally `0.0` even in fp32; the `amin` clamp
   is what stops `Log(0)`. In fp16 that `amin` constant is **below the smallest
   representable value** (fp16 min-normal ≈ 6.1e-5), so on the NPU the effective
   floor rises to ≈6.1e-5 — the log-mel silence floor jumps from **−100 dB (fp32)
   to −42 dB (fp16)**. That 58 dB lift on the many near-zero bins perturbs the
   transformer input enough to invert the one confident-"real" clip. Keeping the
   **log-mel front-end in fp32** preserves the −100 dB floor and fixes it
   (`model.fp16-safe.onnx`, or the `frontend-fp32`/`backbone-fp32` split). This is
   **not** an `-inf`/NaN underflow — see Verification below.

2. **Compile (this repo).** The backbone still crashes OpenVINO's `vpux`
   compiler: it fuses HTSAT windowed attention into SDPA and mis-flattens the bias
   add — `scores [nW,H,L,L] + bias [1,H,L,L]` becomes `IE.Add 1x64x64x64 +
   16x64x64` (non-broadcastable → `LLVM ERROR` process abort). Fix: **expand each
   Softmax-feeding bias-Add's `[1,H,L,L]` constant to the scores' full
   `[nW,H,L,L]` shape** (numerically identical, nothing left to mis-broadcast).
   Applied to 7 attention blocks by `fakeaudio_npu_intel.py`.

Neither fix alone lands it on the Intel NPU: without (1) it mispredicts, without
(2) it won't compile.

## Fully on the NPU?

Not correctly. The whole graph *executes* on the NPU (single compiled model, ~28
ms) but mispredicts, because the fp32-hungry front-end is forced to fp16.

- **bf16 would fix it** — its 8-bit exponent represents `amin = 1e-10` fine, so
  the −100 dB floor survives — **but the Intel NPU rejects it**:
  `INFERENCE_PRECISION_HINT` supports only `f16` and `i8`. There is no
  higher-exponent-range path on the NPU.
- Therefore the log-mel front-end must run off the NPU (CPU, ~4–8 ms) and only the
  transformer backbone goes to the NPU. That already puts ~80% of the work on the
  NPU and is numerically correct.

## Verification (measured, not inferred)

`scripts/experiments/verify_fakeaudio_fp16_underflow.py` instruments the actual
tensors on real labeled audio. Findings:

- **The clamp constant underflows, the tensors don't hit −inf.** `amin = 1e-10`
  casts to `0.0` in fp16, but the NPU floors the `Log` input at fp16 min-normal,
  not zero: on-device `Log` min = **−9.703 = ln(6.104e-5)** for *every* sample,
  vs fp32's **−23.026 = ln(1e-10)**. No `-inf`/NaN is produced on the NPU. (My
  first draft's "MatMul underflows to ~2e-17 → `Log(0)`" was wrong on both counts;
  the measured mel min is exactly `0.0` and the result is a raised floor.)
- **Causal proof.** Raising **only** `amin` (1e-10 → 6.104e-5) in an otherwise
  **fp32 CPU** graph reproduces the failure exactly:

  | sample | label | p(fake) fp32 | p(fake) amin=fp16-floor | |
  |---|---|---|---|---|
  | deepfake_1 | deepfake | 1.0000 | 1.0000 | same |
  | deepfake_2 | deepfake | 0.9981 | 0.9998 | same |
  | real_1 | real | 0.9995 | 1.0000 | same |
  | real_2 | real | 0.9984 | 0.9899 | same |
  | real_3 | real | **0.0048** | **0.9972** | **FLIPPED** |

  `real_3 → 0.9972` matches the full-NPU-fp16 run (0.9972) to four decimals, and
  nothing else flips — so the 58 dB noise-floor lift accounts for the *entire*
  error; the fp16 backbone (attention etc.) contributes nothing to the decision.
- **Why only one flip.** `real_1`/`real_2` are already scored as fake (0.999) in
  fp32 — the model's true fixture accuracy is 3/5. `real_3` is the sole confident
  "real", so it is the only decision a raised floor *can* flip.

## GPU caveat

`fp16-safe` is a genuine drop-in only on **CPU**. On the **GPU** you must pass
`INFERENCE_PRECISION_HINT=f32`; the default (or `f16`) re-downcasts the fp32
front-end and it NaNs again.

## Verdict

- **tsc** → Intel NPU, fp16, correct. Ship it.
- **fakeaudio** → Intel NPU via the **FE-fp32(CPU) + backbone(NPU, surgery)**
  split: correct, ~30 ms, ~80% on the NPU. This is the working on-device path that
  AMD's VAIML could not achieve.
