# fakeaudio log-mel front-end: the `MatMul → Clip → Log` precision trap

Self-contained write-up of *why* the fakeaudio (MS-CLAP / HTSAT) deepfake-audio
classifier is destroyed by fp16/int8, *exactly where* in the graph it happens, and
what does (and does not) fix it. Everything here is measured, not inferred — regenerate
with the scripts in [Reproduce](#reproduce).

## TL;DR

- The whole numerical disaster lives in **5 of the first 40 nodes** of a 2,395-node
  graph: the mel filterbank `MatMul` → `Clip` → `Log` chain in the `logmel_extractor`
  submodule.
- Cause is **underflow**, not overflow: the pre-`Log` mel energies span
  `absmax 434 … min|nonzero| 2.1e-17` — a dynamic range of **~2×10¹⁹**, which is
  ~7 orders of magnitude wider than fp16 can represent *at all* (~10¹² with subnormals).
  ~54 % of the mel tensor sits below fp16's smallest normal (6.1e-5) and snaps to zero.
- The `Clip` floor of `1e-10` — the safeguard meant to keep `Log` finite — is **itself
  unrepresentable in fp16** (below 6.1e-5), so in low precision it rounds to 0,
  `Log(0) = -∞`, and NaN propagates.
- **bf16 fixes the front-end completely** (same 8-bit exponent as fp32, min normal
  ~1.2e-38): every front-end edge goes to `max|Δp| = 0.0000`.
- It does **not** rescue the NPU: on AMD XDNA2 / VitisAI the HTSAT *backbone* diverges
  independently (0.99 fed clean CPU log-mel; 0.51 as calibrated int8). fakeaudio is a
  CPU/GPU model; the NPU is for whisper.

## Exact location

Base graph: `workloads/classifiers/fakeaudio/model.onnx` — 2,395 nodes, opset domains as
exported from MS-CLAP. Node indices are the topological position in `graph.node`.

| node idx | op | node name | note |
|---:|---|---|---|
| 5  | Unsqueeze | `/embedder/base/htsat/spectrogram_extractor/Unsqueeze` | consumes graph input `input [1, 308700]` |
| 22 | Pad | `/embedder/base/htsat/spectrogram_extractor/Pad` | reflect pad for STFT framing |
| 23 | Conv | `/embedder/base/htsat/spectrogram_extractor/conv_real/Conv` | STFT real (cos kernel), weight `…stft.conv_r` |
| —  | Conv | `/embedder/base/htsat/spectrogram_extractor/conv_imag/Conv` | STFT imag (sin kernel), parallel branch |
| 32 | Pow | `/embedder/base/htsat/spectrogram_extractor/Pow` | real² (and parallel `Pow_1` = imag²) |
| 35 | Add | `/embedder/base/htsat/spectrogram_extractor/Add` | **power spectrum** = real² + imag² |
| **36** | **MatMul** | `/embedder/base/htsat/logmel_extractor/MatMul` | **× mel filterbank** initializer `embedder.base.htsat.logmel_extractor.melW` ★ |
| **39** | **Clip** | `/embedder/base/htsat/logmel_extractor/Clip` | **floor = 1e-10** (a `Constant`) ★ |
| **40** | **Log** | `/embedder/base/htsat/logmel_extractor/Log` | → `Log_output_0` (the split seam / CUT) ★ |
| 42 | Div | `/embedder/base/htsat/logmel_extractor/Div` | first backbone consumer of `Log_output_0` |

Key tensor/initializer names (stable handles for programmatic extraction elsewhere):

- seam / CUT tensor: `/embedder/base/htsat/logmel_extractor/Log_output_0`
- mel filterbank weights: `embedder.base.htsat.logmel_extractor.melW`
- power spectrum: `/embedder/base/htsat/spectrogram_extractor/Add_output_0`
- graph input: `input`  (`float32 [1, 308700]`, i.e. 7 s of raw PCM @ 44.1 kHz)
- graph output: `output` (`float32 [1, 2]` raw logits; softmax applied outside the graph)

## Data path into the Log

```
input [1,308700]
  → Unsqueeze[5] → Pad[22]
  → Conv[23] conv_real (cos)  ┐   Conv conv_imag (sin) ┐
  → Pow[32] real²             ┘   Pow_1 imag²          ┘
  → Add[35]  power spectrum = real² + imag²      range 5.7e-16 … 2.4e4
  → MatMul[36] × melW  (mel filterbank pooling)  range 2.1e-17 … 434   ★ underflow origin
  → Clip[39]  floor 1e-10                        floor unrepresentable in fp16 ★
  → Log[40]                                      → log-mel, range [-23, 3.5]
  → Div[42] … HTSAT transformer (2,354 nodes) … → head → logits [1,2]
```

## Module provenance

`spectrogram_extractor` and `logmel_extractor` are **torchlibrosa** module names
(Kong et al., the PANNs/HTSAT audio front-end). This graph is a faithful ONNX export of
MS-CLAP's HTSAT feature extractor: a Conv-based STFT (two Convs with cosine/sine kernels
standing in for the DFT) → magnitude² → mel filterbank matmul → log. The entire librosa
pipeline was traced *into* the network instead of living in CPU preprocessing — which is
why the fragile linear-domain range ends up inside the model.

## Root cause: dynamic range vs fp16 capacity

Per-edge attribution (fp32 graph, one edge round-tripped fp32→low→fp32, rest fp32;
sample `real_3`; metric = `max|Δp(fake)|` vs fp32, which is what the C++ harness logs as
`max_abs_p_diff`):

| edge | absmax | min\|nonzero\| | % below 6.1e-5 | **fp16 max\|Δp\|** | flips | **bf16 max\|Δp\|** | flips |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conv (STFT) | 150 | 9.4e-11 | 15.1 % | 0.0000 | 0 | 0.0000 | 0 |
| Pow (r²) | 2.26e4 | 8.8e-21 | 69.4 % | 0.0040 | 0 | 0.0000 | 0 |
| Add (power spec) | 2.44e4 | 5.7e-16 | 64.1 % | 0.0082 | 0 | 0.0000 | 0 |
| **MatMul (mel)** | 434 | **2.1e-17** | **53.7 %** | **0.9838** | **1** | 0.0000 | 0 |
| **Clip** | 434 | 1e-10 | 53.7 % | **NaN** | **4** | 0.0000 | 0 |
| Log (log-mel) | 23.0 | 3.5e-4 | 0.0 % | 0.0000 | 0 | 0.0001 | 0 |

fp32 baseline `p(fake)`: deepfake_1 0.999951, deepfake_2 0.998115, real_1 0.999452,
real_2 0.998430, real_3 0.004814.

**fp16 capacity wall.** The pre-`Log` mel tensor's dynamic range is `434 / 2.1e-17 ≈
2.07×10¹⁹`. fp16's *entire* representable range is `65504 / 6.1e-5 ≈ 1.07×10⁹` (normals)
or `≈ 1.1×10¹²` including subnormals. The signal is ~7 orders of magnitude wider than the
format's whole window, so **no affine rescale or operator reorder of the existing ops can
fit it into fp16** — you can only slide a 10¹²-wide window along a 10¹⁹-wide signal.

**Why the earlier edges survive but the mel MatMul doesn't.** `Pow`/`Add` have *even more*
values under the floor (69 %, 64 %) yet barely move the result (Δp ≤ 0.008): zeroing quiet
energy *before* the mel sum is harmless because the loud bins dominate the filterbank sum.
It only detonates once those zeroed bins reach `Log`: the mel `MatMul` edge clamps quiet
bins from e.g. `Log(2e-17) = -38` to `Log(1e-10) = -23` (Δp 0.98, one flip); the `Clip`
edge is worse — its `1e-10` floor rounds to 0 in fp16 → `Log(0) = -∞` → NaN (4 flips).

**bf16 zeroes every edge** because it keeps fp32's 8-bit exponent, so 2e-17 and 1e-10 are
both representable, no underflow, the clip floor survives. (bf16 is marginally *worse* than
fp16 at the benign `Log` edge — 0.0001 vs 0.0000 — its 7-bit mantissa is coarser once range
is a non-issue. Range beats mantissa here, decisively.)

## Can reordering fix it (fp16)?

Only one rewrite works: **take the `Log` earlier and do the mel pooling in the log domain**
(logsumexp). Reformulate `mel[m] = Σ_f W[m,f]·P[f]` as
`logmel[m] = max_f ℓ_f + log Σ_f W[m,f]·exp(ℓ_f − max_f ℓ_f)` with `ℓ_f = log P[f]`, and get
`log P` straight from the STFT magnitude (`2·log|STFT|`) so the linear power spectrum is
never materialized in fp16. Then every boundary tensor lives in `~[-35, 10]`, trivially
fp16-safe. Caveats: it does **not** help the NPU (backbone diverges independently), it is
**not needed** for CPU/GPU (the fp32 front-end is 4 MB / microseconds and exact), and it adds
`Exp`/`Log` ops a fixed-function NPU may also mishandle. Worthwhile only if you specifically
need a *uniformly*-fp16 front-end for a runtime that refuses mixed precision.

## NPU reality (AMD XDNA2 / VitisAI, ORT 1.25.1)

The front-end trap is only half the story. Measured on the NPU:

| model on NPU | NPU `max_abs_p_diff` | note |
|---|---:|---|
| fp32 full model | 0.996 | VAIML auto-quantizes; front-end underflows |
| fp16-safe full model | 0.996 | VAIML ignores the fp16 annotations |
| **fp32 backbone only** (CPU-computed mel fed in) | **0.99** | backbone alone still blows up |
| calibrated int8 backbone (our QDQ scales) | 0.51 | + ~12× slower; VAIML won't honor QDQ faithfully |

So VAIML mishandles the **HTSAT backbone itself** (windowed attention, ~30 LayerNorms, 12
Softmaxes, the patch-grid `Resize`) regardless of any ORT-side precision control — a second,
independent failure from the front-end underflow. The backbone region is 2,354 nodes and is
mostly dynamic-shape plumbing (934 `Constant`, 262 `Unsqueeze`, 167 `Gather`, 144 `Concat`,
143 `Reshape`, 134 `Shape`), which is also what broke standalone shape inference until the mel
shape `[1,1,965,64]` was pinned, and what VAIML's partitioner choked on around `Resize`.

**Verdict:** fakeaudio runs on CPU/GPU with `model.fp16-safe.onnx`; the NPU is for whisper
(which ports cleanly). The split / int8-backbone artifacts remain *untested* candidates for
Intel OpenVINO / Qualcomm QNN — different compilers than VAIML, may behave; no such HW here.

## Reproduce

```powershell
# per-edge fp16/bf16 attribution table (ranges + Δp + flips)
.venv\Scripts\python.exe tools\research\fakeaudio_fp16_attribution.py

# static graph structure / op histograms / precision-sensitive ops
.venv\Scripts\python.exe tools\research\inspect_fakeaudio.py

# regenerate the fp16-safe / split / int8 variants from the base model
.venv\Scripts\python.exe tools\research\export_fakeaudio_variants.py
```
