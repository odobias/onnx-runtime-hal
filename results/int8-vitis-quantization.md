# INT8 quantization of the neutral Whisper-tiny.en ONNX (VitisAI attempt)

Goal: produce an INT8 **QDQ** copy of our vendor-neutral `whisper-tiny-en-onnx`
(encoder + decoder) that AMD's Ryzen AI / VitisAI EP can offload to the XDNA NPU,
*without* touching the FP32 model Intel/OpenVINO already runs. The INT8 file lives
in a separate variant dir; Intel keeps eating FP32.

> **CORRECTION (read first).** The premise that AMD *needs* INT8 was wrong.
> Inspecting `models/whisper-tiny-amd/` shows AMD ships a **pure FP32** model
> (all `float32` initializers, **zero** QuantizeLinear/DequantizeLinear nodes) and
> runs it on XDNA via the **VAIML partitioner** (`vaip-pass_vaiml_partition` in the
> `vitisai_config_*.json`), which compiles the FP32 graph to the AI Engine. So
> INT8 is **optional** for AMD, not required — it would only be a size/latency
> optimization. This experiment shows naive PTQ INT8 isn't shippable anyway, but
> the AMD path doesn't hinge on it. See "FP32 is the real AMD path" below.

Script: `scripts/experiments/quantize_int8_vitis.py` (calibration) + `scripts/experiments/_eval_int8.py`
(pure-ORT decode check). Tooling: `onnxruntime.quantization` 1.27 (no `vai_q_onnx`
/ AMD Quark installed). Calibration set: 12 LibriSpeech clips + `jfk.wav`, real mel
features via `WhisperFeatureExtractor`; decoder calibrated on real greedy-decode
prefixes (encoder hidden states + growing `input_ids`).

## What works

- **The pipeline produces valid QDQ INT8.** Encoder 31.4 -> ~8 MB, decoder
  188.8 -> 47.5 MB (per-tensor) / 104.6 MB (per-channel + FP32 LM head).
- **Format portability is real.** The QDQ INT8 model **loads and runs under
  OpenVINO** through our `intel-onnx` C++ backend (CPU verified), and under ONNX
  Runtime. So a single QDQ file *can* be consumed by both the Intel (OpenVINO) and
  AMD (VitisAI EP) stacks at the format level. That was the open question in
  `onnx-portability.md`; answer: yes, QDQ is the common currency.

## What does NOT work: accuracy collapses

Plain post-training static INT8 destroys this model. FP32 reference on `jfk.wav`:
`"And so my fellow Americans ask not what your country can do for you..."`
(25 tokens, avg_logprob -0.13). Every INT8 recipe collapses:

| variant | recipe | jfk decode (ORT, unpadded) | verdict |
|---|---|---|---|
| int8 (v1) | per-tensor, symmetric, MinMax | eot as **first** token (1 tok) | dead |
| int8-v2 | per-channel W, sym, MinMax, FP32 LM head | eot first (1 tok) | dead |
| int8-v3 | per-channel W, **asymmetric act**, FP32 LM head | ` in` (2 tok) | dead |

Component isolation (mix INT8/FP32 halves):

| encoder | decoder | output | read |
|---|---|---|---|
| INT8 v3 | FP32 | `You're not a good guy.` (8 tok) | encoder INT8 = ~100% WER hallucination, but coherent |
| FP32 | INT8 v3 | `a,,,,,,,,,,,...` (55 tok) | decoder INT8 = degenerate repetition / collapse |

So **both halves degrade, and the decoder is catastrophic.** The quantized decoder
predicts `eot` at position 0 with high confidence (the C++ path only limps 2 tokens
because it suppresses first-position `eot` via `begin_suppress_tokens`, then stops
at position 1). This is consistent in **both** ORT and OpenVINO -> it's the
quantization, not a runtime bug (same script's FP32 path decodes perfectly).

## Why (analysis, not excuses)

- Whisper's decoder residual stream + cross-attention over 1500 encoder positions
  is a known-hard PTQ target. 256 INT8 activation levels lose too much; symmetric
  wastes half the range on a non-zero-centered stream (asymmetric helped a hair,
  not enough).
- The usual fixes are exactly what stock ORT static quant does **not** do:
  SmoothQuant / cross-layer equalization, bias correction, per-op mixed precision,
  or QAT. AMD's `vai_q_onnx`/Quark bundles several of these with XDNA-aware defaults
  if you *do* want INT8.

## FP32 is the real AMD path (correcting the premise)

Inspection of `models/whisper-tiny-amd/` (via `scripts/experiments/_inspect_amd.py`):

| file | initializer dtypes | QDQ nodes | I/O | notes |
|---|---|---|---|---|
| `tiny_encoder.onnx` | float32 x68, int64 x4 | **0** | `x` f32 -> f32 | fused `LayerNormalization`, `Gelu` |
| `tiny_decoder.onnx` | float32 x107, int64 x7 | **0** | `x` i64, `xa` f32 -> f32 | no-KV, `Trilu` causal mask |

`config.json`: `torch_dtype: float32`, `_name_or_path: openai/whisper-tiny`
(the **multilingual** tiny, vocab 51865 — not `.en`). The `vitisai_config_*.json`
are VAIML **compiler** directives (`optimize_level: 3`, AIE `--system-stack-size`,
a `LayerNorm2PassAdf` accuracy mode), *external* to the `.onnx`.

Implications, correcting earlier statements:

- **"Run our model through Vitis to add metadata without killing Intel" — yes, and
  no quantization needed.** The "metadata" is a VAIML config JSON that sits *beside*
  the model. It never edits the graph, so Intel/OpenVINO is completely unaffected.
- **Our FP32 static export is precision-compatible with AMD's approach.** The gaps
  are now graph/op lineage and naming, not precision:
  - I/O names: ours `input_features`/`input_ids`/`encoder_hidden_states`/`logits`
    vs AMD's `x`/`xa`; the AMD harness hard-codes names.
  - Ours is an HF-optimum export **with KV-cache outputs**; AMD's is a clean
    **no-KV** OpenAI-style export with fused LayerNorm/Gelu/Trilu. Whether the
    VAIML partitioner compiles our exact ops (or spills more to CPU) is the open
    question — and it needs Ryzen AI hardware to answer.
  - Different checkpoint (multilingual vs `.en`): orthogonal to hardware.

## Calibration gotchas hit along the way

- **Percentile/Entropy calibrators crash** (`inf` / inhomogeneous shape) if you
  feed the eot-padded static decoder input: padded "future" positions produce
  non-finite activations. The runtime tolerates them (it reads only the current
  position), but the calibrator must see **real, unpadded** prefixes. We calibrate
  unpadded with MinMax.

## Verdict / next steps

- **For AMD support, don't quantize.** AMD runs FP32 via VAIML. The right move is
  to pair our FP32 static ONNX with a VAIML config (like AMD's), resolve the I/O
  naming / op-lineage gaps, and test on Ryzen AI hardware. Precision is a non-issue.
- **INT8 remains a *future perf/size optimization*, and naive PTQ won't deliver it.**
  If pursued: use **AMD Quark / `vai_q_onnx`** (SmoothQuant + XDNA-aware config) or
  QAT, keep the LM head (and likely attention matmuls) in higher precision, and
  consider mixed precision (INT8 FFN, FP16 attention).
- **Unverifiable on Intel hardware regardless.** VAIML compile (aiecompiler) and
  XDNA execution need the Ryzen AI stack + NPU.

What this experiment actually proved: (a) our pipeline produces valid, cross-vendor
loadable QDQ INT8; (b) naive INT8 collapses this Whisper decoder; and (c) — the
important one — **INT8 was never the blocker for AMD; FP32 is AMD's own path.**
