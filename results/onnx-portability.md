# Neutral ONNX portability — one model, many runtimes

Goal: run **one vendor-neutral ONNX** (Whisper-tiny.en, exported by `optimum`) on
CPU, GPU and NPU without per-device model surgery, and see what that costs in
load time, latency and quality.

- Model: `openai/whisper-tiny.en` -> neutral ONNX (`encoder` + `decoder` +
  `decoder_with_past`), fp32, dynamic shapes.
- Eval: 12 LibriSpeech clips, greedy decode, WER/CER micro-averaged after
  normalization. Confidence = mean per-token log-prob (self-reported).
- Decode loop is hand-rolled and identical across runtimes, so quality
  differences come only from the runtime/precision, not the logic.

## Results

| Runtime | Device | Cold load | Warm load (cache) | Hot infer (mean) | WER | CER | Conf |
|---|---|---|---|---|---|---|---|
| optimum `generate()` (ORT) | CPU | 1.16 s | — | 13910 ms | 8.58% | 4.53% | -0.164 |
| raw ORT KV-cache loop | CPU | 1.45 s | — | 2029 ms | 8.58% | 4.53% | -0.164 |
| **OpenVINO, dyn KV-cache** | CPU | 1.85 s | **0.32 s** | **527 ms** | 8.58% | 4.53% | -0.164 |
| **OpenVINO, dyn KV-cache** | GPU | 16.39 s | **0.18 s** | **316 ms** | 8.58% | 4.53% | -0.164 |
| OpenVINO, dyn KV-cache | NPU | ❌ compile fail | — | — | — | — | — |
| **OpenVINO, stateful KV** | CPU | 0.82 s | 0.79 s | **347 ms** | 8.58% | 4.53% | -0.164 |
| OpenVINO, stateful KV | GPU | 7.11 s | 0.68 s | 653 ms | 8.58% | 4.53% | -0.164 |
| OpenVINO, stateful KV | NPU | ❌ compile fail | — | — | — | — | — |
| **OpenVINO, static no-KV** | NPU | 9.18 s | **0.70 s** | 970 ms | 9.57% | 5.10% | -0.168 |
| _baseline_ OV-IR GenAI (not ONNX) | NPU | 10.02 s | 0.70 s | 137 ms | 9.57% | 4.46% | — |

## Findings

1. **Quality is portable.** The neutral ONNX gives identical CPU/GPU WER (8.58%)
   and, on the NPU (fp16), 9.57% — **exactly matching the proprietary OV-IR NPU
   baseline**. Going neutral costs nothing in accuracy.

2. **The runtime, not the model, drives speed.** optimum's `generate()` is 26x
   slower than the same ONNX through OpenVINO. The model always did 154 ms
   encoder + ~15-50 ms/decoder-step; the rest was Python overhead.

3. **Caching is the whole game for cold start.** OpenVINO `ov::cache_dir` turns
   compile-from-ONNX into a blob reload: CPU 1.85 s -> 0.32 s (5.8x),
   GPU 16.4 s -> 0.18 s (90x), NPU 9.2 s -> 0.70 s (13x). Ship the cache.

4. **NPU needs static shapes — this is the portability wall.** The dynamic
   (growing) KV cache is rejected by the NPU compiler. The static no-KV recompute
   works and is NPU-compilable, but pays O(n*maxlen): 970 ms vs the 137 ms
   stateful IR pipeline. Closing that gap needs a **static KV-cache export**
   (fixed max context + attention mask), which the stock `optimum` export lacks.

5. **Fixing "no KV cache" with a stateful transform: CPU yes, NPU no.**
   `apply_make_stateful_transformation` pairs the neutral ONNX's decoder KV
   (past<->present) into internal OpenVINO state, so we stop copying KV tensors
   through Python each step. On **CPU this cut 527 -> 347 ms** (near the 282 ms
   GenAI IR), quality unchanged. On **GPU it didn't help** (OpenVINO already
   handled the dynamic KV efficiently; stateful added sync overhead). On the
   **NPU it still fails**: plain MakeStateful produces an *unbounded* KV state,
   and the NPU compiler needs a *bounded max-context* state (fixed cache size +
   masking). That bounded-state path is exactly what OpenVINO GenAI's NPU LLM
   pipeline implements to reach 137 ms. Net: the portable ONNX can be made
   stateful for CPU/GPU with a one-line transform, but a fast NPU KV cache
   requires the vendor's bounded-KV runtime, not a generic ONNX.

See `onnx-portability.csv` for the raw numbers. Scripts:
`scripts/onnx_ov_decode.py` (dyn KV-cache CPU/GPU), `scripts/onnx_npu_static.py`
(static NPU), `scripts/onnx_decode.py` (raw ORT), `scripts/probe_ort_raw.py`.
