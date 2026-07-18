# C++ ONNX-on-NPU — Key Findings

Hard-won lessons from porting the neutral-ONNX Whisper benchmark into the C++ HAL
(`Backend::IntelOnnx`) and optimizing it for the Intel NPU. Companion to the
detailed numbers in `onnx-portability.md`; this doc is the distilled "what we
learned, what to do next time."

_Hardware: Intel Lunar Lake (Core Ultra 200V), NPU "Intel(R) AI Boost", iGPU
"Intel(R) Graphics", 16-thread CPU. OpenVINO 2026.2.1. Model:
`openai/whisper-tiny.en` neutral ONNX (optimum export)._

---

## TL;DR

1. **On the NPU, only the static no-KV decode compiles.** Dynamic (growing) KV and
   OpenVINO generic `MakeStateful` (unbounded state) are both rejected by the NPU
   compiler. Static reshape + full-sequence recompute is the one portable path —
   same strategy AMD's prepared model uses.
2. **The KV cache was a red herring for latency.** A `maxlen` sweep proved the
   decode loop is dominated by *fixed* per-step costs, not the recompute a KV cache
   would remove.
3. **The real bottleneck was our own mel front-end** — a naive DFT eating ~half the
   runtime. Replacing it with a threaded Bluestein FFT cut total latency ~2× on
   NPU/GPU with **bit-identical output**.
4. **Net:** GPU 334→166 ms (≈ GenAI's 137 ms), NPU 452→234 ms. The portable ONNX
   now runs *close* to the vendor GenAI pipeline without any bounded-KV surgery.

---

## 1. NPU decode strategy: what actually compiles

| Strategy | CPU | GPU | NPU | Notes |
|---|:--:|:--:|:--:|---|
| dynamic (growing) KV cache | ✅ | ✅ | ❌ | NPU: "upper bounds not specified" (dynamic dims) |
| generic `MakeStateful` (unbounded state) | ✅ | ✅ | ❌ | Produces *unbounded* state; NPU needs *bounded* |
| **static reshape + no-KV recompute** | ✅ | ✅ | ✅ | The portable path we shipped |
| GenAI bounded-KV + attention mask (OV-IR) | ✅ | ✅ | ✅ | 137 ms reference; **not** the stock ONNX graph |

**Why the stock ONNX can't do bounded-KV:** Whisper's `decoder_with_past` has **no
attention-mask input** for the cached self-attention — it assumes `past_len ==
real_len`. You can't reshape a fixed-size padded past onto it without the softmax
attending to garbage. Bounded-KV requires re-exporting/authoring a graph with a
mask input (i.e. reproducing what GenAI's NPU static-LLM pipeline already does).

---

## 2. `maxlen` is not a useful lever (measured, not guessed)

Sweep on NPU (`WHISPER_ONNX_MAXLEN`, 25-token clip):

| maxlen | total infer | decode/token |
|---|---|---|
| 128 | 358 ms | 7.0 ms |
| 96 | 371 ms | 6.1 ms |
| 48 | 392 ms | 5.5 ms |
| 32 | 393 ms | 5.2 ms |

Shrinking the static context barely moved decode/token and left total latency flat
(noise-dominated). **The per-step decoder cost is fixed-cost bound:**
cross-attention over the **1500 encoder positions** (constant every step) + the
**full-vocab logits projection**. Consequence: even a *perfect* bounded KV cache
would reclaim only ~50–80 ms of the ~170 ms decode — high effort, low yield.

---

## 3. The mel front-end was the real bottleneck

Per-stage split on NPU (`WHISPER_ONNX_PROFILE=1`, naive mel, maxlen=128, 25 tok):

| stage | time | share |
|---|---|---|
| **mel spectrogram (naive DFT)** | ~180 ms | ~48% |
| encoder | ~19 ms | ~5% |
| decode loop | ~170 ms | ~46% |

The naive DFT is `O(frames·bins·nfft)` ≈ 480M mul-adds, single-threaded — nothing to
do with the KV cache.

**Fix — Bluestein FFT, threaded across frames:**

- Whisper's `n_fft = 400` is not a power of two. **Do not zero-pad to 512** — that
  shifts the frequency bins and corrupts the mel output. Bluestein computes the
  *exact* 400-point DFT via radix-2 FFTs (chirp kernel precomputed once).
- The 3000 frames are independent → fan out across all cores.

| device | naive mel | fft mel | speedup |
|---|---|---|---|
| **NPU** | 452 ms (24×) | **234 ms (47×)** | 1.9× |
| **GPU** | 334 ms (33×) | **166 ms (66×)** | 2.0× |
| CPU | 1549 ms | 1616 ms | none |

**CPU sees no win** — OpenVINO inference already saturates every core, so threading
the mel just competes with it, and decode is the CPU wall anyway.

**Correctness verified by A/B:** naive vs FFT produced **byte-identical
transcription** and confidence −0.1329 vs −0.1331 (float noise). Both also match the
Python reference, confirming the in-process mel filterbank (librosa Slaney bank,
computed in C++ because the neutral export omits `mel_filters`) and the FFT are
exact reproductions of whisper's front-end.

---

## 4. Engineering notes for future maintainers

- **Shared front-end:** `include/npu_inference_bench/whisper_frontend.hpp` (header-only) —
  mel (naive + FFT), byte-level BPE detokenizer, minimal JSON readers. Dependency-
  free of ONNX/OpenVINO so any backend can reuse it.
- **Mel filterbank is computed in-process** (`compute_mel_filterbank()`) as a
  fallback, because optimum's neutral export ships no `mel_filters` (only AMD's
  prepared config bakes them in). Don't couple the engine to another model's file.
- **Both mel paths are kept:** naive `log_mel_spectrogram` is the intact reference;
  `log_mel_spectrogram_fft` is the default. Toggle with `WHISPER_ONNX_MEL=naive|fft`.
- **Diagnostics env knobs:** `WHISPER_ONNX_PROFILE=1` (per-stage timings),
  `WHISPER_ONNX_MAXLEN=<8..448>` (context sweep), `WHISPER_ONNX_MEL`.
- **Backends self-describe** run metadata (`runtime`/`model_format`/
  `decode_strategy`/`max_context`) into `benchmark-results.csv` so ONNX vs OV-IR and
  different strategies compare in one shared schema.
- The neutral export ships `decoder_model.onnx` (no-past) + `decoder_with_past` +
  `decoder_model_merged`; the static engine only needs `encoder_model.onnx` +
  `decoder_model.onnx`.

---

## 5. GenAI parity verdict

- **GPU:** effectively at parity (166 ms vs 137 ms) with the FFT mel alone.
- **NPU:** ~1.7× off (234 ms vs 137 ms). The remaining gap is the decode loop's
  fixed costs, which only a bounded static-KV + mask decoder would shrink — and that
  means re-exporting the graph, i.e. rebuilding what GenAI already ships. Not worth
  it unless chasing the last ~70 ms.

**Cheapest remaining wins (if we revisit):**
1. Persistent thread pool for the mel (avoid spawning threads per call; mel showed
   24–85 ms variance suggesting spin-up/scheduling overhead).
2. Cache the **cross-attention** K/V (encoder-derived, constant across steps) — the
   no-KV loop recomputes them every token; `decoder_with_past` already exposes them.
3. Only then consider a bounded static-KV re-export for the last slice of the gap.

---

## 6. What NOT to do (so we don't repeat it)

- Don't assume the KV cache is the latency problem — **profile first.** We nearly
  spent effort on graph surgery worth ~50 ms while a naive DFT burned ~180 ms.
- Don't zero-pad `n_fft=400` to a power of two to "enable FFT" — it silently breaks
  accuracy. Use Bluestein (or a mixed-radix 400 transform).
- Don't thread the mel on the CPU *device* expecting a win — it fights the inference
  threads.
