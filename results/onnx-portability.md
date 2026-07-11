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

_Blank slate — rerun the experiments to repopulate. Raw numbers land in `onnx-portability.csv`._

| Runtime | Device | Cold load | Warm load (cache) | Hot infer (mean) | WER | CER | Conf |
|---|---|---|---|---|---|---|---|

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
`tools/research/onnx_ov_decode.py` (dyn KV-cache CPU/GPU), `tools/research/onnx_npu_static.py`
(static NPU), `tools/research/onnx_decode.py` (raw ORT), `tools/research/probe_ort_raw.py`.

## C++ port (production HAL backend)

The static no-KV path is now a first-class C++ HAL backend
(`Backend::IntelOnnx`, `src/backends/intel/intel_onnx_engine.cpp`): the neutral
ONNX loaded through `ov::Core`, reshaped to static shapes, decoded with the same
recompute loop — the one strategy that compiles on the NPU. Front-end (log-mel +
byte-level BPE) is shared, header-only (`include/npu_inference_bench/whisper_frontend.hpp`);
the mel filterbank is computed in-process (librosa Slaney bank) so the neutral
export needs no baked-in `mel_filters`. Single clip = `jfk.wav` (11 s, 25 tokens),
`maxlen=128`, greedy + suppress. Rows land in `benchmark-results.csv`.

_Blank slate — rerun to repopulate._

| Device | mel | Hot infer (mean) | xRT | WER | Conf |
|---|---|---|---|---|---|

WER 0.0% on `jfk.wav` (vs 9.57% micro-averaged over the 12-clip LibriSpeech set
above — single clean clip, not a contradiction). Confidence -0.133 matches the
Python path, and fft-vs-naive match bit-for-bit, confirming both the in-process
mel bank and the Bluestein FFT reproduce whisper's front-end exactly.

### Where the time actually goes (measured on NPU, `WHISPER_ONNX_PROFILE=1`)

Per-stage split of the naive-mel run (maxlen=128, 25 tokens):

_Blank slate — rerun with `WHISPER_ONNX_PROFILE=1` to repopulate._

| stage | time | note |
|---|---|---|

Two findings that overturned the intuitive "the KV cache is the problem":

1. **`maxlen` is not a useful lever.** Sweeping 128→96→64→48→32 barely moved
   decode/token (7.0→5.2 ms) and left total latency flat. The per-step decoder
   cost is dominated by **fixed** work — cross-attention over the 1500 encoder
   positions (constant every step) and the full-vocab logits projection — not the
   maxlen-dependent self-attention. So even a perfect bounded KV cache would only
   reclaim ~50-80 ms of the 170 ms decode, not the bulk of it.

2. **The real bottleneck was the mel front-end, unrelated to the KV cache.** The
   naive DFT is O(frames·bins·nfft) single-threaded (~480M mul-adds). Replacing it
   with **Bluestein's FFT** (exact for the awkward n_fft=400 — no zero-pad bin
   shift) fanned across CPU cores drops mel to ~24-85 ms and cuts total NPU infer
   ~1.9x (452→234 ms) and GPU ~2.0x (334→166 ms) with **identical output**. GPU is
   now within ~20% of the GenAI IR pipeline; NPU within ~1.7x — closing most of
   the gap without any bounded-KV graph surgery. (CPU sees no win: OpenVINO
   inference already saturates every core, so threading the mel just competes with
   it, and decode is the CPU wall anyway.)

The naive DFT is kept intact as the reference (`log_mel_spectrogram`,
`WHISPER_ONNX_MEL=naive`); FFT is the default (`log_mel_spectrogram_fft`).

Build/run: `.\tools\build\build.ps1`; `.\benchmark\run-whisper.ps1 -Backend intel-onnx -Device npu`.
Diagnostics: `WHISPER_ONNX_PROFILE=1` (per-stage timings), `WHISPER_ONNX_MEL=naive|fft`,
`WHISPER_ONNX_MAXLEN=<8..448>`.
