# Baseline: static multilingual Whisper @ 7s

## Product window (AvastClient Media Scan)

Confirmed on `devel` in `AudioRecognizer.cpp`:

- `seconds_to_process_at_once_asr = 7`
- `seconds_to_process_at_once_fakeaudio = 7` (independent FA path @ 44.1 kHz / 308700 samples)
- Whisper host resamples 44.1 kHz → 16 kHz → **112000 samples** for a full 7s window

## HAL today

| Artifact | Hub key | Model | Mel / PCM | Decode |
|----------|---------|-------|-----------|--------|
| `artifacts/.../static-onnx` | `whisper/en-static-onnx` | **tiny.en** | `[1,80,3000]` / 480000 (30s) | static-no-KV, `input_ids[1,128]` + `encoder_hidden_states[1,1500,384]` |
| `artifacts/.../sherpa-export` | `whisper/sherpa-export` | tiny.en product names | Sherpa mel 30s | Sherpa KV cache |
| Suite workload | `whisper-tiny-static` | onnx-static | NPU/GPU/CPU | `portable.json` |

NPU evidence applies to the **30s tiny.en static** pack, not to multilingual or 7s shapes.

## AvastClient today

- `Whisper` / `WhisperMultilingual` → `WhisperBackend` → `asw::whisper::Session`
- File layout: `{MLM}\whisper[_multilingual]\{encoder,decoder}.onnx` + `tokens.txt`
- Graph contract: **Sherpa** (cross-KV from encoder, self-KV decoder, mel `[1,n_mels,3000]`)
- Multilingual selected via encoder ONNX `metadata_props` (`is_multilingual`)

## Target package (this work)

- Hub / local: `whisper/static-onnx-tiny-multi-7s` → `artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s`
- Weights: multilingual tiny via `onnx-community/whisper-tiny` (same as `openai/whisper-tiny`)
- Graph shapes (NPU-static): mel `[1,80,3000]`, encoder hidden `[1,1500,384]`, decoder `input_ids[1,128]`
- **Product audio window:** 7s / 112000 samples @ 16 kHz, **zero-padded to the 30s mel canvas**
- `npu_hal_package.json`: `product_window_s=7`, `pad_to_s=30`, `multilingual=true`

### Why not mel `[1,80,700]` / enc_seq 350?

Verified: truncating the encoder positional grid to 350 (Torch or ONNX) + stock decoder produces severe token loops on JFK. Whisper public weights assume the 30s canvas; true shape reduction needs fine-tuning. Product still always sends 7s — the runtime pads.

### Language handling: verbatim, never translated

Decoder prompt is `<|sot|> <|lang|> <|transcribe|> <|notimestamps|>`. `<|translate|>` is never used, and `<|lang|>` is **detected per clip**, not assumed:

- one decoder step after `<|sot|>`, then argmax over the 99 `<|xx|>` tokens (same algorithm as the reference `whisper.detect_language()`)
- costs one extra decoder forward per clip (~4% of a 7s-clip decode on CPU)
- `NPU_INFERENCE_BENCH_WHISPER_LANG=<iso>` pins a language for A/B runs; an unknown code is a hard init error, never a silent fallback

Forcing `<|en|>` on non-English speech makes Whisper paraphrase into English instead of transcribing. Measured on the Spanish FLEURS clip over the full 30s canvas: detected `es` → 0.0% WER verbatim, forced `en` → 104.5% WER ("It was so much the amount of people who were concentrated…"). Hence auto-detect is the default. Detection is 6/6 on de/fr/es/cs/it/pl at both the 7s product window and the full canvas.
