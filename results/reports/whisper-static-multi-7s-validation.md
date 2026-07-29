# Validation: static-onnx-tiny-multi-7s

Date: 2026-07-29  
Host: local (NVIDIA T1000 8GB)  
Package: `artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s`  
HF: `gendigital/npu-hal-over-9000` / `whisper/static-onnx-tiny-multi-7s`

## Contract

| Field | Value |
|-------|-------|
| Multilingual | yes (`openai/whisper-tiny` / onnx-community) |
| Product audio | 7s / 112000 @ 16 kHz |
| Mel canvas | 30s / `[1,80,3000]` → enc `[1,1500,384]` (pad; see note) |
| Decode | static-no-KV, `input_ids[1,128]` |
| Prompt | `<\|sot\|><\|en\|><\|transcribe\|><\|notimestamps\|>` |

## Results (this host)

### Smoke (JFK)

| Device | Runtime | EP | Transcript (JFK 7s) | Pass |
|--------|---------|----|---------------------|------|
| CPU | `.venv` onnxruntime 1.28.0 | `CPUExecutionProvider` | `And so my fellow Americans ask not what your country can do` | YES |
| GPU | `.venv-dml` onnxruntime-directml 1.24.4 | `DmlExecutionProvider` | same | YES |
| NPU | — | no VitisAI/QNN/OpenVINO on this host | deferred | — |

### Multi-sample WER (`eval.jsonl` + JFK)

Script: `tools/validate/eval_static_whisper_7s_multi.py`  
Gate: mean/per-clip WER ≤ 35% on scored clips (no decode loops).

| Device | Window | Clips scored | Mean WER | Pass |
|--------|--------|--------------|----------|------|
| CPU | product 7s, clips ≤7s | 5 LibriSpeech | 14.1% | YES |
| GPU | product 7s, clips ≤7s | 5 LibriSpeech | 14.1% (bit-identical hyps vs CPU) | YES |
| CPU | full ≤30s canvas | 13 (JFK + 12 LS) | 9.9% | YES |
| GPU | full ≤30s canvas | 13 | 9.9% (matches CPU) | YES |

JSON dumps: `results/reports/whisper-static-multi-7s-eval-{cpu,gpu}-{product,full}.json`.

Worst scored full-canvas clips ~20% WER (`ls_004`, `ls_010`) — tiny multilingual noise, not loops.

### Verbatim multilingual decode (language auto-detect)

Script: `tools/validate/eval_static_whisper_multilingual.py` (FLEURS clips).  
Prompt is always `<|sot|> <|detected-lang|> <|transcribe|> <|notimestamps|>` — `<|translate|>` is never used.

| Check | Result |
|-------|--------|
| Language detected from audio (de, fr, es, cs, it, pl), 7s product window | 6/6 correct, confidence 0.90–1.00 |
| Same, full 30s canvas | 6/6 correct, confidence 0.90–1.00 |
| Transcripts in the spoken language | 6/6 verbatim; mean WER 44.9% product / 40.0% full |
| English eval set with auto-detect vs pinned `en` | 13/13 detected `en`, hypotheses and 14.1% mean WER identical |
| Detection cost | one extra decoder forward, ~4% of a 7s-clip decode |
| Control: 7s Spanish forced to `<|en|>` | 100.0% WER English paraphrase, vs 31.8% verbatim when detected |

All FLEURS clips run 7.1–9.9s, so every one is cut by the 7s product window; their WER includes the missing tail and is not a language error. Detection is unaffected by the truncation — 7s of speech is plenty.

The C++ static engine previously defaulted to `<|en|>` whenever `NPU_INFERENCE_BENCH_WHISPER_LANG` was unset, which silently translated non-English speech. It now detects per clip; the env var pins a language for A/B runs and rejects unknown codes at init instead of falling back to English. Verified on the built runner:

| Clip | `NPU_INFERENCE_BENCH_WHISPER_LANG` | Transcript |
|------|-----------------------------------|------------|
| `fleurs_es` | unset (auto) | `Fue tanta la cantidad de gente que se concentró…` |
| `fleurs_cs` | unset (auto) | `Kůry jsou na přebně 70 km…` |
| `fleurs_es` | `en` | `It was so much the amount of people who were concentrated…` |
| `fleurs_es` | `klingon` | init error, no silent English fallback |

## How to run elsewhere (incl. NPU)

```powershell
# CPU + DirectML GPU smoke (default)
.\tools\validate\run_static_whisper_7s_multi.ps1

# Multi-sample WER (eval.jsonl + JFK), product-fit + full canvas:
.\tools\validate\run_static_whisper_7s_multi.ps1 -Device cpu,gpu -Eval

# Verbatim multilingual decode with per-clip language detection:
.\.venv\Scripts\python.exe tools\validate\eval_static_whisper_multilingual.py --lang-mode auto --window product --en-control
.\.venv\Scripts\python.exe tools\validate\eval_static_whisper_multilingual.py --lang-mode auto --window full

# NPU host with registered ORT NPU EP in Python:
.\tools\validate\run_static_whisper_7s_multi.ps1 -Device npu -Eval

# Full HAL portable suite (needs npu_inference_bench build):
.\tools\validate\run_static_whisper_7s_multi.ps1 -Device npu -Suite
# or:
.\benchmark\run-suite.ps1 -Device npu -Only whisper -Provider auto
```

Workload id in `benchmark/manifests/portable.json`: `whisper-tiny-static-multi-7s`.

## Notes

- Truncating encoder to 700 frames without fine-tuning **fails** (token loops). Documented in package `note` and baseline doc.
- English-only `whisper-tiny-static` remains the prior NPU golden path; this package reuses the same static-no-KV decode strategy with multilingual weights + SOT language/task tokens.
- Language is detected per clip, never assumed English. Transcripts stay verbatim in the spoken language; translation to English is not a supported mode.
- GPU smoke requires `onnxruntime-directml` (separate `.venv-dml`); stock `onnxruntime` is CPU-only.
