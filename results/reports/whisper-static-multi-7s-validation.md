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

## How to run elsewhere (incl. NPU)

```powershell
# CPU + DirectML GPU smoke (default)
.\tools\validate\run_static_whisper_7s_multi.ps1

# Multi-sample WER (eval.jsonl + JFK), product-fit + full canvas:
.\tools\validate\run_static_whisper_7s_multi.ps1 -Device cpu,gpu -Eval

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
- GPU smoke requires `onnxruntime-directml` (separate `.venv-dml`); stock `onnxruntime` is CPU-only.
