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

| Device | Runtime | EP | Transcript (JFK 7s) | Pass |
|--------|---------|----|---------------------|------|
| CPU | `.venv` onnxruntime 1.28.0 | `CPUExecutionProvider` | `And so my fellow Americans ask not what your country can do` | YES |
| GPU | `.venv-dml` onnxruntime-directml 1.24.4 | `DmlExecutionProvider` | same | YES |
| NPU | — | no VitisAI/QNN/OpenVINO on this host | deferred | — |

## How to run elsewhere (incl. NPU)

```powershell
# CPU + DirectML GPU (default)
.\tools\validate\run_static_whisper_7s_multi.ps1

# NPU host with registered ORT NPU EP in Python:
.\tools\validate\run_static_whisper_7s_multi.ps1 -Device npu

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
