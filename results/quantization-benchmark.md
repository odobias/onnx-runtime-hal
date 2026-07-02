# Whisper NPU HAL - quantization benchmark

- Model: `openai/whisper-tiny`
- Eval clips: 1 | Runs/clip: 1 | Generated: 2026-07-02T13:51:47Z
- Confidence = mean per-token log-prob (self-reported; higher = more confident, not calibrated truth).
- WER/CER micro-averaged over clips after normalization.
- Cold start = first engine creation after cache deletion; hot start = second engine creation in the same process after cache population.

| Variant | Prec | Backend | Device | Status | Size MB | Cold s | Hot s | Mean ms | xRT | tok/s | Conf | WER % | CER % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper-tiny-amd | fp32-static | amd | NPU | ok | 151.8 | 223.31 | 1.82 | 1014.7 | 5.8 | -1 | 0 | 5.88 | 4.49 |
| whisper-tiny-amd | fp32-static | amd | GPU | unsupported/error | 151.8 | - | - | - | - | - | - | - | - |
| whisper-tiny-amd | fp32-static | amd | CPU | ok | 151.8 | 0.26 | 0.29 | 2483.6 | 2.4 | -1 | 0 | 5.88 | 4.49 |

- **GPU** failed: `Engine creation failed: AMD backend currently supports CPU and NPU; GPU is not wired`

_Caching: cold = first compile, hot = same-process reload from populated cache. See `cache/` and backend-specific compiled-model cache hooks._
