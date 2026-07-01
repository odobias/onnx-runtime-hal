# Whisper NPU HAL - quantization benchmark

- Model: `openai/whisper-tiny.en`
- Eval clips: 12 | Runs/clip: 2 | Generated: 2026-07-01T16:45:22
- Confidence = mean per-token log-prob (self-reported; higher = more confident, not calibrated truth).
- WER/CER micro-averaged over clips after normalization.

| Variant | Prec | Backend | Device | Status | Size MB | Cold s | Warm s | Mean ms | xRT | tok/s | Conf | WER % | CER % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wten-ov-fp32 | fp32 | intel | NPU | ok | 150.7 | 10.02 | 0.7 | 137.3 | 71.3 | 403.2 | 0.04 | 9.57 | 4.46 |
| wten-ov-fp32 | fp32 | intel | GPU | ok | 150.7 | 2.35 | 0.25 | 152.9 | 65 | 293.1 | 0.0396 | 8.58 | 3.92 |
| wten-ov-fp32 | fp32 | intel | CPU | ok | 150.7 | 1.03 | 0.85 | 281.6 | 32.1 | 330.2 | 0.0396 | 8.58 | 3.92 |
| wten-ov-fp16 | fp16 | intel | NPU | ok | 78.8 | 9.42 | 0.62 | 139.3 | 71.1 | 423.7 | 0.04 | 9.57 | 4.46 |
| wten-ov-fp16 | fp16 | intel | GPU | ok | 78.8 | 1.82 | 0.26 | 143.1 | 70 | 322.4 | 0.0396 | 8.58 | 3.92 |
| wten-ov-fp16 | fp16 | intel | CPU | ok | 78.8 | 0.68 | 0.49 | 266.7 | 33.3 | 409.2 | 0.0396 | 8.58 | 3.92 |
| wten-ov-int8 | int8 | intel | NPU | ok | 44.9 | 10.46 | 0.7 | 141.5 | 60.1 | 440 | 0.0397 | 10.23 | 5.05 |
| wten-ov-int8 | int8 | intel | GPU | ok | 44.9 | 1.83 | 0.27 | 258.8 | 36.6 | 258.3 | 0.0397 | 10.56 | 5.17 |
| wten-ov-int8 | int8 | intel | CPU | ok | 44.9 | 0.78 | 0.56 | 246.8 | 36.7 | 421 | 0.0397 | 10.56 | 5.17 |
| wten-ov-int4 | int4 | intel | NPU | ok | 37.6 | 21.72 | 0.7 | 119.9 | 83.1 | 436.2 | 0.04 | 14.85 | 7.13 |
| wten-ov-int4 | int4 | intel | GPU | ok | 37.6 | 8.59 | 0.22 | 154.8 | 64.4 | 303.5 | 0.04 | 14.85 | 7.13 |
| wten-ov-int4 | int4 | intel | CPU | ok | 37.6 | 0.77 | 0.54 | 250.1 | 36 | 425.5 | 0.04 | 14.85 | 7.13 |

_Caching: cold = first compile, warm = cache hit. See `cache/` and OpenVINO `ov::cache_dir`._

