# Deepfake model benchmark (ONNX Runtime)

Load/latency on **real labeled inputs** plus a correctness check against ground truth in the same session. `input_kind=labeled` rows are scored for accuracy; see `validate_tsc_onnx.py` / `validate_fakeaudio_onnx.py` docstrings for data provenance and caveats (small samples, best-effort preprocessing -- plausibility signals, not certified accuracy). Per-sample probabilities below also serve as a cross-EP numerical-agreement check (identical p across EPs = the accelerator isn't silently changing outputs).

Host: `AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD` / GPU `NVIDIA T1000 8GB` / arch `AMD64` / Windows 11 / ONNX Runtime `1.24.4`.

## Latency + accuracy summary

| Model | EP | Input | Status | Load s | Mean ms | P95 ms | ips | Accuracy |
|---|---|---|---|---|---|---|---|---|
| fakeaudio | CPU | labeled | ok | 0.2534 | 61.814 | 66.337 | 16.177 | 3/5 (0.6) |
| fakeaudio | DML | labeled | ok | 0.594 | 33.679 | 35.274 | 29.692 | 3/5 (0.6) |
| tsc | CPU | labeled | ok | 0.2405 | 75.08 | 77.928 | 13.319 | 14/15 (0.9333) |
| tsc | DML | labeled | ok | 0.3337 | 35.187 | 36.437 | 28.419 | 14/15 (0.9333) |

### fakeaudio: per-sample p(deepfake)  (MISS = predicted != label)

| Sample | Label | CPU p(deepfake) | DML p(deepfake) |
|---|---|---|---|
| deepfake_1 | deepfake | 1.0 | 1.0 |
| deepfake_2 | deepfake | 0.9981 | 0.9981 |
| real_1 | real | 0.9995  MISS | 0.9995  MISS |
| real_2 | real | 0.9984  MISS | 0.9984  MISS |
| real_3 | real | 0.0048 | 0.0048 |

### tsc: per-sample p(scam)  (MISS = predicted != label)

| Sample | Label | CPU p(scam) | DML p(scam) |
|---|---|---|---|
| scam_0 | scam | 0.0369  MISS | 0.0369  MISS |
| scam_1 | scam | 1.0 | 1.0 |
| scam_2 | scam | 0.9093 | 0.9093 |
| scam_3 | scam | 0.9992 | 0.9992 |
| scam_4 | scam | 0.9998 | 0.9998 |
| scam_5 | scam | 0.9904 | 0.9904 |
| scam_6 | scam | 0.9721 | 0.9721 |
| scam_7 | scam | 0.9404 | 0.9404 |
| clean_8 | clean | 0.0 | 0.0 |
| clean_9 | clean | 0.0 | 0.0 |
| clean_10 | clean | 0.0 | 0.0 |
| clean_11 | clean | 0.0 | 0.0 |
| clean_12 | clean | 0.0131 | 0.0131 |
| clean_13 | clean | 0.0007 | 0.0007 |
| clean_14 | clean | 0.0014 | 0.0014 |
