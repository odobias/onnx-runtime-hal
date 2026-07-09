# Deepfake model benchmark (ONNX Runtime)

Load/latency on **real labeled inputs** plus a correctness check against ground truth in the same session. `input_kind=labeled` rows are scored for accuracy; see `validate_tsc_onnx.py` / `validate_fakeaudio_onnx.py` docstrings for data provenance and caveats (small samples, best-effort preprocessing -- plausibility signals, not certified accuracy). Per-sample probabilities below also serve as a cross-EP numerical-agreement check (identical p across EPs = the accelerator isn't silently changing outputs).

Host: `Intel64 Family 6 Model 204 Stepping 0, GenuineIntel` / GPU `Intel(R) Graphics` / arch `AMD64` / Windows 11 / ONNX Runtime `1.27.0`.

## Latency + accuracy summary

| Model | EP | Input | Status | Load s | Mean ms | P95 ms | ips | Accuracy |
|---|---|---|---|---|---|---|---|---|
| fakeaudio | CPU | synthetic | ok | 0.4382 | 84.739 | 92.952 | 11.801 | - |
| tsc | CPU | labeled | ok | 0.5874 | 134.917 | 148.25 | 7.412 | 14/15 (0.9333) |

### tsc: per-sample p(scam)  (MISS = predicted != label)

| Sample | Label | CPU p(scam) |
|---|---|---|
| scam_0 | scam | 0.0369  MISS |
| scam_1 | scam | 1.0 |
| scam_2 | scam | 0.9093 |
| scam_3 | scam | 0.9992 |
| scam_4 | scam | 0.9998 |
| scam_5 | scam | 0.9904 |
| scam_6 | scam | 0.9721 |
| scam_7 | scam | 0.9404 |
| clean_8 | clean | 0.0 |
| clean_9 | clean | 0.0 |
| clean_10 | clean | 0.0 |
| clean_11 | clean | 0.0 |
| clean_12 | clean | 0.0131 |
| clean_13 | clean | 0.0007 |
| clean_14 | clean | 0.0014 |
