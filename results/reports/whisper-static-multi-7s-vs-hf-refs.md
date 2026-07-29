# WER: static ONNX tiny-multi vs HF Whisper refs

- Device: `cpu`
- Window: full audio ≤30s
- Decode: greedy (`num_beams=1`) for HF; static-no-KV greedy for ONNX
- Larger model: `openai/whisper-small`

| System | Mean WER |
|--------|----------|
| ONNX `static-onnx-tiny-multi-7s` | 9.9% |
| HF `openai/whisper-tiny` | 9.9% |
| HF `openai/whisper-small` | 7.3% |

Fidelity gap (ONNX − tiny): **+0.0 pp**
Ceiling gap (ONNX − larger): **+2.6 pp**

| id | dur | onnx | tiny | larger |
|----|-----|------|------|--------|
| jfk | 11.00 | 0.0% | 0.0% | 0.0% |
| ls_000 | 5.86 | 5.9% | 5.9% | 5.9% |
| ls_001 | 4.82 | 10.0% | 10.0% | 10.0% |
| ls_002 | 12.48 | 3.1% | 3.1% | 3.1% |
| ls_003 | 9.90 | 4.2% | 4.2% | 4.2% |
| ls_004 | 29.40 | 20.6% | 20.6% | 16.2% |
| ls_005 | 9.01 | 5.6% | 5.6% | 5.6% |
| ls_006 | 5.64 | 16.7% | 16.7% | 25.0% |
| ls_007 | 9.24 | 5.3% | 5.3% | 5.3% |
| ls_008 | 5.12 | 18.2% | 18.2% | 0.0% |
| ls_009 | 18.29 | 7.0% | 7.0% | 7.0% |
| ls_010 | 5.60 | 20.0% | 20.0% | 6.7% |
| ls_011 | 15.12 | 11.8% | 11.8% | 5.9% |
