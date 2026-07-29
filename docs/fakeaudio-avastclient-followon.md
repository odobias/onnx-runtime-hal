# FakeAudio follow-on (after Whisper static multi)

## Contract (unchanged)

| Side | Window | Rate | Input | Output |
|------|--------|------|-------|--------|
| HAL + product | 7s | 44.1 kHz | `[1,308700]` float PCM | `[1,2]` logits |
| Softmax | client (`stable_softmax_2`) | — | — | P(synthetic) |
| Product log threshold | 0.95 | VPS uses dynamic confidence | | |
| HAL smoke threshold | 0.5 | | | |

## Identity audit status

No product `{MLM}\fakeaudio\model.onnx` was present on the build machine during this pass.
HAL SHA256 of `artifacts/workloads/classifiers/fakeaudio/model.onnx`:

`D4B2DDE8F4F95862ECB9F5E90821D853ADEA3774E7C48857F60EE6084D5E85C6`

**Next:** hash-compare against MLM when available; if identical, wire HAL `test_audio` fixtures into `model_host` FakeAudio gtests (label @ 0.5 + optional `|Δp|`).

## Explicitly deferred

- NPU FE/BB FakeAudio graphs in product
- Media Scan Tier-B WAV harness
