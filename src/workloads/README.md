# Workloads

Adapters that plug into `NpuInferenceBench` live here. Heavy model and fixture
payloads are fetched or generated beside them, but they are local artifacts —
not source.

## Tracked source

| Path | Role |
|------|------|
| `whisper/backends/` | Whisper engine adapters (Intel, ORT static/dynamic, AMD, Qualcomm) |
| `classifiers/ort_classifier.cpp` | TSC / FakeAudio fixture-replay runner |
| `eval/eval.jsonl` | Eval manifest (clip metadata + baked baselines) |

## Local artifacts (gitignored)

| Path | Produced by |
|------|-------------|
| `whisper/models/` | `tools/fetch/`, `tools/export/` |
| `classifiers/tsc/`, `classifiers/fakeaudio/` | `tools/fetch/get-classifier-models.ps1` |
| `classifiers/fixtures/`, `classifiers/audio-samples/` | `tools/fixtures/generate.py`, fetch scripts |
| `eval/*` except `eval.jsonl` | `tools/fetch/get-eval-set.ps1` |
| `audio/` | `tools/fetch/get-audio.ps1` |

Do not commit ONNX/OpenVINO graphs, WAVs, or fixture tensors under this tree.
See [`LAYOUT.md`](../../LAYOUT.md) for the repo-wide source vs artifact contract.
