# AMD Radeon 890M DirectML results — 2026-07-13

Command:

```powershell
.\benchmark\run-suite.ps1 -Device gpu -Provider DmlExecutionProvider -Runs 5 -ClassifierRuns 20
```

Environment: AMD Radeon 890M, driver 32.0.13058.2, ONNX Runtime 1.25.1,
Windows 11 build 26200, battery power, FP32.

## Whisper tiny static

- Model SHA-256: `9da9f440ca7601e6ddfb6afdbc8e1564dbeb19ff56d0522cefd68e52f8a95bac`
- Cold/hot load: 2.041 s / 0.601 s
- Mean inference: 840.723 ms (6.96x real time)
- WER/CER: 5.88% / 4.49%
- Placement: 2 DirectML nodes, 0 CPU nodes, 0% CPU offload

## Whisper tiny dynamic KV

- Model SHA-256: `12534e24d20072ab1bca394fe2254205058aeae38b88545c1c12b279d1e51ba2`
- Cold/hot load: 1.439 s / 1.412 s
- Mean inference: 780.176 ms (7.50x real time)
- WER/CER: 5.88% / 4.49%
- Placement: 592 DirectML nodes, 180 CPU nodes, 23.32% CPU offload
- CPU nodes: shape/control operations (`Concat`, `Gather`, `Unsqueeze`,
  `Equal`, `Reshape`, `Where`, and `Add`)

## Text Scam Classifier

- Model SHA-256: `90f2ed1b3f1a469f36e416ceeb6121da8de72152f69c61e5a14865ee045834e0`
- Cold/hot load: 0.895 s / 0.549 s
- Mean inference: 61.674 ms
- Accuracy: 14/15 (93.33%)
- Maximum probability difference from CPU: 0.000306
- Placement: 1 DirectML node, 0 CPU nodes, 0% CPU offload

## FakeAudio

- Model SHA-256: `549143bc35c75f7f531dc900a901a5f6c8835d47de3cf833ab2579f8be72dcbb`
- Cold/hot load: 1.342 s / 0.742 s
- Mean inference: 69.821 ms
- Accuracy: 3/5 (60%), matching the CPU reference
- Maximum probability difference from CPU: 0.000006
- Placement: 4 DirectML nodes, 1 CPU node, 20% CPU offload
- CPU node: `Resize`

All four runs resolved to `DmlExecutionProvider` without provider fallback.
Offload percentages are profiler node counts, not compute share. The model
hashes identify complete inference artifact packages, so they are not
necessarily the raw SHA-256 of a single ONNX file.
