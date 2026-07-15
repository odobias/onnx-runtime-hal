# Snapdragon X Elite battery campaign — 2026-07-14

Three complete latency passes were run across the bundled and Windows ML
runtimes, every valid CPU/GPU/NPU execution profile, and both classifier
workloads. The campaign then ran whole-graph FakeAudio NPU latency and accuracy
diagnostics on both runtimes.

- Power: battery, Balanced plan.
- Battery range: 77% at the first snapshot to 61% at the final snapshot.
- Sleep prevention: `SetThreadExecutionState(ES_SYSTEM_REQUIRED)` for the
  campaign process.
- Whisper: 5 measured iterations after one warmup.
- Classifiers: 20 measured iterations per fixture.
- New evidence: 30 ASR rows, 38 classifier rows, and two intentionally invalid
  diagnostic accuracy records.

## Interpretation

The third-pass CPU rows are throttled outliers and must not be averaged with the
first two passes:

- Bundled static Whisper rose from 1.07/1.57 seconds to 12.42 seconds.
- Bundled dynamic Whisper rose from 0.27/0.16 seconds to 1.94 seconds.
- Bundled TSC rose from 175/135 ms to 1,348 ms.
- Bundled FakeAudio rose from 82/130 ms to 1,022 ms.

The corresponding Windows ML CPU rows also slowed, though less dramatically.
GPU and NPU inference remained substantially more stable. Compare rows by their
`environment_snapshot_id` and battery percentage rather than collapsing all
three passes into one mean.

## NPU findings

- Bundled static Whisper inference: 367–409 ms.
- Windows ML static Whisper inference: 459–546 ms.
- Bundled TSC: 19.4–19.7 ms; Windows ML: 20.8–21.3 ms.
- Split FakeAudio: 112.6–114.4 ms bundled and 111.1–111.4 ms Windows ML.
- Split FakeAudio stayed fully on QNN with maximum CPU-reference probability
  drift `0.000839` and no decision flips.
- Whole-graph FakeAudio stayed fully on QNN but remained invalid on both
  runtimes: maximum drift `0.932525` and one reference decision flip.

Bundled QNN hot loading reduced NPU startup to roughly one second. Windows ML
QNN hot loading remained approximately as expensive as cold compilation, so
those rows must not be presented as evidence of effective persistent caching.
