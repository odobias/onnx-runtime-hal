# Repository layout

This repo keeps product source, published benchmark evidence, and local
artifacts in separate places. If a path is not listed under **Source** or
**Published evidence**, treat it as disposable local state.

## Source (commit)

| Path | Role |
|------|------|
| `runner/` | C++ benchmark app, runtime HAL, public headers |
| `workloads/` | Workload adapters (C++) and tracked `eval/eval.jsonl` |
| `projects/` | MSBuild `.vcxproj` files (no `.cpp` here) |
| `msbuild/` | Shared props/targets |
| `tests/` | C++ and packaging tests |
| `benchmark/` | Suite-of-record harness, manifests, schemas |
| `tools/` | Bootstrap, fetch, setup, export, validate, research, repros |
| `packaging/` | Runner catalogs and redistributable **metadata** (not packages) |
| `docs/` | Guides and engineering notes |

Root identity files stay at the repo root for MSBuild and package shims:

- `NpuInferenceBench.sln`, `Directory.Build.props`
- `run-benchmark.ps1` (forwards to `benchmark/run-suite.ps1`)

## Published evidence (commit selectively)

| Path | Role |
|------|------|
| `results/accuracy-runs/` | Immutable accuracy campaigns (default suite mode) |
| `results/run-attempts/` | Published attempt matrix per environment snapshot |
| `results/host-snapshots/` | Host/runtime fingerprints |
| `results/model-compilation/` | Provider-input provenance |
| `results/ledgers/` | Latency CSVs and frozen accuracy history |
| `results/reports/` | Historical human reports (MD/CSV/canvas); not suite-of-record |

See [`results/README.md`](results/README.md) for write contracts and validity rules.

## Local artifacts (never commit)

| Path | Role |
|------|------|
| `build/` | MSBuild output trees |
| `dist/` | Assembled redistributable packages |
| `third_party/` | Downloaded vendor SDKs / ORT drops |
| `cache/` | Provider compilation cache |
| `.venv*`, `__pycache__/` | Python toolchains |
| `workloads/whisper/models/` | Fetched/exported Whisper payloads |
| `workloads/classifiers/{tsc,fakeaudio,fixtures,audio-samples}/` | Classifier assets |
| `workloads/eval/*` except `eval.jsonl` | Eval WAVs and local eval junk |
| `workloads/audio/` | Sample audio |
| `results/local/` | Ephemeral research/sweep outputs |
| `results/ledgers/attempts.jsonl` | Rolling local attempt buffer |

`packaging/` is **not** package output. Built packages live only under `dist/`.

## Authority

- Benchmark-of-record: C++ runner + `benchmark/run-suite.ps1`
- `tools/research/` and `benchmark/research/` are probes/sweeps; their outputs
  belong in `results/local/` (ephemeral) or `results/reports/` (kept snapshots),
  never loose next to evidence directories
