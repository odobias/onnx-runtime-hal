# Repository layout

Top-level folders are grouped so humans can find things without a scavenger hunt.

| Path | Role |
|------|------|
| `src/` | C++ product (`runner/`, `workloads/`, `tests/`) |
| `eng/` | Build system (`projects/`, `msbuild/`) + packaging metadata |
| `benchmark/` | Suite-of-record harness, manifests, schemas |
| `tools/` | Bootstrap, fetch, setup, export, validate, research, repros |
| `docs/` | Guides and engineering notes |
| `results/` | Published evidence (+ `reports/`, gitignored `local/`) |

Root identity files (stay put for MSBuild / package shims):

- `NpuInferenceBench.sln`, `Directory.Build.props`
- `run-benchmark.ps1` → `benchmark/run-suite.ps1`
- `README.md`, `LAYOUT.md`, `.gitignore`

Local-only (gitignored, may be absent): `build/`, `dist/`, `third_party/`, `cache/`, `.venv*`.

## Source detail

| Path | Role |
|------|------|
| `src/runner/` | App, HAL public headers (`include/`), HAL impl (`runtime/`) |
| `src/workloads/` | Whisper/classifier adapters; `eval/eval.jsonl` tracked |
| `src/tests/` | HAL C++ tests + packaging PowerShell tests |
| `eng/projects/` | `.vcxproj` only (no `.cpp`) |
| `eng/msbuild/` | Shared props/targets (C++23, backends, HAL consumer props) |
| `eng/packaging/` | Runner catalogs/schemas — **not** package output |
| `eng/projects/OnnxRuntimeHal/` | Reusable `OnnxRuntimeHal.lib` project |

## Published evidence (`results/`)

| Path | Role |
|------|------|
| `accuracy-runs/`, `run-attempts/` | Immutable per-run campaigns |
| `host-snapshots/`, `model-compilation/` | Environment / provider-input provenance |
| `ledgers/` | Latency CSVs + frozen history (`attempts.jsonl` local-only) |
| `reports/` | Historical human reports (not suite-of-record) |
| `local/` | Ephemeral research output (gitignored) |

## Local artifacts (never commit)

| Path | Role |
|------|------|
| `build/` | MSBuild output trees |
| `dist/npu-inference-bench-*` | Assembled benchmark redistributables |
| `dist/onnx-runtime-hal-*` | Packaged HAL (`include/` + `lib/`) |
| `third_party/` | Downloaded vendor SDKs / ORT drops |
| `cache/` | Provider compilation cache |
| `src/workloads/**` payloads | Models, fixtures, WAVs (see `.gitignore`) |
| `results/local/` | Ephemeral research/sweep outputs |

## Authority

- Benchmark-of-record: C++ runner + `benchmark/run-suite.ps1`
- `tools/research/` / `benchmark/research/` outputs → `results/local/` or `results/reports/`
