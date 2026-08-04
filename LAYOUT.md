# Repository layout

Top-level folders are grouped so humans can find things without a scavenger hunt.

| Path | Role |
|------|------|
| `src/` | C++ product (`runner/`, `workloads/` adapters, `tests/`) |
| `eng/` | Build system (`projects/`, `msbuild/`) + packaging metadata |
| `benchmark/` | Suite-of-record harness, manifests, schemas |
| `tools/` | Bootstrap, fetch, setup, export, validate, research, repros |
| `docs/` | Guides and engineering notes |
| `results/` | Local benchmark output (gitignored; versioned in the private companion) |
| `artifacts/` | **All** generated/local outputs (gitignored; see `artifacts/README.md`) |

Root identity files (stay put for MSBuild / package shims):

- `NpuInferenceBench.sln`, `Directory.Build.props`
- `run-benchmark.ps1` → `benchmark/run-suite.ps1`
- `README.md`, `LAYOUT.md`, `.gitignore`

## Source detail

| Path | Role |
|------|------|
| `src/runner/` | App, HAL public headers (`include/`), HAL impl (`runtime/`) |
| `src/workloads/` | Whisper/classifier C++ adapters; tracked `eval/eval.jsonl` |
| `src/tests/` | HAL C++ tests + packaging PowerShell tests |
| `eng/projects/` | `.vcxproj` only (no `.cpp`) |
| `eng/msbuild/` | Shared props/targets (C++23, backends, HAL consumer props) |
| `eng/packaging/` | Runner catalogs/schemas — **not** package output |
| `eng/projects/OnnxRuntimeHal/` | Reusable `OnnxRuntimeHal.lib` project |

## Local benchmark output (`results/`)

| Path | Role |
|------|------|
| `accuracy-runs/`, `run-attempts/` | Immutable per-run campaigns |
| `host-snapshots/`, `model-compilation/` | Environment / provider-input provenance |
| `ledgers/` | Latency CSVs + frozen history (`attempts.jsonl` local-only) |
| `reports/` | Historical human reports (not suite-of-record) |

Reviewed evidence is copied to the private companion repository. The public
checkout retains only `results/README.md`.

## Local artifacts (`artifacts/`)

| Path | Role |
|------|------|
| `build/` | MSBuild output trees |
| `dist/` | Assembled redistributables (bench + HAL packages) |
| `third_party/` | Downloaded vendor SDKs / ORT drops |
| `cache/` | Provider compilation cache |
| `workloads/` | Fetched models, fixtures, ASR speech WAVs (`speech/`) |
| `scratch/` | Ephemeral research/sweep outputs |
| `venv*/` | Python virtualenvs used by fetch/export tools |

## Authority

- Benchmark-of-record: C++ runner + `benchmark/run-suite.ps1`
- `tools/research/` / `benchmark/research/` outputs → `artifacts/scratch/` or `results/reports/`
