# artifacts/

All generated / downloaded / local-only outputs live here. Nothing under this
tree is source; most of it is gitignored (only this README and
`scratch/.gitkeep` are tracked).

| Path | Role |
|------|------|
| `build/` | MSBuild output (`NpuInferenceBench.exe`, libs, objs) |
| `dist/` | Assembled redistributable packages |
| `third_party/` | Vendor SDKs / ORT drops (setup + fetch) |
| `cache/` | Provider compilation cache |
| `workloads/` | Fetched models, fixtures, eval WAVs, sample audio |
| `scratch/` | Ephemeral research/sweep outputs |
| `venv*/` | Python virtualenvs used by fetch/export tools |

Tracked eval metadata stays in `src/workloads/eval/eval.jsonl`. Published
benchmark evidence stays in repo-root `results/` (not here).
