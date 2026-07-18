# Isolated runner distributions

Vendor ONNX Runtime builds are not ABI-compatible. The distribution therefore
contains several executables instead of combining every provider DLL in one
directory. Each runner loads only the files colocated with that executable.

## Package shape

Packages are architecture-specific:

```text
dist/npu-inference-bench-ARM64/
  runner-package.json
  run-benchmark.ps1
  benchmark/
  tools/                    # allowlisted helpers (fetch/eval/fixtures/validate)
  src/workloads/
    whisper/models/static-onnx/
    eval/                   # eval.jsonl + ls_*.wav (+ baked baselines)
    audio/
    classifiers/
      tsc/
      fakeaudio/
      fixtures/
  runners/
    ort/
      NpuInferenceBench.exe
      onnxruntime.dll
      runner.json
    qualcomm/
      NpuInferenceBench.exe
      onnxruntime.dll
      onnxruntime_providers_qnn.dll
      QnnHtp.dll
      runner.json
    winml/
      NpuInferenceBench.exe
      onnxruntime.dll
      Microsoft.Windows.AI.MachineLearning.dll
      runner.json
```

There is no top-level `models/` tree in the package. All runtime assets live under
`src/workloads/`.

The x64 package uses the same layout and may contain `amd`, `ovep`, `dml`,
`winml`, and portable `ort` runners. A package contains only one PE
architecture; mixed x64/ARM64 artifacts are rejected.

`runner-package.json` records the runners actually assembled, files and SHA-256
hashes, supported vendors/providers, minimum Windows build, and explicit skip
reasons. `eng/packaging/runner-package.schema.json` defines the version 1 contract.

## Build a package

First install the SDKs relevant to the build host. Then build and assemble every
locally viable runner. The published matrix entry point stages SDKs per
architecture and builds both packages:

```powershell
.\tools\build\build-all-runner-packages.ps1 -Clean
.\tools\build\build-all-runner-packages.ps1 -Clean -InstallArm64Tools
```

Or build one architecture at a time:

```powershell
.\tools\build\build-runner-package.ps1 -Architecture ARM64 -Clean
.\tools\build\build-runner-package.ps1 -Architecture x64 -Clean
```

`build-all-runner-packages.ps1` gates third-party staging first, then launches
every runner build across the requested architectures in one parallel pool:

1. Stage SDKs into non-overlapping roots (`third_party/onnxruntime-directml-x64`,
   `third_party/onnxruntime-directml-ARM64`, WinML per-platform bins, OVEP, QNN).
2. Build all `build.ps1` runner variants concurrently (`-ThrottleLimit`, default
   auto). Unique `OutDir`/`IntDir` per `PlatformOutTag` make this safe.
3. Assemble each `dist/npu-inference-bench-<arch>/` package (`-SkipBuild`).

Use `-SkipSdkStage` when those roots are already populated. Vendor ORT ABIs
still cannot share one process, so there is no single Visual Studio "Build All"
configuration — only a parallel matrix of isolated builds. Missing SDKs do not
invalidate the package; they produce entries in `skipped`. A build failure for
an SDK that was detected is fatal. A summary is written to
`dist/runner-packages-summary.json`.

### Redistributable model assets

Packages copy only the allowlist in `eng/packaging/redistributable-assets.json`:
Whisper static, TSC whole, and FakeAudio **whole + generic NPU backbone split**,
all under `src/workloads/`. Eval WAVs are required at assemble time. The package
replaces `benchmark/manifests/portable.json` with
`eng/packaging/portable.redistributable.json` so dynamic Whisper / OV-IR / AMD
vendor Whisper / Intel-only FakeAudio splits stay out. `tools/` is trimmed to
fetch/eval/fixtures/validate helpers (no research/export/build trees).

Whisper eval clips in `src/workloads/eval/eval.jsonl` carry baked
`baseline_hyp` / `baseline_wer` / `baseline_cer` fields (mint with
`tools/eval/bake-whisper-baselines.ps1`). Accuracy-quick reports WER delta vs
that baseline. `-Executor fastest|most-accurate` races packaged runners on those
samples and returns the winner.

Useful assembly options:

```powershell
# Assemble existing build outputs without rebuilding.
.\tools\build\build-runner-package.ps1 -Architecture ARM64 -SkipBuild

# Build or assemble only selected runner IDs.
.\tools\build\build-runner-package.ps1 -Architecture x64 -Runner ort,dml,winml

# Validate runner assembly without copying large shared assets.
.\tools\build\build-runner-package.ps1 -Architecture ARM64 -SkipBuild -SkipAssets
```

The supported IDs are `ort`, `dml`, `winml`, `ovep`, `amd`, and `qualcomm`.
OpenVINO EP and AMD packs are x64-only. Qualcomm is ARM64-only. Windows ML,
DirectML, and portable ORT can be built for either architecture when the
matching SDK payload is installed.

## Merge runners built elsewhere

Every assembled `runners/<id>/` directory is also a standardized runner
artifact. Its `runner.json` lists all required files and hashes, so CI can
publish runner directories independently.

Supply one artifact directory or a parent containing several artifacts:

```powershell
.\tools\build\build-runner-package.ps1 `
  -Architecture x64 `
  -AdditionalRunnerRoot C:\artifacts\ryzen-ai-runners `
  -AdditionalRunnerRoot C:\artifacts\openvino-runners
```

Imports are rejected when:

- the runner ID is unknown or duplicated;
- manifest architecture and package architecture differ;
- the executable PE architecture differs;
- a declared file is missing, duplicated, unsafe, or has the wrong hash;
- the artifact does not declare exactly one `NpuInferenceBench.exe`.

This permits a normal build host to assemble shared assets and portable packs,
then merge AMD, Intel, or Qualcomm outputs produced on SDK-equipped CI agents.

## Run the package

Normal benchmark options are unchanged:

```powershell
.\run-benchmark.ps1
.\run-benchmark.ps1 -Only whisper -Device cpu,gpu
.\run-benchmark.ps1 -Only tsc,fakeaudio -Device npu -Provider auto
.\run-benchmark.ps1 -Runtime winml -Device npu,gpu,cpu
```

Inspect routing without running models:

```powershell
.\run-benchmark.ps1 --list-runners
.\run-benchmark.ps1 --explain -Runtime bundled -Provider auto
.\run-benchmark.ps1 --explain -Provider DmlExecutionProvider -Device gpu
```

Automatic bundled selection prefers the host vendor pack: Qualcomm QNN, AMD
VitisAI, or Intel OpenVINO EP. It falls back to portable ORT when that vendor
pack is absent. Explicit DirectML and OpenVINO requests select compatible
dedicated packs when present; portable ORT may satisfy DirectML when its
manifest declares that provider. `-Runtime winml` selects only the Windows ML
pack. An unsupported explicit request fails with the rejected-runner reasons
instead of silently changing providers.

`-Runtime all` runs an available bundled selection and then Windows ML. Missing
Windows ML is reported and skipped; a missing bundled runner is an error.

Before launching native code, the harness replaces process `PATH` with the
selected runner directory plus Windows system directories and clears vendor SDK
path overrides. The original environment is restored afterward. Environment
snapshots record both `runner_id` and the selected platform tag.

## Publish

Do not merge x64 and ARM64 trees. Publish each directory as a separate archive:

```powershell
Compress-Archive -Path .\dist\npu-inference-bench-ARM64\* `
  -DestinationPath .\dist\npu-inference-bench-ARM64.zip -Force
Compress-Archive -Path .\dist\npu-inference-bench-x64\* `
  -DestinationPath .\dist\npu-inference-bench-x64.zip -Force
```

Review `runner-package.json` before publishing. A package with skipped entries
is honest and usable, but it is not evidence that those omitted execution
providers were tested.
