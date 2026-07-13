# Windows ML benchmark runtime

Windows ML is an isolated runtime option for the existing native C++ benchmark.
It uses the ONNX Runtime and Execution Provider Catalog from
`Microsoft.WindowsAppSDK.ML`, while preserving the same models, fixtures,
metrics, cold/hot-load procedure, and result ledgers as the bundled runtimes.

## Requirements

- Windows 11 24H2 or newer for hardware-accelerated Windows ML execution
  providers.
- Current OEM CPU, GPU, and NPU drivers.
- Visual Studio Build Tools with the C++ toolchain for the target architecture.
- Network access on the first NPU run if Windows ML needs to acquire the
  compatible vendor execution provider through Windows Update.

The native runtime is self-contained; the benchmark does not require a
packaged Windows App SDK application.

## Setup and build

```powershell
.\tools\setup\setup-winml.ps1
.\tools\build\build.ps1 -EnableWinML -DisableIntel
```

Both scripts default to the host architecture. Use `-Platform ARM64` or
`-Platform x64` explicitly when required. The output is isolated from bundled
ONNX Runtime builds:

```text
build/ARM64-winml/Release/NpuInferenceBench.exe
build/x64-winml/Release/NpuInferenceBench.exe
```

Do not copy DLLs between a `*-winml` tree and the plain, `*-dml`, or `*-ovep`
trees. Each contains a different ONNX Runtime distribution.

## Device selection

Run the same suite with a selected processing unit:

```powershell
.\benchmark\run-suite.ps1 -Runtime winml -Device cpu
.\benchmark\run-suite.ps1 -Runtime winml -Device gpu
.\benchmark\run-suite.ps1 -Runtime winml -Device npu
.\benchmark\run-suite.ps1 -Runtime winml -Device cpu,gpu,npu
```

For reproducible measurements, the benchmark explicitly selects a Windows ML
EP device:

- `cpu` uses `CPUExecutionProvider`.
- `gpu` prefers `DmlExecutionProvider`.
- `npu` selects the compatible NPU EP, such as QNN, OpenVINO, or VitisAI.

This avoids a platform-dependent ambiguity: on Snapdragon, Windows ML's generic
`PREFER_GPU` policy may select QNN's GPU device instead of DirectML. To
experiment with Windows ML's automatic policy selector instead, set:

```powershell
$env:NPU_INFERENCE_BENCH_WINML_SELECTION = "policy"
.\benchmark\run-suite.ps1 -Runtime winml -Device npu
```

The policy mode maps the requested device to `PREFER_NPU`, `PREFER_GPU`, or
`PREFER_CPU`. It is less deterministic and is not the benchmark default.

Always verify `resolved_provider`, `fallback_occurred`, `ep_nodes`, and
`cpu_nodes` in the result. A requested accelerator is not proof that the model
actually ran there.

## Validated Snapdragon ARM64 behavior

With Windows ML 2.1.70 (ONNX Runtime 1.24.6):

- CPU resolved to `CPUExecutionProvider`.
- GPU resolved to `DmlExecutionProvider`.
- NPU resolved to `QNNExecutionProvider`.
- Whisper static, TSC, and FakeAudio ran on all three processing units.
- Whisper dynamic-KV ran on CPU and GPU; NPU remains intentionally unsupported.

FakeAudio on QNN still has large numerical divergence from the fp32 reference.
Windows ML changes provider acquisition and selection; it does not fix that
model-level precision problem.
