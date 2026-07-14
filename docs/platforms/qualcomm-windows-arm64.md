# Qualcomm Snapdragon on Windows ARM64

This guide defines the tested requirements for running the benchmark's CPU,
Adreno GPU, and Hexagon NPU executors from one native ARM64 executable.

## Tested platform

- Windows 11 on ARM64.
- A Qualcomm Snapdragon system with a DirectX 12-capable Adreno GPU and Hexagon
  HTP NPU.
- Current OEM GPU and NPU drivers. The repository can stage user-mode runtime
  DLLs, but it cannot replace missing or incompatible device drivers.
- Native ARM64 build output. Do not mix x64, ARM64EC, and pure ARM64 DLLs in the
  same process.

The validated software stack is:

- Microsoft Visual Studio 2022 Build Tools with the C++ workload and ARM64
  compiler/linker tools.
- `Microsoft.ML.OnnxRuntime.DirectML` 1.24.4.
- `Microsoft.AI.DirectML` 1.15.4.
- `Qualcomm.ML.OnnxRuntime.QNN` 2.3.0.

The DirectML package currently determines the ONNX Runtime core version. QNN is
loaded through the execution-provider plugin ABI. Never combine DirectML
headers or an import library from one ORT release with `onnxruntime.dll` from
another.

## One-time setup

Run these commands from a native ARM64 PowerShell session:

```powershell
.\tools\setup\setup-qualcomm.ps1 -Platform ARM64
.\tools\fetch\get-models.ps1
.\tools\build\build.ps1 -Platform ARM64 -EnableQualcomm -DisableIntel
```

The model repository is private. Authenticate first with `hf auth login`.
Python is not required by the C++ runner when the Hugging Face snapshot contains
the classifier fixtures. It is only needed to regenerate fixtures locally.

The resulting executable is:

```text
build/ARM64/Release/NpuInferenceBench.exe
```

Its directory must contain `onnxruntime.dll`, `DirectML.dll`,
`onnxruntime_providers_shared.dll`, `onnxruntime_providers_qnn.dll`, the
required `Qnn*.dll` files, and the Hexagon skeleton libraries. The build stages
these files automatically.

## CPU executor

The CPU path uses `CPUExecutionProvider` from the same DirectML-enabled ONNX
Runtime core. It does not require DirectML or QNN hardware, but the unified
binary still requires its staged runtime DLL set to start.

Supported benchmark workloads:

- Whisper static and dynamic-KV models.
- TSC.
- FakeAudio.

Run it with:

```powershell
.\benchmark\run-suite.ps1 -Device cpu -Provider auto
```

A successful result reports:

- `resolved_provider=CPUExecutionProvider`
- `fallback_occurred=false`

CPU nodes are expected on a CPU request and do not represent accidental
accelerator offload.

## Adreno GPU executor

The GPU path uses `DmlExecutionProvider`. It requires:

- a DirectX 12-capable Adreno GPU;
- a working OEM GPU driver;
- the ARM64 DirectML-enabled ORT core and `DirectML.dll` staged by
  `setup-qualcomm.ps1`.

Supported benchmark workloads:

- Whisper static and dynamic-KV models.
- TSC.
- FakeAudio.

Run it with:

```powershell
.\benchmark\run-suite.ps1 -Device gpu -Provider auto
```

A successful GPU result reports:

- `resolved_provider=DmlExecutionProvider`
- `runtime=onnxruntime-directml`
- `fallback_occurred=false`

Some CPU-assigned shape operations are valid DirectML behavior. On the tested
models, Whisper static and TSC execute without CPU offload; Whisper dynamic
retains shape-related CPU nodes, and FakeAudio retains a CPU `Resize` node.
Inspect `cpu_nodes`, `cpu_offload_pct`, and `cpu_offload_ops` rather than
assuming that selecting GPU means every operation ran there.

If the result resolves to `CPUExecutionProvider`, the GPU benchmark did not
succeed. Check the provider-attempt error, confirm that `DirectML.dll` is beside
the executable, and update the OEM GPU driver.

## Hexagon NPU executor

The NPU path uses the plugin `QNNExecutionProvider` with the HTP backend. It
requires:

- a supported Snapdragon Hexagon HTP NPU;
- a working OEM Qualcomm NPU driver;
- pure ARM64 QNN plugin/runtime DLLs and matching Hexagon skeleton libraries;
- static model shapes accepted by the QNN compiler.

The NuGet package supplies pure ARM64 QNN binaries. The pip QNN wheel may supply
ARM64EC binaries and is not a substitute for a native ARM64 process.

Currently supported benchmark workloads:

- Whisper static.
- TSC.
- FakeAudio using the `npu-split-generic` profile: fp32 CPU log-mel frontend
  fixture plus the generic fp32 backbone on QNN.

Whisper dynamic-KV is intentionally unsupported because its growing cache
shapes cannot be compiled as the fixed HTP graph used by this benchmark.
Whole-graph FakeAudio NPU is known-invalid and is available only as the explicit
`npu-whole-diagnostic` profile. It is excluded from the default matrix. Split
FakeAudio preserves the full-fp32 CPU probabilities as its end-to-end reference;
the trust gate rejects the result if any decision flips or the intended QNN
provider does not resolve.

Run the supported NPU workloads with:

```powershell
.\benchmark\run-suite.ps1 -Runtime bundled -Device npu -Only whisper,tsc,fakeaudio
```

A successful NPU result reports:

- `resolved_provider=QNNExecutionProvider`
- `runtime=onnxruntime-qnn`
- `fallback_occurred=false`
- `cpu_nodes=0` for the currently validated Whisper static and TSC graphs

The first load compiles the graph for HTP and is expected to be slow. The hot
load reuses the generated QNN context. A fast inference result paired with a
CPU-resolved provider is not an NPU result.

## Full platform sweep

Run every declared combination with:

```powershell
.\benchmark\run-suite.ps1 -Device npu,gpu,cpu -Provider auto
```

The default `accuracy-quick` run writes `results/ledgers/accuracy.jsonl`,
immutable host/SDK sidecars, and `results/ledgers/attempts.jsonl`; it does not
touch latency CSVs. Use `-Mode latency` explicitly to append performance rows.

For comparable performance measurements:

- connect AC power and use the same Windows power mode;
- close competing GPU/NPU workloads;
- use the suite's isolated caches rather than reusing arbitrary compiled blobs;
- compare identical model hashes, audio/fixtures, run counts, and runtime
  versions.

## Diagnosis checklist

1. Re-run `setup-qualcomm.ps1 -Platform ARM64` after changing ORT, DirectML, or
   QNN versions.
2. Rebuild with `-Platform ARM64 -EnableQualcomm -DisableIntel`.
3. Confirm the JSON result's `requested_provider`, `resolved_provider`,
   `fallback_occurred`, and `provider_attempts`.
4. Treat `assets-missing` and `unsupported` as skipped measurements, not zeroes.
5. Inspect CPU-offload fields for successful GPU/NPU runs.
6. Keep the executable and every staged DLL on the same architecture.
