# Reusable ONNX Runtime HAL

`OnnxRuntimeHal` is a Windows-first C++17 static library for loading and running
arbitrary ONNX models through the repository's CPU, DirectML, OpenVINO, VitisAI,
QNN, and Windows ML execution-provider policies.

The library owns:

- logical device and explicit provider selection;
- ordered provider attempts and optional fallback;
- strict requested-device enforcement;
- session creation, vendor cache options, and QNN EPContext caching;
- named tensor introspection and inference;
- resolved-provider, resolved-device, profiling, and CPU-offload diagnostics.

It does not own Whisper decoding, audio preprocessing, classifier fixtures,
accuracy scoring, benchmark timing policy, or result ledgers.

## C++ API

Public headers are under
`runner/include/npu_inference_bench/runtime/`. A consumer creates a
`runtime::RuntimeContext`, loads one or more `runtime::ModelSpec` objects, and
runs named `runtime::TensorView` inputs through a `runtime::ModelSession`.

Loading multiple models in one call resolves the provider once and constructs
every session on that provider. This keeps encoder/decoder model sets coherent.

Use `RuntimeOptions::require_requested_device = true` when a request must fail
instead of silently demoting from NPU to GPU/CPU. For Windows ML policy
providers, profiling must be enabled so the concrete provider can be verified.

## Generic CLI

The benchmark executable is also a reference consumer:

```powershell
.\build\ARM64\Release\NpuInferenceBench.exe run-onnx `
  .\models\deepfake\tsc\model.onnx `
  .\tests\runtime-hal-tsc-inputs.json `
  --device cpu `
  --strict-device `
  --output-dir .\build\onnx-output
```

The input manifest contains named binary tensors:

```json
{
  "inputs": [
    {
      "name": "input_ids",
      "dtype": "int64",
      "shape": [1, 512],
      "file": "input_ids.bin"
    }
  ],
  "outputs": ["logits"]
}
```

Input files are resolved relative to the manifest. Omitting `outputs` returns
every model output. Output tensors are written as raw binary files, while stdout
receives one JSON result containing shapes, types, files, runtime versions,
provider resolution, fallback, load/inference timing, and offload counts.

## Build and runtime packs

`projects/OnnxRuntimeHal/OnnxRuntimeHal.vcxproj` produces
`OnnxRuntimeHal.lib`. Consumers can import:

- `msbuild/OnnxRuntimeHal.props` for headers and static-library linkage;
- `msbuild/runtime-pack.targets` for runtime DLL staging.

The consumer must use the same backend properties and runtime flavor that built
the library. Bundled, Windows ML, OpenVINO EP, AMD, and Qualcomm DLL trees are
intentionally isolated; combining their incompatible ONNX Runtime binaries in
one output directory is unsupported.
