# Reusable ONNX Runtime HAL

`OnnxRuntimeHal` is a Windows-first **C++17 static library** for loading and
running arbitrary ONNX models through this repository's CPU, DirectML,
OpenVINO, VitisAI, QNN, and Windows ML execution-provider policies.

It is a normal C++ ABI library for the **same MSVC toolset and CRT** (`/MD` or
`/MDd`) that built the `.lib`. There is no C wrapper and no ABI stability claim
across compiler versions — link it like any other internal static library.

The library owns:

- logical device and explicit provider selection;
- ordered provider attempts and optional fallback;
- strict requested-device enforcement;
- session creation, vendor cache options, and QNN EPContext caching;
- named tensor introspection and inference;
- resolved-provider, resolved-device, profiling, and CPU-offload diagnostics.

It does not own Whisper decoding, audio preprocessing, classifier fixtures,
accuracy scoring, benchmark timing policy, or result ledgers.

## Public C++ API

Include the umbrella header:

```cpp
#include <onnx_runtime_hal.hpp>

using namespace onnx_runtime_hal;

RuntimeOptions options;
options.device = Device::NPU;
options.require_requested_device = true;

RuntimeContext context(options);
ModelSession session = context.load_one(ModelSpec{
    .model_path = model_path,
});

std::vector<Tensor> outputs = session.run(inputs);
const ExecutionDiagnostics& diag = session.diagnostics();
```

Equivalent types also live under `npu_inference_bench::runtime` (and
`npu_inference_bench::Device`). Prefer `onnx_runtime_hal` in new external code.

Public headers (shipped by the package):

| Header | Role |
|--------|------|
| `onnx_runtime_hal.hpp` | Umbrella + `onnx_runtime_hal` aliases |
| `npu_inference_bench/runtime/runtime_context.hpp` | `RuntimeContext` |
| `npu_inference_bench/runtime/model_session.hpp` | `ModelSession`, `ModelSpec` |
| `npu_inference_bench/runtime/tensor.hpp` | Named tensor I/O |
| `npu_inference_bench/runtime/runtime_options.hpp` | Load / provider options |
| `npu_inference_bench/runtime/runtime_error.hpp` | Errors |
| `npu_inference_bench/runtime/device.hpp` | `Device`, `ResolvedDevice` |
| `npu_inference_bench/runtime/execution_diagnostics.hpp` | Provider / offload diagnostics |
| `npu_inference_bench/runtime/precision_policy.hpp` | Precision policy helpers |
| `npu_inference_bench/runtime/detail/ort_session_access.hpp` | Advanced: borrow `Ort::Session&` |

Loading multiple models in one call resolves the provider once and constructs
every session on that provider. This keeps encoder/decoder model sets coherent.

Use `RuntimeOptions::require_requested_device = true` when a request must fail
instead of silently demoting from NPU to GPU/CPU. For Windows ML policy
providers, profiling must be enabled so the concrete provider can be verified.

### Advanced ORT session access

Whisper-style custom decode loops may need the underlying `Ort::Session`. Prefer
`ModelSession::run()` when possible. When not, include
`npu_inference_bench/runtime/detail/ort_session_access.hpp` and use
`runtime::detail::OrtSessionAccess::get(session)` — same-compiler C++ only, and
only when the library was built with `NPU_INFERENCE_BENCH_ORT`.

In-repo Whisper backends still include private `runner/runtime/ort_ep.hpp`
helpers; those headers are **not** part of the redistributable package.

## Package an install tree

```powershell
.\tools\build\package-onnx-runtime-hal.ps1 -Flavor ort -Configuration Release
# -> dist/onnx-runtime-hal-<arch>-ort/
```

Layout:

```text
dist/onnx-runtime-hal-<arch>-<flavor>/
  include/onnx_runtime_hal.hpp
  include/npu_inference_bench/runtime/...
  lib/OnnxRuntimeHal.lib
  msbuild/OnnxRuntimeHal.Flavor.props
  msbuild/OnnxRuntimeHal.props
  msbuild/runtime-pack.targets
  README.md
```

## Consume from another C++ project

```xml
<Import Project="...\onnx-runtime-hal-x64-ort\msbuild\OnnxRuntimeHal.Flavor.props" />
<!-- Import the same ORT/backend SDK props used when the .lib was built. -->
<Import Project="...\onnx-runtime-hal-x64-ort\msbuild\OnnxRuntimeHal.props" />
```

Then `#include <onnx_runtime_hal.hpp>` and link. Stage ORT / vendor DLLs beside
the executable (see `msbuild/runtime-pack.targets` or your own packager).

The consumer must use the **same flavor** that built `OnnxRuntimeHal.lib`
(bundled ORT, Windows ML, OpenVINO EP, AMD, or Qualcomm). Mixing incompatible
ONNX Runtime DLL trees in one output directory is unsupported.

## Generic CLI (reference consumer)

The benchmark executable is also a reference consumer of the public API:

```powershell
.\build\ARM64\Release\NpuInferenceBench.exe run-onnx `
  .\workloads\classifiers\tsc\model.onnx `
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

## Build inside this repo

`projects/OnnxRuntimeHal/OnnxRuntimeHal.vcxproj` produces `OnnxRuntimeHal.lib`
under `build/<PlatformOutTag>/<Configuration>/`.
