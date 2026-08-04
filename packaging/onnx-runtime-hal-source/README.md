# GenDigital.OnnxRuntimeHal.Source

This package distributes the canonical, workload-neutral ONNX Runtime HAL as
native C++ source. A consuming project compiles the HAL against its own ONNX
Runtime SDK, avoiding a prebuilt C++ ABI and an accidental dependency on the
publisher's vendor runtime DLLs.

The package contains:

- the public `onnx_runtime_hal.hpp` and runtime headers;
- the versioned `schema/execution-plan.schema.json` contract;
- whole-graph and split-graph FakeAudio execution-plan examples;
- provider selection, model-session, tensor, and execution-diagnostics code;
- two implementation translation units;
- `build/native/GenDigital.OnnxRuntimeHal.Source.props`, imported by NuGet.

It deliberately excludes model weights, benchmark orchestration, workload
adapters, accuracy baselines, and vendor runtime binaries.

The execution-plan schema is a shared data contract, not an executor. Consumers
must implement and explicitly gate the adapter kinds they support.

## Consume from native MSBuild

Add the package using the repository's normal NuGet restore mechanism and
import the ONNX Runtime SDK before the automatically imported package props.
The package adds its include paths, defines `NPU_INFERENCE_BENCH_ORT`, and
compiles the two HAL translation units as C++ latest without precompiled
headers.

```cpp
#include <onnx_runtime_hal.hpp>

onnx_runtime_hal::RuntimeOptions options;
options.device = onnx_runtime_hal::Device::CPU;

onnx_runtime_hal::RuntimeContext runtime(std::move(options));
auto session = runtime.load_one({.model_path = model_path});
auto outputs = session.run(inputs);
```

Set `OnnxRuntimeHalCompileSources=false` before the package import when a
project only needs the public headers and links the implementation elsewhere.

## Compatibility contract

- Windows and MSVC are the supported build environment.
- The consumer supplies a compatible ONNX Runtime SDK and runtime DLLs.
- Provider-specific compile definitions remain consumer policy.
- Package versions are immutable. Published `.nupkg`, `.sha256`, and
  `.provenance.json` assets identify the same source commit.
