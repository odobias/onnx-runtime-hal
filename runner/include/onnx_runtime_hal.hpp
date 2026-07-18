#pragma once

// Public C++ umbrella for OnnxRuntimeHal.
// Same-compiler, C++23 ABI — link OnnxRuntimeHal.lib built with the matching
// ORT flavor props. No C API; use these types directly.

#include "npu_inference_bench/runtime/device.hpp"
#include "npu_inference_bench/runtime/execution_diagnostics.hpp"
#include "npu_inference_bench/runtime/model_session.hpp"
#include "npu_inference_bench/runtime/runtime_context.hpp"
#include "npu_inference_bench/runtime/runtime_error.hpp"
#include "npu_inference_bench/runtime/runtime_options.hpp"
#include "npu_inference_bench/runtime/tensor.hpp"

namespace onnx_runtime_hal {

using npu_inference_bench::Device;
using npu_inference_bench::to_string;

using npu_inference_bench::runtime::ExecutionDiagnostics;
using npu_inference_bench::runtime::ProviderAttempt;
using npu_inference_bench::runtime::LoadedModelSet;
using npu_inference_bench::runtime::ModelSession;
using npu_inference_bench::runtime::ModelSpec;
using npu_inference_bench::runtime::ResolvedDevice;
using npu_inference_bench::runtime::RuntimeContext;
using npu_inference_bench::runtime::RuntimeError;
using npu_inference_bench::runtime::RuntimeErrorCode;
using npu_inference_bench::runtime::RuntimeOptions;
using npu_inference_bench::runtime::Tensor;
using npu_inference_bench::runtime::TensorDescriptor;
using npu_inference_bench::runtime::TensorElementType;
using npu_inference_bench::runtime::TensorView;
using npu_inference_bench::runtime::element_count;
using npu_inference_bench::runtime::element_size;
using npu_inference_bench::runtime::requested_device_class;
using npu_inference_bench::runtime::to_string;

}  // namespace onnx_runtime_hal
