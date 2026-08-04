#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "npu_inference_bench/runtime/runtime_options.hpp"

namespace npu_inference_bench::runtime {

enum class AdapterKind : std::uint8_t {
  GenericOnnx,
  StagedOnnx,
  WhisperStatic,
  WhisperDynamicKv,
  WhisperSherpa,
};

enum class StageDevice : std::uint8_t {
  Requested,
  Cpu,
};

struct TensorBinding {
  std::string input;
  std::string source_stage;
  std::string source_output;
};

struct GraphStage {
  std::string id;
  std::string artifact;
  StageDevice device = StageDevice::Requested;
  std::vector<std::string> outputs;
  std::vector<TensorBinding> bindings;
};

struct ExecutionPlan {
  std::uint32_t schema_version = 1;
  std::string id;
  AdapterKind adapter = AdapterKind::GenericOnnx;
  RuntimeOptions runtime;
  std::vector<std::string> eligible_architectures;
  std::vector<GraphStage> stages;
};

} // namespace npu_inference_bench::runtime
