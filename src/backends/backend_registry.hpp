// Internal registry: each backend exposes a create() + available() pair. The
// factory (factory.cpp) dispatches to these. Real implementations live behind
// their WHISPER_HAL_* compile macros; otherwise these compile as throwing stubs.
#pragma once

#include "whisper_npu/whisper_engine.hpp"

namespace whisper_npu {

namespace intel {
std::unique_ptr<IWhisperEngine> create(const EngineOptions& options);
bool available();
}  // namespace intel

namespace intel_onnx {
std::unique_ptr<IWhisperEngine> create(const EngineOptions& options);
bool available();
}  // namespace intel_onnx

namespace ort_static {
std::unique_ptr<IWhisperEngine> create(const EngineOptions& options);
bool available();
// Runtime probe of the ORT execution-provider device list: best logical device
// this build can run on now (NPU if a QNN accelerator is present, else CPU).
Device best_available_device();
}  // namespace ort_static

namespace amd {
std::unique_ptr<IWhisperEngine> create(const EngineOptions& options);
bool available();
}  // namespace amd

namespace qualcomm {
std::unique_ptr<IWhisperEngine> create(const EngineOptions& options);
bool available();
}  // namespace qualcomm

}  // namespace whisper_npu
