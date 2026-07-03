#include "whisper_npu/whisper_engine.hpp"

#include <stdexcept>

#include "backends/backend_registry.hpp"

namespace whisper_npu {

const char* to_string(Backend backend) {
    switch (backend) {
        case Backend::Auto:          return "Auto";
        case Backend::IntelOpenVINO: return "IntelOpenVINO";
        case Backend::IntelOnnx:     return "IntelOnnx";
        case Backend::AmdRyzenAI:    return "AmdRyzenAI";
        case Backend::QualcommQNN:   return "QualcommQNN";
    }
    return "Unknown";
}

const char* to_string(Device device) {
    switch (device) {
        case Device::NPU: return "NPU";
        case Device::GPU: return "GPU";
        case Device::CPU: return "CPU";
    }
    return "Unknown";
}

bool backend_available(Backend backend) {
    switch (backend) {
        case Backend::IntelOpenVINO: return intel::available();
        case Backend::IntelOnnx:     return intel_onnx::available();
        case Backend::AmdRyzenAI:    return amd::available();
        case Backend::QualcommQNN:   return qualcomm::available();
        case Backend::Auto:
            return intel::available() || intel_onnx::available() || amd::available() ||
                   qualcomm::available();
    }
    return false;
}

std::vector<Backend> available_backends() {
    std::vector<Backend> out;
    if (intel::available())      out.push_back(Backend::IntelOpenVINO);
    if (intel_onnx::available()) out.push_back(Backend::IntelOnnx);
    if (amd::available())        out.push_back(Backend::AmdRyzenAI);
    if (qualcomm::available())   out.push_back(Backend::QualcommQNN);
    return out;
}

std::unique_ptr<IWhisperEngine> create_engine(Backend backend, const EngineOptions& options) {
    if (backend == Backend::Auto) {
        for (Backend b : available_backends()) {
            return create_engine(b, options);
        }
        throw std::runtime_error(
            "no Whisper HAL backend is compiled into this build "
            "(enable WHISPER_HAL_INTEL / _AMD / _QUALCOMM)");
    }

    switch (backend) {
        case Backend::IntelOpenVINO: return intel::create(options);
        case Backend::IntelOnnx:     return intel_onnx::create(options);
        case Backend::AmdRyzenAI:    return amd::create(options);
        case Backend::QualcommQNN:   return qualcomm::create(options);
        default: break;
    }
    throw std::runtime_error(std::string("unknown backend: ") + to_string(backend));
}

}  // namespace whisper_npu
