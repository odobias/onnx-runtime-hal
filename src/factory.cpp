#include "whisper_npu/whisper_engine.hpp"

#include <stdexcept>

#include "backends/backend_registry.hpp"

namespace whisper_npu {

const char* to_string(Backend backend) {
    switch (backend) {
        case Backend::Auto:          return "Auto";
        case Backend::IntelOpenVINO: return "IntelOpenVINO";
        case Backend::IntelOnnx:     return "IntelOnnx";
        case Backend::OnnxRuntimeStatic: return "OnnxRuntimeStatic";
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
        case Backend::OnnxRuntimeStatic: return ort_static::available();
        case Backend::AmdRyzenAI:    return amd::available();
        case Backend::QualcommQNN:   return qualcomm::available();
        case Backend::Auto:
            return intel::available() || intel_onnx::available() || ort_static::available() ||
                   amd::available() || qualcomm::available();
    }
    return false;
}

std::vector<Backend> available_backends() {
    std::vector<Backend> out;
    if (intel::available())      out.push_back(Backend::IntelOpenVINO);
    if (intel_onnx::available()) out.push_back(Backend::IntelOnnx);
    if (ort_static::available()) out.push_back(Backend::OnnxRuntimeStatic);
    if (amd::available())        out.push_back(Backend::AmdRyzenAI);
    if (qualcomm::available())   out.push_back(Backend::QualcommQNN);
    return out;
}

std::unique_ptr<IWhisperEngine> create_engine(Backend backend, const EngineOptions& options) {
    if (backend == Backend::Auto) {
        // Prefer the unified ONNX Runtime backend: one engine that self-selects
        // its execution provider at runtime (QNN NPU -> DirectML GPU -> CPU) from
        // the real EP device list, so a single binary adapts to whatever hardware
        // and drivers are present. Only fall back to a vendor-specific backend if
        // ORT was not compiled in.
        if (ort_static::available()) {
            EngineOptions o = options;
            // Only auto-pick the device when the caller asked for it (device "auto").
            // An explicitly requested NPU/GPU/CPU must be honored so per-device
            // benchmark rows are not silently collapsed onto the NPU.
            if (o.auto_device && o.device_override.empty()) {
                o.device = ort_static::best_available_device();
            }
            return ort_static::create(o);
        }
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
        case Backend::OnnxRuntimeStatic: return ort_static::create(options);
        case Backend::AmdRyzenAI:    return amd::create(options);
        // Qualcomm QNN is now served by the unified ONNX Runtime backend (it
        // registers the QNN plugin EP internally). The standalone qualcomm engine
        // remains compiled for reference but is no longer on the active path when
        // ORT is available.
        case Backend::QualcommQNN:
            return ort_static::available() ? ort_static::create(options) : qualcomm::create(options);
        default: break;
    }
    throw std::runtime_error(std::string("unknown backend: ") + to_string(backend));
}

}  // namespace whisper_npu
