#include "npu_inference_bench/whisper.hpp"

#include <stdexcept>

#include "backend_registry.hpp"

namespace npu_inference_bench {

const char* to_string(Backend backend) {
    switch (backend) {
        case Backend::Auto:          return "Auto";
        case Backend::IntelOpenVINO: return "IntelOpenVINO";
        case Backend::IntelOnnx:     return "IntelOnnx";
        case Backend::OnnxRuntimeStatic: return "OnnxRuntimeStatic";
        case Backend::OnnxRuntimeDynamic: return "OnnxRuntimeDynamic";
        case Backend::AmdRyzenAI:    return "AmdRyzenAI";
        case Backend::QualcommQNN:   return "QualcommQNN";
    }
    return "Unknown";
}

bool backend_available(Backend backend) {
    switch (backend) {
        case Backend::IntelOpenVINO: return intel::available();
        case Backend::IntelOnnx:     return intel_onnx::available();
        case Backend::OnnxRuntimeStatic: return ort_static::available();
        case Backend::OnnxRuntimeDynamic: return ort_dynamic::available();
        case Backend::AmdRyzenAI:    return amd::available();
        case Backend::QualcommQNN:   return qualcomm::available();
        case Backend::Auto:
            return intel::available() || intel_onnx::available() || ort_static::available() ||
                   ort_dynamic::available() || amd::available() || qualcomm::available();
    }
    return false;
}

std::vector<Backend> available_backends() {
    std::vector<Backend> out;
    if (intel::available())      out.push_back(Backend::IntelOpenVINO);
    if (intel_onnx::available()) out.push_back(Backend::IntelOnnx);
    if (ort_static::available()) out.push_back(Backend::OnnxRuntimeStatic);
    if (ort_dynamic::available()) out.push_back(Backend::OnnxRuntimeDynamic);
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
        std::string failures;
        for (Backend b : available_backends()) {
            try {
                auto engine = create_engine(b, options);
                if (engine) return engine;
                if (!failures.empty()) failures += "; ";
                failures += std::string(to_string(b)) + ": returned no engine";
            } catch (const std::exception& e) {
                if (!failures.empty()) failures += "; ";
                failures += std::string(to_string(b)) + ": " + e.what();
            }
        }
        if (!failures.empty()) {
            throw std::runtime_error(
                "all compiled Whisper workload backends failed: " + failures);
        }
        throw std::runtime_error(
            "no Whisper workload backend is compiled into this build "
            "(enable NPU_INFERENCE_BENCH_INTEL / _AMD / _QUALCOMM)");
    }

    switch (backend) {
        case Backend::IntelOpenVINO: return intel::create(options);
        case Backend::IntelOnnx:     return intel_onnx::create(options);
        case Backend::OnnxRuntimeStatic: return ort_static::create(options);
        case Backend::OnnxRuntimeDynamic: return ort_dynamic::create(options);
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

}  // namespace npu_inference_bench
