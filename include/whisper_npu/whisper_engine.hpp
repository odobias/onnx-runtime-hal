// Public API for the Whisper NPU hardware-abstraction layer (HAL).
//
// One stable C++ interface (IWhisperEngine) for running whisper-tiny(.en) speech
// recognition on different vendor NPUs. Concrete backends are selected at runtime
// via create_engine(); which backends are *available* depends on what the library
// was compiled with (see WHISPER_HAL_INTEL / _AMD / _QUALCOMM).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace whisper_npu {

// Vendor backend / inference runtime.
enum class Backend {
    Auto,          // pick the first compiled-in, available backend
    IntelOpenVINO, // Intel NPU/GPU/CPU via OpenVINO GenAI  (reference impl)
    AmdRyzenAI,    // AMD XDNA NPU via ONNX Runtime + VitisAI EP  (prepared)
    QualcommQNN,   // Qualcomm Hexagon NPU via ONNX Runtime + QNN EP  (prepared)
};

// Logical accelerator target. Mapped to a backend-specific device string.
enum class Device {
    NPU,
    GPU,
    CPU,
};

struct EngineOptions {
    // Path to the exported model directory for the chosen backend.
    // Intel: OpenVINO IR dir (e.g. whisper-tiny-en-hybrid-ov).
    // AMD/Qualcomm: ONNX / QNN context-binary dir (backend-defined).
    std::string model_dir;

    Device device = Device::NPU;

    // Optional raw device string that overrides `device` for the backend
    // (e.g. OpenVINO "NPU", "GPU.1"; ORT EP-specific selectors). Empty = derive
    // from `device`.
    std::string device_override;

    // Directory for the compiled-model cache. When set, the backend persists its
    // device-compiled blob here so subsequent loads skip the (slow) NPU compile
    // step. Empty = no on-disk caching. This is the key to a fast-loading runner.
    std::string cache_dir;
};

struct TranscribeResult {
    std::string text;
    double infer_seconds = 0.0;  // wall-clock of the generate/inference call
};

// 16 kHz, mono, float PCM normalized to [-1, 1].
using AudioSamples = std::vector<float>;

class IWhisperEngine {
public:
    virtual ~IWhisperEngine() = default;

    // Run recognition on a full audio buffer. Throws std::runtime_error on failure.
    virtual TranscribeResult transcribe(const AudioSamples& audio) = 0;

    virtual std::string backend_name() const = 0;
    virtual std::string device_name() const = 0;

    // Wall-clock time spent constructing/compiling this engine (model load +
    // device compile). With a warm cache this should drop dramatically.
    virtual double load_seconds() const = 0;
};

// --- Introspection -----------------------------------------------------------

// True if `backend` was compiled into this build and can be instantiated.
bool backend_available(Backend backend);

// All compiled-in backends (may still fail at create() if no hardware/model).
std::vector<Backend> available_backends();

const char* to_string(Backend backend);
const char* to_string(Device device);

// --- Factory -----------------------------------------------------------------

// Create an engine for `backend`. With Backend::Auto, picks the first available
// backend in preference order (Intel, AMD, Qualcomm).
// Throws std::runtime_error if the backend is unavailable or model load fails.
std::unique_ptr<IWhisperEngine> create_engine(Backend backend, const EngineOptions& options);

}  // namespace whisper_npu
