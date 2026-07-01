// AMD backend: whisper on AMD XDNA NPU (Ryzen AI) via ONNX Runtime + VitisAI EP.
//
// STATUS: PREPARED SCAFFOLD, NOT VERIFIED. We have no AMD hardware to test on.
// The platform-specific wiring (ORT session + VitisAI execution provider) is written
// out under WHISPER_HAL_AMD so it is ready to complete against a real Ryzen AI SDK;
// the whisper encode/decode loop is intentionally left as a clearly-marked TODO
// instead of shipping untested code that merely looks correct.
//
// To activate: install the AMD Ryzen AI SW stack (ONNX Runtime + VitisAI EP), set
// EnableAmd=true (defines WHISPER_HAL_AMD), and point the build at the ORT SDK
// (see msbuild/backend.amd.props). Export whisper-tiny.en to ONNX quantized for XDNA.
#include "backends/backend_registry.hpp"

#include <stdexcept>

#ifdef WHISPER_HAL_AMD
#include <string>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>  // from the Ryzen AI / ONNX Runtime SDK
#endif

namespace whisper_npu {
namespace amd {

#ifdef WHISPER_HAL_AMD

namespace {

// Maps our logical Device to a VitisAI/ORT selection. AMD's NPU is the XDNA/IPU
// target exposed through the VitisAI EP; GPU/CPU fall back to ORT's stock EPs.
Ort::SessionOptions make_session_options(const EngineOptions& options) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (options.device == Device::NPU) {
        // VitisAI EP drives the XDNA NPU. Provider options are SDK/version specific;
        // typical keys: "config_file" (vaip config), "cacheDir", "cacheKey".
        std::unordered_map<std::string, std::string> vitis_opts;
        if (!options.device_override.empty()) {
            vitis_opts["config_file"] = options.device_override;
        }
        so.AppendExecutionProvider("VitisAI", vitis_opts);
    }
    // else: default CPU EP (and DML/ROCm could be wired here for GPU).
    return so;
}

class AmdRyzenAiEngine final : public IWhisperEngine {
public:
    explicit AmdRyzenAiEngine(const EngineOptions& options)
        : env_(ORT_LOGGING_LEVEL_WARNING, "whisper_hal_amd"),
          options_(options),
          session_options_(make_session_options(options)) {
        // TODO(amd): load encoder/decoder ONNX models from options.model_dir and
        // construct the ORT sessions, e.g.:
        //   encoder_ = Ort::Session(env_, enc_path.c_str(), session_options_);
        //   decoder_ = Ort::Session(env_, dec_path.c_str(), session_options_);
        // Session construction is where VitisAI compiles/loads the XDNA binary.
    }

    TranscribeResult transcribe(const AudioSamples& /*audio*/) override {
        // TODO(amd): implement the whisper pipeline on ORT:
        //   1. log-mel spectrogram (80 x 3000) from the 16 kHz audio,
        //   2. run encoder session,
        //   3. autoregressive decoder loop with KV-cache I/O bindings,
        //   4. detokenize to text.
        // Consider onnxruntime-genai (Oga* API) which packages steps 2-4.
        throw std::runtime_error(
            "AMD Ryzen AI backend: ORT/VitisAI wiring is present but the whisper "
            "encode/decode pipeline is not implemented yet. See src/backends/amd.");
    }

    std::string backend_name() const override { return "AMD Ryzen AI (ONNX Runtime + VitisAI EP)"; }
    std::string device_name() const override {
        return options_.device == Device::NPU ? "XDNA-NPU (VitisAI)" : to_string(options_.device);
    }
    // TODO(amd): set load_seconds_ around the Ort::Session construction, and enable
    // ORT EP context caching (VitisAI cacheDir/cacheKey) so warm loads skip compile.
    double load_seconds() const override { return load_seconds_; }

private:
    Ort::Env env_;
    EngineOptions options_;
    Ort::SessionOptions session_options_;
    double load_seconds_ = 0.0;
    // Ort::Session encoder_{nullptr};
    // Ort::Session decoder_{nullptr};
};

}  // namespace

std::unique_ptr<IWhisperEngine> create(const EngineOptions& options) {
    return std::make_unique<AmdRyzenAiEngine>(options);
}

bool available() { return true; }

#else  // !WHISPER_HAL_AMD

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "AMD Ryzen AI backend not compiled. Install the Ryzen AI SW stack "
        "(ONNX Runtime + VitisAI EP), set EnableAmd=true, and provide the ORT SDK "
        "path (msbuild/backend.amd.props).");
}

bool available() { return false; }

#endif

}  // namespace amd
}  // namespace whisper_npu
