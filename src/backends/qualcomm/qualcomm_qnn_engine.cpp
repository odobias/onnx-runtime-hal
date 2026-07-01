// Qualcomm backend: whisper on Snapdragon Hexagon NPU via ONNX Runtime + QNN EP.
//
// STATUS: PREPARED SCAFFOLD, NOT VERIFIED. We have no Snapdragon hardware to test on.
// The platform-specific wiring (ORT session + QNN execution provider targeting the
// HTP/Hexagon backend) is written out under WHISPER_HAL_QUALCOMM so it is ready to
// complete against a real QNN SDK; the whisper encode/decode loop is intentionally
// left as a clearly-marked TODO rather than shipping untested code.
//
// To activate: install the Qualcomm AI Engine Direct (QNN) SDK + ONNX Runtime built
// with the QNN EP, set EnableQualcomm=true (defines WHISPER_HAL_QUALCOMM), and point
// the build at the ORT/QNN SDK (see msbuild/backend.qualcomm.props). On Windows this
// typically targets ARM64. Use a QNN context binary or QDQ ONNX for the HTP backend.
#include "backends/backend_registry.hpp"

#include <stdexcept>

#ifdef WHISPER_HAL_QUALCOMM
#include <string>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>  // from the ONNX Runtime QNN SDK
#endif

namespace whisper_npu {
namespace qualcomm {

#ifdef WHISPER_HAL_QUALCOMM

namespace {

Ort::SessionOptions make_session_options(const EngineOptions& options) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::unordered_map<std::string, std::string> qnn_opts;
    switch (options.device) {
        case Device::NPU:
            // HTP = Hexagon Tensor Processor (the NPU). QnnHtp.dll is the QNN backend.
            qnn_opts["backend_path"] = "QnnHtp.dll";
            qnn_opts["htp_performance_mode"] = "burst";
            break;
        case Device::GPU:
            qnn_opts["backend_path"] = "QnnGpu.dll";
            break;
        case Device::CPU:
        default:
            qnn_opts["backend_path"] = "QnnCpu.dll";
            break;
    }
    if (!options.device_override.empty()) {
        qnn_opts["backend_path"] = options.device_override;
    }
    so.AppendExecutionProvider("QNN", qnn_opts);
    return so;
}

class QualcommQnnEngine final : public IWhisperEngine {
public:
    explicit QualcommQnnEngine(const EngineOptions& options)
        : env_(ORT_LOGGING_LEVEL_WARNING, "whisper_hal_qnn"),
          options_(options),
          session_options_(make_session_options(options)) {
        // TODO(qnn): load encoder/decoder (ONNX QDQ or QNN context binary) from
        // options.model_dir into ORT sessions; session creation triggers QNN graph
        // finalize / HTP prepare.
    }

    TranscribeResult transcribe(const AudioSamples& /*audio*/) override {
        // TODO(qnn): same whisper pipeline as the AMD backend (mel -> encoder ->
        // decoder KV-cache loop -> detokenize). onnxruntime-genai can package this.
        throw std::runtime_error(
            "Qualcomm QNN backend: ORT/QNN wiring is present but the whisper "
            "encode/decode pipeline is not implemented yet. See src/backends/qualcomm.");
    }

    std::string backend_name() const override { return "Qualcomm QNN (ONNX Runtime + QNN EP)"; }
    std::string device_name() const override {
        return options_.device == Device::NPU ? "Hexagon-HTP (QNN)" : to_string(options_.device);
    }
    // TODO(qnn): set load_seconds_ around Ort::Session construction, and use a QNN
    // context binary (ep.context_enable) so warm loads skip HTP graph prepare.
    double load_seconds() const override { return load_seconds_; }

private:
    Ort::Env env_;
    EngineOptions options_;
    Ort::SessionOptions session_options_;
    double load_seconds_ = 0.0;
};

}  // namespace

std::unique_ptr<IWhisperEngine> create(const EngineOptions& options) {
    return std::make_unique<QualcommQnnEngine>(options);
}

bool available() { return true; }

#else  // !WHISPER_HAL_QUALCOMM

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "Qualcomm QNN backend not compiled. Install the QNN SDK + ONNX Runtime QNN EP, "
        "set EnableQualcomm=true, and provide the ORT/QNN SDK path "
        "(msbuild/backend.qualcomm.props).");
}

bool available() { return false; }

#endif

}  // namespace qualcomm
}  // namespace whisper_npu
