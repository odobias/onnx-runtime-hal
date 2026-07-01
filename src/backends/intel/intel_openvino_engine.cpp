// Intel backend: whisper via OpenVINO GenAI. Targets Intel NPU / GPU / CPU.
// This is the reference implementation and the only fully-working backend.
#include "backends/backend_registry.hpp"

#include <chrono>
#include <stdexcept>

#ifdef WHISPER_HAL_INTEL
#include <memory>

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/runtime/properties.hpp"
#endif

namespace whisper_npu {
namespace intel {

#ifdef WHISPER_HAL_INTEL

namespace {

std::string ov_device(const EngineOptions& o) {
    if (!o.device_override.empty()) return o.device_override;
    switch (o.device) {
        case Device::NPU: return "NPU";
        case Device::GPU: return "GPU";
        case Device::CPU: return "CPU";
    }
    return "CPU";
}

class IntelOpenVinoEngine final : public IWhisperEngine {
public:
    explicit IntelOpenVinoEngine(const EngineOptions& options) : device_(ov_device(options)) {
        // Passing ov::cache_dir makes OpenVINO persist the device-compiled blob, so
        // the (slow) NPU compile only happens on the first cold load; warm loads
        // import the cached blob and are near-instant.
        ov::AnyMap properties;
        if (!options.cache_dir.empty()) {
            properties.insert(ov::cache_dir(options.cache_dir));
        }

        const auto t0 = std::chrono::steady_clock::now();
        pipe_ = std::make_unique<ov::genai::WhisperPipeline>(options.model_dir, device_, properties);
        load_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    TranscribeResult transcribe(const AudioSamples& audio) override {
        const auto t0 = std::chrono::steady_clock::now();
        auto result = pipe_->generate(audio);
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        TranscribeResult out;
        out.text = static_cast<std::string>(result);
        out.infer_seconds = secs;
        return out;
    }

    std::string backend_name() const override { return "Intel OpenVINO GenAI"; }
    std::string device_name() const override { return device_; }
    double load_seconds() const override { return load_seconds_; }

private:
    std::string device_;
    std::unique_ptr<ov::genai::WhisperPipeline> pipe_;
    double load_seconds_ = 0.0;
};

}  // namespace

std::unique_ptr<IWhisperEngine> create(const EngineOptions& options) {
    try {
        return std::make_unique<IntelOpenVinoEngine>(options);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Intel OpenVINO backend failed to init: ") +
                                 e.what());
    }
}

bool available() { return true; }

#else  // !WHISPER_HAL_INTEL

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "Intel backend not compiled. Rebuild with EnableIntel=true and the OpenVINO "
        "GenAI SDK available (define WHISPER_HAL_INTEL, link openvino_genai).");
}

bool available() { return false; }

#endif

}  // namespace intel
}  // namespace whisper_npu
