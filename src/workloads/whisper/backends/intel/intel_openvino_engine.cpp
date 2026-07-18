// Intel backend: whisper via OpenVINO GenAI. Targets Intel NPU / GPU / CPU.
// This is the reference implementation and the only fully-working backend.
#include "backend_registry.hpp"
#include "npu_inference_bench/precision_policy.hpp"

#include <chrono>
#include <stdexcept>

#ifdef NPU_INFERENCE_BENCH_INTEL
#include <memory>

#include "openvino/core/version.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#endif

namespace npu_inference_bench {
namespace intel {

#ifdef NPU_INFERENCE_BENCH_INTEL

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

// Real hardware identity (e.g. "Intel(R) AI Boost", "13th Gen Intel(R) Core(TM)
// i7-1370P", "AMD Ryzen Threadripper PRO 7955WX 16-Cores"), as opposed to the
// logical device class in `device_`. Answers "which chip, specifically" instead of
// just "NPU/GPU/CPU" -- the CSV/README previously had no way to tell two NPUs from
// two different machines apart. A throwaway ov::Core is enough; this just reads a
// plugin property, it doesn't compile anything.
std::string query_full_device_name(const std::string& device) {
    try {
        ov::Core core;
        std::string name = core.get_property(device, ov::device::full_name);
        // The CPU plugin returns the raw CPUID brand string, which is
        // space-padded to a fixed width (e.g. "...7955WX 16-Cores     ").
        const auto end = name.find_last_not_of(' ');
        if (end == std::string::npos) return "";
        name.erase(end + 1);
        return name;
    } catch (...) {
        return "";
    }
}

class IntelOpenVinoEngine final : public IWhisperEngine {
public:
    explicit IntelOpenVinoEngine(const EngineOptions& options)
        : device_(ov_device(options)),
          full_device_name_(query_full_device_name(device_)),
          inference_precision_(precision_policy::resolve(options)) {
        // Passing ov::cache_dir makes OpenVINO persist the device-compiled blob, so
        // the (slow) NPU compile only happens on the first cold load; warm loads
        // import the cached blob and are near-instant.
        ov::AnyMap properties;
        if (!options.cache_dir.empty()) {
            properties.insert(ov::cache_dir(options.cache_dir));
        }
        // Thread pinning only makes sense on CPU; NPU/GPU ignore it anyway, but
        // avoid surprising a discrete accelerator with a CPU-shaped hint.
        if (options.cpu_threads > 0 && device_ == "CPU") {
            properties.insert(ov::inference_num_threads(options.cpu_threads));
        }
        if (inference_precision_ == "f32") {
            properties.insert(ov::hint::inference_precision(ov::element::f32));
        } else if (inference_precision_ == "f16") {
            properties.insert(ov::hint::inference_precision(ov::element::f16));
        } else if (inference_precision_ == "bf16") {
            properties.insert(ov::hint::inference_precision(ov::element::bf16));
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

        // Confidence: WhisperDecodedResults.scores holds the summed token log-prob
        // for each hypothesis (index 0 = the returned transcription).
        if (!result.scores.empty()) {
            out.sequence_logprob = static_cast<double>(result.scores[0]);
        }
        auto& pm = result.perf_metrics;
        out.generated_tokens = static_cast<long>(pm.get_num_generated_tokens());
        if (out.generated_tokens > 0) {
            out.avg_logprob = out.sequence_logprob / static_cast<double>(out.generated_tokens);
        }
        out.ttft_ms = pm.get_ttft().mean;
        out.tpot_ms = pm.get_tpot().mean;
        out.throughput_tps = pm.get_throughput().mean;
        out.has_token_metrics = true;
        out.runtime = "openvino-genai";
        out.model_format = "ov-ir";
        out.decode_strategy = "genai-bounded-kv";
        return out;
    }

    std::string backend_name() const override { return "Intel OpenVINO GenAI"; }
    std::string device_name() const override { return device_; }
    std::string full_device_name() const override { return full_device_name_; }
    std::string runtime_version() const override { return ov::get_openvino_version().buildNumber; }
    double load_seconds() const override { return load_seconds_; }
    ExecutionDiagnostics execution_diagnostics() const override {
        ExecutionDiagnostics diagnostics;
        diagnostics.requested_provider = "openvino-genai";
        diagnostics.resolved_provider = "openvino-genai";
        diagnostics.inference_precision = inference_precision_;
        return diagnostics;
    }

private:
    std::string device_;
    std::string full_device_name_;
    std::string inference_precision_;
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

#else  // !NPU_INFERENCE_BENCH_INTEL

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "Intel backend not compiled. Rebuild with EnableIntel=true and the OpenVINO "
        "GenAI SDK available (define NPU_INFERENCE_BENCH_INTEL, link openvino_genai).");
}

bool available() { return false; }

#endif

}  // namespace intel
}  // namespace npu_inference_bench
