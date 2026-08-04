// Product-shaped Sherpa-exported Whisper ONNX backend.
// Model dir contract matches the consumer MLM / framework/whisper layout:
//   encoder.onnx + decoder.onnx + tokens.txt
// Also accepts sherpa-onnx release names:
//   tiny.en-encoder.onnx / tiny.en-decoder.onnx / tiny.en-tokens.txt
//
// Decode stack is vendored from the consumer framework/whisper component via
// tools/export/vendor-product-whisper.ps1. EP selection uses the shared HAL
// ort_ep.hpp so product provider tokens and ORT EP names both work.
#include "backend_registry.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ort_ep.hpp"

#include <asw/framework/whisper.h>

#ifdef NPU_INFERENCE_BENCH_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace npu_inference_bench {
namespace ort_sherpa {

#ifdef NPU_INFERENCE_BENCH_ORT

namespace {

namespace fs = std::filesystem;
namespace ep = npu_inference_bench::ort_common;

struct ModelFiles {
    fs::path encoder;
    fs::path decoder;
    fs::path tokens;
};

ModelFiles resolve_model_files(const fs::path& dir) {
    const std::pair<const char*, const char*> encoder_names[] = {
        {"encoder.onnx", "decoder.onnx"},
        {"tiny.en-encoder.onnx", "tiny.en-decoder.onnx"},
        {"tiny-encoder.onnx", "tiny-decoder.onnx"},
    };
    const char* token_names[] = {"tokens.txt", "tiny.en-tokens.txt", "tiny-tokens.txt"};

    for (const auto& pair : encoder_names) {
        const fs::path enc = dir / pair.first;
        const fs::path dec = dir / pair.second;
        if (!fs::exists(enc) || !fs::exists(dec)) continue;
        for (const char* tok : token_names) {
            const fs::path tokens = dir / tok;
            if (fs::exists(tokens)) {
                return {enc, dec, tokens};
            }
        }
    }
    throw std::runtime_error(
        "Sherpa-export Whisper model_dir must contain encoder.onnx + decoder.onnx + "
        "tokens.txt (or tiny.en-* equivalents): " +
        dir.string());
}

class OrtSherpaEngine final : public IWhisperEngine {
public:
    explicit OrtSherpaEngine(const EngineOptions& options) : options_(options) {
        files_ = resolve_model_files(fs::path(options.model_dir));
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "npu-inference-bench-sherpa");

        EngineOptions runtime = options;
        runtime.device_override = ep::normalize_provider_token(runtime.device_override);
        for (auto& p : runtime.provider_order) {
            p = ep::normalize_provider_token(p);
        }

        const auto chain = ep::fallback_chain(runtime);
        if (chain.empty()) {
            throw std::runtime_error("no execution-provider candidates for sherpa Whisper");
        }

        std::string last_error;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < chain.size(); ++i) {
            const std::string& provider = chain[i];
            try {
                Ort::SessionOptions so;
                so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                if (options.cpu_threads > 0) {
                    so.SetIntraOpNumThreads(options.cpu_threads);
                    so.SetInterOpNumThreads(1);
                }
                ep::append_provider(*env_, so, runtime, provider, files_.encoder.parent_path(),
                                    "sherpa_whisper");

                asw::whisper::Options opts;
                opts.encoder_path = files_.encoder.wstring();
                opts.decoder_path = files_.decoder.wstring();
                opts.tokens_path = files_.tokens.wstring();
                opts.language = "en";
                opts.task = asw::whisper::Task::Transcribe;

                session_ = std::make_unique<asw::whisper::Session>(*env_, so, opts);
                active_provider_ = provider;
                diagnostics_.requested_device = options.device;
                diagnostics_.requested_provider =
                    runtime.device_override.empty() ? "auto" : runtime.device_override;
                diagnostics_.resolved_provider = provider;
                diagnostics_.resolved_device = ep::resolved_device_for_provider(provider);
                diagnostics_.fallback_occurred = (i > 0) ||
                    ep::is_fallback_provider_for_device(options.device, provider);
                diagnostics_.attempts.push_back({provider, true, ""});

                if (options.require_requested_device &&
                    diagnostics_.resolved_device != runtime::ResolvedDevice::Unknown &&
                    diagnostics_.resolved_device != runtime::requested_device_class(options.device)) {
                    throw std::runtime_error(
                        "provider '" + provider + "' does not target requested device");
                }
                break;
            } catch (const std::exception& e) {
                last_error = e.what();
                diagnostics_.attempts.push_back({provider, false, last_error});
                session_.reset();
                std::cerr << "[ort-sherpa] provider " << provider << " failed: " << last_error
                          << "\n";
            }
        }
        if (!session_) {
            throw std::runtime_error("Sherpa Whisper failed to load on any provider: " +
                                     last_error);
        }
        load_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    TranscribeResult transcribe(const AudioSamples& audio) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto result = session_->Run(audio, /*sample_rate=*/16000);
        const double infer =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        TranscribeResult out;
        out.text = result.text;
        out.infer_seconds = infer;
        out.generated_tokens = static_cast<long>(result.token_ids.size());
        out.has_token_metrics = !result.token_ids.empty();
        out.runtime = ep::runtime_for(active_provider_);
        out.model_format = "onnx";
        out.decode_strategy = "sherpa-export-kv";
        out.max_context = session_->GetMetadata().n_text_ctx;
        if (out.has_token_metrics && infer > 0.0) {
            out.throughput_tps = out.generated_tokens / infer;
            out.tpot_ms = (infer * 1000.0) / out.generated_tokens;
        }
        return out;
    }

    std::string backend_name() const override {
        return "ONNX Runtime (Sherpa-export Whisper)";
    }
    std::string device_name() const override {
        return std::string(to_string(options_.device));
    }
    std::string runtime_version() const override {
        return Ort::GetVersionString();
    }
    ExecutionDiagnostics execution_diagnostics() const override { return diagnostics_; }
    double load_seconds() const override { return load_seconds_; }

private:
    EngineOptions options_;
    ModelFiles files_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<asw::whisper::Session> session_;
    std::string active_provider_;
    ExecutionDiagnostics diagnostics_;
    double load_seconds_ = 0.0;
};

}  // namespace

std::unique_ptr<IWhisperEngine> create(const EngineOptions& options) {
    try {
        return std::make_unique<OrtSherpaEngine>(options);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime sherpa backend failed to init: ") +
                                 e.what());
    }
}

bool available() { return true; }

#else  // !NPU_INFERENCE_BENCH_ORT

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "ONNX Runtime sherpa backend not compiled. Build with -EnableOrt / NPU_INFERENCE_BENCH_ORT.");
}

bool available() { return false; }

#endif

}  // namespace ort_sherpa
}  // namespace npu_inference_bench
