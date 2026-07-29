// Generic static ONNX backend: runs whisper static-onnx packages through ONNX
// Runtime providers. Default (en-static-onnx / tiny.en): encoder [1,80,3000] ->
// [1,1500,384], decoder input_ids [1,128] + encoder_hidden_states [1,1500,384].
// Optional npu_hal_package.json overrides window (e.g. 7s multilingual tiny:
// [1,80,700] -> [1,350,384]).
#include "backend_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "npu_inference_bench/whisper_frontend.hpp"
#include "npu_inference_bench/runtime/runtime_context.hpp"
#include "ort_ep.hpp"
#include "ort_offload.hpp"
#include "ort_session_access.hpp"

#ifdef NPU_INFERENCE_BENCH_ORT
#include <onnxruntime_cxx_api.h>
#if defined(_WIN32) && __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define NPU_INFERENCE_BENCH_ORT_HAS_DML 1
#endif
#if defined(NPU_INFERENCE_BENCH_QUALCOMM) && defined(_WIN32)
#include <windows.h>
#endif
#endif

namespace npu_inference_bench {
namespace ort_static {

#ifdef NPU_INFERENCE_BENCH_ORT

namespace {

namespace fs = std::filesystem;
namespace fe = npu_inference_bench::frontend;
namespace ep = npu_inference_bench::ort_common;

constexpr int64_t kEncSeqDefault = 1500;
constexpr int64_t kDModelDefault = 384;
constexpr int64_t kStaticMaxTokens = 128;

struct PackageConfig {
    int n_samples = fe::kSamples;
    int n_frames = fe::kFrames;
    int64_t enc_seq = kEncSeqDefault;
    int64_t d_model = kDModelDefault;
    int64_t max_tokens = kStaticMaxTokens;
    bool multilingual = false;
};

PackageConfig load_package_config(const fs::path& dir) {
    PackageConfig cfg;
    const fs::path pkg = dir / "npu_hal_package.json";
    if (!fs::exists(pkg)) return cfg;
    const std::string text = fe::read_text(pkg);
    cfg.n_samples = static_cast<int>(fe::json_int(text, "n_samples", cfg.n_samples));
    cfg.n_frames = static_cast<int>(fe::json_int(text, "n_frames", cfg.n_frames));
    cfg.enc_seq = fe::json_int(text, "enc_seq", cfg.enc_seq);
    cfg.d_model = fe::json_int(text, "d_model", cfg.d_model);
    cfg.max_tokens = fe::json_int(text, "max_tokens", cfg.max_tokens);
    // multilingual: true/false — cheap string search
    cfg.multilingual = text.find("\"multilingual\": true") != std::string::npos ||
                       text.find("\"multilingual\":true") != std::string::npos;
    return cfg;
}

int64_t env_max_tokens(int64_t cap) {
    const char* v = std::getenv("NPU_INFERENCE_BENCH_ORT_MAX_TOKENS");
    if (!v || !*v) return cap;
    const long n = std::strtol(v, nullptr, 10);
    return (std::max<long>)(2, (std::min<long>)(n, static_cast<long>(cap)));
}

// Look up <|xx|> language token id from generation_config.json lang_to_id map.
int64_t language_token_id(const std::string& gc, const std::string& lang) {
    if (lang.empty()) return -1;
    const std::string key = "\"<|" + lang + "|>\"";
    const size_t pos = gc.find(key);
    if (pos == std::string::npos) return -1;
    size_t colon = gc.find(':', pos + key.size());
    if (colon == std::string::npos) return -1;
    char* end = nullptr;
    const long id = std::strtol(gc.c_str() + colon + 1, &end, 10);
    if (end == gc.c_str() + colon + 1) return -1;
    return id;
}

Ort::Value tensor_float(std::vector<float>& data, const std::vector<int64_t>& shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
}

Ort::Value tensor_int64(std::vector<int64_t>& data, const std::vector<int64_t>& shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(), shape.data(), shape.size());
}

class OrtStaticEngine final : public IWhisperEngine {
public:
    explicit OrtStaticEngine(const EngineOptions& options)
        : options_(options),
          context_(options) {
        const fs::path dir(options.model_dir);
        const fs::path enc_path = dir / "encoder_model.onnx";
        const fs::path dec_path = dir / "decoder_model.onnx";
        if (!fs::exists(enc_path) || !fs::exists(dec_path)) {
            throw std::runtime_error(
                "ONNX static model_dir must contain encoder_model.onnx and decoder_model.onnx: " +
                options.model_dir);
        }

        pkg_ = load_package_config(dir);
        max_tokens_ = env_max_tokens(pkg_.max_tokens);

        mel_filters_ = fe::parse_mel_filters(dir / "preprocessor_config.json");
        vocab_ = fe::load_vocab(dir / "vocab.json");
        const std::string gc = fe::read_text(dir / "generation_config.json");
        // Multilingual tiny uses decoder_start_token_id=50258; tiny.en uses 50257.
        sot_ = fe::json_int(gc, "decoder_start_token_id", pkg_.multilingual ? 50258 : 50257);
        eos_ = fe::json_int(gc, "eos_token_id", 50256);
        pad_ = fe::json_int(gc, "pad_token_id", eos_);
        no_timestamps_ = fe::json_int(gc, "no_timestamps_token_id", pkg_.multilingual ? 50363 : 50362);
        // <|transcribe|> / <|translate|> — multilingual only
        transcribe_ = fe::json_int(gc, "transcribe_token_id", 50359);
        translate_ = fe::json_int(gc, "translate_token_id", 50358);
        if (pkg_.multilingual) {
            // Prefer explicit lang; default en when unset (product Media Scan).
            const char* env_lang = std::getenv("NPU_INFERENCE_BENCH_WHISPER_LANG");
            const std::string lang = (env_lang && *env_lang) ? env_lang : "en";
            lang_token_ = language_token_id(gc, lang);
            if (lang_token_ < 0) lang_token_ = language_token_id(gc, "en");
        }
        suppress_ = fe::json_int_array(gc, "suppress_tokens");
        begin_suppress_ = fe::json_int_array(gc, "begin_suppress_tokens");

        const std::string cache_base = dir.filename().string();
        models_ = context_.load({
            {enc_path, dir, cache_base + "_encoder"},
            {dec_path, dir, cache_base + "_decoder"},
        });
        load_seconds_ = models_.load_seconds;
        diagnostics_ = models_.diagnostics;
        active_provider_ = diagnostics_.resolved_provider;
        encoder_ = &runtime::detail::OrtSessionAccess::get(models_.sessions[0]);
        decoder_ = &runtime::detail::OrtSessionAccess::get(models_.sessions[1]);
    }

    TranscribeResult transcribe(const AudioSamples& audio) override {
        const auto t0 = std::chrono::steady_clock::now();

        std::vector<float> features =
            fe::log_mel_spectrogram_fft(audio, mel_filters_, pkg_.n_samples, pkg_.n_frames);
        auto feature_tensor = tensor_float(features, {1, fe::kMelBins, pkg_.n_frames});
        const char* enc_in[] = {"input_features"};
        const char* enc_out[] = {"last_hidden_state"};
        auto encoder_outputs =
            encoder_->Run(Ort::RunOptions{nullptr}, enc_in, &feature_tensor, 1, enc_out, 1);
        float* encoder_ptr = encoder_outputs[0].GetTensorMutableData<float>();
        const size_t encoder_count = encoder_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> encoder_state(encoder_ptr, encoder_ptr + encoder_count);

        std::vector<int64_t> ids(static_cast<size_t>(max_tokens_), pad_);
        int64_t cur = 0;
        ids[static_cast<size_t>(cur++)] = sot_;
        if (pkg_.multilingual) {
            if (lang_token_ >= 0) ids[static_cast<size_t>(cur++)] = lang_token_;
            ids[static_cast<size_t>(cur++)] = transcribe_;
        }
        ids[static_cast<size_t>(cur++)] = no_timestamps_;
        std::vector<int64_t> generated;
        double logprob_sum = 0.0;
        long n_gen = 0;
        bool first = true;

        const char* dec_in[] = {"input_ids", "encoder_hidden_states"};
        const char* dec_out[] = {"logits"};
        Ort::RunOptions run_opts;
        if (options_.device == Device::NPU && ep::is_qnn(active_provider_)) {
            run_opts.AddConfigEntry("qnn.perf_mode", "burst");
        }
        while (cur < max_tokens_) {
            auto token_tensor = tensor_int64(ids, {1, max_tokens_});
            auto state_tensor = tensor_float(encoder_state, {1, pkg_.enc_seq, pkg_.d_model});
            std::vector<Ort::Value> inputs;
            inputs.emplace_back(std::move(token_tensor));
            inputs.emplace_back(std::move(state_tensor));

            auto outputs =
                decoder_->Run(run_opts, dec_in, inputs.data(), inputs.size(), dec_out, 1);
            float* logits = outputs[0].GetTensorMutableData<float>();
            const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            const int64_t vocab = shape.back();
            float* row = logits + (cur - 1) * vocab;

            std::vector<float> logit(row, row + vocab);
            for (const int64_t s : suppress_)
                if (s >= 0 && s < vocab) logit[static_cast<size_t>(s)] = -std::numeric_limits<float>::infinity();
            if (first) {
                for (const int64_t s : begin_suppress_)
                    if (s >= 0 && s < vocab) logit[static_cast<size_t>(s)] = -std::numeric_limits<float>::infinity();
                first = false;
            }

            float m = -std::numeric_limits<float>::infinity();
            int64_t tok = 0;
            for (int64_t v = 0; v < vocab; ++v) {
                if (logit[static_cast<size_t>(v)] > m) {
                    m = logit[static_cast<size_t>(v)];
                    tok = v;
                }
            }
            double sum_exp = 0.0;
            for (int64_t v = 0; v < vocab; ++v)
                sum_exp += std::exp(static_cast<double>(logit[static_cast<size_t>(v)]) - m);
            const double lse = m + std::log(sum_exp);
            logprob_sum += static_cast<double>(logit[static_cast<size_t>(tok)]) - lse;
            ++n_gen;

            if (tok == eos_) break;
            generated.push_back(tok);
            ids[static_cast<size_t>(cur)] = tok;
            ++cur;
        }

        TranscribeResult out;
        out.text = fe::decode_tokens(generated, vocab_, eos_);
        out.infer_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        out.generated_tokens = n_gen;
        out.sequence_logprob = logprob_sum;
        if (n_gen > 0) {
            out.avg_logprob = logprob_sum / static_cast<double>(n_gen);
            out.throughput_tps = static_cast<double>(n_gen) / out.infer_seconds;
            out.tpot_ms = out.infer_seconds * 1000.0 / static_cast<double>(n_gen);
        }
        out.ttft_ms = -1.0;
        out.has_token_metrics = true;
        out.runtime = ep::runtime_for(active_provider_);
        out.model_format = "onnx";
        out.decode_strategy = pkg_.multilingual ? "static-no-kv-multi-7s" : "static-no-kv";
        out.max_context = static_cast<long>(max_tokens_);

        // Finalize profiling after the first transcription (normally the harness
        // warmup), so measured runs do not accumulate profiler overhead.
        (void)execution_diagnostics();
        return out;
    }

    std::string backend_name() const override { return "ONNX Runtime (unified, static)"; }
    std::string device_name() const override { return active_provider_; }
    std::string runtime_version() const override { return Ort::GetVersionString(); }
    ExecutionDiagnostics execution_diagnostics() const override {
        if (!diagnostics_finalized_) {
            diagnostics_finalized_ = true;
            try {
                models_.finalize_profiling();
                diagnostics_ = models_.diagnostics;
                active_provider_ = diagnostics_.resolved_provider;
            } catch (const std::exception&) {
                // Profiling is diagnostic unless strict device enforcement is enabled.
                if (options_.require_requested_device) throw;
            }
        }
        return diagnostics_;
    }
    double load_seconds() const override { return load_seconds_; }

private:
    EngineOptions options_;
    runtime::RuntimeContext context_;
    PackageConfig pkg_{};
    mutable runtime::LoadedModelSet models_;
    mutable std::string active_provider_;  // EP that actually built/executed the sessions
    mutable ExecutionDiagnostics diagnostics_;
    mutable bool diagnostics_finalized_ = false;
    int64_t max_tokens_ = kStaticMaxTokens;
    int64_t sot_ = 50257;
    int64_t eos_ = 50256;
    int64_t pad_ = 50256;
    int64_t no_timestamps_ = 50362;
    int64_t transcribe_ = 50359;
    int64_t translate_ = 50358;
    int64_t lang_token_ = -1;
    std::vector<int64_t> suppress_;
    std::vector<int64_t> begin_suppress_;
    std::vector<float> mel_filters_;
    std::unordered_map<int64_t, std::string> vocab_;
    double load_seconds_ = 0.0;
    Ort::Session* encoder_ = nullptr;
    Ort::Session* decoder_ = nullptr;
};

}  // namespace

std::unique_ptr<IWhisperEngine> create(const EngineOptions& options) {
    try {
        return std::make_unique<OrtStaticEngine>(options);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime static backend failed to init: ") + e.what());
    }
}

bool available() { return true; }

// Return the best logical device the unified binary should start on. The engine's
// NPU->GPU->CPU fallback chain validates the pick at session-build time, so a wrong
// guess degrades gracefully rather than failing.
Device best_available_device() {
#ifdef NPU_INFERENCE_BENCH_WINML
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "npu_inference_bench_winml_probe");
        ep::register_windows_ml_catalog(env, Device::NPU);
        bool gpu = false;
        for (Ort::ConstEpDevice device : env.GetEpDevices()) {
            if (device.Device().Type() == OrtHardwareDeviceType_NPU) return Device::NPU;
            if (device.Device().Type() == OrtHardwareDeviceType_GPU &&
                ep::lower(device.EpName()).find("dml") != std::string::npos) {
                gpu = true;
            }
        }
        if (gpu) return Device::GPU;
    } catch (...) {
        // Fall through to CPU; explicit device selection will retain the error.
    }
    return Device::CPU;
#elif defined(NPU_INFERENCE_BENCH_QUALCOMM)
    // Qualcomm / ARM64: prefer a real QNN NPU, then the DirectML-backed Adreno GPU.
    // Session construction validates either pick and retains the normal CPU fallback.
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "npu_inference_bench_probe");
        ep::register_qnn_library(env);
        for (Ort::ConstEpDevice d : env.GetEpDevices()) {
            if (std::strcmp(d.EpName(), ep::kQnnEpName) == 0 &&
                d.Device().Type() == OrtHardwareDeviceType_NPU) {
                return Device::NPU;
            }
        }
    } catch (...) {
        // A failed QNN probe must not hide a usable DirectML GPU.
    }
#ifdef NPU_INFERENCE_BENCH_ORT_HAS_DML
    try {
        for (const std::string& provider : Ort::GetAvailableProviders()) {
            const std::string p = ep::lower(provider);
            if (p.find("dml") != std::string::npos ||
                p.find("directml") != std::string::npos) {
                return Device::GPU;
            }
        }
    } catch (...) {
        // Fall through to CPU; the explicit GPU path can still report the error.
    }
#endif
    return Device::CPU;
#else
    // x64 (and any non-Qualcomm build). Self-select using ORT's EP-device list as a
    // real hardware probe where it can see the silicon, with two important limits
    // established empirically on this hardware:
    //   * The VitisAI EP (AMD XDNA NPU) is registered via the legacy
    //     AppendExecutionProvider API and NEVER appears in GetEpDevices(). So a
    //     compiled-in VitisAI EP is the only signal we get; trust it and let the
    //     engine's NPU->GPU->CPU session fallback demote if the NPU is truly absent.
    //     (Detecting AMD-NPU *absence* up front would need an OS-level PnP probe.)
    //   * The OpenVINO EP (Intel) and DirectML (GPU) DO enumerate their devices, so
    //     for those we honor the probe: pick a tier only if the hardware is present.
    // Set NPU_INFERENCE_BENCH_LOG_PROBE=1 to dump the enumerated devices to stderr.
    try {
        const std::vector<std::string> providers = Ort::GetAvailableProviders();
        auto has = [&](const char* needle) {
            for (const std::string& p : providers) {
                if (ep::lower(p).find(needle) != std::string::npos) return true;
            }
            return false;
        };
        const bool has_vitisai = has("vitisai");
        const bool has_openvino = has("openvino");
        const bool can_gpu = has_openvino || has("dml") || has("directml");
        const bool log_probe = !ep::env_or("NPU_INFERENCE_BENCH_LOG_PROBE", "").empty();

        // Hardware enumeration (authoritative for CPU, DirectML GPUs, and OpenVINO
        // NPU/GPU; blind to VitisAI). The EP-device API landed in ORT 1.22
        // (ORT_API_VERSION 22); on older headers we skip the probe and let the
        // compiled-in-EP heuristic below drive the pick.
        bool probed = false, hw_npu = false, hw_gpu = false;
#if defined(ORT_API_VERSION) && ORT_API_VERSION >= 22
        try {
            Ort::Env probe_env(ORT_LOGGING_LEVEL_WARNING, "npu_inference_bench_probe");
            auto devices = probe_env.GetEpDevices();
            probed = !devices.empty();
            for (const auto& d : devices) {
                const OrtHardwareDeviceType t = d.Device().Type();
                if (t == OrtHardwareDeviceType_NPU) hw_npu = true;
                else if (t == OrtHardwareDeviceType_GPU) hw_gpu = true;
                if (log_probe) {
                    const char* tn = t == OrtHardwareDeviceType_NPU   ? "NPU"
                                     : t == OrtHardwareDeviceType_GPU ? "GPU"
                                                                      : "CPU";
                    std::cerr << "[npu-inference-bench probe] EP=" << d.EpName() << " hw=" << tn << "\n";
                }
            }
        } catch (const std::exception& e) {
            if (log_probe) std::cerr << "[npu-inference-bench probe] GetEpDevices unavailable: " << e.what() << "\n";
        }
#else
        if (log_probe) std::cerr << "[npu-inference-bench probe] EP-device API not in this ORT; using heuristic\n";
#endif

        // AMD Ryzen AI: VitisAI is invisible to the probe, so trust the compiled-in
        // EP. The session-build fallback validates and demotes if the NPU is absent.
        if (has_vitisai) return Device::NPU;
        // Intel: the OpenVINO EP enumerates its NPU, so require the probe to see it.
        if (probed && hw_npu && has_openvino) return Device::NPU;
        // GPU: honor the probe when it worked; assume present only if we couldn't probe.
        if (can_gpu && (hw_gpu || !probed)) return Device::GPU;
        // Probe worked and found nothing we can target -> CPU is the honest answer.
        if (probed) return Device::CPU;
        // No usable probe (older ORT): last-resort compiled-in-EP heuristic.
        if (has_openvino) return Device::NPU;
        if (can_gpu) return Device::GPU;
    } catch (...) {
        // Fall through to CPU on any probe failure.
    }
    return Device::CPU;
#endif
}

#else  // !NPU_INFERENCE_BENCH_ORT

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "ONNX Runtime static backend not compiled. Build with NPU_INFERENCE_BENCH_ORT and link ONNX Runtime.");
}

bool available() { return false; }

Device best_available_device() { return Device::CPU; }

#endif

}  // namespace ort_static
}  // namespace npu_inference_bench
