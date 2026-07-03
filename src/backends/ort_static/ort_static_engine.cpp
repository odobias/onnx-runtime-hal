// Generic static ONNX backend: runs the unchanged whisper-tiny-en-static-onnx
// package through ONNX Runtime providers. The model contract is fixed-shape,
// no-KV recompute: encoder [1,80,3000] -> [1,1500,384], decoder consumes
// input_ids [1,128] + encoder_hidden_states [1,1500,384] and returns logits.
#include "backends/backend_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "whisper_npu/whisper_frontend.hpp"

#ifdef WHISPER_HAL_ORT
#include <onnxruntime_cxx_api.h>
#if defined(_WIN32) && __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define WHISPER_HAL_ORT_HAS_DML 1
#endif
#if defined(WHISPER_HAL_QUALCOMM) && defined(_WIN32)
#include <windows.h>
#endif
#endif

namespace whisper_npu {
namespace ort_static {

#ifdef WHISPER_HAL_ORT

namespace {

namespace fs = std::filesystem;
namespace fe = whisper_npu::frontend;

constexpr int64_t kEncSeq = 1500;
constexpr int64_t kDModel = 384;
constexpr int64_t kStaticMaxTokens = 128;
constexpr const char* kQnnEpName = "QNNExecutionProvider";

std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

int64_t env_max_tokens() {
    const char* v = std::getenv("WHISPER_HAL_ORT_MAX_TOKENS");
    if (!v || !*v) return kStaticMaxTokens;
    const long n = std::strtol(v, nullptr, 10);
    return (std::max<long>)(2, (std::min<long>)(n, kStaticMaxTokens));
}

Ort::Value tensor_float(std::vector<float>& data, const std::vector<int64_t>& shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
}

Ort::Value tensor_int64(std::vector<int64_t>& data, const std::vector<int64_t>& shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(), shape.data(), shape.size());
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Logical device -> preferred ORT execution provider. NPU is vendor-aware:
// when the QNN plugin EP is compiled in (Snapdragon), NPU maps to QNN; otherwise
// it falls back to the VitisAI bridge EP (AMD).
std::string provider_for(Device device) {
    switch (device) {
        case Device::CPU: return "CPUExecutionProvider";
        case Device::GPU: return "DmlExecutionProvider";
        case Device::NPU:
#ifdef WHISPER_HAL_QUALCOMM
            return kQnnEpName;
#else
            return "VitisAIExecutionProvider";
#endif
    }
    return "CPUExecutionProvider";
}

// Human-facing runtime tag for the shared benchmark schema, derived from the EP
// that actually built the session.
std::string runtime_for(const std::string& provider) {
    const std::string p = lower(provider);
    if (p.find("qnn") != std::string::npos) return "onnxruntime-qnn";
    if (p.find("vitis") != std::string::npos) return "onnxruntime-vitisai";
    if (p.find("dml") != std::string::npos || p.find("directml") != std::string::npos)
        return "onnxruntime-directml";
    return "onnxruntime";
}

bool is_qnn(const std::string& provider) {
    return lower(provider).find("qnn") != std::string::npos;
}

std::string cache_safe(std::string s) {
    if (s.empty()) s = "static_onnx";
    for (char& c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) c = '_';
    }
    return s;
}

// The runtime EP fallback chain for a requested device. An explicit provider
// override is honored verbatim (single attempt, no fallback). Otherwise a failed
// NPU walks down to GPU then CPU, and GPU walks down to CPU, so a single binary
// always produces a result on whatever hardware/drivers are actually present.
std::vector<std::string> fallback_chain(const EngineOptions& options) {
    if (!options.device_override.empty()) return {options.device_override};
    switch (options.device) {
        case Device::NPU:
            return {provider_for(Device::NPU), provider_for(Device::GPU), provider_for(Device::CPU)};
        case Device::GPU:
            return {provider_for(Device::GPU), provider_for(Device::CPU)};
        case Device::CPU:
        default:
            return {provider_for(Device::CPU)};
    }
}

#ifdef WHISPER_HAL_QUALCOMM
std::string qnn_ep_library_path() {
    return env_or("WHISPER_QNN_EP_DLL", "onnxruntime_providers_qnn.dll");
}

std::string qnn_backend_path(Device device) {
    switch (device) {
        case Device::GPU: return env_or("WHISPER_QNN_GPU_DLL", "QnnGpu.dll");
        case Device::CPU: return env_or("WHISPER_QNN_CPU_DLL", "QnnCpu.dll");
        case Device::NPU:
        default:          return env_or("WHISPER_QNN_HTP_DLL", "QnnHtp.dll");
    }
}

#ifdef _WIN32
std::wstring ort_tstring(const std::string& value) {
    if (value.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (size <= 0) throw std::runtime_error("Failed to convert path to UTF-16: " + value);
    std::wstring out(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, out.data(), size);
    return out;
}
#endif

void register_qnn_library(Ort::Env& env) {
    // EP-library registration lives on the ORT environment. We may call this more
    // than once (probe env + engine env, encoder + decoder), so swallow the
    // "already registered" error rather than gating on a process-wide flag that
    // would leave a fresh env unregistered.
    try {
#ifdef _WIN32
        env.RegisterExecutionProviderLibrary(kQnnEpName, ort_tstring(qnn_ep_library_path()));
#else
        env.RegisterExecutionProviderLibrary(kQnnEpName, qnn_ep_library_path());
#endif
    } catch (const Ort::Exception&) {
        // Already registered on this (or a shared) environment -- fine.
    }
}

Ort::ConstEpDevice find_qnn_device(Ort::Env& env) {
    for (Ort::ConstEpDevice ep_device : env.GetEpDevices()) {
        if (std::strcmp(ep_device.EpName(), kQnnEpName) == 0) return ep_device;
    }
    throw std::runtime_error("QNNExecutionProvider device not found after registration");
}
#endif  // WHISPER_HAL_QUALCOMM

void append_provider(Ort::Env& env, Ort::SessionOptions& so, const EngineOptions& options,
                     const std::string& provider, const fs::path& model_dir,
                     const std::string& cache_key) {
    const std::string p = lower(provider);

    if (p == "cpu" || p == "cpuexecutionprovider") {
        if (options.cpu_threads > 0) {
            so.SetIntraOpNumThreads(options.cpu_threads);
            so.SetInterOpNumThreads(1);
        }
        return;  // CPU EP is the default ORT fallback.
    }

    if (p == "qnn" || p == "qnnexecutionprovider") {
#ifdef WHISPER_HAL_QUALCOMM
        register_qnn_library(env);
        const Ort::ConstEpDevice qnn_device = find_qnn_device(env);  // throws if absent
        std::vector<Ort::ConstEpDevice> selected{qnn_device};
        std::unordered_map<std::string, std::string> opts{
            {"backend_path", qnn_backend_path(options.device)}};
        if (options.device == Device::NPU) opts.emplace("htp_performance_mode", "burst");
        Ort::KeyValuePairs ep_options(opts);
        so.AppendExecutionProvider_V2(env, selected, ep_options);
        return;
#else
        throw std::runtime_error("QNN execution provider not compiled into this build");
#endif
    }

    if (p == "vitisai" || p == "vitisaiexecutionprovider") {
        std::unordered_map<std::string, std::string> vitis_opts;
        const fs::path config = model_dir / "vitisai_config.json";
        if (fs::exists(config)) vitis_opts["config_file"] = config.string();
        if (!options.cache_dir.empty()) {
            vitis_opts["cache_dir"] = options.cache_dir;
            vitis_opts["cache_key"] = cache_key;
        }
        so.AppendExecutionProvider_VitisAI(vitis_opts);
        return;
    }

    if (p == "dml" || p == "directml" || p == "dmlexecutionprovider") {
#ifdef WHISPER_HAL_ORT_HAS_DML
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
        return;
#else
        throw std::runtime_error(
            "generic ONNX Runtime static backend was built without DirectML provider headers; use CPU/NPU");
#endif
    }

    throw std::runtime_error(
        "unsupported ONNX Runtime provider override for static backend: " + provider +
        " (supported: CPUExecutionProvider, QNNExecutionProvider, DmlExecutionProvider, VitisAIExecutionProvider)");
}

Ort::SessionOptions make_session_options(Ort::Env& env, const EngineOptions& options,
                                         const std::string& provider, const fs::path& model_dir,
                                         const std::string& cache_key) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    append_provider(env, so, options, provider, model_dir, cache_key);
    return so;
}

class OrtStaticEngine final : public IWhisperEngine {
public:
    explicit OrtStaticEngine(const EngineOptions& options)
        : env_(ORT_LOGGING_LEVEL_WARNING, "whisper_hal_ort_static"),
          options_(options),
          max_tokens_(env_max_tokens()) {
        const fs::path dir(options.model_dir);
        const fs::path enc_path = dir / "encoder_model.onnx";
        const fs::path dec_path = dir / "decoder_model.onnx";
        if (!fs::exists(enc_path) || !fs::exists(dec_path)) {
            throw std::runtime_error(
                "ONNX static model_dir must contain encoder_model.onnx and decoder_model.onnx: " +
                options.model_dir);
        }

        mel_filters_ = fe::parse_mel_filters(dir / "preprocessor_config.json");
        vocab_ = fe::load_vocab(dir / "vocab.json");
        const std::string gc = fe::read_text(dir / "generation_config.json");
        sot_ = fe::json_int(gc, "decoder_start_token_id", 50257);
        eos_ = fe::json_int(gc, "eos_token_id", 50256);
        pad_ = fe::json_int(gc, "pad_token_id", eos_);
        no_timestamps_ = fe::json_int(gc, "no_timestamps_token_id", 50362);
        suppress_ = fe::json_int_array(gc, "suppress_tokens");
        begin_suppress_ = fe::json_int_array(gc, "begin_suppress_tokens");

        const std::string cache_base = cache_safe(dir.filename().string());
        const std::vector<std::string> chain = fallback_chain(options_);

        // Try each EP in the fallback chain until both sessions build. This is
        // what lets one binary self-select CPU/GPU/NPU at runtime on whatever
        // hardware and drivers are actually present.
        std::string last_err;
        for (size_t i = 0; i < chain.size(); ++i) {
            const std::string& provider = chain[i];
            try {
                const auto t0 = std::chrono::steady_clock::now();
                auto enc_so = make_session_options(env_, options_, provider, dir, cache_base + "_encoder");
                encoder_ = std::make_unique<Ort::Session>(env_, enc_path.c_str(), enc_so);
                auto dec_so = make_session_options(env_, options_, provider, dir, cache_base + "_decoder");
                decoder_ = std::make_unique<Ort::Session>(env_, dec_path.c_str(), dec_so);
                load_seconds_ =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                active_provider_ = provider;
                break;
            } catch (const std::exception& e) {
                last_err = e.what();
                encoder_.reset();
                decoder_.reset();
                if (i + 1 < chain.size()) {
                    std::cerr << "[whisper-hal] EP '" << provider << "' unavailable (" << e.what()
                              << "); falling back to '" << chain[i + 1] << "'\n";
                }
            }
        }

        if (!encoder_ || !decoder_) {
            throw std::runtime_error("no ONNX Runtime execution provider could build a session (last error: " +
                                     last_err + ")");
        }
    }

    TranscribeResult transcribe(const AudioSamples& audio) override {
        const auto t0 = std::chrono::steady_clock::now();

        std::vector<float> features = fe::log_mel_spectrogram_fft(audio, mel_filters_);
        auto feature_tensor = tensor_float(features, {1, fe::kMelBins, fe::kFrames});
        const char* enc_in[] = {"input_features"};
        const char* enc_out[] = {"last_hidden_state"};
        auto encoder_outputs =
            encoder_->Run(Ort::RunOptions{nullptr}, enc_in, &feature_tensor, 1, enc_out, 1);
        float* encoder_ptr = encoder_outputs[0].GetTensorMutableData<float>();
        const size_t encoder_count = encoder_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> encoder_state(encoder_ptr, encoder_ptr + encoder_count);

        std::vector<int64_t> ids(static_cast<size_t>(kStaticMaxTokens), pad_);
        ids[0] = sot_;
        ids[1] = no_timestamps_;
        int64_t cur = 2;
        std::vector<int64_t> generated;
        double logprob_sum = 0.0;
        long n_gen = 0;
        bool first = true;

        const char* dec_in[] = {"input_ids", "encoder_hidden_states"};
        const char* dec_out[] = {"logits"};
        Ort::RunOptions run_opts;
        if (options_.device == Device::NPU && is_qnn(active_provider_)) {
            run_opts.AddConfigEntry("qnn.perf_mode", "burst");
        }
        while (cur < max_tokens_) {
            auto token_tensor = tensor_int64(ids, {1, kStaticMaxTokens});
            auto state_tensor = tensor_float(encoder_state, {1, kEncSeq, kDModel});
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
        out.runtime = runtime_for(active_provider_);
        out.model_format = "onnx";
        out.decode_strategy = "static-no-kv";
        out.max_context = static_cast<long>(kStaticMaxTokens);
        return out;
    }

    std::string backend_name() const override { return "ONNX Runtime (unified, static)"; }
    std::string device_name() const override { return active_provider_; }
    double load_seconds() const override { return load_seconds_; }

private:
    Ort::Env env_;
    EngineOptions options_;
    std::string active_provider_;  // EP that actually built the sessions (post-fallback)
    int64_t max_tokens_ = kStaticMaxTokens;
    int64_t sot_ = 50257;
    int64_t eos_ = 50256;
    int64_t pad_ = 50256;
    int64_t no_timestamps_ = 50362;
    std::vector<int64_t> suppress_;
    std::vector<int64_t> begin_suppress_;
    std::vector<float> mel_filters_;
    std::unordered_map<int64_t, std::string> vocab_;
    double load_seconds_ = 0.0;
    std::unique_ptr<Ort::Session> encoder_;
    std::unique_ptr<Ort::Session> decoder_;
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

// Probe the real ORT execution-provider device list and return the best logical
// device this build can run on right now: NPU if a QNN accelerator is present,
// otherwise CPU. (DirectML/GPU is dormant on ARM64 -- see the fallback chain.)
Device best_available_device() {
#ifdef WHISPER_HAL_QUALCOMM
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "whisper_hal_probe");
        register_qnn_library(env);
        for (Ort::ConstEpDevice d : env.GetEpDevices()) {
            if (std::strcmp(d.EpName(), kQnnEpName) == 0 &&
                d.Device().Type() == OrtHardwareDeviceType_NPU) {
                return Device::NPU;
            }
        }
    } catch (...) {
        // Fall through to CPU if the plugin can't be probed.
    }
#endif
    return Device::CPU;
}

#else  // !WHISPER_HAL_ORT

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "ONNX Runtime static backend not compiled. Build with WHISPER_HAL_ORT and link ONNX Runtime.");
}

bool available() { return false; }

Device best_available_device() { return Device::CPU; }

#endif

}  // namespace ort_static
}  // namespace whisper_npu
