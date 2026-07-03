// Qualcomm backend: whisper-tiny-en-static-onnx on Snapdragon Hexagon NPU via the
// ONNX Runtime Plugin QNN EP (onnxruntime-qnn >= 2.x). Uses the same static no-KV
// decode loop as ort_static / intel_onnx.
#include "backends/backend_registry.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "whisper_npu/whisper_frontend.hpp"

#ifdef WHISPER_HAL_QUALCOMM
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <windows.h>
#endif
#endif

namespace whisper_npu {
namespace qualcomm {

#ifdef WHISPER_HAL_QUALCOMM

namespace {

namespace fs = std::filesystem;
namespace fe = whisper_npu::frontend;

constexpr const char* kQnnEpName = "QNNExecutionProvider";
constexpr int64_t kEncSeq = 1500;
constexpr int64_t kDModel = 384;
constexpr int64_t kStaticMaxTokens = 128;

std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

std::string qnn_ep_library_path() {
    return env_or("WHISPER_QNN_EP_DLL", "onnxruntime_providers_qnn.dll");
}

std::string qnn_backend_path(Device device) {
    switch (device) {
        case Device::GPU:
            return env_or("WHISPER_QNN_GPU_DLL", "QnnGpu.dll");
        case Device::CPU:
            return env_or("WHISPER_QNN_CPU_DLL", "QnnCpu.dll");
        case Device::NPU:
        default:
            return env_or("WHISPER_QNN_HTP_DLL", "QnnHtp.dll");
    }
}

Ort::Value tensor_float(std::vector<float>& data, const std::vector<int64_t>& shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
}

Ort::Value tensor_int64(std::vector<int64_t>& data, const std::vector<int64_t>& shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(), shape.data(), shape.size());
}

Ort::ConstEpDevice find_qnn_device(Ort::Env& env) {
    for (Ort::ConstEpDevice ep_device : env.GetEpDevices()) {
        if (std::strcmp(ep_device.EpName(), kQnnEpName) == 0) return ep_device;
    }
    throw std::runtime_error("QNNExecutionProvider device not found after registration");
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
    static bool registered = false;
    if (registered) return;
#ifdef _WIN32
    env.RegisterExecutionProviderLibrary(kQnnEpName, ort_tstring(qnn_ep_library_path()));
#else
    env.RegisterExecutionProviderLibrary(kQnnEpName, qnn_ep_library_path());
#endif
    registered = true;
}

Ort::SessionOptions make_session_options(Ort::Env& env, Device device) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (device == Device::CPU) {
        return so;
    }

    const Ort::ConstEpDevice qnn_device = find_qnn_device(env);
    std::vector<Ort::ConstEpDevice> selected{qnn_device};
    std::unordered_map<std::string, std::string> opts{{"backend_path", qnn_backend_path(device)}};
    if (device == Device::NPU) {
        opts.emplace("htp_performance_mode", "burst");
    }
    Ort::KeyValuePairs ep_options(opts);
    so.AppendExecutionProvider_V2(env, selected, ep_options);
    return so;
}

class QualcommQnnEngine final : public IWhisperEngine {
public:
    explicit QualcommQnnEngine(const EngineOptions& options)
        : env_(ORT_LOGGING_LEVEL_WARNING, "whisper_hal_qnn"),
          device_(options.device),
          provider_label_(device_ == Device::CPU ? "CPUExecutionProvider" : kQnnEpName) {
        const fs::path dir(options.model_dir);
        const fs::path enc_path = dir / "encoder_model.onnx";
        const fs::path dec_path = dir / "decoder_model.onnx";
        if (!fs::exists(enc_path) || !fs::exists(dec_path)) {
            throw std::runtime_error(
                "Qualcomm QNN model_dir must contain encoder_model.onnx and decoder_model.onnx: " +
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

        if (device_ != Device::CPU) {
            register_qnn_library(env_);
        }

        Ort::SessionOptions so = make_session_options(env_, device_);
        if (options.cpu_threads > 0 && device_ == Device::CPU) {
            so.SetIntraOpNumThreads(options.cpu_threads);
            so.SetInterOpNumThreads(1);
        }

        const auto t0 = std::chrono::steady_clock::now();
        encoder_ = std::make_unique<Ort::Session>(env_, enc_path.c_str(), so);
        decoder_ = std::make_unique<Ort::Session>(env_, dec_path.c_str(), so);
        load_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    ~QualcommQnnEngine() override {
        encoder_.reset();
        decoder_.reset();
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
        if (device_ == Device::NPU) {
            run_opts.AddConfigEntry("qnn.perf_mode", "burst");
        }

        while (cur < kStaticMaxTokens) {
            auto token_tensor = tensor_int64(ids, {1, kStaticMaxTokens});
            auto state_tensor = tensor_float(encoder_state, {1, kEncSeq, kDModel});
            std::vector<Ort::Value> inputs;
            inputs.emplace_back(std::move(token_tensor));
            inputs.emplace_back(std::move(state_tensor));

            auto outputs = decoder_->Run(run_opts, dec_in, inputs.data(), inputs.size(), dec_out, 1);
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
        out.runtime = "onnxruntime-qnn";
        out.model_format = "onnx";
        out.decode_strategy = "static-no-kv";
        out.max_context = static_cast<long>(kStaticMaxTokens);
        return out;
    }

    std::string backend_name() const override { return "Qualcomm QNN (Plugin EP, static)"; }
    std::string device_name() const override { return provider_label_; }
    double load_seconds() const override { return load_seconds_; }

private:
    Ort::Env env_;
    Device device_;
    std::string provider_label_;
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
        return std::make_unique<QualcommQnnEngine>(options);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Qualcomm QNN backend failed to init: ") + e.what());
    }
}

bool available() { return true; }

#else  // !WHISPER_HAL_QUALCOMM

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "Qualcomm QNN backend not compiled. Build with EnableQualcomm=true and point OrtDir "
        "at an ONNX Runtime 1.24+ SDK (see msbuild/backend.qualcomm.props).");
}

bool available() { return false; }

#endif

}  // namespace qualcomm
}  // namespace whisper_npu
