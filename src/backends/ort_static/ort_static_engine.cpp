// Generic static ONNX backend: runs the unchanged whisper/en-static-onnx
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
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "whisper_npu/whisper_frontend.hpp"
#include "backends/ort_common/ort_ep.hpp"

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
namespace ep = whisper_npu::ort_common;

constexpr int64_t kEncSeq = 1500;
constexpr int64_t kDModel = 384;
constexpr int64_t kStaticMaxTokens = 128;

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

std::string cache_safe(std::string s) {
    if (s.empty()) s = "static_onnx";
    for (char& c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) c = '_';
    }
    return s;
}

// Build one ORT session for `provider`, adding QNN context-binary caching when a
// cache_dir is set. The HTP compile is QNN's ~15s cold cost; ORT's EPContext
// mechanism dumps that compiled graph to "<key>_qnn_ctx.onnx" and reloads it on the
// next process, turning the cold compile into a ~1s blob load. OpenVINO/VitisAI keep
// their own cache_dir handling in append_provider, so this fast path is scoped to QNN
// and stays inert on x64 builds (the QNN token never enters the chain there).
std::unique_ptr<Ort::Session> build_session(Ort::Env& env, const EngineOptions& options,
                                            const std::string& provider,
                                            const fs::path& model_path, const fs::path& model_dir,
                                            const std::string& cache_key) {
    fs::path ctx_path;
    if (ep::is_qnn(provider) && !options.cache_dir.empty()) {
        ctx_path = fs::path(options.cache_dir) / (cache_key + "_qnn_ctx.onnx");
    }

    auto make_so = [&](bool generate_ctx) {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ep::append_provider(env, so, options, provider, model_dir, cache_key);
        ep::enable_offload_profiling(so, cache_key);  // audited + stopped after warmup
        if (generate_ctx) {
            const std::string p = ctx_path.string();
            so.AddConfigEntry("ep.context_enable", "1");
            so.AddConfigEntry("ep.context_file_path", p.c_str());
            so.AddConfigEntry("ep.context_embed_mode", "1");  // embed the HTP binary in the .onnx
        }
        return so;
    };

    // Fast path: a compiled QNN context already exists -> load it and skip the HTP compile.
    if (!ctx_path.empty() && fs::exists(ctx_path)) {
        try {
            auto so = make_so(false);
            return std::make_unique<Ort::Session>(env, ctx_path.c_str(), so);
        } catch (const std::exception& e) {
            // The context is EP/arch/QNN-version specific; a stale blob can't load. Recompile.
            std::cerr << "[whisper-hal] QNN context cache '" << ctx_path.string()
                      << "' unusable (" << e.what() << "); recompiling\n";
            std::error_code ec;
            fs::remove(ctx_path, ec);
        }
    }

    // Compile path. For QNN with a cache_dir this session-create also dumps the context binary.
    if (!ctx_path.empty()) {
        std::error_code ec;
        fs::create_directories(ctx_path.parent_path(), ec);
    }
    auto so = make_so(!ctx_path.empty());
    return std::make_unique<Ort::Session>(env, model_path.c_str(), so);
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
        const std::vector<std::string> chain = ep::fallback_chain(options_);

        // Try each EP in the fallback chain until both sessions build. This is
        // what lets one binary self-select CPU/GPU/NPU at runtime on whatever
        // hardware and drivers are actually present.
        std::string last_err;
        for (size_t i = 0; i < chain.size(); ++i) {
            const std::string& provider = chain[i];
            try {
                const auto t0 = std::chrono::steady_clock::now();
                encoder_ = build_session(env_, options_, provider, enc_path, dir, cache_base + "_encoder");
                decoder_ = build_session(env_, options_, provider, dec_path, dir, cache_base + "_decoder");
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
        if (options_.device == Device::NPU && ep::is_qnn(active_provider_)) {
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
        out.runtime = ep::runtime_for(active_provider_);
        out.model_format = "onnx";
        out.decode_strategy = "static-no-kv";
        out.max_context = static_cast<long>(kStaticMaxTokens);

        // Audit CPU offload once, off the back of this (warmup) inference. Profiling
        // stops inside measure_offload, so the timed runs that follow are unprofiled.
        if (!offload_done_) {
            offload_done_ = true;
            offload_info_ = ep::measure_offload({encoder_.get(), decoder_.get()});
        }
        return out;
    }

    std::string backend_name() const override { return "ONNX Runtime (unified, static)"; }
    std::string device_name() const override { return active_provider_; }
    std::string runtime_version() const override { return Ort::GetVersionString(); }
    double load_seconds() const override { return load_seconds_; }
    OffloadInfo offload_info() const override { return offload_info_; }

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
    bool offload_done_ = false;
    OffloadInfo offload_info_;
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

// Return the best logical device the unified binary should start on. The engine's
// NPU->GPU->CPU fallback chain validates the pick at session-build time, so a wrong
// guess degrades gracefully rather than failing.
Device best_available_device() {
#ifdef WHISPER_HAL_QUALCOMM
    // Qualcomm / ARM64: confirm a real QNN NPU via the EP device list. DirectML/GPU
    // is dormant on ARM64, so it is NPU-or-CPU here.
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "whisper_hal_probe");
        ep::register_qnn_library(env);
        for (Ort::ConstEpDevice d : env.GetEpDevices()) {
            if (std::strcmp(d.EpName(), ep::kQnnEpName) == 0 &&
                d.Device().Type() == OrtHardwareDeviceType_NPU) {
                return Device::NPU;
            }
        }
    } catch (...) {
        // Fall through to CPU if the plugin can't be probed.
    }
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
    // Set WHISPER_HAL_LOG_PROBE=1 to dump the enumerated devices to stderr.
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
        const bool log_probe = !ep::env_or("WHISPER_HAL_LOG_PROBE", "").empty();

        // Hardware enumeration (authoritative for CPU, DirectML GPUs, and OpenVINO
        // NPU/GPU; blind to VitisAI). The EP-device API landed in ORT 1.22
        // (ORT_API_VERSION 22); on older headers we skip the probe and let the
        // compiled-in-EP heuristic below drive the pick.
        bool probed = false, hw_npu = false, hw_gpu = false;
#if defined(ORT_API_VERSION) && ORT_API_VERSION >= 22
        try {
            Ort::Env probe_env(ORT_LOGGING_LEVEL_WARNING, "whisper_hal_probe");
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
                    std::cerr << "[whisper-hal probe] EP=" << d.EpName() << " hw=" << tn << "\n";
                }
            }
        } catch (const std::exception& e) {
            if (log_probe) std::cerr << "[whisper-hal probe] GetEpDevices unavailable: " << e.what() << "\n";
        }
#else
        if (log_probe) std::cerr << "[whisper-hal probe] EP-device API not in this ORT; using heuristic\n";
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
