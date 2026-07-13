// Generic static ONNX backend: runs the unchanged whisper/en-static-onnx
// package through ONNX Runtime providers. The model contract is fixed-shape,
// no-KV recompute: encoder [1,80,3000] -> [1,1500,384], decoder consumes
// input_ids [1,128] + encoder_hidden_states [1,1500,384] and returns logits.
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
#include "ort_ep.hpp"
#include "ort_offload.hpp"

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

constexpr int64_t kEncSeq = 1500;
constexpr int64_t kDModel = 384;
constexpr int64_t kStaticMaxTokens = 128;

int64_t env_max_tokens() {
    const char* v = std::getenv("NPU_INFERENCE_BENCH_ORT_MAX_TOKENS");
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
        const fs::path profile_prefix =
            fs::temp_directory_path() / ("npu_bench_" + cache_key);
        const std::wstring profile_prefix_w = profile_prefix.wstring();
        so.EnableProfiling(profile_prefix_w.c_str());
        ep::append_provider(env, so, options, provider, model_dir, cache_key);
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
            std::cerr << "[npu-inference-bench] QNN context cache '" << ctx_path.string()
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
        : env_(ORT_LOGGING_LEVEL_WARNING, "npu_inference_bench_ort_static"),
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
        diagnostics_.requested_provider =
            options_.device_override.empty() ? std::string("auto") : options_.device_override;

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
                diagnostics_.attempts.push_back({provider, true, {}});
                diagnostics_.resolved_provider = provider;
                diagnostics_.inference_precision =
                    ep::resolved_inference_precision(options_, provider);
                diagnostics_.fallback_occurred = provider != chain.front();
                break;
            } catch (const std::exception& e) {
                last_err = e.what();
                diagnostics_.attempts.push_back({provider, false, last_err});
                encoder_.reset();
                decoder_.reset();
                if (i + 1 < chain.size()) {
                    std::cerr << "[npu-inference-bench] EP '" << provider << "' unavailable (" << e.what()
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
            auto audit = [&](const std::unique_ptr<Ort::Session>& session) {
                if (!session) return;
                try {
                    Ort::AllocatorWithDefaultOptions alloc;
                    const std::string profile_path =
                        session->EndProfilingAllocated(alloc).get();
                    const auto stats = ep::parse_ort_profile(profile_path);
                    if (stats.measured) {
                        diagnostics_.offload_measured = true;
                        diagnostics_.ep_nodes =
                            std::max(0, diagnostics_.ep_nodes) + stats.ep_nodes;
                        diagnostics_.cpu_nodes =
                            std::max(0, diagnostics_.cpu_nodes) + stats.cpu_nodes;
                        if (!stats.cpu_ops.empty()) {
                            if (!diagnostics_.cpu_offload_ops.empty())
                                diagnostics_.cpu_offload_ops += "; ";
                            diagnostics_.cpu_offload_ops += stats.cpu_ops;
                        }
                    }
                    std::error_code ec;
                    fs::remove(profile_path, ec);
                } catch (const std::exception&) {
                    // Profiling is diagnostic only; inference results remain valid.
                }
            };
            audit(encoder_);
            audit(decoder_);
        }
        return diagnostics_;
    }
    double load_seconds() const override { return load_seconds_; }

private:
    Ort::Env env_;
    EngineOptions options_;
    std::string active_provider_;  // EP that actually built the sessions (post-fallback)
    mutable ExecutionDiagnostics diagnostics_;
    mutable bool diagnostics_finalized_ = false;
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

// Return the best logical device the unified binary should start on. The engine's
// NPU->GPU->CPU fallback chain validates the pick at session-build time, so a wrong
// guess degrades gracefully rather than failing.
Device best_available_device() {
#ifdef NPU_INFERENCE_BENCH_QUALCOMM
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
