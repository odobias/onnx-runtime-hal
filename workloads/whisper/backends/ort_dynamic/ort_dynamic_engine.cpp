// Dynamic (with-past KV-cache) ONNX Whisper backend. Runs the stock optimum
// `--task automatic-speech-recognition-with-past` export (encoder_model.onnx +
// decoder_model.onnx + decoder_with_past_model.onnx) through ONNX Runtime with a
// real KV cache: the encoder cross-attention K/V are computed ONCE in the prefill
// and reused, and only the decoder self-attention K/V grow one row per generated
// token. That is dramatically less compute than the static no-KV path (which
// recomputes the whole 128-wide context every step), so it is faster on CPU/GPU.
//
// The trade-off is portability: the growing KV cache means dynamic tensor shapes,
// which the NPU compilers (VitisAI / QNN / OpenVINO-NPU) reject -- so this backend
// is a CPU/GPU citizen by design. The static backend (ort_static) remains the
// NPU-compilable path. Both share EP selection via backends/ort_common/ort_ep.hpp.
//
// This is a direct C++ port of the validated reference decode loop in
// tools/research/onnx_decode.py (same prompt, suppress, greedy logic).
#include "backend_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "npu_inference_bench/whisper_frontend.hpp"
#include "ort_ep.hpp"
#include "ort_offload.hpp"

#ifdef NPU_INFERENCE_BENCH_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace npu_inference_bench {
namespace ort_dynamic {

#ifdef NPU_INFERENCE_BENCH_ORT

namespace {

namespace fs = std::filesystem;
namespace fe = npu_inference_bench::frontend;
namespace ep = npu_inference_bench::ort_common;

constexpr int64_t kMaxTokens = 224;  // whisper decoder ceiling; decode stops at EOS

// A cached key/value tensor: raw data + its current shape. Kept in our own
// buffers (not ORT-owned Values) so we can cheaply re-feed them each step and
// grow the decoder ones -- mirrors how ort_static re-creates the encoder tensor.
struct Kv {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

std::vector<std::string> session_input_names(Ort::Session& s) {
    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<std::string> names;
    for (size_t i = 0; i < s.GetInputCount(); ++i)
        names.emplace_back(s.GetInputNameAllocated(i, alloc).get());
    return names;
}

std::vector<std::string> session_output_names(Ort::Session& s) {
    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<std::string> names;
    for (size_t i = 0; i < s.GetOutputCount(); ++i)
        names.emplace_back(s.GetOutputNameAllocated(i, alloc).get());
    return names;
}

Kv read_kv(const Ort::Value& v) {
    const auto info = v.GetTensorTypeAndShapeInfo();
    Kv kv;
    kv.shape = info.GetShape();
    const size_t n = info.GetElementCount();
    const float* p = v.GetTensorData<float>();
    kv.data.assign(p, p + n);
    return kv;
}

// present.<L>.<decoder|encoder>.<key|value> -> past_key_values.<L>.<...>
std::string present_to_past(const std::string& present_name) {
    static const std::string kPrefix = "present.";
    return "past_key_values." + present_name.substr(kPrefix.size());
}

bool is_present(const std::string& name) { return name.rfind("present.", 0) == 0; }

class OrtDynamicEngine final : public IWhisperEngine {
public:
    explicit OrtDynamicEngine(const EngineOptions& options)
        : env_(ORT_LOGGING_LEVEL_WARNING, "npu_inference_bench_ort_dynamic"), options_(options) {
        const fs::path dir(options.model_dir);
        const fs::path enc_path = dir / "encoder_model.onnx";
        const fs::path dec_path = dir / "decoder_model.onnx";
        const fs::path decp_path = dir / "decoder_with_past_model.onnx";
        if (!fs::exists(enc_path) || !fs::exists(dec_path) || !fs::exists(decp_path)) {
            throw std::runtime_error(
                "ONNX dynamic model_dir must contain encoder_model.onnx, decoder_model.onnx and "
                "decoder_with_past_model.onnx (stock optimum with-past export): " + options.model_dir);
        }

        mel_filters_ = fe::parse_mel_filters(dir / "preprocessor_config.json");
        vocab_ = fe::load_vocab(dir / "vocab.json");
        const std::string gc = fe::read_text(dir / "generation_config.json");
        sot_ = fe::json_int(gc, "decoder_start_token_id", 50257);
        eos_ = fe::json_int(gc, "eos_token_id", 50256);
        no_timestamps_ = fe::json_int(gc, "no_timestamps_token_id", 50362);
        suppress_ = fe::json_int_array(gc, "suppress_tokens");
        begin_suppress_ = fe::json_int_array(gc, "begin_suppress_tokens");

        // Walk the device fallback chain; keep the first EP that builds ALL three
        // sessions. On NPU EPs this is expected to fail (dynamic KV shapes) and
        // demote to GPU/CPU -- the explicit, recorded benchmark behavior.
        const std::vector<std::string> chain = ep::fallback_chain(options_);
        diagnostics_.requested_provider =
            options_.device_override.empty() ? std::string("auto") : options_.device_override;
        std::string last_err;
        for (size_t i = 0; i < chain.size(); ++i) {
            const std::string& provider = chain[i];
            try {
                const auto t0 = std::chrono::steady_clock::now();
                encoder_ = build_session(provider, enc_path, dir, "dyn_encoder");
                decoder_ = build_session(provider, dec_path, dir, "dyn_decoder");
                decoder_past_ = build_session(provider, decp_path, dir, "dyn_decoder_past");
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
                decoder_past_.reset();
                if (i + 1 < chain.size()) {
                    std::cerr << "[npu-inference-bench] dynamic EP '" << provider << "' unavailable ("
                              << e.what() << "); falling back to '" << chain[i + 1] << "'\n";
                }
            }
        }
        if (!encoder_ || !decoder_ || !decoder_past_) {
            throw std::runtime_error(
                "no ONNX Runtime EP could build the dynamic (with-past) sessions -- the NPU EPs "
                "reject the growing KV cache, so this backend needs CPU or GPU (last error: " +
                last_err + ")");
        }

        decp_input_names_ = session_input_names(*decoder_past_);
        dec_output_names_ = session_output_names(*decoder_);
        decp_output_names_ = session_output_names(*decoder_past_);
    }

    TranscribeResult transcribe(const AudioSamples& audio) override {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const auto t0 = std::chrono::steady_clock::now();

        // --- encoder ---
        std::vector<float> features = fe::log_mel_spectrogram_fft(audio, mel_filters_);
        std::vector<int64_t> feat_shape{1, fe::kMelBins, fe::kFrames};
        auto feat_tensor = Ort::Value::CreateTensor<float>(mem, features.data(), features.size(),
                                                           feat_shape.data(), feat_shape.size());
        const char* enc_in[] = {"input_features"};
        const char* enc_out[] = {"last_hidden_state"};
        auto enc_outputs = encoder_->Run(Ort::RunOptions{nullptr}, enc_in, &feat_tensor, 1, enc_out, 1);
        Kv ehs = read_kv(enc_outputs[0]);  // [1,1500,384]

        // --- prefill (no-past decoder) on the forced prompt ---
        std::vector<int64_t> prompt{sot_, no_timestamps_};
        std::vector<int64_t> prompt_shape{1, static_cast<int64_t>(prompt.size())};
        std::vector<Ort::Value> pre_inputs;
        pre_inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, prompt.data(), prompt.size(),
                                                              prompt_shape.data(), prompt_shape.size()));
        pre_inputs.push_back(Ort::Value::CreateTensor<float>(mem, ehs.data.data(), ehs.data.size(),
                                                            ehs.shape.data(), ehs.shape.size()));
        const char* pre_in[] = {"input_ids", "encoder_hidden_states"};
        std::vector<const char*> dec_out_c;
        for (const auto& n : dec_output_names_) dec_out_c.push_back(n.c_str());
        auto pre_outputs = decoder_->Run(Ort::RunOptions{nullptr}, pre_in, pre_inputs.data(),
                                         pre_inputs.size(), dec_out_c.data(), dec_out_c.size());

        // Seed the KV cache: decoder + encoder present -> past.
        std::unordered_map<std::string, Kv> past;
        std::vector<float> logits;
        int64_t vocab = 0;
        for (size_t i = 0; i < dec_output_names_.size(); ++i) {
            const std::string& name = dec_output_names_[i];
            if (name == "logits") {
                const auto info = pre_outputs[i].GetTensorTypeAndShapeInfo();
                const auto shp = info.GetShape();  // [1, plen, vocab]
                vocab = shp.back();
                const float* base = pre_outputs[i].GetTensorData<float>();
                const int64_t last = shp[1] - 1;  // logits of the final prompt position
                logits.assign(base + last * vocab, base + (last + 1) * vocab);
            } else if (is_present(name)) {
                past[present_to_past(name)] = read_kv(pre_outputs[i]);
            }
        }
        if (vocab == 0) throw std::runtime_error("dynamic decoder produced no logits output");

        // --- greedy decode with KV cache ---
        std::vector<int64_t> generated;
        double logprob_sum = 0.0;
        long n_gen = 0;
        double ttft_ms = -1.0;
        for (int64_t step = 0; step < kMaxTokens; ++step) {
            apply_suppress(logits, vocab, /*first=*/step == 0);

            float m = -std::numeric_limits<float>::infinity();
            int64_t tok = 0;
            for (int64_t v = 0; v < vocab; ++v) {
                if (logits[static_cast<size_t>(v)] > m) { m = logits[static_cast<size_t>(v)]; tok = v; }
            }
            double sum_exp = 0.0;
            for (int64_t v = 0; v < vocab; ++v)
                sum_exp += std::exp(static_cast<double>(logits[static_cast<size_t>(v)]) - m);
            logprob_sum += static_cast<double>(logits[static_cast<size_t>(tok)]) - (m + std::log(sum_exp));
            ++n_gen;
            if (ttft_ms < 0.0)
                ttft_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();

            if (tok == eos_) break;
            generated.push_back(tok);

            // Step the with-past decoder: input_ids=[tok], feed the whole past.
            std::vector<int64_t> tok_ids{tok};
            std::vector<int64_t> tok_shape{1, 1};
            std::vector<const char*> in_names;
            std::vector<Ort::Value> in_values;
            in_names.reserve(decp_input_names_.size());
            in_values.reserve(decp_input_names_.size());
            for (const std::string& name : decp_input_names_) {
                in_names.push_back(name.c_str());
                if (name == "input_ids") {
                    in_values.push_back(Ort::Value::CreateTensor<int64_t>(
                        mem, tok_ids.data(), tok_ids.size(), tok_shape.data(), tok_shape.size()));
                } else {
                    Kv& kv = past.at(name);
                    in_values.push_back(Ort::Value::CreateTensor<float>(
                        mem, kv.data.data(), kv.data.size(), kv.shape.data(), kv.shape.size()));
                }
            }

            std::vector<const char*> out_names;
            for (const auto& n : decp_output_names_) out_names.push_back(n.c_str());
            auto step_out = decoder_past_->Run(Ort::RunOptions{nullptr}, in_names.data(),
                                               in_values.data(), in_values.size(),
                                               out_names.data(), out_names.size());

            // Refresh logits + grow the decoder KV (encoder KV are not re-emitted).
            for (size_t i = 0; i < decp_output_names_.size(); ++i) {
                const std::string& name = decp_output_names_[i];
                if (name == "logits") {
                    const float* base = step_out[i].GetTensorData<float>();
                    logits.assign(base, base + vocab);  // [1,1,vocab]
                } else if (is_present(name)) {
                    past[present_to_past(name)] = read_kv(step_out[i]);
                }
            }
        }

        TranscribeResult out;
        out.text = fe::decode_tokens(generated, vocab_, eos_);
        out.infer_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        out.generated_tokens = n_gen;
        out.sequence_logprob = logprob_sum;
        if (n_gen > 0) {
            out.avg_logprob = logprob_sum / static_cast<double>(n_gen);
            out.throughput_tps = static_cast<double>(n_gen) / out.infer_seconds;
            out.tpot_ms = out.infer_seconds * 1000.0 / static_cast<double>(n_gen);
        }
        out.ttft_ms = ttft_ms;
        out.has_token_metrics = true;
        out.runtime = ep::runtime_for(active_provider_);
        out.model_format = "onnx";
        out.decode_strategy = "dynamic-kv";
        out.max_context = -1;  // grows with the sequence

        // Finalize profiling after the first transcription (normally the harness
        // warmup), so measured runs do not accumulate profiler overhead.
        (void)execution_diagnostics();
        return out;
    }

    std::string backend_name() const override { return "ONNX Runtime (unified, dynamic KV)"; }
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
                    // Profiling is diagnostic only.
                }
            };
            audit(encoder_);
            audit(decoder_);
            audit(decoder_past_);
        }
        return diagnostics_;
    }
    double load_seconds() const override { return load_seconds_; }

private:
    std::unique_ptr<Ort::Session> build_session(const std::string& provider, const fs::path& model_path,
                                                const fs::path& model_dir, const std::string& cache_key) {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const fs::path profile_prefix =
            fs::temp_directory_path() / ("npu_bench_" + cache_key);
        const std::wstring profile_prefix_w = profile_prefix.wstring();
        so.EnableProfiling(profile_prefix_w.c_str());
        ep::append_provider(env_, so, options_, provider, model_dir, cache_key);
        const std::wstring wpath = model_path.wstring();
        return std::make_unique<Ort::Session>(env_, wpath.c_str(), so);
    }

    void apply_suppress(std::vector<float>& logit, int64_t vocab, bool first) {
        const float neg = -std::numeric_limits<float>::infinity();
        for (const int64_t s : suppress_)
            if (s >= 0 && s < vocab) logit[static_cast<size_t>(s)] = neg;
        if (first)
            for (const int64_t s : begin_suppress_)
                if (s >= 0 && s < vocab) logit[static_cast<size_t>(s)] = neg;
    }

    Ort::Env env_;
    EngineOptions options_;
    std::string active_provider_;
    mutable ExecutionDiagnostics diagnostics_;
    mutable bool diagnostics_finalized_ = false;
    int64_t sot_ = 50257;
    int64_t eos_ = 50256;
    int64_t no_timestamps_ = 50362;
    std::vector<int64_t> suppress_;
    std::vector<int64_t> begin_suppress_;
    std::vector<float> mel_filters_;
    std::unordered_map<int64_t, std::string> vocab_;
    double load_seconds_ = 0.0;
    std::unique_ptr<Ort::Session> encoder_;
    std::unique_ptr<Ort::Session> decoder_;       // no-past prefill
    std::unique_ptr<Ort::Session> decoder_past_;  // with-past steps
    std::vector<std::string> decp_input_names_;
    std::vector<std::string> dec_output_names_;
    std::vector<std::string> decp_output_names_;
};

}  // namespace

std::unique_ptr<IWhisperEngine> create(const EngineOptions& options) {
    try {
        return std::make_unique<OrtDynamicEngine>(options);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime dynamic backend failed to init: ") + e.what());
    }
}

bool available() { return true; }

#else  // !NPU_INFERENCE_BENCH_ORT

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "ONNX Runtime dynamic backend not compiled. Build with NPU_INFERENCE_BENCH_ORT and link ONNX Runtime.");
}

bool available() { return false; }

#endif

}  // namespace ort_dynamic
}  // namespace npu_inference_bench
