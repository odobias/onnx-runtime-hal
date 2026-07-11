// Intel backend variant: the VENDOR-NEUTRAL ONNX whisper run through OpenVINO
// (ov::Core), NOT OpenVINO GenAI. This is the portable path -- the same ONNX we
// run on AMD -- and the whole point is that it also compiles+runs on the Intel
// NPU. NPUs reject dynamic (growing) KV-cache shapes, so we use a STATIC decode:
// reshape the encoder + no-past decoder to fixed shapes and recompute the whole
// (padded) causal sequence each step. O(n * MaxTokens), but fully static and
// NPU-compilable -- identical strategy to the AMD backend, just via OpenVINO.
#include "backend_registry.hpp"

#include <chrono>
#include <stdexcept>

#ifdef NPU_INFERENCE_BENCH_INTEL
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/version.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "npu_inference_bench/whisper_frontend.hpp"
#endif

namespace npu_inference_bench {
namespace intel_onnx {

#ifdef NPU_INFERENCE_BENCH_INTEL

namespace {

namespace fs = std::filesystem;
namespace fe = npu_inference_bench::frontend;

constexpr int kEncSeq = 1500;
constexpr int kDModel = 384;
// Static decoder context. NPUs need a fixed length; keep it small (whisper's
// clips here generate < ~60 tokens) since per-step cost scales with this. AMD's
// prepared model hard-codes 448, which is wasteful; 128 is plenty and faster.
// Overridable via WHISPER_ONNX_MAXLEN to sweep the O(n*maxlen) recompute cost.
constexpr int64_t kDefaultMaxTokens = 128;

int64_t env_max_tokens() {
    const char* v = std::getenv("WHISPER_ONNX_MAXLEN");
    if (!v || !*v) return kDefaultMaxTokens;
    const long n = std::strtol(v, nullptr, 10);
    if (n < 8 || n > 448) return kDefaultMaxTokens;  // clamp to sane whisper range
    return static_cast<int64_t>(n);
}

bool env_profile() {
    const char* v = std::getenv("WHISPER_ONNX_PROFILE");
    return v && *v && std::string(v) != "0";
}

// Mel front-end: FFT (Bluestein, threaded) by default; the naive reference DFT is
// kept intact and selectable via WHISPER_ONNX_MEL=naive for A/B comparison.
bool env_use_fft_mel() {
    const char* v = std::getenv("WHISPER_ONNX_MEL");
    return !(v && std::string(v) == "naive");
}

std::string ov_device(const EngineOptions& o) {
    if (!o.device_override.empty()) return o.device_override;
    switch (o.device) {
        case Device::NPU: return "NPU";
        case Device::GPU: return "GPU";
        case Device::CPU: return "CPU";
    }
    return "CPU";
}

std::string query_full_device_name(const std::string& device) {
    try {
        ov::Core core;
        std::string name = core.get_property(device, ov::device::full_name);
        const auto end = name.find_last_not_of(' ');
        if (end == std::string::npos) return "";
        name.erase(end + 1);
        return name;
    } catch (...) {
        return "";
    }
}

class IntelOnnxEngine final : public IWhisperEngine {
public:
    explicit IntelOnnxEngine(const EngineOptions& options)
        : device_(ov_device(options)),
          full_device_name_(query_full_device_name(device_)),
          max_tokens_(env_max_tokens()),
          profile_(env_profile()),
          use_fft_mel_(env_use_fft_mel()) {
        const fs::path dir(options.model_dir);
        const fs::path enc_path = dir / "encoder_model.onnx";
        const fs::path dec_path = dir / "decoder_model.onnx";
        if (!fs::exists(enc_path) || !fs::exists(dec_path)) {
            throw std::runtime_error(
                "Intel ONNX model_dir must contain encoder_model.onnx and decoder_model.onnx "
                "(neutral optimum export): " + options.model_dir);
        }

        mel_filters_ = fe::parse_mel_filters(dir / "preprocessor_config.json");
        vocab_ = fe::load_vocab(dir / "vocab.json");
        const std::string gc = fe::read_text(dir / "generation_config.json");
        const int64_t sot = fe::json_int(gc, "decoder_start_token_id", 50257);
        eos_ = fe::json_int(gc, "eos_token_id", 50256);
        const int64_t no_ts = fe::json_int(gc, "no_timestamps_token_id", 50362);
        prompt_ = {sot, no_ts};
        suppress_ = fe::json_int_array(gc, "suppress_tokens");
        begin_suppress_ = fe::json_int_array(gc, "begin_suppress_tokens");

        ov::AnyMap props;
        if (!options.cache_dir.empty()) props.insert(ov::cache_dir(options.cache_dir));
        if (options.cpu_threads > 0 && device_ == "CPU")
            props.insert(ov::inference_num_threads(options.cpu_threads));

        const auto t0 = std::chrono::steady_clock::now();
        auto enc_model = core_.read_model(enc_path.string());
        enc_model->reshape(std::map<std::string, ov::PartialShape>{
            {"input_features", ov::PartialShape{1, fe::kMelBins, fe::kFrames}}});
        encoder_ = core_.compile_model(enc_model, device_, props);

        auto dec_model = core_.read_model(dec_path.string());
        dec_model->reshape(std::map<std::string, ov::PartialShape>{
            {"input_ids", ov::PartialShape{1, max_tokens_}},
            {"encoder_hidden_states", ov::PartialShape{1, kEncSeq, kDModel}}});
        decoder_ = core_.compile_model(dec_model, device_, props);
        load_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    TranscribeResult transcribe(const AudioSamples& audio) override {
        const auto t0 = std::chrono::steady_clock::now();

        std::vector<float> features = use_fft_mel_
                                          ? fe::log_mel_spectrogram_fft(audio, mel_filters_)
                                          : fe::log_mel_spectrogram(audio, mel_filters_);
        const auto t_mel = std::chrono::steady_clock::now();

        auto er = encoder_.create_infer_request();
        er.set_input_tensor(ov::Tensor(ov::element::f32,
                                       ov::Shape{1, fe::kMelBins, fe::kFrames}, features.data()));
        er.infer();
        const ov::Tensor enc_out = er.get_output_tensor(0);
        std::vector<float> ehs(enc_out.data<const float>(),
                               enc_out.data<const float>() + enc_out.get_size());
        const auto t_enc = std::chrono::steady_clock::now();

        std::vector<int64_t> ids(max_tokens_, eos_);
        for (size_t i = 0; i < prompt_.size() && i < static_cast<size_t>(max_tokens_); ++i)
            ids[i] = prompt_[i];
        int64_t cur = static_cast<int64_t>(prompt_.size());

        auto dr = decoder_.create_infer_request();
        ov::Tensor ehs_t(ov::element::f32, ov::Shape{1, kEncSeq, kDModel}, ehs.data());

        std::vector<int64_t> gen;
        double logprob_sum = 0.0;
        long n_gen = 0;
        bool first = true;
        while (cur < max_tokens_) {
            ov::Tensor id_t(ov::element::i64, ov::Shape{1, static_cast<size_t>(max_tokens_)}, ids.data());
            dr.set_tensor("input_ids", id_t);
            dr.set_tensor("encoder_hidden_states", ehs_t);
            dr.infer();

            const ov::Tensor logits_t = dr.get_output_tensor(0);  // [1, kMaxTokens, vocab]
            const ov::Shape lshape = logits_t.get_shape();
            const size_t vocab = lshape.back();
            const float* row = logits_t.data<const float>() + static_cast<size_t>(cur - 1) * vocab;

            std::vector<float> logit(row, row + vocab);
            for (const int64_t s : suppress_)
                if (s >= 0 && static_cast<size_t>(s) < vocab) logit[s] = -std::numeric_limits<float>::infinity();
            if (first)
                for (const int64_t s : begin_suppress_)
                    if (s >= 0 && static_cast<size_t>(s) < vocab) logit[s] = -std::numeric_limits<float>::infinity();
            first = false;

            float m = -std::numeric_limits<float>::infinity();
            int64_t tok = 0;
            for (size_t v = 0; v < vocab; ++v) {
                if (logit[v] > m) { m = logit[v]; tok = static_cast<int64_t>(v); }
            }
            double sum_exp = 0.0;
            for (size_t v = 0; v < vocab; ++v) sum_exp += std::exp(static_cast<double>(logit[v]) - m);
            const double lse = m + std::log(sum_exp);
            logprob_sum += static_cast<double>(logit[tok]) - lse;
            ++n_gen;

            if (tok == eos_) break;
            gen.push_back(tok);
            ids[cur] = tok;
            ++cur;
        }

        const auto t_dec = std::chrono::steady_clock::now();

        if (profile_) {
            const double mel_ms = std::chrono::duration<double, std::milli>(t_mel - t0).count();
            const double enc_ms = std::chrono::duration<double, std::milli>(t_enc - t_mel).count();
            const double dec_ms = std::chrono::duration<double, std::milli>(t_dec - t_enc).count();
            std::fprintf(stderr,
                         "[onnx-profile] maxlen=%lld tokens=%ld | mel(%s) %.1f ms | encoder %.1f ms | "
                         "decode %.1f ms (%.2f ms/tok) | total %.1f ms\n",
                         static_cast<long long>(max_tokens_), n_gen,
                         use_fft_mel_ ? "fft" : "naive", mel_ms, enc_ms, dec_ms,
                         n_gen > 0 ? dec_ms / static_cast<double>(n_gen) : 0.0,
                         mel_ms + enc_ms + dec_ms);
        }

        TranscribeResult out;
        out.text = fe::decode_tokens(gen, vocab_, eos_);
        out.infer_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        out.generated_tokens = n_gen;
        out.sequence_logprob = logprob_sum;
        if (n_gen > 0) {
            out.avg_logprob = logprob_sum / static_cast<double>(n_gen);
            out.throughput_tps = static_cast<double>(n_gen) / out.infer_seconds;
            out.tpot_ms = out.infer_seconds * 1000.0 / static_cast<double>(n_gen);
        }
        out.ttft_ms = -1.0;  // not separately measured in this static loop
        out.has_token_metrics = true;
        out.runtime = "openvino";
        out.model_format = "onnx";
        out.decode_strategy = "static-no-kv";
        out.max_context = static_cast<long>(max_tokens_);
        return out;
    }

    std::string backend_name() const override { return "Intel ONNX (OpenVINO, static)"; }
    std::string device_name() const override { return device_; }
    std::string full_device_name() const override { return full_device_name_; }
    std::string runtime_version() const override { return ov::get_openvino_version().buildNumber; }
    double load_seconds() const override { return load_seconds_; }

private:
    ov::Core core_;
    std::string device_;
    std::string full_device_name_;
    ov::CompiledModel encoder_;
    ov::CompiledModel decoder_;
    std::vector<float> mel_filters_;
    std::unordered_map<int64_t, std::string> vocab_;
    std::vector<int64_t> prompt_;
    std::vector<int64_t> suppress_;
    std::vector<int64_t> begin_suppress_;
    int64_t eos_ = 50256;
    int64_t max_tokens_ = kDefaultMaxTokens;
    bool profile_ = false;
    bool use_fft_mel_ = true;
    double load_seconds_ = 0.0;
};

}  // namespace

std::unique_ptr<IWhisperEngine> create(const EngineOptions& options) {
    try {
        return std::make_unique<IntelOnnxEngine>(options);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Intel ONNX backend failed to init: ") + e.what());
    }
}

bool available() { return true; }

#else  // !NPU_INFERENCE_BENCH_INTEL

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "Intel ONNX backend not compiled. Rebuild with EnableIntel=true and the OpenVINO "
        "runtime available (define NPU_INFERENCE_BENCH_INTEL, link openvino).");
}

bool available() { return false; }

#endif

}  // namespace intel_onnx
}  // namespace npu_inference_bench
