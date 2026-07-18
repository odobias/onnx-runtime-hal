// AMD backend: whisper on AMD XDNA NPU (Ryzen AI) via ONNX Runtime + VitisAI EP.
//
// Model directory convention:
//   tiny_encoder.onnx
//   tiny_decoder.onnx
//   preprocessor_config.json
//   vocab.json
// plus optional tokenizer sidecars. The ONNX files are AMD's NPU-exported
// whisper-tiny models; tokenizer/preprocessor files come from openai/whisper-tiny.
#include "backend_registry.hpp"
#include "npu_inference_bench/precision_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef NPU_INFERENCE_BENCH_AMD
#include <string>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>  // from the Ryzen AI / ONNX Runtime SDK
#endif

namespace npu_inference_bench {
namespace amd {

#ifdef NPU_INFERENCE_BENCH_AMD

namespace {

namespace fs = std::filesystem;

constexpr int kSampleRate = 16000;
constexpr int kNfft = 400;
constexpr int kHop = 160;
constexpr int kMelBins = 80;
constexpr int kFftBins = 201;
constexpr int kSamples = 480000;
constexpr int kFrames = 3000;
constexpr int64_t kEos = 50257;
constexpr int64_t kSot = 50258;
constexpr int64_t kLangEn = 50259;
constexpr int64_t kTranscribe = 50359;
constexpr int64_t kNoTimestamps = 50363;
constexpr int64_t kMaxTokens = 448;

bool progress_enabled() {
    const char* v = std::getenv("NPU_INFERENCE_BENCH_PROGRESS");
    return v && *v && std::string(v) != "0";
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void progress(std::chrono::steady_clock::time_point t0, const std::string& msg) {
    if (!progress_enabled()) return;
    std::cerr << "[amd-progress +" << seconds_since(t0) << "s] " << msg << "\n";
}

std::string read_text(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::vector<float> parse_mel_filters(const fs::path& preprocessor_config) {
    const std::string text = read_text(preprocessor_config);
    const size_t key = text.find("\"mel_filters\"");
    if (key == std::string::npos) throw std::runtime_error("mel_filters not found in preprocessor_config.json");
    size_t pos = text.find('[', key);
    if (pos == std::string::npos) throw std::runtime_error("mel_filters array not found");

    std::vector<float> values;
    values.reserve(kMelBins * kFftBins);
    char* end = nullptr;
    while (pos < text.size() && values.size() < kMelBins * kFftBins) {
        const char ch = text[pos];
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+') {
            const char* start = text.c_str() + pos;
            const float value = std::strtof(start, &end);
            if (end != start) {
                values.push_back(value);
                pos = static_cast<size_t>(end - text.c_str());
                continue;
            }
        }
        ++pos;
    }
    if (values.size() != kMelBins * kFftBins) {
        throw std::runtime_error("expected 80x201 mel filters, got " + std::to_string(values.size()));
    }
    return values;
}

std::vector<float> log_mel_spectrogram(const AudioSamples& input, const std::vector<float>& mel_filters) {
    std::vector<float> audio(kSamples, 0.0f);
    std::copy_n(input.begin(), (std::min)(input.size(), audio.size()), audio.begin());

    constexpr double pi = 3.14159265358979323846;
    std::vector<float> window(kNfft);
    for (int i = 0; i < kNfft; ++i) {
        window[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * i / kNfft));
    }

    std::vector<float> cos_table(kFftBins * kNfft);
    std::vector<float> sin_table(kFftBins * kNfft);
    for (int k = 0; k < kFftBins; ++k) {
        for (int n = 0; n < kNfft; ++n) {
            const double angle = 2.0 * pi * k * n / kNfft;
            cos_table[k * kNfft + n] = static_cast<float>(std::cos(angle));
            sin_table[k * kNfft + n] = static_cast<float>(std::sin(angle));
        }
    }

    std::vector<float> features(kMelBins * kFrames);
    float global_max = -std::numeric_limits<float>::infinity();

    for (int frame = 0; frame < kFrames; ++frame) {
        float power[kFftBins] = {};
        const int base = frame * kHop - kNfft / 2;
        for (int k = 0; k < kFftBins; ++k) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int n = 0; n < kNfft; ++n) {
                const int idx = base + n;
                float sample = (idx >= 0 && idx < kSamples) ? audio[idx] : 0.0f;
                sample *= window[n];
                real += sample * cos_table[k * kNfft + n];
                imag -= sample * sin_table[k * kNfft + n];
            }
            power[k] = real * real + imag * imag;
        }

        for (int mel = 0; mel < kMelBins; ++mel) {
            float energy = 0.0f;
            const float* filter = mel_filters.data() + mel * kFftBins;
            for (int k = 0; k < kFftBins; ++k) energy += filter[k] * power[k];
            const float logv = std::log10((std::max)(energy, 1.0e-10f));
            features[mel * kFrames + frame] = logv;
            global_max = (std::max)(global_max, logv);
        }
    }

    const float floor = global_max - 8.0f;
    for (float& value : features) {
        value = ((std::max)(value, floor) + 4.0f) / 4.0f;
    }
    return features;
}

std::string parse_json_string(const std::string& text, size_t& pos) {
    if (text[pos] != '"') throw std::runtime_error("expected JSON string");
    ++pos;
    std::string out;
    while (pos < text.size()) {
        const char ch = text[pos++];
        if (ch == '"') return out;
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (pos >= text.size()) break;
        const char esc = text[pos++];
        if (esc == '"' || esc == '\\' || esc == '/') out.push_back(esc);
        else if (esc == 'n') out.push_back('\n');
        else if (esc == 'r') out.push_back('\r');
        else if (esc == 't') out.push_back('\t');
        else if (esc == 'u') throw std::runtime_error("\\u escapes are not supported in tokenizer vocab");
        else out.push_back(esc);
    }
    throw std::runtime_error("unterminated JSON string");
}

std::unordered_map<int64_t, std::string> load_vocab(const fs::path& vocab_path) {
    const std::string text = read_text(vocab_path);
    std::unordered_map<int64_t, std::string> vocab;
    size_t pos = 0;
    while (pos < text.size()) {
        if (text[pos] != '"') {
            ++pos;
            continue;
        }
        std::string token = parse_json_string(text, pos);
        while (pos < text.size() && (text[pos] == ':' || std::isspace(static_cast<unsigned char>(text[pos])))) ++pos;
        char* end = nullptr;
        const long id = std::strtol(text.c_str() + pos, &end, 10);
        if (end != text.c_str() + pos) {
            vocab[id] = std::move(token);
            pos = static_cast<size_t>(end - text.c_str());
        }
    }
    return vocab;
}

std::vector<uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = static_cast<uint8_t>(s[i]);
        if (c < 0x80) {
            out.push_back(c);
            ++i;
        } else if ((c >> 5) == 0x6 && i + 1 < s.size()) {
            out.push_back(((c & 0x1F) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3F));
            i += 2;
        } else if ((c >> 4) == 0xE && i + 2 < s.size()) {
            out.push_back(((c & 0x0F) << 12) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                          (static_cast<uint8_t>(s[i + 2]) & 0x3F));
            i += 3;
        } else if ((c >> 3) == 0x1E && i + 3 < s.size()) {
            out.push_back(((c & 0x07) << 18) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                          ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) |
                          (static_cast<uint8_t>(s[i + 3]) & 0x3F));
            i += 4;
        } else {
            ++i;
        }
    }
    return out;
}

std::unordered_map<uint32_t, uint8_t> byte_decoder() {
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    std::unordered_map<uint32_t, uint8_t> map;
    for (size_t i = 0; i < bs.size(); ++i) map[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    return map;
}

std::string decode_tokens(const std::vector<int64_t>& tokens, const std::unordered_map<int64_t, std::string>& vocab) {
    static const auto decoder = byte_decoder();
    std::string bytes;
    for (const int64_t id : tokens) {
        if (id >= kEos) continue;
        const auto it = vocab.find(id);
        if (it == vocab.end()) continue;
        for (const uint32_t cp : utf8_codepoints(it->second)) {
            const auto bit = decoder.find(cp);
            if (bit != decoder.end()) bytes.push_back(static_cast<char>(bit->second));
        }
    }
    while (!bytes.empty() && std::isspace(static_cast<unsigned char>(bytes.front()))) bytes.erase(bytes.begin());
    while (!bytes.empty() && std::isspace(static_cast<unsigned char>(bytes.back()))) bytes.pop_back();
    return bytes;
}

Ort::Value tensor_float(std::vector<float>& data, const std::vector<int64_t>& shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
}

Ort::Value tensor_int64(std::vector<int64_t>& data, const std::vector<int64_t>& shape) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(), shape.data(), shape.size());
}

std::vector<std::string> input_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    }
    return names;
}

std::vector<std::string> output_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    }
    return names;
}

Ort::Session make_session(Ort::Env& env, const fs::path& model, const EngineOptions& options,
                          const fs::path& config, const std::string& cache_key,
                          std::chrono::steady_clock::time_point t0) {
    const auto step0 = std::chrono::steady_clock::now();
    progress(t0, "session options begin: " + model.filename().string());
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (options.device == Device::NPU) {
        std::unordered_map<std::string, std::string> vitis_opts;
        vitis_opts["config_file"] = config.string();
        if (!options.cache_dir.empty()) {
            vitis_opts["cache_dir"] = options.cache_dir;
            vitis_opts["cache_key"] = cache_key;
        }
        progress(t0, "append VitisAI EP begin: key=" + cache_key +
                         " cache=" + (options.cache_dir.empty() ? std::string("(disabled)") : options.cache_dir));
        so.AppendExecutionProvider_VitisAI(vitis_opts);
        progress(t0, "append VitisAI EP done: key=" + cache_key +
                         " step=" + std::to_string(seconds_since(step0)) + "s");
    } else {
        progress(t0, "CPUExecutionProvider session path: " + model.filename().string());
    }

    progress(t0, "Ort::Session create begin: " + model.filename().string());
    Ort::Session session(env, model.c_str(), so);
    progress(t0, "Ort::Session create done: " + model.filename().string() +
                     " step=" + std::to_string(seconds_since(step0)) + "s");
    return session;
}

class AmdRyzenAiEngine final : public IWhisperEngine {
public:
    explicit AmdRyzenAiEngine(const EngineOptions& options)
        : env_(ORT_LOGGING_LEVEL_WARNING, "npu_inference_bench_amd"),
          options_(options),
          inference_precision_(precision_policy::resolve(options)) {
        const fs::path model_dir(options.model_dir);
        if (options.device == Device::GPU) {
            throw std::runtime_error("AMD backend currently supports CPU and NPU; GPU is not wired");
        }
        if (options.device == Device::NPU && inference_precision_ != "preferred") {
            throw std::runtime_error(
                "AMD VitisAI precision is fixed by its compiled model/config; "
                "use precision=preferred");
        }
        if (options.device == Device::CPU && inference_precision_ != "f32" &&
            inference_precision_ != "preferred") {
            throw std::runtime_error(
                "ORT CPU cannot apply a provider-wide precision conversion; "
                "use f32/preferred or export another model");
        }

        encoder_path_ = model_dir / "tiny_encoder.onnx";
        decoder_path_ = model_dir / "tiny_decoder.onnx";
        if (!fs::exists(encoder_path_) || !fs::exists(decoder_path_)) {
            throw std::runtime_error(
                "AMD model_dir must contain tiny_encoder.onnx and tiny_decoder.onnx: " + options.model_dir);
        }

        encoder_config_ = model_dir / "vitisai_config_whisper_encoder.json";
        decoder_config_ = model_dir / "vitisai_config_whisper_decoder.json";
        if (!fs::exists(encoder_config_)) encoder_config_ = model_dir / "vitisai_config.json";
        if (!fs::exists(decoder_config_)) decoder_config_ = model_dir / "vitisai_config.json";
        if (options.device == Device::NPU && (!fs::exists(encoder_config_) || !fs::exists(decoder_config_))) {
            throw std::runtime_error(
                "AMD NPU model_dir must contain VitisAI config JSON files "
                "(vitisai_config_whisper_encoder.json / decoder.json or vitisai_config.json)");
        }

        const auto t0 = std::chrono::steady_clock::now();
        progress(t0, "AMD engine load begin: model_dir=" + model_dir.string() +
                     " device=" + std::string(options.device == Device::NPU ? "NPU" : "CPU"));
        const auto parse0 = std::chrono::steady_clock::now();
        progress(t0, "parse mel filters begin");
        mel_filters_ = parse_mel_filters(model_dir / "preprocessor_config.json");
        progress(t0, "parse mel filters done: step=" + std::to_string(seconds_since(parse0)) + "s");
        const auto vocab0 = std::chrono::steady_clock::now();
        progress(t0, "load vocab begin");
        vocab_ = load_vocab(model_dir / "vocab.json");
        progress(t0, "load vocab done: step=" + std::to_string(seconds_since(vocab0)) + "s");
        progress(t0, "encoder session begin");
        encoder_ = std::make_unique<Ort::Session>(
            make_session(env_, encoder_path_, options_, encoder_config_, "whisper_tiny_amd_encoder", t0));
        progress(t0, "encoder session done");
        progress(t0, "decoder session begin");
        decoder_ = std::make_unique<Ort::Session>(
            make_session(env_, decoder_path_, options_, decoder_config_, "whisper_tiny_amd_decoder", t0));
        progress(t0, "decoder session done");
        load_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        progress(t0, "AMD engine load done: total=" + std::to_string(load_seconds_) + "s");
    }

    TranscribeResult transcribe(const AudioSamples& audio) override {
        const auto t0 = std::chrono::steady_clock::now();

        auto features = log_mel_spectrogram(audio, mel_filters_);
        auto enc_inputs = input_names(*encoder_);
        auto enc_outputs = output_names(*encoder_);
        std::vector<const char*> enc_in = {enc_inputs[0].c_str()};
        std::vector<const char*> enc_out = {enc_outputs[0].c_str()};
        auto feature_tensor = tensor_float(features, {1, kMelBins, kFrames});
        auto encoder_outputs =
            encoder_->Run(Ort::RunOptions{nullptr}, enc_in.data(), &feature_tensor, 1, enc_out.data(), 1);
        float* encoder_ptr = encoder_outputs[0].GetTensorMutableData<float>();
        const size_t encoder_count = encoder_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> encoder_state(encoder_ptr, encoder_ptr + encoder_count);

        std::vector<int64_t> tokens = {kSot, kLangEn, kTranscribe, kNoTimestamps};
        std::vector<int64_t> token_buffer(kMaxTokens, kEos);
        auto dec_inputs = input_names(*decoder_);
        auto dec_outputs = output_names(*decoder_);
        std::vector<const char*> dec_in(dec_inputs.size());
        for (size_t i = 0; i < dec_inputs.size(); ++i) dec_in[i] = dec_inputs[i].c_str();
        std::vector<const char*> dec_out = {dec_outputs[0].c_str()};

        for (int step = static_cast<int>(tokens.size()); step < kMaxTokens; ++step) {
            std::fill(token_buffer.begin(), token_buffer.end(), kEos);
            std::copy(tokens.begin(), tokens.end(), token_buffer.begin());

            auto token_tensor = tensor_int64(token_buffer, {1, kMaxTokens});
            auto state_tensor = tensor_float(encoder_state, {1, 1500, 384});
            std::vector<Ort::Value> inputs;
            if (dec_inputs[0] == "x") {
                inputs.emplace_back(std::move(token_tensor));
                inputs.emplace_back(std::move(state_tensor));
            } else {
                inputs.emplace_back(std::move(state_tensor));
                inputs.emplace_back(std::move(token_tensor));
            }

            auto outputs =
                decoder_->Run(Ort::RunOptions{nullptr}, dec_in.data(), inputs.data(), inputs.size(), dec_out.data(), 1);
            float* logits = outputs[0].GetTensorMutableData<float>();
            const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            const int64_t vocab_size = shape.back();
            const int64_t row = static_cast<int64_t>(tokens.size()) - 1;
            float* row_ptr = logits + row * vocab_size;
            const int64_t next = std::max_element(row_ptr, row_ptr + vocab_size) - row_ptr;
            if (next == kEos) break;
            tokens.push_back(next);
        }

        std::vector<int64_t> text_tokens(tokens.begin() + 4, tokens.end());
        TranscribeResult out;
        out.text = decode_tokens(text_tokens, vocab_);
        out.infer_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        out.runtime = "onnxruntime-vitisai";
        out.model_format = "onnx";
        out.decode_strategy = "static-no-kv";
        out.max_context = static_cast<long>(kMaxTokens);
        return out;
    }

    std::string backend_name() const override { return "AMD Ryzen AI (ONNX Runtime + VitisAI EP)"; }
    std::string device_name() const override {
        return options_.device == Device::NPU ? "XDNA NPU (VitisAI EP)" : "CPUExecutionProvider";
    }
    std::string runtime_version() const override { return Ort::GetVersionString(); }
    double load_seconds() const override { return load_seconds_; }
    ExecutionDiagnostics execution_diagnostics() const override {
        ExecutionDiagnostics diagnostics;
        diagnostics.requested_provider =
            options_.device == Device::NPU ? "VitisAIExecutionProvider"
                                           : "CPUExecutionProvider";
        diagnostics.resolved_provider = diagnostics.requested_provider;
        diagnostics.inference_precision = inference_precision_;
        return diagnostics;
    }

private:
    Ort::Env env_;
    EngineOptions options_;
    std::string inference_precision_;
    fs::path encoder_path_;
    fs::path decoder_path_;
    fs::path encoder_config_;
    fs::path decoder_config_;
    std::vector<float> mel_filters_;
    std::unordered_map<int64_t, std::string> vocab_;
    double load_seconds_ = 0.0;
    std::unique_ptr<Ort::Session> encoder_;
    std::unique_ptr<Ort::Session> decoder_;
};

}  // namespace

std::unique_ptr<IWhisperEngine> create(const EngineOptions& options) {
    return std::make_unique<AmdRyzenAiEngine>(options);
}

bool available() { return true; }

#else  // !NPU_INFERENCE_BENCH_AMD

std::unique_ptr<IWhisperEngine> create(const EngineOptions&) {
    throw std::runtime_error(
        "AMD Ryzen AI backend not compiled. Install the Ryzen AI SW stack "
        "(ONNX Runtime + VitisAI EP), set EnableAmd=true, and provide the ORT SDK "
        "path (eng/msbuild/backend.amd.props).");
}

bool available() { return false; }

#endif

}  // namespace amd
}  // namespace npu_inference_bench
