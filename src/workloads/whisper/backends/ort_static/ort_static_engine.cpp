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
    int sample_rate = 16000;
    int product_samples = 0;  // 0 = package declares no product window
    int product_hop = 0;      // stride between window starts (window - overlap)
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

    // Product window: the caller streams fixed-length windows rather than whole
    // utterances, so audio longer than one window is decoded as several overlapping
    // windows. Overlap exists so a word straddling a boundary is heard intact by at
    // least one window.
    cfg.sample_rate = static_cast<int>(fe::json_int(text, "sample_rate", cfg.sample_rate));
    if (cfg.sample_rate <= 0) cfg.sample_rate = 16000;
    cfg.product_samples = static_cast<int>(fe::json_int(text, "product_n_samples", 0));
    if (cfg.product_samples <= 0) {
        const int64_t window_s = fe::json_int(text, "product_window_s", 0);
        if (window_s > 0) cfg.product_samples = static_cast<int>(window_s * cfg.sample_rate);
    }
    // A product window wider than the mel canvas would be silently truncated anyway.
    if (cfg.product_samples > cfg.n_samples) cfg.product_samples = cfg.n_samples;
    if (cfg.product_samples > 0) {
        const int64_t overlap_s = fe::json_int(text, "product_overlap_s", 1);
        int overlap = static_cast<int>(overlap_s * cfg.sample_rate);
        overlap = (std::max)(0, (std::min)(overlap, cfg.product_samples - cfg.sample_rate / 10));
        cfg.product_hop = cfg.product_samples - overlap;
    }
    return cfg;
}

int64_t env_max_tokens(int64_t cap) {
    const char* v = std::getenv("NPU_INFERENCE_BENCH_ORT_MAX_TOKENS");
    if (!v || !*v) return cap;
    const long n = std::strtol(v, nullptr, 10);
    return (std::max<long>)(2, (std::min<long>)(n, static_cast<long>(cap)));
}

// Parse the whole lang_to_id map ("<|en|>": 50259, ...) so the decoder can pick
// the language from the audio instead of assuming one.
std::vector<std::pair<std::string, int64_t>> parse_language_table(const std::string& gc) {
    std::vector<std::pair<std::string, int64_t>> table;
    const size_t map_start = gc.find("\"lang_to_id\"");
    if (map_start == std::string::npos) return table;
    const size_t open = gc.find('{', map_start);
    if (open == std::string::npos) return table;
    const size_t close = gc.find('}', open);
    size_t pos = open;
    while (true) {
        const size_t key = gc.find("\"<|", pos);
        if (key == std::string::npos || (close != std::string::npos && key > close)) break;
        const size_t key_end = gc.find("|>\"", key);
        if (key_end == std::string::npos) break;
        const std::string code = gc.substr(key + 3, key_end - (key + 3));
        const size_t colon = gc.find(':', key_end);
        if (colon == std::string::npos) break;
        char* end = nullptr;
        const long id = std::strtol(gc.c_str() + colon + 1, &end, 10);
        if (end != gc.c_str() + colon + 1 && !code.empty()) {
            table.emplace_back(code, static_cast<int64_t>(id));
        }
        pos = colon + 1;
    }
    return table;
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

// Task id from generation_config.json's task_to_id map
// ("task_to_id": {"transcribe": 50359, "translate": 50358}), which is what the HF
// exporter actually writes. Older packages carried scalar <task>_token_id keys, so
// those are still honoured before falling back to the hardcoded id.
int64_t task_token_id(const std::string& gc, const std::string& task, int64_t fallback) {
    const size_t map_start = gc.find("\"task_to_id\"");
    if (map_start != std::string::npos) {
        const size_t open = gc.find('{', map_start);
        if (open != std::string::npos) {
            const size_t close = gc.find('}', open);
            const std::string key = "\"" + task + "\"";
            const size_t pos = gc.find(key, open);
            if (pos != std::string::npos && (close == std::string::npos || pos < close)) {
                const size_t colon = gc.find(':', pos + key.size());
                if (colon != std::string::npos) {
                    char* end = nullptr;
                    const long id = std::strtol(gc.c_str() + colon + 1, &end, 10);
                    if (end != gc.c_str() + colon + 1) return static_cast<int64_t>(id);
                }
            }
        }
    }
    return fe::json_int(gc, task + "_token_id", fallback);
}

// Overlapping windows re-transcribe the shared audio, so consecutive transcripts
// repeat the words in the overlap. The seam is found by sliding the head of the new
// transcript against the tail of what we have and keeping the alignment that agrees
// on the most words.
//
// The agreement is deliberately fuzzy: the two windows heard the shared audio with
// different amounts of context, so they routinely render a word or two differently
// ("zum Rest" against "zum Reste"). Demanding an exact run means one such word
// rejects the whole alignment and the overlap gets emitted twice.
constexpr size_t kMaxOverlapWords = 48;
constexpr double kOverlapMatchRatio = 0.6;

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
        const size_t start = pos;
        while (pos < text.size() && !std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
        if (pos > start) words.push_back(text.substr(start, pos - start));
    }
    return words;
}

// Case- and punctuation-insensitive comparison key. Only ASCII is folded: UTF-8
// continuation bytes must pass through untouched or accented words stop matching.
std::string overlap_key(const std::string& word) {
    std::string out;
    out.reserve(word.size());
    for (const char ch : word) {
        const auto uch = static_cast<unsigned char>(ch);
        if (uch < 0x80) {
            if (std::isalnum(uch)) out.push_back(static_cast<char>(std::tolower(uch)));
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

// Two windows heard the same word with different context and often spell it slightly
// differently, so the seam has to survive near-misses: comparing keys for equality
// loses the alignment in exactly the garbled cases that need it most. Edit distance is
// counted in bytes, so a differing accented letter costs two -- that only makes the
// test stricter, never looser.
bool words_agree(const std::string& a, const std::string& b) {
    if (a == b) return true;
    if (a.empty() || b.empty()) return false;
    const size_t longest = (std::max)(a.size(), b.size());
    const size_t budget = (longest / 4 > 1) ? longest / 4 : size_t{1};
    if (a.size() > b.size() + budget || b.size() > a.size() + budget) return false;

    std::vector<size_t> prev(b.size() + 1);
    std::vector<size_t> cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        size_t row_best = cur[0];
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t cost = (a[i - 1] == b[j - 1]) ? 0u : 1u;
            cur[j] = (std::min)((std::min)(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            row_best = (std::min)(row_best, cur[j]);
        }
        if (row_best > budget) return false;  // no completion can come back under budget
        prev.swap(cur);
    }
    return prev[b.size()] <= budget;
}

std::string stitch_transcripts(const std::string& accumulated, const std::string& next) {
    if (accumulated.empty()) return next;
    if (next.empty()) return accumulated;
    const std::vector<std::string> have_raw = split_words(accumulated);
    const std::vector<std::string> add = split_words(next);
    std::vector<std::string> have;
    have.reserve(have_raw.size());
    for (const std::string& word : have_raw) have.push_back(overlap_key(word));
    std::vector<std::string> add_keys;
    add_keys.reserve(add.size());
    for (const std::string& word : add) add_keys.push_back(overlap_key(word));
    const size_t limit = (std::min)((std::min)(have.size(), add.size()), kMaxOverlapWords);

    // Descending k so that, among alignments agreeing on equally many words, the
    // longest overlap wins -- a long agreement is stronger evidence of the seam.
    size_t shared = 0;
    size_t best_hits = 0;
    for (size_t k = limit; k >= 1; --k) {
        size_t hits = 0;
        for (size_t i = 0; i < k; ++i) {
            if (words_agree(have[have.size() - k + i], add_keys[i])) ++hits;
        }
        if (static_cast<double>(hits) < kOverlapMatchRatio * static_cast<double>(k)) continue;
        if (hits > best_hits) {
            best_hits = hits;
            shared = k;
        }
    }

    std::string out = accumulated;
    for (size_t i = shared; i < add.size(); ++i) {
        out.push_back(' ');
        out += add[i];
    }
    return out;
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

        // The product streams fixed windows, so that is the default whenever the
        // package declares one. "full" decodes the whole clip in a single pass (the
        // old behaviour), which is only useful for measuring what windowing costs.
        const char* env_window = std::getenv("NPU_INFERENCE_BENCH_WHISPER_WINDOW");
        const std::string window_mode = (env_window && *env_window) ? env_window : "product";
        if (window_mode != "product" && window_mode != "full") {
            throw std::runtime_error(
                "NPU_INFERENCE_BENCH_WHISPER_WINDOW=" + window_mode +
                " is not a window mode (use 'product' or 'full')");
        }
        windowed_ = (window_mode == "product") && pkg_.product_samples > 0 && pkg_.product_hop > 0;
        strategy_ = pkg_.multilingual ? "static-no-kv-multi-7s" : "static-no-kv";
        if (windowed_) {
            const int window_s = pkg_.product_samples / pkg_.sample_rate;
            const int overlap_s = (pkg_.product_samples - pkg_.product_hop) / pkg_.sample_rate;
            strategy_ += "-sliding" + std::to_string(window_s) + "s-ov" + std::to_string(overlap_s) + "s";
        }

        mel_filters_ = fe::parse_mel_filters(dir / "preprocessor_config.json");
        vocab_ = fe::load_vocab(dir / "vocab.json");
        const std::string gc = fe::read_text(dir / "generation_config.json");
        // Multilingual tiny uses decoder_start_token_id=50258; tiny.en uses 50257.
        sot_ = fe::json_int(gc, "decoder_start_token_id", pkg_.multilingual ? 50258 : 50257);
        eos_ = fe::json_int(gc, "eos_token_id", 50256);
        pad_ = fe::json_int(gc, "pad_token_id", eos_);
        no_timestamps_ = fe::json_int(gc, "no_timestamps_token_id", pkg_.multilingual ? 50363 : 50362);
        // <|transcribe|> / <|translate|> — multilingual only
        transcribe_ = task_token_id(gc, "transcribe", 50359);
        translate_ = task_token_id(gc, "translate", 50358);
        if (pkg_.multilingual) {
            if (transcribe_ == translate_) {
                throw std::runtime_error(
                    "generation_config.json gives <|transcribe|> and <|translate|> the same id (" +
                    std::to_string(transcribe_) + "): task selection would be meaningless");
            }
            // Transcribe keeps the speaker's own language (what the product wants);
            // translate asks Whisper for English out of foreign speech.
            const char* env_task = std::getenv("NPU_INFERENCE_BENCH_WHISPER_TASK");
            task_ = (env_task && *env_task) ? env_task : "transcribe";
            if (task_ != "transcribe" && task_ != "translate") {
                throw std::runtime_error(
                    "NPU_INFERENCE_BENCH_WHISPER_TASK=" + task_ +
                    " is not a Whisper task (use 'transcribe' or 'translate')");
            }
            // Translation needs the whole utterance: rendering half a sentence into
            // English, then the other half, and gluing the fragments measured 203% WER
            // against 92% for a single pass. Refuse the combination rather than return
            // a transcript that looks plausible and is not.
            if (windowed_ && task_ == "translate") {
                throw std::runtime_error(
                    "NPU_INFERENCE_BENCH_WHISPER_TASK=translate cannot be combined with the "
                    "product window (" + std::to_string(pkg_.product_samples / pkg_.sample_rate) +
                    "s): stitched fragments are not a translation. Set "
                    "NPU_INFERENCE_BENCH_WHISPER_WINDOW=full to translate whole clips.");
            }
            // Language is detected per clip unless pinned. Defaulting to <|en|> makes
            // Whisper paraphrase non-English speech into English instead of
            // transcribing it verbatim, so an unset env var means "auto".
            lang_table_ = parse_language_table(gc);
            const char* env_lang = std::getenv("NPU_INFERENCE_BENCH_WHISPER_LANG");
            const std::string want = (env_lang && *env_lang) ? env_lang : "auto";
            if (want != "auto") {
                lang_token_ = language_token_id(gc, want);
                if (lang_token_ < 0) {
                    throw std::runtime_error(
                        "NPU_INFERENCE_BENCH_WHISPER_LANG=" + want +
                        " is not a language token of this model (use 'auto' or an ISO code)");
                }
                language_ = want;
            } else if (lang_table_.empty()) {
                throw std::runtime_error(
                    "multilingual model without a lang_to_id table in generation_config.json: "
                    "cannot detect language, pin one via NPU_INFERENCE_BENCH_WHISPER_LANG");
            }
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
        // Language is detected on the first window and reused: re-detecting per window
        // lets one noisy window switch language mid-utterance.
        detected_token_ = -1;

        const std::vector<std::vector<float>> windows = split_windows(audio);
        std::string text;
        long total_tokens = 0;
        double total_logprob = 0.0;
        for (const std::vector<float>& window : windows) {
            const WindowDecode decoded = decode_window(window);
            text = stitch_transcripts(text, decoded.text);
            total_tokens += decoded.generated_tokens;
            total_logprob += decoded.logprob_sum;
        }

        TranscribeResult out;
        out.text = text;
        out.infer_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        out.generated_tokens = total_tokens;
        out.sequence_logprob = total_logprob;
        if (total_tokens > 0) {
            out.avg_logprob = total_logprob / static_cast<double>(total_tokens);
            out.throughput_tps = static_cast<double>(total_tokens) / out.infer_seconds;
            out.tpot_ms = out.infer_seconds * 1000.0 / static_cast<double>(total_tokens);
        }
        out.ttft_ms = -1.0;
        out.has_token_metrics = true;
        out.runtime = ep::runtime_for(active_provider_);
        out.model_format = "onnx";
        out.decode_strategy = strategy_;
        out.max_context = static_cast<long>(max_tokens_);
        out.language = language_;
        out.task = pkg_.multilingual ? task_ : "";
        out.windows = static_cast<long>(windows.size());

        // Finalize profiling after the first transcription (normally the harness
        // warmup), so measured runs do not accumulate profiler overhead.
        (void)execution_diagnostics();
        return out;
    }

    std::string backend_name() const override { return "ONNX Runtime (unified, static)"; }

private:
    struct WindowDecode {
        std::string text;
        long generated_tokens = 0;
        double logprob_sum = 0.0;
    };

    // Cut the clip into product-sized windows. Audio that already fits goes through
    // untouched, so short-utterance behaviour is unchanged.
    std::vector<std::vector<float>> split_windows(const AudioSamples& audio) const {
        const size_t total = audio.size();
        const size_t window = static_cast<size_t>(pkg_.product_samples);
        if (!windowed_ || total == 0 || total <= window) {
            return {std::vector<float>(audio.begin(), audio.end())};
        }
        const size_t hop = static_cast<size_t>(pkg_.product_hop);
        const size_t min_new = static_cast<size_t>(pkg_.sample_rate) / 4;

        std::vector<std::vector<float>> out;
        size_t covered = 0;
        for (size_t start = 0;; start += hop) {
            if (start + window < total) {
                out.emplace_back(audio.begin() + static_cast<std::ptrdiff_t>(start),
                                 audio.begin() + static_cast<std::ptrdiff_t>(start + window));
                covered = start + window;
                continue;
            }
            // Final window is aligned to the end of the audio rather than left as a
            // stub: a fraction-of-a-second window invites hallucinated tokens. If it
            // would add essentially no new audio, the previous window already had it.
            if (!out.empty() && total - covered < min_new) break;
            out.emplace_back(audio.begin() + static_cast<std::ptrdiff_t>(total - window),
                             audio.end());
            break;
        }
        return out;
    }

    WindowDecode decode_window(const std::vector<float>& audio) {
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

        int64_t lang_token = lang_token_;
        if (pkg_.multilingual && lang_token < 0) {
            if (detected_token_ < 0) detected_token_ = detect_language(encoder_state);
            lang_token = detected_token_;
        }

        std::vector<int64_t> ids(static_cast<size_t>(max_tokens_), pad_);
        int64_t cur = 0;
        ids[static_cast<size_t>(cur++)] = sot_;
        if (pkg_.multilingual) {
            if (lang_token >= 0) ids[static_cast<size_t>(cur++)] = lang_token;
            ids[static_cast<size_t>(cur++)] = (task_ == "translate") ? translate_ : transcribe_;
        }
        ids[static_cast<size_t>(cur++)] = no_timestamps_;
        std::vector<int64_t> generated;
        double logprob_sum = 0.0;
        long n_gen = 0;
        bool first = true;

        const char* dec_in[] = {"input_ids", "encoder_hidden_states"};
        const char* dec_out[] = {"logits"};
        Ort::RunOptions run_opts;
        configure_run_options(run_opts);
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

        WindowDecode out;
        out.text = fe::decode_tokens(generated, vocab_, eos_);
        out.generated_tokens = n_gen;
        out.logprob_sum = logprob_sum;
        return out;
    }

public:
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
    void configure_run_options(Ort::RunOptions& run_opts) const {
        if (options_.device == Device::NPU && ep::is_qnn(active_provider_)) {
            run_opts.AddConfigEntry("qnn.perf_mode", "burst");
        }
    }

    // Whisper language ID: one decoder step with only <|startoftranscript|>, then
    // take the highest-scoring <|xx|> token. Same algorithm as the reference
    // whisper.detect_language(); costs one extra decoder forward per clip.
    int64_t detect_language(std::vector<float>& encoder_state) {
        std::vector<int64_t> ids(static_cast<size_t>(max_tokens_), pad_);
        ids[0] = sot_;
        auto token_tensor = tensor_int64(ids, {1, max_tokens_});
        auto state_tensor = tensor_float(encoder_state, {1, pkg_.enc_seq, pkg_.d_model});
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(token_tensor));
        inputs.emplace_back(std::move(state_tensor));

        const char* dec_in[] = {"input_ids", "encoder_hidden_states"};
        const char* dec_out[] = {"logits"};
        Ort::RunOptions run_opts;
        configure_run_options(run_opts);
        auto outputs = decoder_->Run(run_opts, dec_in, inputs.data(), inputs.size(), dec_out, 1);
        const float* logits = outputs[0].GetTensorMutableData<float>();
        const int64_t vocab = outputs[0].GetTensorTypeAndShapeInfo().GetShape().back();

        int64_t best_token = -1;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (const auto& entry : lang_table_) {
            if (entry.second < 0 || entry.second >= vocab) continue;
            const float value = logits[entry.second];
            if (value > best_logit) {
                best_logit = value;
                best_token = entry.second;
                language_ = entry.first;
            }
        }
        return best_token;
    }

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
    int64_t lang_token_ = -1;  // >= 0 only when pinned; otherwise detected per clip
    std::vector<std::pair<std::string, int64_t>> lang_table_;
    std::string language_;  // pinned or last detected ISO code
    std::string task_ = "transcribe";
    int64_t detected_token_ = -1;  // language detected for the clip being transcribed
    bool windowed_ = false;
    std::string strategy_ = "static-no-kv";
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
