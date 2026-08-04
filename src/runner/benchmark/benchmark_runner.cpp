// npu_inference_bench CLI: multi-workload inference benchmark runner.
//
// Usage: npu_inference_bench <model_dir> <audio.wav> [backend] [device] [runs]
//                    [--cache <dir>] [--ref "<reference text>"] [--threads N] [--json]
//                    [--results <csv>] [--label <text>]
//   backend: auto | intel | amd | qualcomm     (default: auto)
//   device : npu  | gpu | cpu                   (default: npu)
//   runs   : timed iterations                   (default: 5)
//   --cache <dir> : persist compiled model; loads twice (cold vs hot)
//   --hot-only    : load once from a populated --cache (skip cold compile; cold=n/a)
//   --ref "<text>": reference transcript -> compute WER/CER
//   --threads N   : CPU inference thread count (CPU device only; default: runtime default)
//   --json        : emit one machine-readable JSON record (for the harness)
//   --results <csv>: append a shared-schema benchmark result row
//   --label <text>: tag the results row (e.g. quantization variant)
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>  // GetSystemPowerStatus (AC vs battery detection)
#endif

#include "npu_inference_bench/audio.hpp"
#include "npu_inference_bench/benchmark_runner.hpp"
#include "npu_inference_bench/benchmark_meta.hpp"
#include "npu_inference_bench/classifier.hpp"
#include "npu_inference_bench/metrics.hpp"
#include "npu_inference_bench/model_hash.hpp"
#include "npu_inference_bench/whisper.hpp"

namespace {

#include "benchmark_support.ipp"
#include "classifier_benchmark.ipp"

void append_utf8(std::string& out, unsigned int cp) {
    if (cp < 0x80u) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

unsigned int parse_hex4(const std::string& text, size_t& pos) {
    unsigned int value = 0;
    for (int i = 0; i < 4; ++i) {
        if (pos >= text.size()) throw std::runtime_error("truncated \\u escape");
        const char ch = text[pos++];
        value <<= 4;
        if (ch >= '0' && ch <= '9') value |= static_cast<unsigned>(ch - '0');
        else if (ch >= 'a' && ch <= 'f') value |= static_cast<unsigned>(ch - 'a' + 10);
        else if (ch >= 'A' && ch <= 'F') value |= static_cast<unsigned>(ch - 'A' + 10);
        else throw std::runtime_error("bad \\u escape");
    }
    return value;
}

std::string parse_json_text(const std::string& text, size_t& pos) {
    if (pos >= text.size() || text[pos] != '"') throw std::runtime_error("expected string");
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
        switch (esc) {
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'u': {
                unsigned int cp = parse_hex4(text, pos);
                // Surrogate pair: ConvertTo-Json emits these for astral characters.
                if (cp >= 0xD800u && cp <= 0xDBFFu && pos + 1 < text.size() &&
                    text[pos] == '\\' && text[pos + 1] == 'u') {
                    pos += 2;
                    const unsigned int low = parse_hex4(text, pos);
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (low - 0xDC00u);
                }
                append_utf8(out, cp);
                break;
            }
            default: out.push_back(esc); break;
        }
    }
    throw std::runtime_error("unterminated string");
}

void skip_json_value(const std::string& text, size_t& pos) {
    if (pos >= text.size()) return;
    if (text[pos] == '"') {
        parse_json_text(text, pos);
        return;
    }
    if (text[pos] == '{' || text[pos] == '[') {
        int depth = 0;
        while (pos < text.size()) {
            const char ch = text[pos];
            if (ch == '"') {
                parse_json_text(text, pos);
                continue;
            }
            ++pos;
            if (ch == '{' || ch == '[') ++depth;
            else if (ch == '}' || ch == ']') {
                if (--depth == 0) return;
            }
        }
        return;
    }
    while (pos < text.size() && text[pos] != ',' && text[pos] != '}') ++pos;
}

// String-valued members of one flat JSON object. Parsed member-by-member rather than
// by searching for `"key"`, so a key name appearing inside a reference transcript
// cannot be mistaken for the field itself.
std::unordered_map<std::string, std::string> json_object_strings(const std::string& line) {
    std::unordered_map<std::string, std::string> out;
    size_t pos = line.find('{');
    if (pos == std::string::npos) return out;
    ++pos;
    while (pos < line.size()) {
        while (pos < line.size() && (std::isspace(static_cast<unsigned char>(line[pos])) ||
                                     line[pos] == ',')) {
            ++pos;
        }
        if (pos >= line.size() || line[pos] == '}') break;
        if (line[pos] != '"') break;
        const std::string key = parse_json_text(line, pos);
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
        if (pos >= line.size() || line[pos] != ':') break;
        ++pos;
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
        if (pos < line.size() && line[pos] == '"') {
            out[key] = parse_json_text(line, pos);
        } else {
            skip_json_value(line, pos);
        }
    }
    return out;
}
}  // namespace

int npu_inference_bench::benchmark::run_cli(int argc, char* argv[]) {
    using namespace npu_inference_bench;

    struct EvalClipSpec {
        std::string id;
        std::string audio;
        std::string reference;
    };
    std::vector<std::string> pos;
    std::vector<EvalClipSpec> eval_clips;
    std::string cache_dir, reference, results_csv, label, provider_override, classify_dir;
    std::string json_output_path;
    bool json_out = false, have_ref = false, hot_only = false;
    int cpu_threads = 0;
    if (argc > 1 && std::string(argv[1]) == "suite") {
        std::cerr << "Suite orchestration lives in benchmark\\run-suite.ps1 so one manifest "
                     "drives every workload/device attempt.\n";
        return 1;
    }
    if (argc < 3 || std::string(argv[1]) != "run") {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " run whisper <model_dir> <audio.wav> [backend] [device] [runs] [options]\n"
                  << "  " << argv[0]
                  << " run <tsc|fakeaudio> <fixture_dir> [device] [runs] [options]\n"
                  << "  .\\benchmark\\run-suite.ps1   # manifest-driven complete suite\n";
        return 1;
    }
    const std::string workload = lower(argv[2]);
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cache" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if (a == "--ref" && i + 1 < argc) {
            reference = argv[++i];
            have_ref = true;
        } else if (a == "--eval-clip" && i + 3 < argc) {
            EvalClipSpec clip;
            clip.id = argv[++i];
            clip.audio = argv[++i];
            clip.reference = argv[++i];
            eval_clips.push_back(std::move(clip));
        } else if (a == "--eval-set" && i + 1 < argc) {
            // References routinely contain double quotes, which the Windows command
            // line silently truncates at -- a reference cut short inflates WER without
            // any error. Reading clips from a UTF-8 JSONL file avoids argv entirely.
            const std::string set_path = argv[++i];
            std::ifstream in(set_path, std::ios::binary);
            if (!in) {
                std::cerr << "Failed to open eval set: " << set_path << "\n";
                return 1;
            }
            std::string line;
            int line_no = 0;
            while (std::getline(in, line)) {
                ++line_no;
                if (line_no == 1 && line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
                    static_cast<unsigned char>(line[1]) == 0xBB &&
                    static_cast<unsigned char>(line[2]) == 0xBF) {
                    line.erase(0, 3);
                }
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (line.find_first_not_of(" \t") == std::string::npos) continue;
                std::unordered_map<std::string, std::string> fields;
                try {
                    fields = json_object_strings(line);
                } catch (const std::exception& e) {
                    std::cerr << "Bad eval-set line " << line_no << " in " << set_path << ": "
                              << e.what() << "\n";
                    return 1;
                }
                EvalClipSpec clip;
                clip.id = fields.count("id") ? fields["id"] : "";
                clip.audio = fields.count("audio") ? fields["audio"] : "";
                clip.reference = fields.count("ref") ? fields["ref"] : "";
                if (clip.id.empty() || clip.audio.empty()) {
                    std::cerr << "Eval-set line " << line_no << " in " << set_path
                              << " needs both \"id\" and \"audio\"\n";
                    return 1;
                }
                eval_clips.push_back(std::move(clip));
            }
        } else if (a == "--threads" && i + 1 < argc) {
            cpu_threads = std::max(0, std::atoi(argv[++i]));
        } else if ((a == "--provider" || a == "--device-override") && i + 1 < argc) {
            provider_override = argv[++i];
        } else if (a == "--hot-only") {
            hot_only = true;
        } else if (a == "--json") {
            json_out = true;
        } else if (a == "--json-output" && i + 1 < argc) {
            json_output_path = argv[++i];
            json_out = true;
        } else if (a == "--results" && i + 1 < argc) {
            results_csv = argv[++i];
        } else if (a == "--label" && i + 1 < argc) {
            label = argv[++i];
        } else {
            pos.push_back(a);
        }
    }

    if (workload == "tsc" || workload == "fakeaudio") {
        if (pos.empty()) {
            std::cerr << "Classifier workload requires a fixture directory.\n";
            return 1;
        }
        classify_dir = pos.front();
        pos.erase(pos.begin());
    } else if (workload != "whisper") {
        std::cerr << "Unknown workload: " << workload << " (whisper|tsc|fakeaudio)\n";
        return 1;
    }

    // Classifier mode: replay pre-baked fixture tensors through ORT EPs.
    // Positionals here are [device] [runs]; no model_dir/audio needed.
    if (!classify_dir.empty()) {
        Device cdev = Device::CPU;
        if (!pos.empty() && !parse_device(pos[0], cdev)) {
            std::cerr << "Unknown device: " << pos[0] << " (npu|gpu|cpu)\n";
            return 1;
        }
        const int cruns = pos.size() > 1 ? std::max(1, std::atoi(pos[1].c_str())) : 20;
        return run_classify(classify_dir, cdev, provider_override, cruns, cpu_threads, cache_dir,
                            results_csv, json_out);
    }

    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " run whisper <model_dir> <audio.wav> [backend] [device] [runs]"
                     " [--cache <dir>] [--ref \"text\"] [--threads N] [--provider <ort-ep>] [--json] [--results <csv>]\n"
                  << "  backend: auto | intel | intel-onnx | onnx-static | onnx-dynamic | onnx-sherpa | amd | qualcomm   (default auto)\n"
                  << "  device : npu | gpu | cpu                 (default npu)\n"
                  << "  runs   : timed iterations                (default 5)\n"
                  << "  --cache <dir>: persist compiled model; loads twice (cold/hot)\n"
                  << "  --hot-only  : load once from a populated --cache (skip cold compile; cold=n/a)\n"
                  << "  --ref \"text\": reference transcript -> compute WER/CER\n"
                  << "  --eval-clip <id> <audio> <ref>: repeat to evaluate clips in one loaded session\n"
                  << "  --eval-set <file.jsonl>: same, read from UTF-8 JSONL {id,audio,ref} lines\n"
                  << "      (use this when references contain quotes -- argv would truncate them)\n"
                  << "  --threads N: CPU inference thread count (CPU device only)\n"
                  << "  --provider <ort-ep>: backend-specific provider override (e.g. VitisAIExecutionProvider)\n"
                  << "  --json: emit one machine-readable JSON record\n"
                  << "  --json-output <path>: write JSON to a file, isolated from provider logs\n"
                  << "  --results <csv>: append a benchmark result row\n"
                  << "  --label <text>: tag the results row (e.g. quantization variant)\n"
                  << "\nClassifiers: run <tsc|fakeaudio> <fixture_dir> [device] [runs]\n"
                  << "  Replays tools/fixtures/generate.py tensors through ORT providers.\n\n";
        std::cerr << "Compiled-in backends:";
        for (Backend b : available_backends()) std::cerr << " " << to_string(b);
        if (available_backends().empty()) std::cerr << " (none!)";
        std::cerr << "\n";
        return 1;
    }

    EngineOptions opt;
    opt.model_dir = pos[0];
    const std::string audio_path = pos[1];
    opt.cache_dir = cache_dir;
    opt.cpu_threads = cpu_threads;
    opt.device_override = provider_override;

    Backend backend = Backend::Auto;
    if (pos.size() > 2 && !parse_backend(pos[2], backend)) {
        std::cerr << "Unknown backend: " << pos[2] << "\n";
        return 1;
    }
    if (pos.size() > 3 && !parse_device(pos[3], opt.device)) {
        std::cerr << "Unknown device: " << pos[3] << "\n";
        return 1;
    }
    // Distinguish "auto" (let Backend::Auto self-pick) from an explicit NPU/GPU/CPU,
    // which parse_device also resolves to a concrete device but must be honored as-is.
    opt.auto_device = pos.size() > 3 && lower(pos[3]) == "auto";
    const int runs = pos.size() > 4 ? std::max(1, std::atoi(pos[4].c_str())) : 5;
    const int warmup = 1;
    const std::string requested_backend = to_string(backend);

    auto emit_json = [&](const std::string& payload) {
        if (json_output_path.empty()) {
            std::cout << payload << "\n";
            return;
        }
        std::ofstream output(json_output_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Failed to open JSON output: " + json_output_path);
        }
        output << payload << "\n";
        if (!output) {
            throw std::runtime_error("Failed to write JSON output: " + json_output_path);
        }
    };
    auto fail = [&](int code, const std::string& msg) {
        if (json_out) {
            emit_json("{\"ok\":false,\"error\":\"" + json_escape(msg) + "\"}");
        } else {
            std::cerr << msg << "\n";
        }
        return code;
    };

    WavData wav;
    try {
        wav = read_wav(audio_path);
    } catch (const std::exception& e) {
        return fail(2, std::string("Failed to read WAV: ") + e.what());
    }
    const double audio_len =
        static_cast<double>(wav.samples.size()) /
        static_cast<double>(wav.sample_rate ? wav.sample_rate : 16000);

    if (!json_out) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Audio  : " << audio_path << "  (" << audio_len << "s @ "
                  << wav.sample_rate << " Hz, " << wav.channels << " ch)\n";
        std::cout << "Backend: " << to_string(backend) << "   Device: " << to_string(opt.device)
                  << "   Runs: " << runs << "\n";
        std::cout << "Cache  : " << (cache_dir.empty() ? std::string("(disabled)") : cache_dir)
                  << "\n\n";
    }

    // cold_load == -1 signals "not measured" (hot-only mode); downstream tools treat
    // a negative load as N/A rather than a real zero-second cold start.
    std::unique_ptr<IWhisperEngine> engine;
    double cold_load = -1.0, warm_load = -1.0;

    if (hot_only) {
        // Hot-only: skip the cold compile entirely and load once straight from a
        // pre-populated cache. There is nothing to be "hot" from without a cache.
        if (cache_dir.empty()) {
            return fail(4, "--hot-only requires --cache <dir> with a populated compiled cache");
        }
        try {
            engine = create_engine(backend, opt);
            warm_load = engine->load_seconds();
        } catch (const std::exception& e) {
            return fail(3, std::string("Engine creation failed: ") + e.what());
        }
    } else {
        // Cold start (compiles; writes cache if enabled).
        opt.profile_execution = false;
        try {
            engine = create_engine(backend, opt);
            cold_load = engine->load_seconds();
        } catch (const std::exception& e) {
            return fail(3, std::string("Engine creation failed: ") + e.what());
        }

        // Hot start (imports cached blob) to quantify the caching win.
        if (!cache_dir.empty()) {
            try {
                const auto cold_diagnostics = engine->execution_diagnostics();
                if (!cold_diagnostics.resolved_provider.empty()) {
#ifdef NPU_INFERENCE_BENCH_WINML
                    // Keep the user-visible request as "auto" while pinning the
                    // exact catalog policy selected by the cold load.
                    opt.provider_order = {cold_diagnostics.resolved_provider};
#else
                    opt.device_override = cold_diagnostics.resolved_provider;
#endif
                }
                engine.reset();
                opt.profile_execution = true;
                auto hot = create_engine(backend, opt);
                warm_load = hot->load_seconds();
                engine = std::move(hot);
            } catch (const std::exception& e) {
                if (!json_out) std::cerr << "Hot reload failed: " << e.what() << "\n";
            }
        }
    }

    if (!eval_clips.empty()) {
        const double size_mb = model_size_mb(opt.model_dir);
        std::string model_sha256;
        try {
            model_sha256 = npu_inference_bench::model_artifacts_sha256(opt.model_dir);
        } catch (const std::exception& e) {
            return fail(5, std::string("Failed to hash model artifacts: ") + e.what());
        }

        std::ostringstream output;
        output << std::fixed << std::setprecision(6);
        output << "{\"ok\":true,\"clips\":[";
        for (std::size_t index = 0; index < eval_clips.size(); ++index) {
            const auto& spec = eval_clips[index];
            WavData clip_wav;
            try {
                clip_wav = read_wav(spec.audio);
            } catch (const std::exception& e) {
                return fail(2, std::string("Failed to read WAV '") + spec.id + "': " + e.what());
            }
            const double clip_audio_len =
                static_cast<double>(clip_wav.samples.size()) /
                static_cast<double>(clip_wav.sample_rate ? clip_wav.sample_rate : 16000);

            TranscribeResult clip_result;
            std::vector<double> clip_latencies;
            clip_latencies.reserve(static_cast<std::size_t>(runs));
            try {
                if (index == 0) engine->transcribe(clip_wav.samples);
                for (int run = 0; run < runs; ++run) {
                    clip_result = engine->transcribe(clip_wav.samples);
                    clip_latencies.push_back(clip_result.infer_seconds);
                }
            } catch (const std::exception& e) {
                return fail(4, std::string("Transcription failed for '") + spec.id + "': " + e.what());
            }
            const std::string clip_text = trim(clip_result.text);
            double clip_mean = 0.0;
            for (double latency : clip_latencies) clip_mean += latency;
            clip_mean /= static_cast<double>(clip_latencies.size());
            const double clip_median = percentile(clip_latencies, 50.0);
            const double clip_p90 = percentile(clip_latencies, 90.0);
            const double clip_rtf = clip_mean / clip_audio_len;
            const ErrorRate clip_error = compute_error_rate(spec.reference, clip_text);
            const auto clip_meta =
                npu_inference_bench::benchmark_meta::resolve(opt.model_dir, label, *engine, clip_result);

            if (index) output << ",";
            output << "{";
            output << "\"ok\":true";
            output << ",\"eval_id\":\"" << json_escape(spec.id) << "\"";
            output << ",\"backend\":\"" << json_escape(engine->backend_name()) << "\"";
            output << ",\"device\":\"" << json_escape(engine->device_name()) << "\"";
            output << ",\"device_full_name\":\"" << json_escape(engine->full_device_name()) << "\"";
            output << ",\"power_source\":\"" << json_escape(power_source()) << "\"";
            output << ",\"host_arch\":\"" << json_escape(host_arch()) << "\"";
            output << ",\"host_os\":\"" << json_escape(host_os()) << "\"";
            output << ",\"runtime_version\":\"" << json_escape(engine->runtime_version()) << "\"";
            output << ",\"inference_precision\":\""
                   << json_escape(engine->execution_diagnostics().inference_precision) << "\"";
            output << ",\"cpu_threads_requested\":" << cpu_threads;
            output << ",\"hw_concurrency\":" << std::thread::hardware_concurrency();
            output << ",\"model_dir\":\"" << json_escape(published_path(opt.model_dir)) << "\"";
            output << ",\"model_package\":\"" << json_escape(clip_meta.model_package) << "\"";
            output << ",\"model_sha256\":\"" << json_escape(model_sha256) << "\"";
            output << ",\"variant_id\":\"" << json_escape(clip_meta.variant_id) << "\"";
            output << ",\"base_model\":\"" << json_escape(clip_meta.base_model) << "\"";
            output << ",\"precision\":\"" << json_escape(clip_meta.precision) << "\"";
            output << ",\"quant_method\":\"" << json_escape(clip_meta.quant_method) << "\"";
            output << ",\"execution_provider\":\"" << json_escape(clip_meta.execution_provider) << "\"";
            append_diagnostics_json(output, engine->execution_diagnostics());
            output << ",\"runtime\":\"" << json_escape(clip_meta.runtime) << "\"";
            output << ",\"model_format\":\"" << json_escape(clip_meta.model_format) << "\"";
            output << ",\"decode_strategy\":\"" << json_escape(clip_meta.decode_strategy) << "\"";
            if (clip_meta.max_context > 0) output << ",\"max_context\":" << clip_meta.max_context;
            output << ",\"model_size_mb\":" << size_mb;
            output << ",\"audio\":\"" << json_escape(published_path(spec.audio)) << "\"";
            output << ",\"audio_len_s\":" << clip_audio_len;
            output << ",\"runs\":" << runs;
            output << ",\"warmup\":" << (index == 0 ? 1 : 0);
            output << ",\"hot_only\":false";
            output << ",\"load_cold_s\":" << (index == 0 ? cold_load : -1.0);
            output << ",\"load_hot_s\":" << (index == 0 ? warm_load : -1.0);
            output << ",\"cold_load_seconds\":" << (index == 0 ? cold_load : -1.0);
            output << ",\"hot_load_seconds\":" << (index == 0 ? warm_load : -1.0);
            output << ",\"warm_load_seconds\":" << (index == 0 ? warm_load : -1.0);
            output << ",\"mean_ms\":" << clip_mean * 1000.0;
            output << ",\"median_ms\":" << clip_median * 1000.0;
            output << ",\"p90_ms\":" << clip_p90 * 1000.0;
            output << ",\"rtf\":" << clip_rtf;
            output << ",\"xrt\":" << (clip_rtf > 0 ? 1.0 / clip_rtf : 0.0);
            output << ",\"avg_logprob\":" << clip_result.avg_logprob;
            output << ",\"sequence_logprob\":" << clip_result.sequence_logprob;
            output << ",\"generated_tokens\":" << clip_result.generated_tokens;
            output << ",\"ttft_ms\":" << clip_result.ttft_ms;
            output << ",\"tpot_ms\":" << clip_result.tpot_ms;
            output << ",\"throughput_tps\":" << clip_result.throughput_tps;
            output << ",\"has_token_metrics\":"
                   << (clip_result.has_token_metrics ? "true" : "false");
            output << ",\"wer\":" << clip_error.wer;
            output << ",\"cer\":" << clip_error.cer;
            output << ",\"ref_words\":" << clip_error.ref_words;
            output << ",\"word_edits\":" << clip_error.word_edits;
            output << ",\"ref_chars\":" << clip_error.ref_chars;
            output << ",\"char_edits\":" << clip_error.char_edits;
            output << ",\"ref\":\"" << json_escape(spec.reference) << "\"";
            output << ",\"text\":\"" << json_escape(clip_text) << "\"";
            output << ",\"language\":\"" << json_escape(clip_result.language) << "\"";
            output << ",\"task\":\"" << json_escape(clip_result.task) << "\"";
            if (clip_result.windows > 0) output << ",\"windows\":" << clip_result.windows;
            output << "}";
        }
        output << "]}";
        emit_json(output.str());
        return 0;
    }

    // Inference benchmark.
    std::vector<double> lat;
    std::string text;
    TranscribeResult last;
    try {
        for (int i = 0; i < warmup; ++i) engine->transcribe(wav.samples);
        for (int i = 0; i < runs; ++i) {
            last = engine->transcribe(wav.samples);
            lat.push_back(last.infer_seconds);
            text = trim(last.text);
        }
    } catch (const std::exception& e) {
        return fail(4, std::string("Transcription failed: ") + e.what());
    }

    double mean = 0.0;
    for (double l : lat) mean += l;
    mean /= static_cast<double>(lat.size());
    const double median = percentile(lat, 50.0);
    const double p90 = percentile(lat, 90.0);
    const double rtf = mean / audio_len;
    const double size_mb = model_size_mb(opt.model_dir);
    std::string model_sha256;
    try {
        model_sha256 = npu_inference_bench::model_artifacts_sha256(opt.model_dir);
    } catch (const std::exception& e) {
        return fail(5, std::string("Failed to hash model artifacts: ") + e.what());
    }

    ErrorRate er;
    if (have_ref) er = compute_error_rate(reference, text);

    if (!results_csv.empty()) {
        try {
            append_result_csv(results_csv, requested_backend, *engine, opt.device, opt.model_dir,
                              audio_path, audio_len, runs, warmup, cache_dir,
                              cold_load, warm_load, mean, rtf, text,
                              label, size_mb, model_sha256, last, have_ref, er);
        } catch (const std::exception& e) {
            return fail(5, std::string("Failed to append results CSV: ") + e.what());
        }
    }

    if (json_out) {
        // Only the JSON payload needs the resolved metadata; append_result_csv
        // resolves its own copy internally, so keep this off the text-output path.
        const auto bench_meta =
            npu_inference_bench::benchmark_meta::resolve(opt.model_dir, label, *engine, last);
        std::ostringstream js;
        js << std::fixed << std::setprecision(6);
        js << "{";
        js << "\"ok\":true";
        js << ",\"backend\":\"" << json_escape(engine->backend_name()) << "\"";
        js << ",\"device\":\"" << json_escape(engine->device_name()) << "\"";
        js << ",\"device_full_name\":\"" << json_escape(engine->full_device_name()) << "\"";
        js << ",\"power_source\":\"" << json_escape(power_source()) << "\"";
        js << ",\"host_arch\":\"" << json_escape(host_arch()) << "\"";
        js << ",\"host_os\":\"" << json_escape(host_os()) << "\"";
        js << ",\"runtime_version\":\"" << json_escape(engine->runtime_version()) << "\"";
        js << ",\"inference_precision\":\""
           << json_escape(engine->execution_diagnostics().inference_precision) << "\"";
        js << ",\"cpu_threads_requested\":" << cpu_threads;
        js << ",\"hw_concurrency\":" << std::thread::hardware_concurrency();
        js << ",\"model_dir\":\"" << json_escape(published_path(opt.model_dir)) << "\"";
        js << ",\"model_package\":\"" << json_escape(bench_meta.model_package) << "\"";
        js << ",\"model_sha256\":\"" << json_escape(model_sha256) << "\"";
        js << ",\"variant_id\":\"" << json_escape(bench_meta.variant_id) << "\"";
        js << ",\"base_model\":\"" << json_escape(bench_meta.base_model) << "\"";
        js << ",\"precision\":\"" << json_escape(bench_meta.precision) << "\"";
        js << ",\"quant_method\":\"" << json_escape(bench_meta.quant_method) << "\"";
        js << ",\"execution_provider\":\"" << json_escape(bench_meta.execution_provider) << "\"";
        append_diagnostics_json(js, engine->execution_diagnostics());
        js << ",\"runtime\":\"" << json_escape(bench_meta.runtime) << "\"";
        js << ",\"model_format\":\"" << json_escape(bench_meta.model_format) << "\"";
        js << ",\"decode_strategy\":\"" << json_escape(bench_meta.decode_strategy) << "\"";
        if (bench_meta.max_context > 0) {
            js << ",\"max_context\":" << bench_meta.max_context;
        }
        js << ",\"model_size_mb\":" << size_mb;
        js << ",\"audio\":\"" << json_escape(published_path(audio_path)) << "\"";
        js << ",\"audio_len_s\":" << audio_len;
        js << ",\"runs\":" << runs;
        js << ",\"warmup\":" << warmup;
        js << ",\"hot_only\":" << (hot_only ? "true" : "false");
        js << ",\"load_cold_s\":" << cold_load;
        js << ",\"load_hot_s\":" << warm_load;
        js << ",\"cold_load_seconds\":" << cold_load;
        js << ",\"hot_load_seconds\":" << warm_load;
        js << ",\"warm_load_seconds\":" << warm_load;  // migration alias
        js << ",\"mean_ms\":" << mean * 1000.0;
        js << ",\"median_ms\":" << median * 1000.0;
        js << ",\"p90_ms\":" << p90 * 1000.0;
        js << ",\"rtf\":" << rtf;
        js << ",\"xrt\":" << (rtf > 0 ? 1.0 / rtf : 0.0);
        js << ",\"avg_logprob\":" << last.avg_logprob;
        js << ",\"sequence_logprob\":" << last.sequence_logprob;
        js << ",\"generated_tokens\":" << last.generated_tokens;
        js << ",\"ttft_ms\":" << last.ttft_ms;
        js << ",\"tpot_ms\":" << last.tpot_ms;
        js << ",\"throughput_tps\":" << last.throughput_tps;
        js << ",\"has_token_metrics\":" << (last.has_token_metrics ? "true" : "false");
        if (have_ref) {
            js << ",\"wer\":" << er.wer;
            js << ",\"cer\":" << er.cer;
            js << ",\"ref_words\":" << er.ref_words;
            js << ",\"word_edits\":" << er.word_edits;
            js << ",\"ref_chars\":" << er.ref_chars;
            js << ",\"char_edits\":" << er.char_edits;
            js << ",\"ref\":\"" << json_escape(reference) << "\"";
        }
        js << ",\"text\":\"" << json_escape(text) << "\"";
        js << ",\"language\":\"" << json_escape(last.language) << "\"";
        js << ",\"task\":\"" << json_escape(last.task) << "\"";
        if (last.windows > 0) js << ",\"windows\":" << last.windows;
        js << "}";
        std::cout << js.str() << "\n";
    } else {
        std::cout << "Engine : " << engine->backend_name() << "  [" << engine->device_name() << "]";
        if (!engine->full_device_name().empty()) {
            std::cout << "  (" << engine->full_device_name() << ")";
        }
        if (opt.device == Device::CPU) {
            std::cout << "  (threads " << (cpu_threads > 0 ? std::to_string(cpu_threads) : "default")
                      << " / " << std::thread::hardware_concurrency() << " logical)";
        }
        std::cout << "\n";
        std::cout << "power        : " << power_source() << "\n";
        std::cout << "host         : " << host_arch() << " / " << host_os() << "\n";
        if (!engine->runtime_version().empty())
            std::cout << "runtime ver  : " << engine->runtime_version() << "\n";
        std::cout << std::setprecision(3);
        if (cold_load >= 0.0) std::cout << "load (cold)  : " << cold_load << " s\n";
        else                  std::cout << "load (cold)  : n/a (hot-only)\n";
        if (warm_load >= 0.0) {
            std::cout << "load (hot)   : " << warm_load << " s";
            if (cold_load > 0.0 && warm_load > 0.0)
                std::cout << "   (" << std::setprecision(1) << (cold_load / warm_load)
                          << "x faster)" << std::setprecision(3);
            std::cout << "\n";
        }
        std::cout << std::setprecision(1);
        std::cout << "infer mean   : " << mean * 1000.0 << " ms  (median " << median * 1000.0
                  << ", p90 " << p90 * 1000.0 << ")\n";
        std::cout << "RTF          : " << std::setprecision(3) << rtf << std::setprecision(1)
                  << "  (" << 1.0 / rtf << "x real time)\n";
        if (last.has_token_metrics) {
            std::cout << "confidence   : avg_logprob " << std::setprecision(4) << last.avg_logprob
                      << " over " << last.generated_tokens << " tokens\n";
            std::cout << std::setprecision(1);
            std::cout << "token perf   : ttft " << last.ttft_ms << " ms, tpot " << last.tpot_ms
                      << " ms, " << last.throughput_tps << " tok/s\n";
        }
        if (size_mb >= 0) std::cout << "model size   : " << std::setprecision(1) << size_mb << " MB\n";
        if (have_ref) {
            std::cout << "accuracy     : WER " << std::setprecision(2) << er.wer * 100.0 << "%  CER "
                      << er.cer * 100.0 << "%  (ref " << er.ref_words << " words)\n";
        }
        std::cout << "transcription: " << text << "\n";
        if (!results_csv.empty()) std::cout << "results csv  : " << results_csv << "\n";
    }
    return 0;
}
