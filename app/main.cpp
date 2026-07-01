// whisper_hal CLI: backend-agnostic Whisper runner / benchmark built on the HAL.
//
// Usage: whisper_hal <model_dir> <audio.wav> [backend] [device] [runs]
//                    [--cache <dir>] [--ref "<reference text>"] [--json]
//                    [--results <csv>]
//   backend: auto | intel | amd | qualcomm     (default: auto)
//   device : npu  | gpu | cpu                   (default: npu)
//   runs   : timed iterations                   (default: 5)
//   --cache <dir> : persist compiled model; loads twice (cold vs warm)
//   --ref "<text>": reference transcript -> compute WER/CER
//   --json        : emit one machine-readable JSON record (for the harness)
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "whisper_npu/audio.hpp"
#include "whisper_npu/metrics.hpp"
#include "whisper_npu/whisper_engine.hpp"

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parse_backend(const std::string& s, whisper_npu::Backend& out) {
    const std::string v = lower(s);
    if (v == "auto")  { out = whisper_npu::Backend::Auto; return true; }
    if (v == "intel") { out = whisper_npu::Backend::IntelOpenVINO; return true; }
    if (v == "amd")   { out = whisper_npu::Backend::AmdRyzenAI; return true; }
    if (v == "qualcomm" || v == "qnn") { out = whisper_npu::Backend::QualcommQNN; return true; }
    return false;
}

bool parse_device(const std::string& s, whisper_npu::Device& out) {
    const std::string v = lower(s);
    if (v == "npu") { out = whisper_npu::Device::NPU; return true; }
    if (v == "gpu") { out = whisper_npu::Device::GPU; return true; }
    if (v == "cpu") { out = whisper_npu::Device::CPU; return true; }
    return false;
}

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

double model_size_mb(const std::string& dir) {
    std::error_code ec;
    uintmax_t total = 0;
    std::filesystem::path p(dir);
    if (!std::filesystem::exists(p, ec)) return -1.0;
    for (auto it = std::filesystem::recursive_directory_iterator(p, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec)) total += it->file_size(ec);
    }
    return static_cast<double>(total) / 1e6;
}

double percentile(std::vector<double> v, double pct) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = pct / 100.0 * (v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

std::string csv_escape(const std::string& s) {
    bool quote = s.find_first_of(",\"\r\n") != std::string::npos;
    if (!quote) return s;

    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

std::string utc_now_iso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

// Format an optional numeric CSV cell: empty string when the backend didn't
// provide it (keeps the shared schema backend-neutral -- a backend fills only the
// metrics it can measure).
std::string opt_num(double v, bool present, int prec = 6) {
    if (!present) return "";
    std::ostringstream o;
    o << std::setprecision(prec) << v;
    return o.str();
}

// Shared, backend-neutral results schema. The trailing metric columns (label +
// confidence/perf/accuracy) are populated only when the backend/run can supply
// them, so rows from Intel/AMD/Qualcomm machines share one schema and concatenate.
void append_result_csv(const std::string& path,
                       const std::string& requested_backend,
                       const whisper_npu::IWhisperEngine& engine,
                       whisper_npu::Device device,
                       const std::string& model_dir,
                       const std::string& audio_path,
                       double audio_seconds,
                       int runs,
                       int warmup,
                       const std::string& cache_dir,
                       double cold_load,
                       double warm_load,
                       double mean_seconds,
                       double rtf,
                       const std::string& text,
                       const std::string& label,
                       double model_size_mb_val,
                       const whisper_npu::TranscribeResult& last,
                       bool have_ref,
                       const whisper_npu::ErrorRate& er) {
    namespace fs = std::filesystem;
    const fs::path csv_path(path);
    if (csv_path.has_parent_path()) fs::create_directories(csv_path.parent_path());

    const bool write_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
    std::ofstream out(csv_path, std::ios::app);
    if (!out) throw std::runtime_error("cannot open results CSV for append: " + path);

    if (write_header) {
        out << "timestamp_utc,requested_backend,resolved_backend,device,device_name,"
               "model_dir,audio_path,audio_seconds,runs,warmup,cache_dir,"
               "cold_load_seconds,warm_load_seconds,mean_infer_seconds,rtf,"
               "realtime_factor,label,model_size_mb,avg_logprob,ttft_ms,tpot_ms,"
               "throughput_tps,wer,cer,transcription\n";
    }

    const bool tok = last.has_token_metrics;
    out << csv_escape(utc_now_iso8601()) << ','
        << csv_escape(requested_backend) << ','
        << csv_escape(engine.backend_name()) << ','
        << csv_escape(to_string(device)) << ','
        << csv_escape(engine.device_name()) << ','
        << csv_escape(model_dir) << ','
        << csv_escape(audio_path) << ','
        << std::setprecision(9) << audio_seconds << ','
        << runs << ','
        << warmup << ','
        << csv_escape(cache_dir) << ','
        << cold_load << ','
        << warm_load << ','
        << mean_seconds << ','
        << rtf << ','
        << (rtf > 0.0 ? 1.0 / rtf : 0.0) << ','
        << csv_escape(label) << ','
        << opt_num(model_size_mb_val, model_size_mb_val >= 0.0, 6) << ','
        << opt_num(last.avg_logprob, tok, 6) << ','
        << opt_num(last.ttft_ms, tok, 6) << ','
        << opt_num(last.tpot_ms, tok, 6) << ','
        << opt_num(last.throughput_tps, tok, 6) << ','
        << opt_num(er.wer, have_ref, 6) << ','
        << opt_num(er.cer, have_ref, 6) << ','
        << csv_escape(text) << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    using namespace whisper_npu;

    std::vector<std::string> pos;
    std::string cache_dir, reference, results_csv, label;
    bool json_out = false, have_ref = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cache" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if (a == "--ref" && i + 1 < argc) {
            reference = argv[++i];
            have_ref = true;
        } else if (a == "--json") {
            json_out = true;
        } else if (a == "--results" && i + 1 < argc) {
            results_csv = argv[++i];
        } else if (a == "--label" && i + 1 < argc) {
            label = argv[++i];
        } else {
            pos.push_back(a);
        }
    }

    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_dir> <audio.wav> [backend] [device] [runs]"
                     " [--cache <dir>] [--ref \"text\"] [--json] [--results <csv>]\n"
                  << "  backend: auto | intel | amd | qualcomm   (default auto)\n"
                  << "  device : npu | gpu | cpu                 (default npu)\n"
                  << "  runs   : timed iterations                (default 5)\n"
                  << "  --cache <dir>: persist compiled model; loads twice (cold/warm)\n"
                  << "  --ref \"text\": reference transcript -> compute WER/CER\n"
                  << "  --json: emit one machine-readable JSON record\n"
                  << "  --results <csv>: append a benchmark result row\n"
                  << "  --label <text>: tag the results row (e.g. quantization variant)\n\n";
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

    Backend backend = Backend::Auto;
    if (pos.size() > 2 && !parse_backend(pos[2], backend)) {
        std::cerr << "Unknown backend: " << pos[2] << "\n";
        return 1;
    }
    if (pos.size() > 3 && !parse_device(pos[3], opt.device)) {
        std::cerr << "Unknown device: " << pos[3] << "\n";
        return 1;
    }
    const int runs = pos.size() > 4 ? std::max(1, std::atoi(pos[4].c_str())) : 5;
    const int warmup = 1;
    const std::string requested_backend = to_string(backend);

    auto fail = [&](int code, const std::string& msg) {
        if (json_out) {
            std::cout << "{\"ok\":false,\"error\":\"" << json_escape(msg) << "\"}\n";
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

    // Cold load (compiles; writes cache if enabled).
    std::unique_ptr<IWhisperEngine> engine;
    double cold_load = 0.0, warm_load = -1.0;
    try {
        engine = create_engine(backend, opt);
        cold_load = engine->load_seconds();
    } catch (const std::exception& e) {
        return fail(3, std::string("Engine creation failed: ") + e.what());
    }

    // Warm load (imports cached blob) to quantify the caching win.
    if (!cache_dir.empty()) {
        try {
            auto warm = create_engine(backend, opt);
            warm_load = warm->load_seconds();
            engine = std::move(warm);
        } catch (const std::exception& e) {
            if (!json_out) std::cerr << "Warm reload failed: " << e.what() << "\n";
        }
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

    ErrorRate er;
    if (have_ref) er = compute_error_rate(reference, text);

    if (!results_csv.empty()) {
        try {
            append_result_csv(results_csv, requested_backend, *engine, opt.device, opt.model_dir,
                              audio_path, audio_len, runs, warmup, cache_dir,
                              cold_load, warm_load, mean, rtf, text,
                              label, size_mb, last, have_ref, er);
        } catch (const std::exception& e) {
            return fail(5, std::string("Failed to append results CSV: ") + e.what());
        }
    }

    if (json_out) {
        std::ostringstream js;
        js << std::fixed << std::setprecision(6);
        js << "{";
        js << "\"ok\":true";
        js << ",\"backend\":\"" << json_escape(engine->backend_name()) << "\"";
        js << ",\"device\":\"" << json_escape(engine->device_name()) << "\"";
        js << ",\"model_dir\":\"" << json_escape(opt.model_dir) << "\"";
        js << ",\"model_size_mb\":" << size_mb;
        js << ",\"audio\":\"" << json_escape(audio_path) << "\"";
        js << ",\"audio_len_s\":" << audio_len;
        js << ",\"runs\":" << runs;
        js << ",\"load_cold_s\":" << cold_load;
        js << ",\"load_warm_s\":" << warm_load;
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
        js << "}";
        std::cout << js.str() << "\n";
    } else {
        std::cout << "Engine : " << engine->backend_name() << "  [" << engine->device_name() << "]\n";
        std::cout << std::setprecision(3);
        std::cout << "load (cold)  : " << cold_load << " s\n";
        if (warm_load >= 0.0) {
            std::cout << "load (warm)  : " << warm_load << " s";
            if (warm_load > 0.0) std::cout << "   (" << std::setprecision(1) << (cold_load / warm_load)
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
