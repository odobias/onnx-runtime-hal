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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
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
#include "npu_inference_bench/whisper.hpp"

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parse_backend(const std::string& s, npu_inference_bench::Backend& out) {
    const std::string v = lower(s);
    if (v == "auto")  { out = npu_inference_bench::Backend::Auto; return true; }
    if (v == "intel") { out = npu_inference_bench::Backend::IntelOpenVINO; return true; }
    if (v == "intel-onnx" || v == "intelonnx") {
        out = npu_inference_bench::Backend::IntelOnnx; return true;
    }
    if (v == "onnx" || v == "onnx-static" || v == "ort" || v == "ort-static") {
        out = npu_inference_bench::Backend::OnnxRuntimeStatic; return true;
    }
    if (v == "onnx-dynamic" || v == "onnx-dyn" || v == "ort-dynamic") {
        out = npu_inference_bench::Backend::OnnxRuntimeDynamic; return true;
    }
    if (v == "amd")   { out = npu_inference_bench::Backend::AmdRyzenAI; return true; }
    if (v == "qualcomm" || v == "qnn") { out = npu_inference_bench::Backend::QualcommQNN; return true; }
    return false;
}

bool parse_device(const std::string& s, npu_inference_bench::Device& out) {
    const std::string v = lower(s);
    if (v == "npu") { out = npu_inference_bench::Device::NPU; return true; }
    if (v == "gpu") { out = npu_inference_bench::Device::GPU; return true; }
    if (v == "cpu") { out = npu_inference_bench::Device::CPU; return true; }
    // "auto" is a placeholder: with the Auto backend the engine self-selects the
    // device (best_available_device), so the value here is overridden anyway.
    if (v == "auto") { out = npu_inference_bench::Device::NPU; return true; }
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

void append_diagnostics_json(
    std::ostringstream& js,
    const npu_inference_bench::ExecutionDiagnostics& diagnostics) {
    js << ",\"requested_provider\":\"" << json_escape(diagnostics.requested_provider) << "\"";
    js << ",\"resolved_provider\":\"" << json_escape(diagnostics.resolved_provider) << "\"";
    js << ",\"fallback_occurred\":" << (diagnostics.fallback_occurred ? "true" : "false");
    js << ",\"provider_attempts\":[";
    for (size_t i = 0; i < diagnostics.attempts.size(); ++i) {
        if (i) js << ',';
        const auto& attempt = diagnostics.attempts[i];
        js << "{\"provider\":\"" << json_escape(attempt.provider)
           << "\",\"success\":" << (attempt.success ? "true" : "false")
           << ",\"error\":\"" << json_escape(attempt.error) << "\"}";
    }
    js << "]";
    if (diagnostics.offload_measured) {
        const int total = diagnostics.ep_nodes + diagnostics.cpu_nodes;
        js << ",\"ep_nodes\":" << diagnostics.ep_nodes;
        js << ",\"cpu_nodes\":" << diagnostics.cpu_nodes;
        js << ",\"cpu_offload_pct\":"
           << (total > 0 ? 100.0 * diagnostics.cpu_nodes / total : 0.0);
        js << ",\"cpu_offload_ops\":\"" << json_escape(diagnostics.cpu_offload_ops) << "\"";
    }
}

std::string provider_attempts_json(
    const npu_inference_bench::ExecutionDiagnostics& diagnostics) {
    std::ostringstream js;
    js << '[';
    for (size_t i = 0; i < diagnostics.attempts.size(); ++i) {
        if (i) js << ',';
        const auto& attempt = diagnostics.attempts[i];
        js << "{\"provider\":\"" << json_escape(attempt.provider)
           << "\",\"success\":" << (attempt.success ? "true" : "false")
           << ",\"error\":\"" << json_escape(attempt.error) << "\"}";
    }
    js << ']';
    return js.str();
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

// Best-effort AC vs battery detection, read at run time. Power state changes
// CPU/NPU clocking (battery = throttled), so an unlabeled battery run can silently
// skew a comparison; record it per row. "ac" / "battery" / "unknown".
std::string power_source() {
#if defined(_WIN32)
    SYSTEM_POWER_STATUS s{};
    if (GetSystemPowerStatus(&s)) {
        if (s.ACLineStatus == 1) return "ac";       // plugged in (also desktops w/o battery)
        if (s.ACLineStatus == 0) return "battery";  // running on battery
    }
    return "unknown";  // ACLineStatus 255 = unknown, or call failed
#else
    return "unknown";
#endif
}

// Compile-time target ISA of this binary. This is the axis that decides which
// vendor DLL pack a build belongs to (x64 = Intel/AMD; arm64 = Qualcomm/QNN), so
// it's recorded per row to keep cross-ISA results attributable in one ledger.
std::string host_arch() {
#if defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#elif defined(_M_X64) || defined(__x86_64__)
    return "x64";
#elif defined(_M_IX86) || defined(__i386__)
    return "x86";
#else
    return "unknown";
#endif
}

// Best-effort OS identity. On Windows, RtlGetVersion is the only non-deprecated
// way to read the real build number (GetVersionEx lies without a manifest).
std::string host_os() {
#if defined(_WIN32)
    if (HMODULE nt = GetModuleHandleW(L"ntdll.dll")) {
        using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
        if (auto fn = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(nt, "RtlGetVersion"))) {
            RTL_OSVERSIONINFOW vi{};
            vi.dwOSVersionInfoSize = sizeof(vi);
            if (fn(&vi) == 0) {
                const char* name = (vi.dwMajorVersion == 10 && vi.dwBuildNumber >= 22000)
                                       ? "Windows 11"
                                       : (vi.dwMajorVersion == 10 ? "Windows 10" : "Windows");
                std::ostringstream o;
                o << name << " (build " << vi.dwBuildNumber << ")";
                return o.str();
            }
        }
    }
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "unknown";
#endif
}

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else if (c == '"') {
            in_quotes = true;
        } else if (c == ',') {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    fields.push_back(cur);
    return fields;
}

constexpr const char* kBenchmarkCsvHeader =
    "timestamp_utc,requested_backend,resolved_backend,device,device_name,"
    "device_full_name,"
    "model_package,variant_id,base_model,precision,quant_method,execution_provider,"
    "model_dir,audio_path,audio_seconds,runs,warmup,cache_dir,"
    "cold_load_seconds,warm_load_seconds,mean_infer_seconds,rtf,"
    "realtime_factor,label,model_size_mb,avg_logprob,ttft_ms,tpot_ms,"
    "throughput_tps,wer,cer,transcription,"
    "runtime,model_format,decode_strategy,max_context,eval_clips,status,"
    "cold_start_seconds,hot_start_seconds,power_source,"
    "host_arch,host_os,runtime_version,"
    "requested_provider,resolved_provider,fallback_occurred,provider_attempts,"
    "ep_nodes,cpu_nodes,cpu_offload_pct,cpu_offload_ops,error";

// Classifier ledger schema (see append_classifier_csv). Kept as a named constant so
// the writer and the additive migration below agree on column order.
constexpr const char* kClassifierCsvHeader =
    "timestamp_utc,model,requested_device,execution_provider,runtime,runtime_version,"
    "host_arch,host_os,backend_name,runs,cache_dir,load_seconds,cold_load_seconds,hot_load_seconds,"
    "mean_infer_ms,median_infer_ms,p90_infer_ms,"
    "model_size_mb,eval_samples,correct,accuracy,max_abs_p_diff,"
    "ep_nodes,cpu_nodes,cpu_offload_pct,cpu_offload_ops,power_source,eval_detail,"
    "status,requested_provider,resolved_provider,fallback_occurred,provider_attempts,error";

// Additive, name-keyed CSV upgrade: rewrite `csv_path` under `new_header_line`,
// mapping each existing row by column name (new columns become empty). Used to fold
// the CPU-offload columns into a pre-existing classifier ledger without corrupting it.
void migrate_named_csv_schema(const std::filesystem::path& csv_path,
                              const std::string& new_header_line) {
    std::ifstream in(csv_path);
    if (!in) return;
    std::string header;
    std::getline(in, header);
    if (!header.empty() && header.back() == '\r') header.pop_back();
    if (header == new_header_line) return;  // already current

    const std::vector<std::string> old_header = split_csv_row(header);
    const std::vector<std::string> new_header = split_csv_row(new_header_line);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) rows.push_back(split_csv_row(line));
    }
    in.close();

    std::ofstream out(csv_path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot upgrade classifier CSV schema: " + csv_path.string());
    out << new_header_line << '\n';
    for (const auto& row : rows) {
        std::unordered_map<std::string, std::string> by_name;
        for (size_t i = 0; i < old_header.size() && i < row.size(); ++i) by_name[old_header[i]] = row[i];
        for (size_t i = 0; i < new_header.size(); ++i) {
            if (i) out << ',';
            const auto it = by_name.find(new_header[i]);
            // Preserve the historical single classifier load as the cold value
            // when upgrading older ledgers; hot remains unknown for those rows.
            if (it == by_name.end() && new_header[i] == "cold_load_seconds") {
                const auto legacy = by_name.find("load_seconds");
                out << csv_escape(legacy == by_name.end() ? "" : legacy->second);
            } else {
                out << csv_escape(it == by_name.end() ? "" : it->second);
            }
        }
        out << '\n';
    }
}

void migrate_benchmark_csv_schema(const std::filesystem::path& csv_path) {
    namespace fs = std::filesystem;
    std::ifstream in(csv_path);
    if (!in) return;
    std::string header;
    std::getline(in, header);
    // The PowerShell harness writes CRLF; strip a trailing CR before comparing so a
    // byte-identical header isn't spuriously "migrated" on every C++ append.
    if (!header.empty() && header.back() == '\r') header.pop_back();
    if (header == kBenchmarkCsvHeader) return;  // already the current schema

    std::vector<std::string> old_header = split_csv_row(header);
    std::vector<std::string> new_header = split_csv_row(kBenchmarkCsvHeader);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) rows.push_back(split_csv_row(line));
    }
    in.close();

    std::ofstream out(csv_path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot upgrade results CSV schema: " + csv_path.string());
    out << kBenchmarkCsvHeader << '\n';

    for (const auto& row : rows) {
        std::unordered_map<std::string, std::string> by_name;
        for (size_t i = 0; i < old_header.size() && i < row.size(); ++i) {
            by_name[old_header[i]] = row[i];
        }

        const std::string model_dir = by_name.count("model_dir") ? by_name["model_dir"] : "";
        const std::string label = by_name.count("label") ? by_name["label"] : "";
        npu_inference_bench::TranscribeResult last;
        last.runtime = by_name.count("runtime") ? by_name["runtime"] : "";
        last.model_format = by_name.count("model_format") ? by_name["model_format"] : "";
        last.decode_strategy = by_name.count("decode_strategy") ? by_name["decode_strategy"] : "";
        if (by_name.count("max_context") && !by_name["max_context"].empty()) {
            last.max_context = std::strtol(by_name["max_context"].c_str(), nullptr, 10);
        }

        struct MigrationEngine : npu_inference_bench::IWhisperEngine {
            std::string backend;
            std::string device;
            std::string full_device;
            std::string backend_name() const override { return backend; }
            std::string device_name() const override { return device; }
            std::string full_device_name() const override { return full_device; }
            double load_seconds() const override { return 0.0; }
            npu_inference_bench::TranscribeResult transcribe(const std::vector<float>&) override {
                return {};
            }
        };
        MigrationEngine engine;
        engine.backend = by_name.count("resolved_backend") ? by_name["resolved_backend"] : "";
        engine.device = by_name.count("device_name") ? by_name["device_name"] : "";
        engine.full_device = by_name.count("device_full_name") ? by_name["device_full_name"] : "";

        const auto meta = npu_inference_bench::benchmark_meta::resolve(model_dir, label, engine, last);

        if (by_name["runtime"].empty() || by_name["runtime"] == by_name["resolved_backend"]) {
            by_name["runtime"] = meta.runtime;
        }
        if (by_name["model_format"].empty()) by_name["model_format"] = meta.model_format;
        if (by_name["decode_strategy"].empty()) by_name["decode_strategy"] = meta.decode_strategy;
        if ((!by_name.count("max_context") || by_name["max_context"].empty()) && meta.max_context > 0) {
            by_name["max_context"] = std::to_string(meta.max_context);
        }
        if (by_name["label"].empty()) by_name["label"] = meta.variant_id;

        by_name["model_package"] = meta.model_package;
        by_name["variant_id"] = meta.variant_id;
        by_name["base_model"] = meta.base_model;
        by_name["precision"] = meta.precision;
        by_name["quant_method"] = meta.quant_method;
        by_name["execution_provider"] = meta.execution_provider;

        for (size_t i = 0; i < new_header.size(); ++i) {
            if (i) out << ',';
            const auto it = by_name.find(new_header[i]);
            out << csv_escape(it == by_name.end() ? "" : it->second);
        }
        out << '\n';
    }
}

// Shared, backend-neutral results schema. Filter columns (model_package, variant_id,
// precision, runtime, ...) are always populated so rows concatenate and filter cleanly.
//
// AUTHORITATIVE COLUMN ORDER lives in benchmark/lib/harness.ps1
// (Get-BenchmarkResultColumns). The header emitted below (kBenchmarkCsvHeader) MUST
// match that list; the harness and this executor are the two writers of the same
// ledger. If you add/reorder a column, change it in both places (and results/README.md).
void append_result_csv(const std::string& path,
                       const std::string& requested_backend,
                       const npu_inference_bench::IWhisperEngine& engine,
                       npu_inference_bench::Device device,
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
                       const npu_inference_bench::TranscribeResult& last,
                       bool have_ref,
                       const npu_inference_bench::ErrorRate& er) {
    namespace fs = std::filesystem;
    const fs::path csv_path(path);
    if (csv_path.has_parent_path()) fs::create_directories(csv_path.parent_path());

    const bool write_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
    if (!write_header) migrate_benchmark_csv_schema(csv_path);

    std::ofstream out(csv_path, std::ios::app);
    if (!out) throw std::runtime_error("cannot open results CSV for append: " + path);

    if (write_header) {
        out << kBenchmarkCsvHeader << '\n';
    }

    const auto meta = npu_inference_bench::benchmark_meta::resolve(model_dir, label, engine, last);
    const std::string effective_label = label.empty() ? meta.variant_id : label;
    const std::string runtime = meta.runtime;
    const std::string model_format = meta.model_format;
    const std::string decode_strategy = meta.decode_strategy;
    const long max_context = meta.max_context;
    const auto diagnostics = engine.execution_diagnostics();
    const int diagnostic_nodes = diagnostics.ep_nodes + diagnostics.cpu_nodes;
    const double cpu_offload_pct =
        diagnostics.offload_measured && diagnostic_nodes > 0
            ? 100.0 * diagnostics.cpu_nodes / diagnostic_nodes
            : 0.0;

    const bool tok = last.has_token_metrics;
    out << csv_escape(utc_now_iso8601()) << ','
        << csv_escape(requested_backend) << ','
        << csv_escape(engine.backend_name()) << ','
        << csv_escape(to_string(device)) << ','
        << csv_escape(engine.device_name()) << ','
        << csv_escape(engine.full_device_name()) << ','
        << csv_escape(meta.model_package) << ','
        << csv_escape(meta.variant_id) << ','
        << csv_escape(meta.base_model) << ','
        << csv_escape(meta.precision) << ','
        << csv_escape(meta.quant_method) << ','
        << csv_escape(meta.execution_provider) << ','
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
        << csv_escape(effective_label) << ','
        << opt_num(model_size_mb_val, model_size_mb_val >= 0.0, 6) << ','
        << opt_num(last.avg_logprob, tok, 6) << ','
        << opt_num(last.ttft_ms, tok, 6) << ','
        << opt_num(last.tpot_ms, tok, 6) << ','
        << opt_num(last.throughput_tps, tok, 6) << ','
        << opt_num(er.wer, have_ref, 6) << ','
        << opt_num(er.cer, have_ref, 6) << ','
        << csv_escape(text) << ','
        << csv_escape(runtime) << ','
        << csv_escape(model_format) << ','
        << csv_escape(decode_strategy) << ','
        << (max_context > 0 ? std::to_string(max_context) : std::string()) << ','
        << 1 << ','          // eval_clips: this CLI benchmarks a single audio file
        << "ok" << ','       // reached here => run succeeded
        << cold_load << ','  // cold_start_seconds
        << warm_load << ','  // hot_start_seconds
        << csv_escape(power_source()) << ','
        << csv_escape(host_arch()) << ','
        << csv_escape(host_os()) << ','
        << csv_escape(engine.runtime_version()) << ','
        << csv_escape(diagnostics.requested_provider) << ','
        << csv_escape(diagnostics.resolved_provider) << ','
        << (diagnostics.fallback_occurred ? "true" : "false") << ','
        << csv_escape(provider_attempts_json(diagnostics)) << ','
        << (diagnostics.offload_measured ? std::to_string(diagnostics.ep_nodes) : std::string()) << ','
        << (diagnostics.offload_measured ? std::to_string(diagnostics.cpu_nodes) : std::string()) << ','
        << opt_num(cpu_offload_pct, diagnostics.offload_measured, 4) << ','
        << csv_escape(diagnostics.cpu_offload_ops) << ','
        << "" << '\n';
}

// --- deepfake classifier mode -----------------------------------------------
// Separate ledger from the ASR benchmark: classifiers have a different I/O
// contract and metric set (accuracy + cross-EP probability agreement, not
// WER/RTF), so mixing them into the ASR ledger would produce a ragged,
// meaning-diluted schema. One row per (model, EP) run.
void append_classifier_csv(const std::string& path, const npu_inference_bench::classifier::Result& r) {
    namespace fs = std::filesystem;
    const fs::path csv_path(path);
    if (csv_path.has_parent_path()) fs::create_directories(csv_path.parent_path());
    const bool write_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
    if (!write_header) migrate_named_csv_schema(csv_path, kClassifierCsvHeader);
    std::ofstream out(csv_path, std::ios::app);
    if (!out) throw std::runtime_error("cannot open classifier results CSV for append: " + path);

    if (write_header) out << kClassifierCsvHeader << '\n';

    // eval_detail packs per-sample results without CSV-hostile commas: samples
    // separated by ';', fields within a sample by '|' -> id|label|pred|p|expected_p.
    std::ostringstream detail;
    detail << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < r.samples.size(); ++i) {
        const auto& s = r.samples[i];
        if (i) detail << ';';
        detail << s.id << '|' << s.label << '|' << s.predicted << '|' << s.p << '|' << s.expected_p;
    }

    const double acc = r.eval_samples ? static_cast<double>(r.correct) / r.eval_samples : 0.0;
    const double offload_pct = (r.offload_measured && (r.ep_nodes + r.cpu_nodes) > 0)
                                   ? 100.0 * r.cpu_nodes / (r.ep_nodes + r.cpu_nodes)
                                   : 0.0;
    out << csv_escape(utc_now_iso8601()) << ','
        << csv_escape(r.model) << ','
        << csv_escape(r.requested_device) << ','
        << csv_escape(r.execution_provider) << ','
        << csv_escape(r.runtime) << ','
        << csv_escape(r.runtime_version) << ','
        << csv_escape(r.host_arch) << ','
        << csv_escape(r.host_os) << ','
        << csv_escape(r.backend_name) << ','
        << r.runs << ','
        << csv_escape(r.cache_dir) << ','
        << std::setprecision(6) << r.cold_load_seconds << ','  // legacy load_seconds alias
        << r.cold_load_seconds << ','
        << r.hot_load_seconds << ','
        << r.mean_infer_ms << ','
        << r.median_infer_ms << ','
        << r.p90_infer_ms << ','
        << r.model_size_mb << ','
        << r.eval_samples << ','
        << r.correct << ','
        << acc << ','
        << r.max_abs_p_diff << ','
        << (r.offload_measured ? std::to_string(r.ep_nodes) : std::string()) << ','
        << (r.offload_measured ? std::to_string(r.cpu_nodes) : std::string()) << ','
        << opt_num(offload_pct, r.offload_measured, 4) << ','
        << csv_escape(r.cpu_offload_ops) << ','
        << csv_escape(power_source()) << ','
        << csv_escape(detail.str()) << ','
        << "ok" << ','
        << csv_escape(r.diagnostics.requested_provider) << ','
        << csv_escape(r.diagnostics.resolved_provider) << ','
        << (r.diagnostics.fallback_occurred ? "true" : "false") << ','
        << csv_escape(provider_attempts_json(r.diagnostics)) << ','
        << "" << '\n';
}

int run_classify(const std::string& dir, npu_inference_bench::Device device, const std::string& provider,
                 int runs, int cpu_threads, const std::string& cache_dir,
                 const std::string& results_csv, bool json_out) {
    using namespace npu_inference_bench;
    auto fail = [&](int code, const std::string& msg) {
        if (json_out) std::cout << "{\"ok\":false,\"error\":\"" << json_escape(msg) << "\"}\n";
        else std::cerr << msg << "\n";
        return code;
    };
    if (!classifier::available()) {
        return fail(3, "classifier mode needs an ONNX Runtime build "
                       "(tools\\build\\build.ps1 -EnableOrt -DisableIntel)");
    }

    classifier::Result r;
    try {
        r = classifier::run(dir, device, provider, runs, cpu_threads, cache_dir);
    } catch (const std::exception& e) {
        return fail(4, std::string("classifier run failed: ") + e.what());
    }
    r.host_arch = host_arch();
    r.host_os = host_os();

    if (!results_csv.empty()) {
        try {
            append_classifier_csv(results_csv, r);
        } catch (const std::exception& e) {
            std::cerr << "classifier results append failed: " << e.what() << "\n";
        }
    }

    const double acc = r.eval_samples ? 100.0 * r.correct / r.eval_samples : 0.0;
    if (json_out) {
        std::ostringstream js;
        js << std::fixed << std::setprecision(6) << "{";
        js << "\"ok\":true";
        js << ",\"model\":\"" << json_escape(r.model) << "\"";
        js << ",\"requested_device\":\"" << json_escape(r.requested_device) << "\"";
        js << ",\"execution_provider\":\"" << json_escape(r.execution_provider) << "\"";
        js << ",\"runtime\":\"" << json_escape(r.runtime) << "\"";
        js << ",\"runtime_version\":\"" << json_escape(r.runtime_version) << "\"";
        js << ",\"host_arch\":\"" << json_escape(r.host_arch) << "\"";
        js << ",\"host_os\":\"" << json_escape(r.host_os) << "\"";
        js << ",\"power_source\":\"" << json_escape(power_source()) << "\"";
        js << ",\"cache_dir\":\"" << json_escape(r.cache_dir) << "\"";
        js << ",\"load_seconds\":" << r.cold_load_seconds;  // legacy alias
        js << ",\"cold_load_seconds\":" << r.cold_load_seconds;
        js << ",\"hot_load_seconds\":" << r.hot_load_seconds;
        js << ",\"model_size_mb\":" << r.model_size_mb;
        js << ",\"runs\":" << r.runs;
        js << ",\"mean_infer_ms\":" << r.mean_infer_ms;
        js << ",\"median_infer_ms\":" << r.median_infer_ms;
        js << ",\"p90_infer_ms\":" << r.p90_infer_ms;
        js << ",\"eval_samples\":" << r.eval_samples;
        js << ",\"correct\":" << r.correct;
        js << ",\"accuracy_pct\":" << acc;
        js << ",\"max_abs_p_diff\":" << r.max_abs_p_diff;
        if (r.offload_measured) {
            const int total = r.ep_nodes + r.cpu_nodes;
            js << ",\"ep_nodes\":" << r.ep_nodes;
            js << ",\"cpu_nodes\":" << r.cpu_nodes;
            js << ",\"cpu_offload_pct\":" << (total > 0 ? 100.0 * r.cpu_nodes / total : 0.0);
            js << ",\"cpu_offload_ops\":\"" << json_escape(r.cpu_offload_ops) << "\"";
        }
        append_diagnostics_json(js, r.diagnostics);
        js << ",\"samples\":[";
        for (size_t i = 0; i < r.samples.size(); ++i) {
            const auto& s = r.samples[i];
            if (i) js << ',';
            js << "{\"id\":\"" << json_escape(s.id) << "\",\"label\":\"" << json_escape(s.label)
               << "\",\"pred\":\"" << json_escape(s.predicted) << "\",\"p\":" << s.p
               << ",\"expected_p\":" << s.expected_p << ",\"correct\":" << (s.correct ? "true" : "false")
               << "}";
        }
        js << "]}";
        std::cout << js.str() << "\n";
        return 0;
    }

    std::cout << std::fixed;
    std::cout << "Classifier : " << r.model << "   EP " << r.execution_provider << "  ["
              << r.requested_device << "]\n";
    std::cout << "runtime    : " << r.runtime << "  " << r.runtime_version << "\n";
    std::cout << "host       : " << r.host_arch << " / " << r.host_os << "\n";
    std::cout << "power      : " << power_source() << "\n";
    std::cout << std::setprecision(3);
    std::cout << "load cold  : " << r.cold_load_seconds << " s";
    if (r.model_size_mb >= 0) std::cout << "   (model " << std::setprecision(1) << r.model_size_mb << " MB)";
    std::cout << std::setprecision(3) << "\n";
    std::cout << "load hot   : " << r.hot_load_seconds << " s";
    if (r.cold_load_seconds > 0.0 && r.hot_load_seconds > 0.0) {
        std::cout << "   (" << std::setprecision(1)
                  << (r.cold_load_seconds / r.hot_load_seconds) << "x faster)";
    }
    std::cout << std::setprecision(3) << "\n";
    std::cout << std::setprecision(2);
    std::cout << "infer      : mean " << r.mean_infer_ms << " ms (median " << r.median_infer_ms
              << ", p90 " << r.p90_infer_ms << ")  over " << r.runs << " runs x " << r.samples.size()
              << " samples\n\n";

    std::cout << "  " << std::left << std::setw(14) << "sample" << std::setw(10) << "label"
              << std::setw(10) << "pred" << std::right << std::setw(10) << "p" << std::setw(12)
              << "cpu_ref_p" << std::setw(10) << "|dp|" << "  hit\n";
    std::cout << "  " << std::string(72, '-') << "\n";
    for (const auto& s : r.samples) {
        const std::string hit = !s.has_label ? "-" : (s.correct ? "OK" : "MISS");
        std::cout << "  " << std::left << std::setw(14) << s.id << std::setw(10) << s.label
                  << std::setw(10) << s.predicted << std::right << std::setprecision(4)
                  << std::setw(10) << s.p << std::setw(12) << s.expected_p << std::setw(10)
                  << std::abs(s.p - s.expected_p) << "  " << hit << "\n";
    }
    std::cout << "\n";
    if (r.eval_samples > 0) {
        std::cout << "accuracy   : " << r.correct << "/" << r.eval_samples << "  ("
                  << std::setprecision(1) << acc << "%)\n";
    } else {
        std::cout << "accuracy   : n/a (no ground-truth labels in fixture)\n";
    }
    std::cout << std::setprecision(6);
    std::cout << "agreement  : max |p - cpu_ref_p| = " << r.max_abs_p_diff
              << (r.max_abs_p_diff < 1e-3 ? "  (matches CPU)" : "  (EP diverges from CPU!)") << "\n";
    if (r.offload_measured) {
        const int total = r.ep_nodes + r.cpu_nodes;
        std::cout << "cpu offload: " << r.cpu_nodes << " / " << total
                  << " profiler nodes on the CPU EP";
        if (r.cpu_nodes == 0) {
            std::cout << "  (none -- fully on " << r.execution_provider << ")";
        } else {
            std::cout << std::setprecision(1) << "  ("
                      << (total > 0 ? 100.0 * r.cpu_nodes / total : 0.0) << "%: " << r.cpu_offload_ops
                      << ")";
        }
        std::cout << " [node ratio, not compute share]";
        std::cout << "\n";
    }
    if (!results_csv.empty()) std::cout << "results csv: " << results_csv << "\n";
    return 0;
}

}  // namespace

int npu_inference_bench::benchmark::run_cli(int argc, char* argv[]) {
    using namespace npu_inference_bench;

    std::vector<std::string> pos;
    std::string cache_dir, reference, results_csv, label, provider_override, classify_dir;
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
        } else if (a == "--threads" && i + 1 < argc) {
            cpu_threads = std::max(0, std::atoi(argv[++i]));
        } else if ((a == "--provider" || a == "--device-override") && i + 1 < argc) {
            provider_override = argv[++i];
        } else if (a == "--hot-only") {
            hot_only = true;
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
                  << "  backend: auto | intel | intel-onnx | onnx-static | onnx-dynamic | amd | qualcomm   (default auto)\n"
                  << "  device : npu | gpu | cpu                 (default npu)\n"
                  << "  runs   : timed iterations                (default 5)\n"
                  << "  --cache <dir>: persist compiled model; loads twice (cold/hot)\n"
                  << "  --hot-only  : load once from a populated --cache (skip cold compile; cold=n/a)\n"
                  << "  --ref \"text\": reference transcript -> compute WER/CER\n"
                  << "  --threads N: CPU inference thread count (CPU device only)\n"
                  << "  --provider <ort-ep>: backend-specific provider override (e.g. VitisAIExecutionProvider)\n"
                  << "  --json: emit one machine-readable JSON record\n"
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
        try {
            engine = create_engine(backend, opt);
            cold_load = engine->load_seconds();
        } catch (const std::exception& e) {
            return fail(3, std::string("Engine creation failed: ") + e.what());
        }

        // Hot start (imports cached blob) to quantify the caching win.
        if (!cache_dir.empty()) {
            try {
                auto warm = create_engine(backend, opt);
                warm_load = warm->load_seconds();
                engine = std::move(warm);
            } catch (const std::exception& e) {
                if (!json_out) std::cerr << "Hot reload failed: " << e.what() << "\n";
            }
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
        js << ",\"cpu_threads_requested\":" << cpu_threads;
        js << ",\"hw_concurrency\":" << std::thread::hardware_concurrency();
        js << ",\"model_dir\":\"" << json_escape(opt.model_dir) << "\"";
        js << ",\"model_package\":\"" << json_escape(bench_meta.model_package) << "\"";
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
        js << ",\"audio\":\"" << json_escape(audio_path) << "\"";
        js << ",\"audio_len_s\":" << audio_len;
        js << ",\"runs\":" << runs;
        js << ",\"warmup\":" << warmup;
        js << ",\"hot_only\":" << (hot_only ? "true" : "false");
        js << ",\"load_cold_s\":" << cold_load;
        js << ",\"load_warm_s\":" << warm_load;
        js << ",\"load_hot_s\":" << warm_load;
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
