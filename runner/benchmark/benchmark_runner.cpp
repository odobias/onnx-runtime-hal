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
#include "npu_inference_bench/model_hash.hpp"
#include "npu_inference_bench/whisper.hpp"

namespace {

#include "benchmark_support.ipp"
#include "classifier_benchmark.ipp"
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
                    opt.device_override = cold_diagnostics.resolved_provider;
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
        js << ",\"model_dir\":\"" << json_escape(opt.model_dir) << "\"";
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
        js << ",\"audio\":\"" << json_escape(audio_path) << "\"";
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
