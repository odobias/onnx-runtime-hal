// Internal classifier benchmark path; included by benchmark_runner.cpp.
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
        js << ",\"model_sha256\":\"" << json_escape(r.model_sha256) << "\"";
        js << ",\"requested_device\":\"" << json_escape(r.requested_device) << "\"";
        js << ",\"execution_provider\":\"" << json_escape(r.execution_provider) << "\"";
        js << ",\"runtime\":\"" << json_escape(r.runtime) << "\"";
        js << ",\"runtime_version\":\"" << json_escape(r.runtime_version) << "\"";
        js << ",\"inference_precision\":\"" << json_escape(r.inference_precision) << "\"";
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
        append_diagnostics_json(js, r.diagnostics);
        js << ",\"samples\":[";
        for (size_t i = 0; i < r.samples.size(); ++i) {
            const auto& s = r.samples[i];
            if (i) js << ',';
            js << "{\"id\":\"" << json_escape(s.id) << "\",\"label\":\"" << json_escape(s.label)
               << "\",\"pred\":\"" << json_escape(s.predicted) << "\",\"p\":" << s.p
               << ",\"expected_pred\":\"" << json_escape(s.expected_predicted) << "\""
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

