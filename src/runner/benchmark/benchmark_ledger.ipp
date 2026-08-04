// Internal CSV schemas, migrations, and ledger writers.
constexpr const char* kBenchmarkCsvHeader =
    "timestamp_utc,requested_backend,resolved_backend,device,device_name,"
    "device_full_name,"
    "model_package,model_sha256,variant_id,base_model,precision,quant_method,execution_provider,"
    "model_dir,audio_path,audio_seconds,runs,warmup,cache_dir,"
    "cold_load_seconds,hot_load_seconds,warm_load_seconds,mean_infer_seconds,rtf,"
    "realtime_factor,label,model_size_mb,avg_logprob,ttft_ms,tpot_ms,"
    "throughput_tps,wer,cer,transcription,"
    "runtime,model_format,decode_strategy,max_context,eval_clips,status,"
    "cold_start_seconds,hot_start_seconds,power_source,"
    "host_arch,host_os,runtime_version,inference_precision,"
    "requested_provider,resolved_provider,fallback_occurred,provider_attempts,"
    "ep_nodes,cpu_nodes,cpu_offload_pct,cpu_offload_ops,"
    "assigned_ops_cpu,assigned_ops_npu,operation_assignment_source,"
    "measurement_purpose,execution_profile,graph_role,environment_snapshot_id,"
    "model_compilation_provenance_ids,error";

// Classifier ledger schema (see append_classifier_csv). Kept as a named constant so
// the writer and the additive migration below agree on column order.
constexpr const char* kClassifierCsvHeader =
    "timestamp_utc,model,model_sha256,requested_device,execution_provider,runtime,runtime_version,"
    "inference_precision,host_arch,host_os,backend_name,runs,cache_dir,load_seconds,cold_load_seconds,hot_load_seconds,"
    "mean_infer_ms,median_infer_ms,p90_infer_ms,"
    "model_size_mb,eval_samples,correct,accuracy,max_abs_p_diff,"
    "ep_nodes,cpu_nodes,cpu_offload_pct,cpu_offload_ops,"
    "assigned_ops_cpu,assigned_ops_npu,operation_assignment_source,"
    "power_source,eval_detail,"
    "status,requested_provider,resolved_provider,fallback_occurred,provider_attempts,"
    "measurement_purpose,execution_profile,graph_role,environment_snapshot_id,"
    "model_compilation_provenance_ids,error";

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
        if (by_name["hot_load_seconds"].empty()) {
            if (!by_name["hot_start_seconds"].empty()) {
                by_name["hot_load_seconds"] = by_name["hot_start_seconds"];
            } else if (!by_name["warm_load_seconds"].empty()) {
                by_name["hot_load_seconds"] = by_name["warm_load_seconds"];
            }
        }

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
                       const std::string& model_sha256,
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
        << csv_escape(model_sha256) << ','
        << csv_escape(meta.variant_id) << ','
        << csv_escape(meta.base_model) << ','
        << csv_escape(meta.precision) << ','
        << csv_escape(meta.quant_method) << ','
        << csv_escape(meta.execution_provider) << ','
        << csv_escape(published_path(model_dir)) << ','
        << csv_escape(published_path(audio_path)) << ','
        << std::setprecision(9) << audio_seconds << ','
        << runs << ','
        << warmup << ','
        << csv_escape(published_path(cache_dir)) << ','
        << cold_load << ','
        << warm_load << ','
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
        << csv_escape(diagnostics.inference_precision) << ','
        << csv_escape(diagnostics.requested_provider) << ','
        << csv_escape(diagnostics.resolved_provider) << ','
        << (diagnostics.fallback_occurred ? "true" : "false") << ','
        << csv_escape(provider_attempts_json(diagnostics)) << ','
        << (diagnostics.offload_measured ? std::to_string(diagnostics.ep_nodes) : std::string()) << ','
        << (diagnostics.offload_measured ? std::to_string(diagnostics.cpu_nodes) : std::string()) << ','
        << opt_num(cpu_offload_pct, diagnostics.offload_measured, 4) << ','
        << csv_escape(diagnostics.cpu_offload_ops) << ','
        << (diagnostics.operation_assignment_measured
                ? std::to_string(diagnostics.assigned_ops_cpu) : std::string()) << ','
        << (diagnostics.operation_assignment_measured
                ? std::to_string(diagnostics.assigned_ops_npu) : std::string()) << ','
        << csv_escape(diagnostics.operation_assignment_source) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_MEASUREMENT_PURPOSE")) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_EXECUTION_PROFILE")) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_GRAPH_ROLE")) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_ENVIRONMENT_SNAPSHOT_ID")) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_MODEL_PROVENANCE_IDS")) << ','
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
        << csv_escape(r.model_sha256) << ','
        << csv_escape(r.requested_device) << ','
        << csv_escape(r.execution_provider) << ','
        << csv_escape(r.runtime) << ','
        << csv_escape(r.runtime_version) << ','
        << csv_escape(r.inference_precision) << ','
        << csv_escape(r.host_arch) << ','
        << csv_escape(r.host_os) << ','
        << csv_escape(r.backend_name) << ','
        << r.runs << ','
        << csv_escape(published_path(r.cache_dir)) << ','
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
        << (r.operation_assignment_measured
                ? std::to_string(r.assigned_ops_cpu) : std::string()) << ','
        << (r.operation_assignment_measured
                ? std::to_string(r.assigned_ops_npu) : std::string()) << ','
        << csv_escape(r.operation_assignment_source) << ','
        << csv_escape(power_source()) << ','
        << csv_escape(detail.str()) << ','
        << "ok" << ','
        << csv_escape(r.diagnostics.requested_provider) << ','
        << csv_escape(r.diagnostics.resolved_provider) << ','
        << (r.diagnostics.fallback_occurred ? "true" : "false") << ','
        << csv_escape(provider_attempts_json(r.diagnostics)) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_MEASUREMENT_PURPOSE")) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_EXECUTION_PROFILE")) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_GRAPH_ROLE")) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_ENVIRONMENT_SNAPSHOT_ID")) << ','
        << csv_escape(environment_value("NPU_INFERENCE_BENCH_MODEL_PROVENANCE_IDS")) << ','
        << "" << '\n';
}
