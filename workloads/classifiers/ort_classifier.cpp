// ONNX Runtime classifier harness: replays the deepfake pipeline's pre-baked,
// validated fixture tensors (tools/fixtures/generate.py) across ORT
// execution providers and scores per-EP latency + correctness + numerical
// agreement vs the CPU reference. See runner/include/npu_inference_bench/classifier.hpp.
#include "npu_inference_bench/classifier.hpp"
#include "npu_inference_bench/model_hash.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef NPU_INFERENCE_BENCH_ORT
#include <onnxruntime_cxx_api.h>
#include "ort_ep.hpp"
#include "ort_offload.hpp"
#endif

namespace npu_inference_bench {
namespace classifier {

#ifdef NPU_INFERENCE_BENCH_ORT

namespace {

namespace fs = std::filesystem;

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string rstrip_cr(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
    return s;
}

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == '\t') { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

std::vector<std::string> read_lines(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open fixture file: " + p.string());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        line = rstrip_cr(line);
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

std::vector<int64_t> parse_shape(const std::string& csv) {
    std::vector<int64_t> dims;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) dims.push_back(std::strtoll(tok.c_str(), nullptr, 10));
    }
    return dims;
}

int64_t elem_count(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

struct InputTensor {
    std::string name;
    bool is_float = true;
    std::vector<int64_t> shape;
    std::vector<float> f;
    std::vector<int64_t> i;
};

struct Sample {
    std::string id;
    std::string label;
    std::string expected_pred;
    double expected_p = 0.0;
    std::vector<InputTensor> inputs;
};

struct Fixture {
    fs::path onnx_path;
    int positive_index = 1;
    double threshold = 0.5;
    std::string positive_label;
    std::string negative_label;
    std::vector<Sample> samples;
};

void load_tensor(const fs::path& data_dir, const std::string& file, InputTensor& t) {
    const fs::path p = data_dir / file;
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open tensor bin: " + p.string());
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    const int64_t n = elem_count(t.shape);
    if (t.is_float) {
        if (bytes != n * 4) throw std::runtime_error("f32 tensor size mismatch: " + p.string());
        t.f.resize(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(t.f.data()), bytes);
    } else {
        if (bytes != n * 8) throw std::runtime_error("i64 tensor size mismatch: " + p.string());
        t.i.resize(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(t.i.data()), bytes);
    }
}

Fixture load_fixture(const fs::path& dir) {
    Fixture fx;
    const auto model = read_lines(dir / "model.tsv");
    if (model.empty()) throw std::runtime_error("empty model.tsv in " + dir.string());
    const auto m = split_tab(model[0]);
    if (m.size() < 5) throw std::runtime_error("malformed model.tsv (need 5 cols) in " + dir.string());
    fx.onnx_path = m[0];
    // dump_fixtures.py writes a fixture-relative path so the tree is portable across
    // machines/repo roots; resolve it here. Absolute paths (older fixtures) pass through.
    if (fx.onnx_path.is_relative()) fx.onnx_path = (dir / fx.onnx_path).lexically_normal();
    fx.positive_index = std::atoi(m[1].c_str());
    fx.threshold = std::strtod(m[2].c_str(), nullptr);
    fx.positive_label = m[3];
    fx.negative_label = m[4];

    std::unordered_map<std::string, Sample> by_id;
    std::vector<std::string> order;

    const auto samples = read_lines(dir / "samples.tsv");
    for (size_t r = 1; r < samples.size(); ++r) {  // skip header
        const auto s = split_tab(samples[r]);
        if (s.size() < 4) continue;
        Sample smp;
        smp.id = s[0];
        smp.label = s[1];
        smp.expected_pred = s[2];
        smp.expected_p = std::strtod(s[3].c_str(), nullptr);
        by_id[smp.id] = smp;
        order.push_back(smp.id);
    }

    const fs::path data_dir = dir / "data";
    const auto inputs = read_lines(dir / "inputs.tsv");
    for (size_t r = 1; r < inputs.size(); ++r) {  // skip header
        const auto in = split_tab(inputs[r]);
        if (in.size() < 5) continue;
        auto it = by_id.find(in[0]);
        if (it == by_id.end()) continue;
        InputTensor t;
        t.name = in[1];
        t.is_float = (in[2] == "f32");
        t.shape = parse_shape(in[3]);
        load_tensor(data_dir, in[4], t);
        it->second.inputs.push_back(std::move(t));
    }

    for (const auto& id : order) fx.samples.push_back(std::move(by_id[id]));
    if (fx.samples.empty()) throw std::runtime_error("no samples in fixture " + dir.string());
    return fx;
}

std::vector<Ort::Value> make_values(const Sample& s) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> vals;
    vals.reserve(s.inputs.size());
    for (const auto& t : s.inputs) {
        if (t.is_float) {
            vals.push_back(Ort::Value::CreateTensor<float>(
                mem, const_cast<float*>(t.f.data()), t.f.size(), t.shape.data(), t.shape.size()));
        } else {
            vals.push_back(Ort::Value::CreateTensor<int64_t>(
                mem, const_cast<int64_t*>(t.i.data()), t.i.size(), t.shape.data(), t.shape.size()));
        }
    }
    return vals;
}

double softmax_positive(const float* logits, int64_t count, int positive_index) {
    if (positive_index < 0 || positive_index >= count) return 0.0;
    float m = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < count; ++i) m = std::max(m, logits[i]);
    double sum = 0.0;
    for (int64_t i = 0; i < count; ++i) sum += std::exp(static_cast<double>(logits[i]) - m);
    return std::exp(static_cast<double>(logits[positive_index]) - m) / sum;
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

double file_size_mb(const fs::path& p) {
    std::error_code ec;
    const auto sz = fs::file_size(p, ec);
    return ec ? -1.0 : static_cast<double>(sz) / 1e6;
}

}  // namespace

Result run(const std::string& fixture_dir, Device device, const std::string& provider_override,
           int runs, int cpu_threads, const std::string& cache_dir) {
    const fs::path dir(fixture_dir);
    Fixture fx = load_fixture(dir);

    Result res;
    res.model = dir.filename().string();
    res.backend_name = "ONNX Runtime (classifier)";
    res.requested_device = to_string(device);
    res.runtime_version = Ort::GetVersionString();
    res.runs = std::max(1, runs);
    res.model_size_mb = file_size_mb(fx.onnx_path);
    res.model_sha256 = model_artifacts_sha256(fx.onnx_path);
    res.cache_dir = cache_dir;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "npu_inference_bench_classifier");
#ifdef NPU_INFERENCE_BENCH_WINML
    ort_common::register_windows_ml_catalog(env, device);
#endif

    // Optional VitisAI compile config: prefer one next to the fixture, else next to
    // the model. Lets us pass VAIML accuracy knobs without any CLI change.
    std::string vitis_config_file;
    for (const auto& cand : {dir / "vitisai_config.json",
                             fx.onnx_path.parent_path() / "vitisai_config.json"}) {
        std::error_code ec;
        if (fs::exists(cand, ec)) { vitis_config_file = cand.string(); break; }
    }

    std::unique_ptr<Ort::Session> session;
    std::string active_provider, last_err;
    ort_common::OperationAssignmentStats recorded_assignment;
    EngineOptions provider_options;
    provider_options.device = device;
    provider_options.device_override = provider_override;
    provider_options.cpu_threads = cpu_threads;
    provider_options.cache_dir = cache_dir;
    const auto chain = ort_common::fallback_chain(provider_options);
    res.diagnostics.requested_provider =
        provider_override.empty() ? std::string("auto") : provider_override;
    // Only the hot session is profiled: it is the one that executes inference and
    // therefore owns the provider-assignment events used by the CPU-offload audit.
    const fs::path prof_prefix = fs::temp_directory_path() / ("whal_ofl_" + res.model);
    const std::wstring prof_prefix_w = prof_prefix.wstring();

    // QNN needs ORT's EPContext mechanism in addition to the generic EP options:
    // cold session creation compiles the HTP graph and writes an embedded context
    // model; hot session creation loads that context model and skips recompilation.
    // OpenVINO and VitisAI consume cache_dir directly in append_provider().
    const auto build_session = [&](const std::string& provider, bool enable_profile) {
        fs::path qnn_ctx;
        if (lower(provider).find("qnn") != std::string::npos && !cache_dir.empty()) {
            qnn_ctx = fs::path(cache_dir) / (res.model + "_qnn_ctx.onnx");
        }

        const auto make_options = [&](bool generate_qnn_ctx) {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if ORT_API_VERSION >= 24
            if (device == Device::NPU) {
                so.AddConfigEntry("session.record_ep_graph_assignment_info", "1");
            }
#endif
            if (enable_profile) so.EnableProfiling(prof_prefix_w.c_str());
            const fs::path provider_model_dir =
                vitis_config_file.empty() ? fx.onnx_path.parent_path()
                                          : fs::path(vitis_config_file).parent_path();
            ort_common::append_provider(
                env, so, provider_options, provider, provider_model_dir, "classifier");
            if (generate_qnn_ctx) {
                const std::string p = qnn_ctx.string();
                so.AddConfigEntry("ep.context_enable", "1");
                so.AddConfigEntry("ep.context_file_path", p.c_str());
                so.AddConfigEntry("ep.context_embed_mode", "1");
            }
            return so;
        };

        if (!qnn_ctx.empty() && fs::exists(qnn_ctx)) {
            try {
                auto so = make_options(false);
                const std::wstring wctx = qnn_ctx.wstring();
                return std::make_unique<Ort::Session>(env, wctx.c_str(), so);
            } catch (const std::exception& e) {
                std::cerr << "[classifier] QNN context cache '" << qnn_ctx.string()
                          << "' unusable (" << e.what() << "); recompiling\n";
                std::error_code ec;
                fs::remove(qnn_ctx, ec);
            }
        }

        if (!qnn_ctx.empty()) {
            std::error_code ec;
            fs::create_directories(qnn_ctx.parent_path(), ec);
        }
        auto so = make_options(!qnn_ctx.empty());
        const std::wstring wpath = fx.onnx_path.wstring();
        return std::make_unique<Ort::Session>(env, wpath.c_str(), so);
    };

    // Cold creation selects the first EP that can build the model. benchmark-onnx
    // removes cache_dir before calling us, making this a genuine compile.
    for (size_t i = 0; i < chain.size(); ++i) {
        try {
            const auto t0 = std::chrono::steady_clock::now();
            session = build_session(chain[i], false);
            res.cold_load_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            active_provider = chain[i];
#if ORT_API_VERSION >= 24
            if (device == Device::NPU &&
                lower(active_provider).find("vitis") == std::string::npos) {
                recorded_assignment =
                    ort_common::query_ort_operation_assignment(*session);
            }
#endif
            res.diagnostics.attempts.push_back({chain[i], true, {}});
            break;
        } catch (const std::exception& e) {
            last_err = e.what();
            res.diagnostics.attempts.push_back({chain[i], false, last_err});
            session.reset();
            if (i + 1 < chain.size()) {
                std::cerr << "[classifier] EP '" << chain[i] << "' unavailable (" << e.what()
                          << "); falling back to '" << chain[i + 1] << "'\n";
            }
        }
    }
    if (!session) {
        throw std::runtime_error("no ONNX Runtime EP could build a classifier session (last error: " +
                                 last_err + ")");
    }

    // Destroy the cold session, then recreate the same selected EP from its newly
    // populated cache. Do not run the fallback chain again: changing EPs would make
    // the cold/hot pair incomparable.
    session.reset();
    {
        const auto t0 = std::chrono::steady_clock::now();
        session = build_session(active_provider, true);
        res.hot_load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    res.execution_provider = active_provider;
    res.runtime = ort_common::runtime_for(active_provider);
    res.inference_precision =
        ort_common::resolved_inference_precision(provider_options, active_provider);
    res.diagnostics.resolved_provider = active_provider;
    res.diagnostics.fallback_occurred =
        !chain.empty() && active_provider != chain.front();

    Ort::AllocatorWithDefaultOptions alloc;
    std::string out_name = session->GetOutputNameAllocated(0, alloc).get();
    const char* out_names[] = {out_name.c_str()};

    std::vector<double> lat_ms;
    lat_ms.reserve(fx.samples.size() * static_cast<size_t>(res.runs));

    for (const auto& s : fx.samples) {
        std::vector<const char*> in_names;
        in_names.reserve(s.inputs.size());
        for (const auto& t : s.inputs) in_names.push_back(t.name.c_str());

        auto values = make_values(s);
        // Warmup (untimed): first touch compiles/loads the EP graph for this shape.
        session->Run(Ort::RunOptions{nullptr}, in_names.data(), values.data(), values.size(),
                     out_names, 1);

        std::vector<Ort::Value> last_out;
        for (int r = 0; r < res.runs; ++r) {
            auto vals = make_values(s);
            const auto t0 = std::chrono::steady_clock::now();
            last_out = session->Run(Ort::RunOptions{nullptr}, in_names.data(), vals.data(),
                                    vals.size(), out_names, 1);
            lat_ms.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }

        const float* logits = last_out[0].GetTensorMutableData<float>();
        const auto shape = last_out[0].GetTensorTypeAndShapeInfo().GetShape();
        const int64_t classes = shape.empty() ? 0 : shape.back();
        const double p = softmax_positive(logits, classes, fx.positive_index);
        if (!std::isfinite(p)) {
            double measured_ms = 0.0;
            for (double latency : lat_ms) measured_ms += latency;
            if (!lat_ms.empty()) measured_ms /= static_cast<double>(lat_ms.size());
            throw std::runtime_error(
                "non-finite classifier probability for sample '" + s.id +
                "' on provider '" + active_provider +
                "' (mean inference before rejection: " +
                std::to_string(measured_ms) + " ms)");
        }

        SampleResult sr;
        sr.id = s.id;
        sr.label = s.label;
        sr.expected_predicted = s.expected_pred;
        sr.p = p;
        sr.expected_p = s.expected_p;
        sr.predicted = (p > fx.threshold) ? fx.positive_label : fx.negative_label;
        sr.has_label = (s.label != "unknown" && !s.label.empty() && s.label != "?");
        sr.correct = sr.has_label && (sr.predicted == s.label);
        if (sr.has_label) { res.eval_samples += 1; res.correct += sr.correct ? 1 : 0; }
        res.max_abs_p_diff = std::max(res.max_abs_p_diff, std::abs(p - s.expected_p));
        res.samples.push_back(std::move(sr));
    }

    double mean = 0.0;
    for (double l : lat_ms) mean += l;
    res.mean_infer_ms = lat_ms.empty() ? 0.0 : mean / static_cast<double>(lat_ms.size());
    res.median_infer_ms = percentile(lat_ms, 50.0);
    res.p90_infer_ms = percentile(lat_ms, 90.0);

    // CPU-offload audit: flush the profile trace and tally the per-node provider
    // assignments (deduped across all runs). Best-effort -- never fail the benchmark.
    try {
        const std::string prof_path = session->EndProfilingAllocated(alloc).get();
        const auto st = ort_common::parse_ort_profile(prof_path);
        if (st.measured) {
#ifdef NPU_INFERENCE_BENCH_WINML
            if (!st.providers.empty()) {
                const auto comma = st.providers.find(',');
                const std::string resolved = st.providers.substr(0, comma);
                res.execution_provider = resolved;
                res.diagnostics.resolved_provider = resolved;
                res.diagnostics.fallback_occurred =
                    res.diagnostics.fallback_occurred ||
                    ort_common::is_fallback_provider_for_device(device, resolved);
            }
#endif
            res.offload_measured = true;
            res.ep_nodes = st.ep_nodes;
            res.cpu_nodes = st.cpu_nodes;
            res.cpu_offload_ops = st.cpu_ops;
            res.diagnostics.offload_measured = true;
            res.diagnostics.ep_nodes = st.ep_nodes;
            res.diagnostics.cpu_nodes = st.cpu_nodes;
            res.diagnostics.cpu_offload_ops = st.cpu_ops;
        }
        std::error_code ec;
        fs::remove(prof_path, ec);
    } catch (const std::exception&) {
        // profiling unavailable / parse failed -> leave offload unmeasured (-1)
    }
    if (lower(res.execution_provider).find("vitis") != std::string::npos) {
        const auto assignment = ort_common::parse_vitis_operation_assignment(
            fs::path(cache_dir), {"classifier"});
        if (assignment.measured) {
            res.operation_assignment_measured = true;
            res.assigned_ops_cpu = assignment.cpu_operations;
            res.assigned_ops_npu = assignment.npu_operations;
            res.operation_assignment_source = assignment.source;
            res.diagnostics.operation_assignment_measured = true;
            res.diagnostics.assigned_ops_cpu = assignment.cpu_operations;
            res.diagnostics.assigned_ops_npu = assignment.npu_operations;
            res.diagnostics.operation_assignment_source = assignment.source;
        }
    }
#if ORT_API_VERSION >= 24
    else if (device == Device::NPU && !res.diagnostics.fallback_occurred) {
        const auto assignment = recorded_assignment.measured
                                    ? recorded_assignment
                                    : ort_common::query_ort_operation_assignment(*session);
        if (assignment.measured) {
            res.operation_assignment_measured = true;
            res.assigned_ops_cpu = assignment.cpu_operations;
            res.assigned_ops_npu = assignment.npu_operations;
            res.operation_assignment_source = assignment.source;
            res.diagnostics.operation_assignment_measured = true;
            res.diagnostics.assigned_ops_cpu = assignment.cpu_operations;
            res.diagnostics.assigned_ops_npu = assignment.npu_operations;
            res.diagnostics.operation_assignment_source = assignment.source;
        }
    }
#endif
    return res;
}

bool available() { return true; }

#else  // !NPU_INFERENCE_BENCH_ORT

Result run(const std::string&, Device, const std::string&, int, int, const std::string&) {
    throw std::runtime_error(
        "classifier harness not compiled: build with an ONNX Runtime backend "
        "(e.g. tools\\build\\build.ps1 -EnableOrt -DisableIntel).");
}

bool available() { return false; }

#endif

}  // namespace classifier
}  // namespace npu_inference_bench
