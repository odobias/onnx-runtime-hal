// ONNX Runtime classifier harness: replays the deepfake pipeline's pre-baked,
// validated fixture tensors (scripts/experiments/dump_fixtures.py) across ORT
// execution providers and scores per-EP latency + correctness + numerical
// agreement vs the CPU reference. See include/whisper_npu/classifier.hpp.
#include "whisper_npu/classifier.hpp"

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

#ifdef WHISPER_HAL_ORT
#include <onnxruntime_cxx_api.h>
#include "backends/ort_common/ort_offload.hpp"
#if defined(_WIN32) && __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define WHISPER_HAL_CLS_HAS_DML 1
#endif
#if defined(WHISPER_HAL_QUALCOMM) && defined(_WIN32)
#include <windows.h>
#endif
#endif

namespace whisper_npu {
namespace classifier {

#ifdef WHISPER_HAL_ORT

namespace {

namespace fs = std::filesystem;

constexpr const char* kQnnEpName = "QNNExecutionProvider";

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

std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

std::string runtime_for(const std::string& provider) {
    const std::string p = lower(provider);
    if (p.find("openvino") != std::string::npos) return "onnxruntime-openvino";
    if (p.find("qnn") != std::string::npos) return "onnxruntime-qnn";
    if (p.find("vitis") != std::string::npos) return "onnxruntime-vitisai";
    if (p.find("dml") != std::string::npos || p.find("directml") != std::string::npos)
        return "onnxruntime-directml";
    return "onnxruntime";
}

std::vector<std::string> providers_for(Device device) {
    switch (device) {
        case Device::NPU: {
            std::vector<std::string> v;
#ifdef WHISPER_HAL_QUALCOMM
            v.push_back(kQnnEpName);
#endif
            v.push_back("VitisAIExecutionProvider");
            v.push_back("openvino:NPU");
            return v;
        }
        case Device::GPU:
            return {"DmlExecutionProvider", "openvino:GPU"};
        case Device::CPU:
        default:
            return {"CPUExecutionProvider"};
    }
}

std::vector<std::string> fallback_chain(Device device, const std::string& provider_override) {
    if (!provider_override.empty()) return {provider_override};
    std::vector<std::string> chain;
    const auto add = [&](Device d) { for (auto& p : providers_for(d)) chain.push_back(p); };
    switch (device) {
        case Device::NPU: add(Device::NPU); add(Device::GPU); add(Device::CPU); break;
        case Device::GPU: add(Device::GPU); add(Device::CPU); break;
        case Device::CPU:
        default:          add(Device::CPU); break;
    }
    return chain;
}

#if defined(WHISPER_HAL_QUALCOMM) && defined(_WIN32)
std::wstring ort_tstring(const std::string& value) {
    if (value.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (size <= 0) throw std::runtime_error("Failed to convert path to UTF-16: " + value);
    std::wstring out(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, out.data(), size);
    return out;
}
#endif

#ifdef WHISPER_HAL_QUALCOMM
void register_qnn_library(Ort::Env& env) {
    try {
#ifdef _WIN32
        env.RegisterExecutionProviderLibrary(
            kQnnEpName, ort_tstring(env_or("WHISPER_QNN_EP_DLL", "onnxruntime_providers_qnn.dll")));
#else
        env.RegisterExecutionProviderLibrary(
            kQnnEpName, env_or("WHISPER_QNN_EP_DLL", "onnxruntime_providers_qnn.dll"));
#endif
    } catch (const Ort::Exception&) {
        // Already registered on this environment -- fine.
    }
}

Ort::ConstEpDevice find_qnn_device(Ort::Env& env) {
    for (Ort::ConstEpDevice d : env.GetEpDevices()) {
        if (std::strcmp(d.EpName(), kQnnEpName) == 0) return d;
    }
    throw std::runtime_error("QNNExecutionProvider device not found after registration");
}
#endif

void append_provider(Ort::Env& env, Ort::SessionOptions& so, const std::string& provider,
                     Device device, int cpu_threads, const std::string& cache_dir) {
    const std::string p = lower(provider);

    if (p.rfind("openvino", 0) == 0) {
        std::string device_type;
        const auto colon = provider.find(':');
        if (colon != std::string::npos) device_type = provider.substr(colon + 1);
        else device_type = device == Device::NPU ? "NPU" : device == Device::GPU ? "GPU" : "CPU";
        std::unordered_map<std::string, std::string> ov_opts{{"device_type", device_type}};
        if (!cache_dir.empty()) ov_opts["cache_dir"] = cache_dir;
        so.AppendExecutionProvider_OpenVINO_V2(ov_opts);
        return;
    }
    if (p == "cpu" || p == "cpuexecutionprovider") {
        if (cpu_threads > 0) { so.SetIntraOpNumThreads(cpu_threads); so.SetInterOpNumThreads(1); }
        return;
    }
    if (p == "dml" || p == "directml" || p == "dmlexecutionprovider") {
#ifdef WHISPER_HAL_CLS_HAS_DML
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
        return;
#else
        throw std::runtime_error("classifier harness built without DirectML provider headers");
#endif
    }
    if (p == "vitisai" || p == "vitisaiexecutionprovider") {
        std::unordered_map<std::string, std::string> vitis_opts;
        if (!cache_dir.empty()) { vitis_opts["cache_dir"] = cache_dir; vitis_opts["cache_key"] = "classifier"; }
        so.AppendExecutionProvider_VitisAI(vitis_opts);
        return;
    }
    if (p == "qnn" || p == "qnnexecutionprovider") {
#ifdef WHISPER_HAL_QUALCOMM
        register_qnn_library(env);
        const Ort::ConstEpDevice qnn_device = find_qnn_device(env);
        std::vector<Ort::ConstEpDevice> selected{qnn_device};
        std::unordered_map<std::string, std::string> opts{
            {"backend_path", env_or("WHISPER_QNN_HTP_DLL", "QnnHtp.dll")}};
        if (device == Device::NPU) opts.emplace("htp_performance_mode", "burst");
        Ort::KeyValuePairs ep_options(opts);
        so.AppendExecutionProvider_V2(env, selected, ep_options);
        return;
#else
        throw std::runtime_error("QNN execution provider not compiled into this build");
#endif
    }
    throw std::runtime_error("unsupported classifier EP override: " + provider);
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

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "whisper_hal_classifier");

    std::unique_ptr<Ort::Session> session;
    std::string active_provider, last_err;
    const auto chain = fallback_chain(device, provider_override);
    // Profile the session so we can audit CPU offload after the run. The accelerator
    // EPs fuse their subgraph into one node, so profiling adds a sub-microsecond
    // per-run cost on the interesting (NPU/GPU) paths -- well inside timing noise.
    const fs::path prof_prefix = fs::temp_directory_path() / ("whal_ofl_" + res.model);
    const std::wstring prof_prefix_w = prof_prefix.wstring();
    for (size_t i = 0; i < chain.size(); ++i) {
        try {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            so.EnableProfiling(prof_prefix_w.c_str());
            append_provider(env, so, chain[i], device, cpu_threads, cache_dir);
            const auto t0 = std::chrono::steady_clock::now();
            const std::wstring wpath = fx.onnx_path.wstring();
            session = std::make_unique<Ort::Session>(env, wpath.c_str(), so);
            res.load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            active_provider = chain[i];
            break;
        } catch (const std::exception& e) {
            last_err = e.what();
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

    res.execution_provider = active_provider;
    res.runtime = runtime_for(active_provider);

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

        SampleResult sr;
        sr.id = s.id;
        sr.label = s.label;
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
            res.offload_measured = true;
            res.ep_nodes = st.ep_nodes;
            res.cpu_nodes = st.cpu_nodes;
            res.cpu_offload_ops = st.cpu_ops;
        }
        std::error_code ec;
        fs::remove(prof_path, ec);
    } catch (const std::exception&) {
        // profiling unavailable / parse failed -> leave offload unmeasured (-1)
    }
    return res;
}

bool available() { return true; }

#else  // !WHISPER_HAL_ORT

Result run(const std::string&, Device, const std::string&, int, int, const std::string&) {
    throw std::runtime_error(
        "classifier harness not compiled: build with an ONNX Runtime backend "
        "(e.g. scripts\\build.ps1 -EnableOrt -DisableIntel).");
}

bool available() { return false; }

#endif

}  // namespace classifier
}  // namespace whisper_npu
