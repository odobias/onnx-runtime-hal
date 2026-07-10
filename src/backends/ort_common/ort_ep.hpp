// Shared ONNX Runtime execution-provider plumbing for the ORT-based Whisper
// backends (ort_static = static no-KV recompute; ort_dynamic = with-past KV
// cache). Both self-select across Intel (OpenVINO EP) / AMD (VitisAI) / Qualcomm
// (QNN) / any-DX12 GPU (DirectML) / CPU from ONE binary: the engine walks a
// device fallback chain and keeps the first EP that actually builds a session.
// This header is the single source of truth for that EP selection so the two
// engines can't drift apart. All functions are inline (header-only, included by
// multiple TUs).
#pragma once

#include "whisper_npu/whisper_engine.hpp"

#ifdef WHISPER_HAL_ORT

#include <onnxruntime_cxx_api.h>
#if defined(_WIN32) && __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define WHISPER_HAL_ORT_HAS_DML 1
#endif
#if defined(WHISPER_HAL_QUALCOMM) && defined(_WIN32)
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "backends/ort_common/ort_offload.hpp"

namespace whisper_npu {
namespace ort_common {

namespace fs = std::filesystem;

inline constexpr const char* kQnnEpName = "QNNExecutionProvider";

inline std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline bool is_qnn(const std::string& provider) {
    return lower(provider).find("qnn") != std::string::npos;
}

// Ordered EP candidates for a logical device. Every vendor EP is compiled into
// the same binary (their Append* calls resolve through the ORT API table at
// runtime), so a single build self-selects across Intel / AMD / Qualcomm / any
// DX12 GPU: the caller walks these and keeps the first that builds a session.
inline std::vector<std::string> providers_for(Device device) {
    switch (device) {
        case Device::NPU: {
            std::vector<std::string> v;
#ifdef WHISPER_HAL_QUALCOMM
            v.push_back(kQnnEpName);                  // Qualcomm Hexagon (Snapdragon)
#endif
            v.push_back("VitisAIExecutionProvider");  // AMD XDNA (Ryzen AI)
            v.push_back("openvino:NPU");              // Intel AI Boost
            return v;
        }
        case Device::GPU:
            // Portable GPU: DirectML drives any DX12 GPU; OpenVINO GPU is the
            // Intel-specific alternative when DML is absent.
            return {"DmlExecutionProvider", "openvino:GPU"};
        case Device::CPU:
        default:
            return {"CPUExecutionProvider"};
    }
}

// Human-facing runtime tag for the shared benchmark schema, from the EP that
// actually built the session.
inline std::string runtime_for(const std::string& provider) {
    const std::string p = lower(provider);
    if (p.find("openvino") != std::string::npos) return "onnxruntime-openvino";
    if (p.find("qnn") != std::string::npos) return "onnxruntime-qnn";
    if (p.find("vitis") != std::string::npos) return "onnxruntime-vitisai";
    if (p.find("dml") != std::string::npos || p.find("directml") != std::string::npos)
        return "onnxruntime-directml";
    return "onnxruntime";
}

// The runtime EP fallback chain for a requested device. An explicit provider
// override is honored verbatim (single attempt). Otherwise NPU walks down to GPU
// then CPU, and GPU walks down to CPU, so one binary always produces a result on
// whatever hardware/drivers are actually present.
inline std::vector<std::string> fallback_chain(const EngineOptions& options) {
    if (!options.device_override.empty()) return {options.device_override};
    std::vector<std::string> chain;
    const auto add = [&](Device d) {
        for (auto& p : providers_for(d)) chain.push_back(p);
    };
    switch (options.device) {
        case Device::NPU: add(Device::NPU); add(Device::GPU); add(Device::CPU); break;
        case Device::GPU: add(Device::GPU); add(Device::CPU); break;
        case Device::CPU:
        default:          add(Device::CPU); break;
    }
    return chain;
}

#ifdef WHISPER_HAL_QUALCOMM
inline std::string qnn_ep_library_path() {
    return env_or("WHISPER_QNN_EP_DLL", "onnxruntime_providers_qnn.dll");
}

inline std::string qnn_backend_path(Device device) {
    switch (device) {
        case Device::GPU: return env_or("WHISPER_QNN_GPU_DLL", "QnnGpu.dll");
        case Device::CPU: return env_or("WHISPER_QNN_CPU_DLL", "QnnCpu.dll");
        case Device::NPU:
        default:          return env_or("WHISPER_QNN_HTP_DLL", "QnnHtp.dll");
    }
}

#ifdef _WIN32
inline std::wstring ort_tstring(const std::string& value) {
    if (value.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (size <= 0) throw std::runtime_error("Failed to convert path to UTF-16: " + value);
    std::wstring out(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, out.data(), size);
    return out;
}
#endif

inline void register_qnn_library(Ort::Env& env) {
    // EP-library registration lives on the ORT environment; may be called more
    // than once (probe + engine env), so swallow the "already registered" error.
    try {
#ifdef _WIN32
        env.RegisterExecutionProviderLibrary(kQnnEpName, ort_tstring(qnn_ep_library_path()));
#else
        env.RegisterExecutionProviderLibrary(kQnnEpName, qnn_ep_library_path());
#endif
    } catch (const Ort::Exception&) {
        // Already registered on this (or a shared) environment -- fine.
    }
}

inline Ort::ConstEpDevice find_qnn_device(Ort::Env& env) {
    for (Ort::ConstEpDevice ep_device : env.GetEpDevices()) {
        if (std::strcmp(ep_device.EpName(), kQnnEpName) == 0) return ep_device;
    }
    throw std::runtime_error("QNNExecutionProvider device not found after registration");
}
#endif  // WHISPER_HAL_QUALCOMM

// Append the requested execution provider to `so`. `model_dir`/`cache_key` feed
// VitisAI's config + blob cache; `options.cache_dir` also drives OpenVINO's blob
// cache. Unknown provider strings throw.
inline void append_provider(Ort::Env& env, Ort::SessionOptions& so, const EngineOptions& options,
                            const std::string& provider, const fs::path& model_dir,
                            const std::string& cache_key) {
    const std::string p = lower(provider);

    // Intel native path: OpenVINO EP. Accepts "openvino", "openvinoexecutionprovider",
    // or an "openvino:<DEVICE>" token; device_type comes from the suffix when present.
    if (p.rfind("openvino", 0) == 0) {
        std::string device_type;
        const auto colon = provider.find(':');
        if (colon != std::string::npos) {
            device_type = provider.substr(colon + 1);
        } else {
            device_type = options.device == Device::NPU   ? "NPU"
                          : options.device == Device::GPU ? "GPU"
                                                          : "CPU";
        }
        std::unordered_map<std::string, std::string> ov_opts{{"device_type", device_type}};
        if (!options.cache_dir.empty()) ov_opts["cache_dir"] = options.cache_dir;
        so.AppendExecutionProvider_OpenVINO_V2(ov_opts);
        return;
    }

    if (p == "cpu" || p == "cpuexecutionprovider") {
        if (options.cpu_threads > 0) {
            so.SetIntraOpNumThreads(options.cpu_threads);
            so.SetInterOpNumThreads(1);
        }
        return;  // CPU EP is the default ORT fallback.
    }

    if (p == "qnn" || p == "qnnexecutionprovider") {
#ifdef WHISPER_HAL_QUALCOMM
        register_qnn_library(env);
        const Ort::ConstEpDevice qnn_device = find_qnn_device(env);  // throws if absent
        std::vector<Ort::ConstEpDevice> selected{qnn_device};
        std::unordered_map<std::string, std::string> opts{
            {"backend_path", qnn_backend_path(options.device)}};
        if (options.device == Device::NPU) opts.emplace("htp_performance_mode", "burst");
        Ort::KeyValuePairs ep_options(opts);
        so.AppendExecutionProvider_V2(env, selected, ep_options);
        return;
#else
        throw std::runtime_error("QNN execution provider not compiled into this build");
#endif
    }

    if (p == "vitisai" || p == "vitisaiexecutionprovider") {
        std::unordered_map<std::string, std::string> vitis_opts;
        const fs::path config = model_dir / "vitisai_config.json";
        if (fs::exists(config)) vitis_opts["config_file"] = config.string();
        if (!options.cache_dir.empty()) {
            vitis_opts["cache_dir"] = options.cache_dir;
            vitis_opts["cache_key"] = cache_key;
        }
        so.AppendExecutionProvider_VitisAI(vitis_opts);
        return;
    }

    if (p == "dml" || p == "directml" || p == "dmlexecutionprovider") {
#ifdef WHISPER_HAL_ORT_HAS_DML
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
        return;
#else
        throw std::runtime_error(
            "ONNX Runtime backend was built without DirectML provider headers; use CPU/NPU");
#endif
    }

    throw std::runtime_error(
        "unsupported ONNX Runtime provider override: " + provider +
        " (supported: OpenVINOExecutionProvider[:NPU|GPU|CPU], CPUExecutionProvider, "
        "QNNExecutionProvider, DmlExecutionProvider, VitisAIExecutionProvider)");
}

// --- CPU-offload audit -------------------------------------------------------

// Turn on ORT profiling for a session so a later measure_offload() can tell which
// EP each executed node actually ran on. The prefix must be unique per session
// (ORT appends a timestamp + ".json"); callers derive it from the cache_key. Kept
// here so both Whisper engines profile identically.
inline void enable_offload_profiling(Ort::SessionOptions& so, const std::string& tag) {
    const std::wstring prefix = (fs::temp_directory_path() / ("whal_ofl_" + tag)).wstring();
    so.EnableProfiling(prefix.c_str());
}

// End profiling on each session (built with enable_offload_profiling), parse the
// emitted traces, and fold them into ONE pipeline-wide OffloadInfo. Call this once,
// after the first (warmup) inference: profiling stops here, so the timed runs that
// follow are pristine. Robust to profiling being unavailable (-> unmeasured).
inline OffloadInfo measure_offload(std::initializer_list<Ort::Session*> sessions) {
    Ort::AllocatorWithDefaultOptions alloc;
    OffloadStats agg;
    for (Ort::Session* s : sessions) {
        if (!s) continue;
        try {
            const std::string path = s->EndProfilingAllocated(alloc).get();
            if (path.empty()) continue;
            agg.add(parse_ort_profile(path));
            std::error_code ec;
            fs::remove(path, ec);
        } catch (const std::exception&) {
            // profiling unavailable / parse failed -> skip this session
        }
    }
    OffloadInfo info;
    if (agg.measured) {
        info.measured = true;
        info.ep_nodes = agg.ep_nodes;
        info.cpu_nodes = agg.cpu_nodes;
        info.cpu_ops = agg.cpu_ops;
    }
    return info;
}

}  // namespace ort_common
}  // namespace whisper_npu

#endif  // WHISPER_HAL_ORT
