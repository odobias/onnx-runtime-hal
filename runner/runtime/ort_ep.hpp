// Shared ONNX Runtime execution-provider plumbing for the ORT-based Whisper
// backends (ort_static = static no-KV recompute; ort_dynamic = with-past KV
// cache). Both self-select across Intel (OpenVINO EP) / AMD (VitisAI) / Qualcomm
// (QNN) / any-DX12 GPU (DirectML) / CPU from ONE binary: the engine walks a
// device fallback chain and keeps the first EP that actually builds a session.
// This header is the single source of truth for that EP selection so the two
// engines can't drift apart. All functions are inline (header-only, included by
// multiple TUs).
#pragma once

#include "npu_inference_bench/whisper.hpp"
#include "npu_inference_bench/precision_policy.hpp"

#ifdef NPU_INFERENCE_BENCH_ORT

#include <onnxruntime_cxx_api.h>
#ifdef NPU_INFERENCE_BENCH_WINML
#include <windows.h>
#include <WinMLEpCatalog.h>
#endif
#if defined(_WIN32) && __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define NPU_INFERENCE_BENCH_ORT_HAS_DML 1
#endif
#if defined(NPU_INFERENCE_BENCH_QUALCOMM) && defined(_WIN32) && !defined(NPU_INFERENCE_BENCH_WINML)
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace npu_inference_bench {
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

inline std::string openvino_device_type(const EngineOptions& options,
                                        const std::string& provider) {
    const auto colon = provider.find(':');
    if (colon != std::string::npos) return provider.substr(colon + 1);
    return options.device == Device::NPU   ? "NPU"
           : options.device == Device::GPU ? "GPU"
                                           : "CPU";
}

#ifdef NPU_INFERENCE_BENCH_WINML
inline Device winml_token_device(const std::string& provider, Device fallback) {
    const std::string p = lower(provider);
    if (p.find("prefer_cpu") != std::string::npos) return Device::CPU;
    if (p.find("prefer_gpu") != std::string::npos) return Device::GPU;
    if (p.find("prefer_npu") != std::string::npos) return Device::NPU;
    return fallback;
}
#endif

// Provider-neutral policy. CPU/GPU default to reproducible FP32; NPU defaults
// to device-preferred because accelerator compilers generally fix precision.
inline std::string inference_precision_policy(const EngineOptions& options,
                                              const std::string& provider) {
    const std::string p = lower(provider);
#ifdef NPU_INFERENCE_BENCH_WINML
    const Device winml_device = winml_token_device(provider, options.device);
    if (p.rfind("windowsml:", 0) == 0 && winml_device != Device::CPU) {
        // The policy selector owns accelerator precision. Use "preferred" by
        // default; an explicit f32/f16/bf16 request remains explicit and is
        // rejected below because Windows ML cannot guarantee it provider-wide.
        return precision_policy::resolve(options, Device::NPU);
    }
#endif
    Device actual_device = options.device;
#ifdef NPU_INFERENCE_BENCH_WINML
    if (p.rfind("windowsml:", 0) == 0) actual_device = winml_device;
#endif
    if (p.find("cpu") != std::string::npos) {
        actual_device = Device::CPU;
    } else if (p.find("dml") != std::string::npos ||
               p.find("directml") != std::string::npos) {
        actual_device = Device::GPU;
    } else if (p.rfind("openvino", 0) == 0) {
        const std::string device = lower(openvino_device_type(options, provider));
        actual_device = device.rfind("cpu", 0) == 0 ? Device::CPU
                      : device.rfind("gpu", 0) == 0 ? Device::GPU
                                                    : Device::NPU;
    }
    return precision_policy::resolve(options, actual_device);
}

inline void validate_inference_precision(const EngineOptions& options,
                                         const std::string& provider) {
    const std::string precision = inference_precision_policy(options, provider);
    const std::string p = lower(provider);
    if (p.find("openvino") != std::string::npos) return;
    if (p.find("cpu") != std::string::npos || p.find("dml") != std::string::npos ||
        p.find("directml") != std::string::npos) {
        if (precision == "f32" || precision == "preferred") return;
        throw std::runtime_error(
            provider + " cannot apply a provider-wide " + precision +
            " conversion; use a model exported in that precision");
    }
    if (precision != "preferred") {
        throw std::runtime_error(
            provider + " cannot guarantee provider-wide " + precision +
            "; use precision=preferred or a compiler/model-specific configuration");
    }
}

inline std::string resolved_inference_precision(const EngineOptions& options,
                                                const std::string& provider) {
    validate_inference_precision(options, provider);
    return inference_precision_policy(options, provider);
}

inline bool is_qnn(const std::string& provider) {
    return lower(provider).find("qnn") != std::string::npos;
}

inline bool is_fallback_provider_for_device(Device requested_device,
                                            const std::string& resolved_provider) {
    const std::string provider = lower(resolved_provider);
    const bool cpu = provider == "cpuexecutionprovider";
    if (requested_device == Device::NPU) {
        return cpu || provider.find("dml") != std::string::npos ||
               provider.find("directml") != std::string::npos;
    }
    return requested_device == Device::GPU && cpu;
}

#ifdef NPU_INFERENCE_BENCH_WINML
inline std::string winml_policy_token(Device device) {
    switch (device) {
        case Device::NPU: return "WindowsML:PREFER_NPU";
        case Device::GPU: return "WindowsML:PREFER_GPU";
        case Device::CPU:
        default:          return "WindowsML:PREFER_CPU";
    }
}
#endif

// Ordered EP candidates for a logical device. Every vendor EP is compiled into
// the same binary (their Append* calls resolve through the ORT API table at
// runtime), so a single build self-selects across Intel / AMD / Qualcomm / any
// DX12 GPU: the caller walks these and keeps the first that builds a session.
inline std::vector<std::string> providers_for(Device device) {
#ifdef NPU_INFERENCE_BENCH_WINML
    return {winml_policy_token(device)};
#else
    switch (device) {
        case Device::NPU: {
            std::vector<std::string> v;
#ifdef NPU_INFERENCE_BENCH_QUALCOMM
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
#endif
}

// Human-facing runtime tag for the shared benchmark schema, from the EP that
// actually built the session.
inline std::string runtime_for(const std::string& provider) {
#ifdef NPU_INFERENCE_BENCH_WINML
    (void)provider;
    return "windows-ml";
#else
    const std::string p = lower(provider);
    if (p.find("openvino") != std::string::npos) return "onnxruntime-openvino";
    if (p.find("qnn") != std::string::npos) return "onnxruntime-qnn";
    if (p.find("vitis") != std::string::npos) return "onnxruntime-vitisai";
    if (p.find("dml") != std::string::npos || p.find("directml") != std::string::npos)
        return "onnxruntime-directml";
    return "onnxruntime";
#endif
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

#ifdef NPU_INFERENCE_BENCH_WINML
namespace winml_detail {
struct RegisterContext {
    Ort::Env* env = nullptr;
    Device requested_device = Device::CPU;
    std::string errors;
};

inline BOOL CALLBACK register_provider(WinMLEpHandle handle, const WinMLEpInfo* info,
                                       void* opaque) {
    auto* ctx = static_cast<RegisterContext*>(opaque);
    if (!ctx || !ctx->env || !info || !info->name ||
        info->certification != WinMLEpCertification_Certified) {
        return TRUE;
    }

    // CPU and GPU are supplied by the Windows ML runtime itself. NPU preference
    // may acquire the compatible vendor EP on first use; otherwise only register
    // providers already installed on this machine.
    if (info->readyState == WinMLEpReadyState_NotPresent &&
        ctx->requested_device != Device::NPU) {
        return TRUE;
    }

    HRESULT hr = WinMLEpEnsureReady(handle);
    if (FAILED(hr)) {
        ctx->errors += std::string(info->name) + " ensure failed (HRESULT " +
                       std::to_string(static_cast<unsigned long>(hr)) + "); ";
        return TRUE;
    }

    size_t path_size = 0;
    hr = WinMLEpGetLibraryPathSize(handle, &path_size);
    if (FAILED(hr) || path_size == 0) return TRUE;
    std::string path(path_size, '\0');
    hr = WinMLEpGetLibraryPath(handle, path.size(), path.data(), nullptr);
    if (FAILED(hr)) return TRUE;
    if (!path.empty() && path.back() == '\0') path.pop_back();

    try {
        ctx->env->RegisterExecutionProviderLibrary(info->name, fs::path(path).wstring());
    } catch (const Ort::Exception& e) {
        // Duplicate registration is harmless; a real incompatibility is retained
        // for the session-policy failure message.
        const std::string message = e.what();
        if (lower(message).find("already") == std::string::npos) {
            ctx->errors += std::string(info->name) + " register failed (" + message + "); ";
        }
    }
    return TRUE;
}
}  // namespace winml_detail

inline void register_windows_ml_catalog(Ort::Env& env, Device requested_device) {
    WinMLEpCatalogHandle catalog = nullptr;
    const HRESULT create_hr = WinMLEpCatalogCreate(&catalog);
    if (FAILED(create_hr) || !catalog) {
        throw std::runtime_error(
            "Windows ML Execution Provider Catalog initialization failed (HRESULT " +
            std::to_string(static_cast<unsigned long>(create_hr)) + ")");
    }
    winml_detail::RegisterContext context{&env, requested_device, {}};
    const HRESULT enum_hr =
        WinMLEpCatalogEnumProviders(catalog, winml_detail::register_provider, &context);
    WinMLEpCatalogRelease(catalog);
    if (FAILED(enum_hr)) {
        throw std::runtime_error(
            "Windows ML Execution Provider Catalog enumeration failed (HRESULT " +
            std::to_string(static_cast<unsigned long>(enum_hr)) + ")");
    }
    if (!context.errors.empty()) {
        std::cerr << "[npu-inference-bench] Windows ML catalog: " << context.errors << "\n";
    }
}
#endif

#ifdef NPU_INFERENCE_BENCH_QUALCOMM
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
#endif  // NPU_INFERENCE_BENCH_QUALCOMM

// Append the requested execution provider to `so`. `model_dir`/`cache_key` feed
// VitisAI's config + blob cache; `options.cache_dir` also drives OpenVINO's blob
// cache. Unknown provider strings throw.
inline void append_provider(Ort::Env& env, Ort::SessionOptions& so, const EngineOptions& options,
                            const std::string& provider, const fs::path& model_dir,
                            const std::string& cache_key) {
    const std::string p = lower(provider);

#ifdef NPU_INFERENCE_BENCH_WINML
    if (p.rfind("windowsml:", 0) == 0 || p == "windowsml" || p == "winml") {
        const Device selected_device = winml_token_device(provider, options.device);
        const OrtExecutionProviderDevicePolicy policy =
            selected_device == Device::NPU ? OrtExecutionProviderDevicePolicy_PREFER_NPU
            : selected_device == Device::GPU ? OrtExecutionProviderDevicePolicy_PREFER_GPU
                                             : OrtExecutionProviderDevicePolicy_PREFER_CPU;
        if (lower(env_or("NPU_INFERENCE_BENCH_WINML_SELECTION", "explicit")) == "policy") {
            so.SetEpSelectionPolicy(policy);
            return;
        }

        if (selected_device != Device::CPU) {
            const OrtHardwareDeviceType target =
                selected_device == Device::NPU ? OrtHardwareDeviceType_NPU
                                               : OrtHardwareDeviceType_GPU;
            std::vector<Ort::ConstEpDevice> compatible;
            for (Ort::ConstEpDevice candidate : env.GetEpDevices()) {
                if (candidate.Device().Type() == target) compatible.push_back(candidate);
            }
            if (compatible.empty()) {
                throw std::runtime_error(
                    "Windows ML has no registered EP device for " +
                    std::string(selected_device == Device::NPU ? "NPU" : "GPU"));
            }
            if (selected_device == Device::NPU) {
                // Multiple catalog EPs may expose NPU devices. ORT rejects a
                // mixed-EP device vector, so prefer a vendor NPU EP over DML
                // and append one hardware-typed NPU device.
                std::stable_sort(
                    compatible.begin(), compatible.end(),
                    [](Ort::ConstEpDevice a, Ort::ConstEpDevice b) {
                        const bool a_dml = lower(a.EpName()).find("dml") != std::string::npos;
                        const bool b_dml = lower(b.EpName()).find("dml") != std::string::npos;
                        return !a_dml && b_dml;
                    });
                compatible.resize(1);
            } else if (selected_device == Device::GPU) {
                // PREFER_GPU may choose Qualcomm's QNN GPU device. DirectML is
                // the portable and stable Windows GPU path, so prefer it when
                // explicitly selecting a benchmark device.
                std::stable_sort(
                    compatible.begin(), compatible.end(),
                    [](Ort::ConstEpDevice a, Ort::ConstEpDevice b) {
                        const bool a_dml = lower(a.EpName()).find("dml") != std::string::npos;
                        const bool b_dml = lower(b.EpName()).find("dml") != std::string::npos;
                        return a_dml && !b_dml;
                    });
                compatible.resize(1);
            }
            std::unordered_map<std::string, std::string> empty_options;
            Ort::KeyValuePairs ep_options(empty_options);
            so.AppendExecutionProvider_V2(env, compatible, ep_options);
            return;
        }

        if (selected_device == Device::CPU && options.cpu_threads > 0) {
            so.SetIntraOpNumThreads(options.cpu_threads);
            so.SetInterOpNumThreads(1);
        }
        return;
    }

    // Explicit --provider remains available in WinML builds. Select the catalog
    // EP device by its ORT name instead of calling a provider-specific factory.
    if (p != "cpu" && p != "cpuexecutionprovider") {
        std::string requested = p;
        const auto colon = requested.find(':');
        if (colon != std::string::npos) requested.resize(colon);
        std::vector<Ort::ConstEpDevice> selected;
        for (Ort::ConstEpDevice candidate : env.GetEpDevices()) {
            std::string ep_name = lower(candidate.EpName());
            std::string short_name = ep_name;
            const std::string suffix = "executionprovider";
            if (short_name.size() >= suffix.size() &&
                short_name.compare(short_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                short_name.resize(short_name.size() - suffix.size());
            }
            if (requested == ep_name || requested == short_name ||
                (requested == "directml" && short_name == "dml")) {
                selected.push_back(candidate);
            }
        }
        if (selected.empty()) {
            throw std::runtime_error("Windows ML provider is not registered or compatible: " + provider);
        }
        std::unordered_map<std::string, std::string> empty_options;
        Ort::KeyValuePairs ep_options(empty_options);
        so.AppendExecutionProvider_V2(env, selected, ep_options);
        return;
    }
#endif

    // Intel native path: OpenVINO EP. Accepts "openvino", "openvinoexecutionprovider",
    // or an "openvino:<DEVICE>" token; device_type comes from the suffix when present.
    if (p.rfind("openvino", 0) == 0) {
        const std::string device_type = openvino_device_type(options, provider);
        std::unordered_map<std::string, std::string> ov_opts{{"device_type", device_type}};
        if (!options.cache_dir.empty()) ov_opts["cache_dir"] = options.cache_dir;
        const std::string precision = inference_precision_policy(options, provider);
        if (precision != "preferred") {
            std::string config_device = device_type;
            const auto dot = config_device.find('.');
            if (dot != std::string::npos) config_device.resize(dot);
            ov_opts["load_config"] =
                "{\"" + config_device +
                "\":{\"INFERENCE_PRECISION_HINT\":\"" + precision + "\"}}";
        }
        so.AppendExecutionProvider_OpenVINO_V2(ov_opts);
        return;
    }

    if (p == "cpu" || p == "cpuexecutionprovider") {
        validate_inference_precision(options, provider);
        if (options.cpu_threads > 0) {
            so.SetIntraOpNumThreads(options.cpu_threads);
            so.SetInterOpNumThreads(1);
        }
        return;  // CPU EP is the default ORT fallback.
    }

    if (p == "qnn" || p == "qnnexecutionprovider") {
        validate_inference_precision(options, provider);
#ifdef NPU_INFERENCE_BENCH_QUALCOMM
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
        validate_inference_precision(options, provider);
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
        validate_inference_precision(options, provider);
#ifdef NPU_INFERENCE_BENCH_ORT_HAS_DML
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

}  // namespace ort_common
}  // namespace npu_inference_bench

#endif  // NPU_INFERENCE_BENCH_ORT
