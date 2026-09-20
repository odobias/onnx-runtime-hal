// Shared ONNX Runtime execution-provider plumbing for the ORT-based Whisper
// backends (ort_static = static no-KV recompute; ort_dynamic = with-past KV
// cache). Both self-select across Intel (OpenVINO EP) / AMD (VitisAI) / Qualcomm
// (QNN) / any-DX12 GPU (DirectML) / CPU from ONE binary: the engine walks a
// device fallback chain and keeps the first EP that actually builds a session.
// This header is the single source of truth for that EP selection so the two
// engines can't drift apart. Still header-only for in-repo Whisper workloads;
// external library consumers should use RuntimeContext / ModelSession instead.
#pragma once

#include "npu_inference_bench/runtime/precision_policy.hpp"
#include "npu_inference_bench/runtime/runtime_options.hpp"

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

// ORT 1.21's generic AppendExecutionProvider API identifies the Qualcomm
// provider as "QNN".  The longer "QNNExecutionProvider" label is used by
// provider enumeration and diagnostics, but is rejected as the API name.
inline constexpr const char* kQnnEpName = "QNN";

inline std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Map consumer model_host EP tokens (ep_config.cpp) onto the ORT provider
// names used by this header. Pass-through for already-normalized ORT names.
inline std::string normalize_provider_token(std::string provider) {
    const std::string p = lower(provider);
    if (p.empty()) return provider;
    if (p == "cpu") return "CPUExecutionProvider";
    if (p == "directml" || p == "dml") return "DmlExecutionProvider";
    // Product OpenVINO default: AUTO prefers NPU then CPU (see ep_config.cpp).
    if (p == "openvino") return "openvino:AUTO:NPU,CPU";
    if (p == "qnnnpu" || p == "qnncpu") return p;
    if (p == "qnn") return kQnnEpName;
    return provider;
}

inline std::string openvino_device_type(const runtime::RuntimeOptions& options,
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
    if (p.contains("prefer_cpu")) return Device::CPU;
    if (p.contains("prefer_gpu")) return Device::GPU;
    if (p.contains("prefer_npu")) return Device::NPU;
    return fallback;
}
#endif

// Provider-neutral policy. CPU/GPU default to reproducible FP32; NPU defaults
// to device-preferred because accelerator compilers generally fix precision.
inline std::string inference_precision_policy(const runtime::RuntimeOptions& options,
                                              const std::string& provider) {
    const std::string p = lower(provider);
#ifdef NPU_INFERENCE_BENCH_WINML
    const Device winml_device = winml_token_device(provider, options.device);
    if (p.starts_with("windowsml:") && winml_device != Device::CPU) {
        // The policy selector owns accelerator precision. Use "preferred" by
        // default; an explicit f32/f16/bf16 request remains explicit and is
        // rejected below because Windows ML cannot guarantee it provider-wide.
        return precision_policy::resolve(options, Device::NPU);
    }
#endif
    Device actual_device = options.device;
#ifdef NPU_INFERENCE_BENCH_WINML
    if (p.starts_with("windowsml:")) actual_device = winml_device;
#endif
    if (p.contains("cpu")) {
        actual_device = Device::CPU;
    } else if (p.contains("dml") ||
               p.contains("directml")) {
        actual_device = Device::GPU;
    } else if (p.starts_with("openvino")) {
        const std::string device = lower(openvino_device_type(options, provider));
        actual_device = device.starts_with("cpu") ? Device::CPU
                      : device.starts_with("gpu") ? Device::GPU
                                                    : Device::NPU;
    }
    return precision_policy::resolve(options, actual_device);
}

inline void validate_inference_precision(const runtime::RuntimeOptions& options,
                                         const std::string& provider) {
    const std::string precision = inference_precision_policy(options, provider);
    const std::string p = lower(provider);
    if (p.contains("openvino")) return;
    if (p.contains("cpu") || p.contains("dml") ||
        p.contains("directml")) {
        if (precision == "f32" || precision == "preferred") return;
        throw runtime::RuntimeError(
            runtime::RuntimeErrorCode::InvalidArgument,
            provider + " cannot apply a provider-wide " + precision +
            " conversion; use a model exported in that precision");
    }
    if (precision != "preferred") {
        throw runtime::RuntimeError(
            runtime::RuntimeErrorCode::InvalidArgument,
            provider + " cannot guarantee provider-wide " + precision +
            "; use precision=preferred or a compiler/model-specific configuration");
    }
}

inline std::string resolved_inference_precision(const runtime::RuntimeOptions& options,
                                                const std::string& provider) {
    validate_inference_precision(options, provider);
    return inference_precision_policy(options, provider);
}

inline bool is_qnn(const std::string& provider) {
    return lower(provider).contains("qnn");
}

inline runtime::ResolvedDevice resolved_device_for_provider(const std::string& provider) {
    const std::string p = lower(provider);
    if (p.starts_with("windowsml:")) return runtime::ResolvedDevice::Unknown;
    if (p == "qnncpu") return runtime::ResolvedDevice::CPU;
    if (p == "cpuexecutionprovider" || p == "openvino:cpu")
        return runtime::ResolvedDevice::CPU;
    if (p.contains("dml") || p.contains("directml") ||
        p == "openvino:gpu")
        return runtime::ResolvedDevice::GPU;
    if (p == "qnnnpu" || p.contains("qnn") || p.contains("vitis") ||
        p == "openvino:npu" || p.contains("openvino:auto"))
        return runtime::ResolvedDevice::NPU;
    return runtime::ResolvedDevice::Unknown;
}

inline bool is_fallback_provider_for_device(Device requested_device,
                                            const std::string& resolved_provider) {
    const runtime::ResolvedDevice resolved =
        resolved_device_for_provider(resolved_provider);
    return resolved != runtime::ResolvedDevice::Unknown &&
           resolved != runtime::requested_device_class(requested_device);
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
    if (p.contains("openvino")) return "onnxruntime-openvino";
    if (p.contains("qnn")) return "onnxruntime-qnn";
    if (p.contains("vitis")) return "onnxruntime-vitisai";
    if (p.contains("dml") || p.contains("directml"))
        return "onnxruntime-directml";
    return "onnxruntime";
#endif
}

// The runtime EP fallback chain for a requested device. An explicit provider
// override is honored verbatim (single attempt). Otherwise NPU walks down to GPU
// then CPU, and GPU walks down to CPU, so one binary always produces a result on
// whatever hardware/drivers are actually present.
inline std::vector<std::string> fallback_chain(const runtime::RuntimeOptions& options) {
    if (!options.device_override.empty()) {
        return {normalize_provider_token(options.device_override)};
    }
    if (!options.provider_order.empty()) {
        std::vector<std::string> order;
        order.reserve(options.provider_order.size());
        for (const auto& p : options.provider_order) {
            order.push_back(normalize_provider_token(p));
        }
        return order;
    }
    std::vector<std::string> chain;
    const auto add = [&](Device d) {
        for (auto& p : providers_for(d)) chain.push_back(p);
    };
    switch (options.device) {
        case Device::NPU:
            add(Device::NPU);
            if (options.allow_fallback && !options.require_requested_device) {
                add(Device::GPU);
                add(Device::CPU);
            }
            break;
        case Device::GPU:
            add(Device::GPU);
            if (options.allow_fallback && !options.require_requested_device) add(Device::CPU);
            break;
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
        if (!lower(message).contains("already")) {
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
inline std::string qnn_backend_path(Device device) {
    switch (device) {
        case Device::GPU: return env_or("WHISPER_QNN_GPU_DLL", "QnnGpu.dll");
        case Device::CPU: return env_or("WHISPER_QNN_CPU_DLL", "QnnCpu.dll");
        case Device::NPU:
        default:          return env_or("WHISPER_QNN_HTP_DLL", "QnnHtp.dll");
    }
}

#endif  // NPU_INFERENCE_BENCH_QUALCOMM

// Append the requested execution provider to `so`. `model_dir`/`cache_key` feed
// VitisAI's config + blob cache; `options.cache_dir` also drives OpenVINO's blob
// cache. Unknown provider strings throw.
inline std::unordered_map<std::string, std::string> vitis_provider_options(
    const runtime::RuntimeOptions& options, const fs::path& model_dir,
    const std::string& cache_key) {
    std::unordered_map<std::string, std::string> vitis_opts;
    const fs::path config = model_dir / "vitisai_config.json";
    if (fs::exists(config)) vitis_opts["config_file"] = config.string();
    if (!options.cache_dir.empty()) {
        vitis_opts["cache_dir"] = options.cache_dir;
        vitis_opts["cache_key"] = cache_key;
        vitis_opts["enable_cache_file_io_in_mem"] = "0";
    }
    return vitis_opts;
}

#ifdef NPU_INFERENCE_BENCH_WINML
inline std::unordered_map<std::string, std::string> winml_catalog_provider_options(
    const runtime::RuntimeOptions& options, const std::string& ep_name,
    Device selected_device, const fs::path& model_dir,
    const std::string& cache_key) {
    const std::string selected_ep = lower(ep_name);
    if (selected_ep.contains("vitis")) {
        return vitis_provider_options(options, model_dir, cache_key);
    }
    std::unordered_map<std::string, std::string> provider_options;
    if (selected_device == Device::NPU &&
        selected_ep.contains("qnn")) {
        // Match the direct QNN pack's session policy. Without this, WinML
        // leaves HTP clocks at a platform default that can vary substantially
        // between AC/DC power overlays.
        provider_options.emplace("htp_performance_mode", "burst");
    }
    return provider_options;
}
#endif

inline void append_provider(Ort::Env& env, Ort::SessionOptions& so,
                            const runtime::RuntimeOptions& options,
                            const std::string& provider, const fs::path& model_dir,
                            const std::string& cache_key) {
    const std::string p = lower(provider);

#ifdef NPU_INFERENCE_BENCH_WINML
    if (p.starts_with("windowsml:") || p == "windowsml" || p == "winml") {
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
                        const bool a_dml = lower(a.EpName()).contains("dml");
                        const bool b_dml = lower(b.EpName()).contains("dml");
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
                        const bool a_dml = lower(a.EpName()).contains("dml");
                        const bool b_dml = lower(b.EpName()).contains("dml");
                        return a_dml && !b_dml;
                    });
                compatible.resize(1);
            }
            auto provider_options = winml_catalog_provider_options(
                options, compatible.front().EpName(), selected_device,
                model_dir, cache_key);
            Ort::KeyValuePairs ep_options(provider_options);
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
        auto provider_options = winml_catalog_provider_options(
            options, selected.front().EpName(), options.device,
            model_dir, cache_key);
        Ort::KeyValuePairs ep_options(provider_options);
        so.AppendExecutionProvider_V2(env, selected, ep_options);
        return;
    }
#endif

    // Intel native path: OpenVINO EP. Accepts "openvino", "openvinoexecutionprovider",
    // or an "openvino:<DEVICE>" token; device_type comes from the suffix when present.
    if (p.starts_with("openvino")) {
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

    // Product tokens qnnnpu/qnncpu (model_host ep_config) plus ORT QNN names.
    if (p == "qnn" || p == "qnnexecutionprovider" || p == "qnnnpu" || p == "qnncpu") {
        validate_inference_precision(options, provider);
#ifdef NPU_INFERENCE_BENCH_QUALCOMM
        Device qnn_device_class = options.device;
        if (p == "qnnnpu") qnn_device_class = Device::NPU;
        if (p == "qnncpu") qnn_device_class = Device::CPU;
        std::unordered_map<std::string, std::string> opts{
            {"backend_path", qnn_backend_path(qnn_device_class)}};
        if (qnn_device_class == Device::NPU) {
            opts.emplace("htp_performance_mode", "burst");
            opts.emplace("enable_htp_fp16_precision", "1");
        }
        // ORT 1.21 exposes QNN through the generic provider append API. The
        // newer Env::RegisterExecutionProviderLibrary/EpDevice API is not
        // available in the product's pinned runtime headers.
        so.AppendExecutionProvider(kQnnEpName, opts);
        return;
#else
        throw std::runtime_error("QNN execution provider not compiled into this build");
#endif
    }

    if (p == "vitisai" || p == "vitisaiexecutionprovider") {
        validate_inference_precision(options, provider);
        auto vitis_opts = vitis_provider_options(options, model_dir, cache_key);
        so.AppendExecutionProvider_VitisAI(vitis_opts);
        return;
    }

    if (p == "dml" || p == "directml" || p == "dmlexecutionprovider") {
        validate_inference_precision(options, provider);
#ifdef NPU_INFERENCE_BENCH_ORT_HAS_DML
        // Match the model_host DirectML setup (sequential + no mem pattern).
        so.DisableMemPattern();
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
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
        "QNNExecutionProvider / qnnnpu / qnncpu, DmlExecutionProvider / directml, "
        "VitisAIExecutionProvider)");
}

}  // namespace ort_common
}  // namespace npu_inference_bench

#endif  // NPU_INFERENCE_BENCH_ORT
