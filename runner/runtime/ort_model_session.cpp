#include "npu_inference_bench/runtime/runtime_context.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <system_error>
#include <utility>

#include "npu_inference_bench/runtime/runtime_error.hpp"
#include "ort_ep.hpp"
#include "ort_offload.hpp"

#ifdef NPU_INFERENCE_BENCH_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace npu_inference_bench {
namespace runtime {

namespace fs = std::filesystem;

#ifdef NPU_INFERENCE_BENCH_ORT
namespace {

std::atomic<unsigned long> g_profile_id{0};

std::string cache_safe(std::string value) {
    if (value.empty()) value = "model";
    for (char& c : value) {
        const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!safe) c = '_';
    }
    return value;
}

ONNXTensorElementDataType to_ort_type(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float32:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case TensorElementType::Float64:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case TensorElementType::Float16:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case TensorElementType::BFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        case TensorElementType::Int8:     return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case TensorElementType::UInt8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case TensorElementType::Int16:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case TensorElementType::UInt16:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case TensorElementType::Int32:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case TensorElementType::UInt32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case TensorElementType::Int64:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case TensorElementType::UInt64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        case TensorElementType::Bool:     return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        default:
            throw RuntimeError(RuntimeErrorCode::TensorMismatch,
                               "unsupported input tensor element type");
    }
}

TensorElementType from_ort_type(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    return TensorElementType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:   return TensorElementType::Float64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:  return TensorElementType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return TensorElementType::BFloat16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:     return TensorElementType::Int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:    return TensorElementType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:    return TensorElementType::Int16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:   return TensorElementType::UInt16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    return TensorElementType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:   return TensorElementType::UInt32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    return TensorElementType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:   return TensorElementType::UInt64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:     return TensorElementType::Bool;
        default:
            throw RuntimeError(RuntimeErrorCode::TensorMismatch,
                               "model exposes unsupported tensor element type " +
                                   std::to_string(static_cast<int>(type)));
    }
}

std::vector<TensorDescriptor> describe_inputs(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<TensorDescriptor> descriptors;
    descriptors.reserve(session.GetInputCount());
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
        auto type_info = session.GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        descriptors.push_back({
            session.GetInputNameAllocated(i, allocator).get(),
            from_ort_type(info.GetElementType()),
            info.GetShape(),
        });
    }
    return descriptors;
}

std::vector<TensorDescriptor> describe_outputs(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<TensorDescriptor> descriptors;
    descriptors.reserve(session.GetOutputCount());
    for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto type_info = session.GetOutputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        descriptors.push_back({
            session.GetOutputNameAllocated(i, allocator).get(),
            from_ort_type(info.GetElementType()),
            info.GetShape(),
        });
    }
    return descriptors;
}

const TensorDescriptor* find_descriptor(const std::vector<TensorDescriptor>& descriptors,
                                        const std::string& name) {
    const auto it = std::find_if(
        descriptors.begin(), descriptors.end(),
        [&](const TensorDescriptor& descriptor) { return descriptor.name == name; });
    return it == descriptors.end() ? nullptr : &*it;
}

void validate_shape(const TensorView& input, const TensorDescriptor& descriptor) {
    if (input.type != descriptor.type) {
        throw RuntimeError(
            RuntimeErrorCode::TensorMismatch,
            "input '" + input.name + "' type does not match model metadata");
    }
    if (input.shape.size() != descriptor.shape.size()) {
        throw RuntimeError(
            RuntimeErrorCode::TensorMismatch,
            "input '" + input.name + "' rank does not match model metadata");
    }
    for (std::size_t i = 0; i < input.shape.size(); ++i) {
        // ORT distributions encode symbolic dimensions as either -1 or 0.
        // Concrete positive dimensions remain strict.
        if (descriptor.shape[i] > 0 && descriptor.shape[i] != input.shape[i]) {
            throw RuntimeError(
                RuntimeErrorCode::TensorMismatch,
                "input '" + input.name + "' shape does not match model metadata");
        }
    }
}

std::string first_accelerator_provider(const std::string& providers) {
    const auto comma = providers.find(',');
    const std::string first = providers.substr(0, comma);
    return first == "CPUExecutionProvider" ? std::string{} : first;
}

void enforce_requested_device(const RuntimeOptions& options,
                              const ExecutionDiagnostics& diagnostics) {
    if (!options.require_requested_device) return;
    const ResolvedDevice expected = requested_device_class(options.device);
    if (diagnostics.resolved_device == ResolvedDevice::Unknown) {
        throw RuntimeError(
            RuntimeErrorCode::RequestedDeviceNotResolved,
            "the actual hardware device could not be established for provider '" +
                diagnostics.resolved_provider + "'",
            diagnostics.resolved_provider);
    }
    if (diagnostics.resolved_device != expected) {
        throw RuntimeError(
            RuntimeErrorCode::RequestedDeviceNotResolved,
            "requested " + std::string(npu_inference_bench::to_string(options.device)) +
                " but provider '" + diagnostics.resolved_provider + "' resolved " +
                runtime::to_string(diagnostics.resolved_device),
            diagnostics.resolved_provider);
    }
}

}  // namespace
#endif

struct ModelSession::Impl {
#ifdef NPU_INFERENCE_BENCH_ORT
    std::unique_ptr<Ort::Session> session;
#endif
    RuntimeOptions options;
    std::vector<TensorDescriptor> input_descriptors;
    std::vector<TensorDescriptor> output_descriptors;
    ExecutionDiagnostics diagnostics;
    std::string runtime_name;
    std::string runtime_version;
    std::string cache_key;
    double load_seconds = 0.0;
    bool profiling_enabled = false;
    bool profiling_finalized = false;
};

ModelSession::ModelSession() = default;
ModelSession::~ModelSession() = default;
ModelSession::ModelSession(ModelSession&&) noexcept = default;
ModelSession& ModelSession::operator=(ModelSession&&) noexcept = default;
ModelSession::ModelSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

const std::vector<TensorDescriptor>& ModelSession::inputs() const {
    static const std::vector<TensorDescriptor> empty;
    return impl_ ? impl_->input_descriptors : empty;
}

const std::vector<TensorDescriptor>& ModelSession::outputs() const {
    static const std::vector<TensorDescriptor> empty;
    return impl_ ? impl_->output_descriptors : empty;
}

std::vector<Tensor> ModelSession::run(const std::vector<TensorView>& inputs,
                                      const std::vector<std::string>& output_names) {
#ifndef NPU_INFERENCE_BENCH_ORT
    (void)inputs;
    (void)output_names;
    throw RuntimeError(RuntimeErrorCode::ProviderUnavailable,
                       "ONNX Runtime is not compiled into this build");
#else
    if (!impl_ || !impl_->session) {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument, "model session is not loaded");
    }
    if (inputs.size() != impl_->input_descriptors.size()) {
        throw RuntimeError(RuntimeErrorCode::TensorMismatch,
                           "input count does not match model metadata");
    }

    std::vector<std::string> input_name_storage;
    std::vector<const char*> input_name_ptrs;
    std::vector<Ort::Value> input_values;
    input_name_storage.reserve(inputs.size());
    input_name_ptrs.reserve(inputs.size());
    input_values.reserve(inputs.size());
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (const TensorView& input : inputs) {
        try {
            input.validate();
        } catch (const std::exception& error) {
            throw RuntimeError(RuntimeErrorCode::TensorMismatch, error.what());
        }
        const TensorDescriptor* descriptor = find_descriptor(impl_->input_descriptors, input.name);
        if (!descriptor) {
            throw RuntimeError(RuntimeErrorCode::TensorMismatch,
                               "model has no input named '" + input.name + "'");
        }
        validate_shape(input, *descriptor);
        input_name_storage.push_back(input.name);
        input_name_ptrs.push_back(input_name_storage.back().c_str());
        input_values.push_back(Ort::Value::CreateTensor(
            memory, const_cast<void*>(input.data), input.byte_size, input.shape.data(),
            input.shape.size(), to_ort_type(input.type)));
    }

    std::vector<std::string> selected_outputs = output_names;
    if (selected_outputs.empty()) {
        for (const TensorDescriptor& output : impl_->output_descriptors) {
            selected_outputs.push_back(output.name);
        }
    }
    std::vector<const char*> output_name_ptrs;
    output_name_ptrs.reserve(selected_outputs.size());
    for (const std::string& output : selected_outputs) {
        if (!find_descriptor(impl_->output_descriptors, output)) {
            throw RuntimeError(RuntimeErrorCode::TensorMismatch,
                               "model has no output named '" + output + "'");
        }
        output_name_ptrs.push_back(output.c_str());
    }

    std::vector<Ort::Value> values;
    try {
        values = impl_->session->Run(
            Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_values.data(),
            input_values.size(), output_name_ptrs.data(), output_name_ptrs.size());
    } catch (const std::exception& error) {
        throw RuntimeError(RuntimeErrorCode::InferenceFailed, error.what(),
                           impl_->diagnostics.resolved_provider);
    }

    std::vector<Tensor> result;
    result.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        auto info = values[i].GetTensorTypeAndShapeInfo();
        Tensor tensor;
        tensor.name = selected_outputs[i];
        tensor.type = from_ort_type(info.GetElementType());
        tensor.shape = info.GetShape();
        tensor.bytes.resize(info.GetElementCount() * element_size(tensor.type));
        void* raw = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetTensorMutableData(values[i], &raw));
        if (!tensor.bytes.empty()) std::memcpy(tensor.bytes.data(), raw, tensor.bytes.size());
        result.push_back(std::move(tensor));
    }
    return result;
#endif
}

const ExecutionDiagnostics& ModelSession::diagnostics() const {
    static const ExecutionDiagnostics empty;
    return impl_ ? impl_->diagnostics : empty;
}

void ModelSession::finalize_profiling() {
#ifdef NPU_INFERENCE_BENCH_ORT
    if (!impl_ || !impl_->session || !impl_->profiling_enabled || impl_->profiling_finalized) return;
    impl_->profiling_finalized = true;
    Ort::AllocatorWithDefaultOptions allocator;
    try {
        auto profile = impl_->session->EndProfilingAllocated(allocator);
        const fs::path profile_path(profile.get());
        const ort_common::OffloadStats stats = ort_common::parse_ort_profile(profile_path);
        std::error_code ec;
        fs::remove(profile_path, ec);
        if (stats.measured) {
            impl_->diagnostics.offload_measured = true;
            impl_->diagnostics.ep_nodes = stats.ep_nodes;
            impl_->diagnostics.cpu_nodes = stats.cpu_nodes;
            impl_->diagnostics.cpu_offload_ops = stats.cpu_ops;
            const std::string actual_provider = first_accelerator_provider(stats.providers);
            if (!actual_provider.empty()) {
                impl_->diagnostics.resolved_provider = actual_provider;
                impl_->diagnostics.resolved_device =
                    ort_common::resolved_device_for_provider(actual_provider);
                impl_->runtime_name = ort_common::runtime_for(actual_provider);
            } else if (stats.providers == "CPUExecutionProvider") {
                impl_->diagnostics.resolved_provider = "CPUExecutionProvider";
                impl_->diagnostics.resolved_device = ResolvedDevice::CPU;
                impl_->runtime_name = ort_common::runtime_for("CPUExecutionProvider");
            }
            impl_->diagnostics.fallback_occurred =
                impl_->diagnostics.resolved_device != requested_device_class(impl_->options.device);
        }
        if (!impl_->options.cache_dir.empty() &&
            ort_common::lower(impl_->diagnostics.resolved_provider).find("vitis") !=
                std::string::npos) {
            const auto assignment = ort_common::parse_vitis_operation_assignment(
                fs::path(impl_->options.cache_dir), {impl_->cache_key});
            if (assignment.measured) {
                impl_->diagnostics.operation_assignment_measured = true;
                impl_->diagnostics.assigned_ops_cpu = assignment.cpu_operations;
                impl_->diagnostics.assigned_ops_npu = assignment.npu_operations;
                impl_->diagnostics.operation_assignment_source = assignment.source;
            }
        }
        enforce_requested_device(impl_->options, impl_->diagnostics);
    } catch (const RuntimeError&) {
        throw;
    } catch (const std::exception& error) {
        throw RuntimeError(RuntimeErrorCode::ProfilingFailed, error.what(),
                           impl_->diagnostics.resolved_provider);
    }
#endif
}

const std::string& ModelSession::runtime_name() const {
    static const std::string empty;
    return impl_ ? impl_->runtime_name : empty;
}

const std::string& ModelSession::runtime_version() const {
    static const std::string empty;
    return impl_ ? impl_->runtime_version : empty;
}

double ModelSession::load_seconds() const { return impl_ ? impl_->load_seconds : 0.0; }

void* ModelSession::native_handle() noexcept {
#ifdef NPU_INFERENCE_BENCH_ORT
    return impl_ && impl_->session ? impl_->session.get() : nullptr;
#else
    return nullptr;
#endif
}

void LoadedModelSet::finalize_profiling() {
    diagnostics.offload_measured = false;
    diagnostics.ep_nodes = 0;
    diagnostics.cpu_nodes = 0;
    diagnostics.operation_assignment_measured = false;
    diagnostics.assigned_ops_cpu = 0;
    diagnostics.assigned_ops_npu = 0;
    std::set<std::string> cpu_ops;
    for (ModelSession& session : sessions) {
        session.finalize_profiling();
        const ExecutionDiagnostics& current = session.diagnostics();
        diagnostics.resolved_provider = current.resolved_provider;
        diagnostics.resolved_device = current.resolved_device;
        diagnostics.fallback_occurred = current.fallback_occurred;
        if (current.offload_measured) {
            diagnostics.offload_measured = true;
            diagnostics.ep_nodes += current.ep_nodes;
            diagnostics.cpu_nodes += current.cpu_nodes;
            if (!current.cpu_offload_ops.empty()) cpu_ops.insert(current.cpu_offload_ops);
        }
        if (current.operation_assignment_measured) {
            diagnostics.operation_assignment_measured = true;
            diagnostics.assigned_ops_cpu += current.assigned_ops_cpu;
            diagnostics.assigned_ops_npu += current.assigned_ops_npu;
            diagnostics.operation_assignment_source =
                current.operation_assignment_source;
        }
    }
    if (!diagnostics.offload_measured) {
        diagnostics.ep_nodes = -1;
        diagnostics.cpu_nodes = -1;
    }
    if (!diagnostics.operation_assignment_measured) {
        diagnostics.assigned_ops_cpu = -1;
        diagnostics.assigned_ops_npu = -1;
        diagnostics.operation_assignment_source.clear();
    }
    std::string combined;
    for (const std::string& ops : cpu_ops) {
        if (!combined.empty()) combined += "; ";
        combined += ops;
    }
    diagnostics.cpu_offload_ops = std::move(combined);
}

struct RuntimeContext::Impl {
    explicit Impl(RuntimeOptions value)
#ifdef NPU_INFERENCE_BENCH_ORT
        : options(std::move(value)), env(ORT_LOGGING_LEVEL_WARNING, "onnx_runtime_hal")
#else
        : options(std::move(value))
#endif
    {
#if defined(NPU_INFERENCE_BENCH_ORT) && defined(NPU_INFERENCE_BENCH_WINML)
        ort_common::register_windows_ml_catalog(env, options.device);
#endif
    }

    RuntimeOptions options;
#ifdef NPU_INFERENCE_BENCH_ORT
    Ort::Env env;
#endif
};

RuntimeContext::RuntimeContext(RuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
RuntimeContext::~RuntimeContext() = default;
RuntimeContext::RuntimeContext(RuntimeContext&&) noexcept = default;
RuntimeContext& RuntimeContext::operator=(RuntimeContext&&) noexcept = default;

const RuntimeOptions& RuntimeContext::options() const { return impl_->options; }

LoadedModelSet RuntimeContext::load(const std::vector<ModelSpec>& models) {
#ifndef NPU_INFERENCE_BENCH_ORT
    (void)models;
    throw RuntimeError(RuntimeErrorCode::ProviderUnavailable,
                       "ONNX Runtime is not compiled into this build");
#else
    if (models.empty()) {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "at least one ONNX model is required");
    }
    for (const ModelSpec& model : models) {
        if (model.model_path.empty() || !fs::exists(model.model_path)) {
            throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                               "ONNX model does not exist: " + model.model_path.string());
        }
    }

    const std::vector<std::string> chain = ort_common::fallback_chain(impl_->options);
    if (chain.empty()) {
        throw RuntimeError(RuntimeErrorCode::ProviderUnavailable,
                           "provider policy produced no execution-provider candidates");
    }
    LoadedModelSet loaded;
    loaded.diagnostics.requested_device = impl_->options.device;
    loaded.diagnostics.requested_provider =
        impl_->options.device_override.empty() ? std::string("auto")
                                               : impl_->options.device_override;
    std::string last_error;

    for (std::size_t provider_index = 0; provider_index < chain.size(); ++provider_index) {
        const std::string& provider = chain[provider_index];
        std::vector<ModelSession> candidate;
        candidate.reserve(models.size());
        const auto started = std::chrono::steady_clock::now();
        try {
            const ResolvedDevice resolved = ort_common::resolved_device_for_provider(provider);
            if (impl_->options.require_requested_device &&
                resolved != ResolvedDevice::Unknown &&
                resolved != requested_device_class(impl_->options.device)) {
                throw RuntimeError(
                    RuntimeErrorCode::RequestedDeviceNotResolved,
                    "provider '" + provider + "' does not target requested " +
                        npu_inference_bench::to_string(impl_->options.device),
                    provider);
            }

            for (const ModelSpec& model : models) {
                auto state = std::make_unique<ModelSession::Impl>();
                state->options = impl_->options;
                state->diagnostics.requested_device = impl_->options.device;
                state->diagnostics.resolved_device = resolved;
                state->diagnostics.requested_provider = loaded.diagnostics.requested_provider;
                state->diagnostics.resolved_provider = provider;
                state->diagnostics.inference_precision =
                    ort_common::resolved_inference_precision(impl_->options, provider);
                state->diagnostics.fallback_occurred =
                    ort_common::is_fallback_provider_for_device(
                        impl_->options.device, provider);
                state->runtime_name = ort_common::runtime_for(provider);
                state->runtime_version = Ort::GetVersionString();
                state->profiling_enabled = impl_->options.profile_execution;

                const std::string cache_key = cache_safe(
                    !model.cache_key.empty()
                        ? model.cache_key
                        : (!impl_->options.cache_key.empty()
                               ? impl_->options.cache_key
                               : model.model_path.stem().string()));
                state->cache_key = cache_key;
                fs::path qnn_context;
                if (ort_common::is_qnn(provider) && !impl_->options.cache_dir.empty()) {
                    qnn_context = fs::path(impl_->options.cache_dir) /
                                  (cache_key + "_qnn_ctx.onnx");
                }
                const auto create_options = [&](bool generate_qnn_context) {
                    Ort::SessionOptions session_options;
                    session_options.SetGraphOptimizationLevel(
                        GraphOptimizationLevel::ORT_ENABLE_ALL);
                    if (impl_->options.profile_execution) {
                        const fs::path prefix =
                            fs::temp_directory_path() /
                            ("onnx_hal_" + cache_key + "_" +
                             std::to_string(++g_profile_id));
                        const std::wstring prefix_w = prefix.wstring();
                        session_options.EnableProfiling(prefix_w.c_str());
                    }
                    const fs::path auxiliary =
                        model.auxiliary_dir.empty() ? model.model_path.parent_path()
                                                    : model.auxiliary_dir;
                    ort_common::append_provider(
                        impl_->env, session_options, impl_->options, provider, auxiliary,
                        cache_key);
                    if (generate_qnn_context) {
                        const std::string context_path = qnn_context.string();
                        session_options.AddConfigEntry("ep.context_enable", "1");
                        session_options.AddConfigEntry(
                            "ep.context_file_path", context_path.c_str());
                        session_options.AddConfigEntry("ep.context_embed_mode", "1");
                    }
                    return session_options;
                };

                const auto session_started = std::chrono::steady_clock::now();
                if (!qnn_context.empty() && fs::exists(qnn_context)) {
                    try {
                        auto session_options = create_options(false);
                        state->session = std::make_unique<Ort::Session>(
                            impl_->env, qnn_context.c_str(), session_options);
                    } catch (const std::exception& error) {
                        std::cerr << "[onnx-runtime-hal] QNN context cache '"
                                  << qnn_context.string() << "' unusable (" << error.what()
                                  << "); recompiling\n";
                        std::error_code ec;
                        fs::remove(qnn_context, ec);
                    }
                }
                if (!state->session) {
                    if (!qnn_context.empty()) {
                        std::error_code ec;
                        fs::create_directories(qnn_context.parent_path(), ec);
                    }
                    auto session_options = create_options(!qnn_context.empty());
                    state->session = std::make_unique<Ort::Session>(
                        impl_->env, model.model_path.c_str(), session_options);
                }
                state->load_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - session_started).count();
                state->input_descriptors = describe_inputs(*state->session);
                state->output_descriptors = describe_outputs(*state->session);
                candidate.push_back(ModelSession(std::move(state)));
            }

            loaded.sessions = std::move(candidate);
            loaded.load_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            loaded.diagnostics.resolved_provider = provider;
            loaded.diagnostics.resolved_device = resolved;
            loaded.diagnostics.inference_precision =
                ort_common::resolved_inference_precision(impl_->options, provider);
            loaded.diagnostics.fallback_occurred =
                ort_common::is_fallback_provider_for_device(
                    impl_->options.device, provider);
            loaded.diagnostics.attempts.push_back({provider, true, {}});
            for (ModelSession& session : loaded.sessions) {
                session.impl_->diagnostics.attempts = loaded.diagnostics.attempts;
                session.impl_->diagnostics.fallback_occurred =
                    loaded.diagnostics.fallback_occurred;
            }
            if (impl_->options.require_requested_device &&
                resolved == ResolvedDevice::Unknown &&
                !impl_->options.profile_execution) {
                throw RuntimeError(
                    RuntimeErrorCode::RequestedDeviceNotResolved,
                    "strict device enforcement requires profiling for provider policy '" +
                        provider + "'",
                    provider);
            }
            return loaded;
        } catch (const RuntimeError& error) {
            last_error = error.what();
            loaded.diagnostics.attempts.push_back({provider, false, last_error});
            if (error.code() == RuntimeErrorCode::RequestedDeviceNotResolved ||
                error.code() == RuntimeErrorCode::InvalidArgument ||
                error.code() == RuntimeErrorCode::TensorMismatch) {
                throw;
            }
            if (provider_index + 1 < chain.size()) {
                std::cerr << "[onnx-runtime-hal] provider '" << provider
                          << "' unavailable (" << last_error << "); trying '"
                          << chain[provider_index + 1] << "'\n";
            }
        } catch (const std::exception& error) {
            last_error = error.what();
            loaded.diagnostics.attempts.push_back({provider, false, last_error});
            if (provider_index + 1 < chain.size()) {
                std::cerr << "[onnx-runtime-hal] provider '" << provider
                          << "' unavailable (" << last_error << "); trying '"
                          << chain[provider_index + 1] << "'\n";
            }
        }
    }
    throw RuntimeError(
        RuntimeErrorCode::SessionCreationFailed,
        "no ONNX Runtime provider could build the requested model set (last error: " +
            last_error + ")");
#endif
}

ModelSession RuntimeContext::load_one(const ModelSpec& model) {
    LoadedModelSet loaded = load({model});
    return std::move(loaded.sessions.front());
}

Device RuntimeContext::probe_best_device() {
#ifdef NPU_INFERENCE_BENCH_ORT
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_runtime_hal_probe");
#ifdef NPU_INFERENCE_BENCH_WINML
    ort_common::register_windows_ml_catalog(env, Device::NPU);
#endif
#ifdef NPU_INFERENCE_BENCH_QUALCOMM
    ort_common::register_qnn_library(env);
#endif
    for (Ort::ConstEpDevice candidate : env.GetEpDevices()) {
        if (candidate.Device().Type() == OrtHardwareDeviceType_NPU) return Device::NPU;
    }
    for (Ort::ConstEpDevice candidate : env.GetEpDevices()) {
        if (candidate.Device().Type() == OrtHardwareDeviceType_GPU) return Device::GPU;
    }
#ifdef NPU_INFERENCE_BENCH_AMD
    // The legacy VitisAI factory does not participate in GetEpDevices().
    return Device::NPU;
#endif
#endif
    return Device::CPU;
}

}  // namespace runtime
}  // namespace npu_inference_bench
