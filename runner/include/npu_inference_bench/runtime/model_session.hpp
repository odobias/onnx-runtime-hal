#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "npu_inference_bench/runtime/execution_diagnostics.hpp"
#include "npu_inference_bench/runtime/runtime_options.hpp"
#include "npu_inference_bench/runtime/tensor.hpp"

namespace npu_inference_bench {
namespace runtime {

namespace detail {
class OrtSessionAccess;
}

struct ModelSpec {
    std::filesystem::path model_path;
    std::filesystem::path auxiliary_dir;
    std::string cache_key;
};

class ModelSession {
public:
    ModelSession();
    ~ModelSession();
    ModelSession(ModelSession&&) noexcept;
    ModelSession& operator=(ModelSession&&) noexcept;
    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;

    const std::vector<TensorDescriptor>& inputs() const;
    const std::vector<TensorDescriptor>& outputs() const;
    std::vector<Tensor> run(const std::vector<TensorView>& inputs,
                            const std::vector<std::string>& output_names = {});

    const ExecutionDiagnostics& diagnostics() const;
    void finalize_profiling();
    const std::string& runtime_name() const;
    const std::string& runtime_version() const;
    double load_seconds() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit ModelSession(std::unique_ptr<Impl> impl);
    void* native_handle() noexcept;

    friend class RuntimeContext;
    friend class detail::OrtSessionAccess;
};

struct LoadedModelSet {
    std::vector<ModelSession> sessions;
    ExecutionDiagnostics diagnostics;
    double load_seconds = 0.0;

    void finalize_profiling();
};

}  // namespace runtime
}  // namespace npu_inference_bench
