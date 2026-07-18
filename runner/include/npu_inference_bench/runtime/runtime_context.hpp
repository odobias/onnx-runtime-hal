#pragma once

#include <memory>
#include <vector>

#include "npu_inference_bench/runtime/model_session.hpp"

namespace npu_inference_bench::runtime {

// Owns one ORT environment and resolves a provider once for every model in a
// model set. Multi-graph workloads therefore cannot accidentally split their
// encoder/decoder sessions across different devices.
class RuntimeContext {
public:
    explicit RuntimeContext(RuntimeOptions options);
    ~RuntimeContext();
    RuntimeContext(RuntimeContext&&) noexcept;
    RuntimeContext& operator=(RuntimeContext&&) noexcept;
    RuntimeContext(const RuntimeContext&) = delete;
    RuntimeContext& operator=(const RuntimeContext&) = delete;

    [[nodiscard]] LoadedModelSet load(const std::vector<ModelSpec>& models);
    [[nodiscard]] ModelSession load_one(const ModelSpec& model);

    [[nodiscard]] const RuntimeOptions& options() const;
    [[nodiscard]] static Device probe_best_device();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace npu_inference_bench::runtime
