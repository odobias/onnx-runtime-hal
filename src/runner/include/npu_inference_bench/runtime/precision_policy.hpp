#pragma once

#include "npu_inference_bench/runtime/runtime_options.hpp"
#include "npu_inference_bench/runtime/runtime_error.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <format>
#include <string>

namespace npu_inference_bench::runtime::precision_policy {

[[nodiscard]] inline std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

// Resolve the provider-neutral precision policy string.
// Env fallback: NPU_INFERENCE_BENCH_PRECISION (legacy name kept for existing runs).
[[nodiscard]] inline std::string resolve(const RuntimeOptions& options, Device default_device) {
    std::string precision = options.precision_policy;
    if (precision.empty()) {
        const char* environment = std::getenv("NPU_INFERENCE_BENCH_PRECISION");
        if (environment && *environment) precision = environment;
    }
    if (precision.empty()) {
        precision = (default_device == Device::CPU || default_device == Device::GPU)
                        ? "f32"
                        : "preferred";
    }
    precision = lower(precision);
    if (precision == "auto" || precision == "default") precision = "preferred";
    if (precision != "preferred" && precision != "f32" &&
        precision != "f16" && precision != "bf16") {
        throw RuntimeError(
            RuntimeErrorCode::InvalidArgument,
            std::format(
                "unsupported inference precision policy: {} (expected f32, f16, bf16, or preferred)",
                precision));
    }
    return precision;
}

[[nodiscard]] inline std::string resolve(const RuntimeOptions& options) {
    return resolve(options, options.device);
}

}  // namespace npu_inference_bench::runtime::precision_policy

namespace npu_inference_bench::precision_policy {
using runtime::precision_policy::lower;
using runtime::precision_policy::resolve;
}  // namespace npu_inference_bench::precision_policy
