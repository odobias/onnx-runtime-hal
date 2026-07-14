#pragma once

#include "npu_inference_bench/whisper.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace npu_inference_bench {
namespace precision_policy {

inline std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline std::string resolve(const EngineOptions& options, Device default_device) {
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
        throw std::runtime_error(
            "unsupported inference precision policy: " + precision +
            " (expected f32, f16, bf16, or preferred)");
    }
    return precision;
}

inline std::string resolve(const EngineOptions& options) {
    return resolve(options, options.device);
}

}  // namespace precision_policy
}  // namespace npu_inference_bench
