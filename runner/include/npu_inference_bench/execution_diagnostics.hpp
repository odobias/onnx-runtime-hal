#pragma once

#include <string>
#include <vector>

#include "npu_inference_bench/runtime/device.hpp"

namespace npu_inference_bench {

struct ProviderAttempt {
    std::string provider;
    bool success = false;
    std::string error;
};

struct ExecutionDiagnostics {
    Device requested_device = Device::NPU;
    runtime::ResolvedDevice resolved_device = runtime::ResolvedDevice::Unknown;
    std::string requested_provider;
    std::string resolved_provider;
    std::string inference_precision;
    bool fallback_occurred = false;
    std::vector<ProviderAttempt> attempts;
    bool offload_measured = false;
    int ep_nodes = -1;
    int cpu_nodes = -1;
    std::string cpu_offload_ops;
};

}  // namespace npu_inference_bench
