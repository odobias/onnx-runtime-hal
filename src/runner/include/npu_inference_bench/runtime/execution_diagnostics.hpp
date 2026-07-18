#pragma once

#include <string>
#include <vector>

#include "npu_inference_bench/runtime/device.hpp"

namespace npu_inference_bench::runtime {

struct ProviderAttempt {
    std::string provider;
    bool success = false;
    std::string error;
};

struct ExecutionDiagnostics {
    Device requested_device = Device::NPU;
    ResolvedDevice resolved_device = ResolvedDevice::Unknown;
    std::string requested_provider;
    std::string resolved_provider;
    std::string inference_precision;
    bool fallback_occurred = false;
    std::vector<ProviderAttempt> attempts;
    bool offload_measured = false;
    int ep_nodes = -1;
    int cpu_nodes = -1;
    std::string cpu_offload_ops;
    bool operation_assignment_measured = false;
    int assigned_ops_cpu = -1;
    int assigned_ops_npu = -1;
    std::string operation_assignment_source;
};

}  // namespace npu_inference_bench::runtime

namespace npu_inference_bench {

// Compatibility aliases for existing workload / benchmark code.
using runtime::ExecutionDiagnostics;
using runtime::ProviderAttempt;

}  // namespace npu_inference_bench
