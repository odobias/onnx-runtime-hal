#pragma once

#include <string>
#include <vector>

#include "npu_inference_bench/runtime/device.hpp"

namespace npu_inference_bench::runtime {

struct RuntimeOptions {
    Device device = Device::NPU;

    // Exact provider token. When populated, no alternative provider is tried.
    // Kept under the historical name so workload adapters remain source-compatible.
    std::string device_override;

    // Optional explicit provider order. Empty uses the runtime's device policy.
    std::vector<std::string> provider_order;

    // Permit demotion from NPU to GPU/CPU or GPU to CPU. Providers for the same
    // requested device may still be tried when this is false.
    bool allow_fallback = true;

    // Reject a session if the resolved hardware class differs from `device`.
    // This is stronger than merely recording fallback and is appropriate for
    // hardware qualification or benchmark trust gates.
    bool require_requested_device = false;

    std::string cache_dir;
    std::string cache_key;
    int cpu_threads = 0;
    std::string precision_policy;

    // Profiling belongs to an individual session load, not to a workload.
    // The old name is retained while callers migrate to ModelSession.
    bool profile_execution = true;
};

}  // namespace npu_inference_bench::runtime
