#pragma once

#include <utility>

namespace npu_inference_bench {

// Logical hardware target requested by a caller.
enum class Device {
    NPU,
    GPU,
    CPU,
};

[[nodiscard]] inline const char* to_string(Device device) {
    using enum Device;
    switch (device) {
        case NPU: return "NPU";
        case GPU: return "GPU";
        case CPU: return "CPU";
    }
    std::unreachable();
}

namespace runtime {

// Hardware class inferred from the provider that actually built the session.
// Unknown is intentional: catalog/policy providers may require profiler evidence
// before their concrete device can be established.
enum class ResolvedDevice {
    Unknown,
    NPU,
    GPU,
    CPU,
};

[[nodiscard]] inline const char* to_string(ResolvedDevice device) {
    using enum ResolvedDevice;
    switch (device) {
        case NPU: return "NPU";
        case GPU: return "GPU";
        case CPU: return "CPU";
        case Unknown: return "Unknown";
    }
    std::unreachable();
}

[[nodiscard]] inline ResolvedDevice requested_device_class(Device device) {
    using enum Device;
    switch (device) {
        case NPU: return ResolvedDevice::NPU;
        case GPU: return ResolvedDevice::GPU;
        case CPU: return ResolvedDevice::CPU;
    }
    std::unreachable();
}

}  // namespace runtime
}  // namespace npu_inference_bench
