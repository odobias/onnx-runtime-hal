#pragma once

#include <string>

namespace npu_inference_bench {

// Logical hardware target requested by a caller.
enum class Device {
    NPU,
    GPU,
    CPU,
};

inline const char* to_string(Device device) {
    switch (device) {
        case Device::NPU: return "NPU";
        case Device::GPU: return "GPU";
        case Device::CPU:
        default:          return "CPU";
    }
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

inline const char* to_string(ResolvedDevice device) {
    switch (device) {
        case ResolvedDevice::NPU: return "NPU";
        case ResolvedDevice::GPU: return "GPU";
        case ResolvedDevice::CPU: return "CPU";
        case ResolvedDevice::Unknown:
        default:                  return "Unknown";
    }
}

inline ResolvedDevice requested_device_class(Device device) {
    switch (device) {
        case Device::NPU: return ResolvedDevice::NPU;
        case Device::GPU: return ResolvedDevice::GPU;
        case Device::CPU:
        default:          return ResolvedDevice::CPU;
    }
}

}  // namespace runtime
}  // namespace npu_inference_bench
