#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace npu_inference_bench {
namespace runtime {

enum class RuntimeErrorCode {
    InvalidArgument,
    ProviderUnavailable,
    RequestedDeviceNotResolved,
    SessionCreationFailed,
    TensorMismatch,
    InferenceFailed,
    ProfilingFailed,
};

class RuntimeError : public std::runtime_error {
public:
    RuntimeError(RuntimeErrorCode code, std::string message, std::string provider = {})
        : std::runtime_error(std::move(message)), code_(code), provider_(std::move(provider)) {}

    RuntimeErrorCode code() const noexcept { return code_; }
    const std::string& provider() const noexcept { return provider_; }

private:
    RuntimeErrorCode code_;
    std::string provider_;
};

}  // namespace runtime
}  // namespace npu_inference_bench
