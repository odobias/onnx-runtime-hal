#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npu_inference_bench/runtime/runtime_options.hpp"
#include "npu_inference_bench/runtime/tensor.hpp"
#include "ort_ep.hpp"
#include "ort_offload.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_tensor_contracts() {
    using namespace npu_inference_bench::runtime;
    require(element_count({2, 3, 4}) == 24, "tensor element count");
    require(element_count({-1, 3}, true) == 0, "dynamic tensor descriptor");

    std::vector<float> data(6, 1.0f);
    TensorView valid{"input", TensorElementType::Float32, {2, 3},
                     data.data(), data.size() * sizeof(float)};
    valid.validate();

    bool rejected = false;
    try {
        TensorView invalid{"input", TensorElementType::Float32, {2, 4},
                           data.data(), data.size() * sizeof(float)};
        invalid.validate();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "tensor byte-size mismatch must be rejected");
}

void test_profile_parser() {
    const auto path = std::filesystem::temp_directory_path() / "onnx_hal_profile_test.json";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output
            << "["
            << "{\"name\":\"fused_kernel_time\",\"args\":{\"provider\":\"DmlExecutionProvider\",\"op_name\":\"Fused\"}},"
            << "{\"name\":\"shape_kernel_time\",\"args\":{\"provider\":\"CPUExecutionProvider\",\"op_name\":\"Shape\"}}"
            << "]";
    }
    const auto stats = npu_inference_bench::ort_common::parse_ort_profile(path);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    require(stats.measured, "profile should be measured");
    require(stats.ep_nodes == 1 && stats.cpu_nodes == 1, "profile node counts");
    require(stats.cpu_ops == "Shape x1", "profile CPU operation histogram");
}

#ifdef NPU_INFERENCE_BENCH_ORT
void test_provider_policy() {
    using namespace npu_inference_bench;
    runtime::RuntimeOptions options;
    options.device = Device::NPU;
    options.allow_fallback = false;
    const auto strict_chain = ort_common::fallback_chain(options);
    require(!strict_chain.empty(), "NPU provider chain");
    for (const std::string& provider : strict_chain) {
        require(ort_common::resolved_device_for_provider(provider) != runtime::ResolvedDevice::GPU &&
                    ort_common::resolved_device_for_provider(provider) != runtime::ResolvedDevice::CPU,
                "no lower-device provider when fallback is disabled");
    }

    options.allow_fallback = true;
    options.require_requested_device = true;
    const auto required_chain = ort_common::fallback_chain(options);
    for (const std::string& provider : required_chain) {
        require(ort_common::resolved_device_for_provider(provider) != runtime::ResolvedDevice::GPU &&
                    ort_common::resolved_device_for_provider(provider) != runtime::ResolvedDevice::CPU,
                "strict device mode must not append lower-device providers");
    }

    options.device_override = "DmlExecutionProvider";
    const auto override_chain = ort_common::fallback_chain(options);
    require(override_chain.size() == 1 && override_chain[0] == "DmlExecutionProvider",
            "explicit provider override");
    require(ort_common::resolved_device_for_provider("DmlExecutionProvider") ==
                runtime::ResolvedDevice::GPU,
            "DirectML device classification");
    require(ort_common::resolved_device_for_provider("VitisAIExecutionProvider") ==
                runtime::ResolvedDevice::NPU,
            "VitisAI device classification");
    require(!ort_common::is_fallback_provider_for_device(
                Device::NPU, "VitisAIExecutionProvider"),
            "same-device provider retry is not fallback");
    require(ort_common::is_fallback_provider_for_device(
                Device::NPU, "DmlExecutionProvider"),
            "NPU to GPU demotion is fallback");

    options = runtime::RuntimeOptions{};
    options.device = Device::CPU;
    options.precision_policy = "fp8";
    bool precision_rejected = false;
    try {
        (void)ort_common::resolved_inference_precision(
            options, "CPUExecutionProvider");
    } catch (const runtime::RuntimeError& error) {
        precision_rejected =
            error.code() == runtime::RuntimeErrorCode::InvalidArgument;
    }
    require(precision_rejected, "unsupported precision must be a typed error");
}
#endif

}  // namespace

int main() {
    try {
        test_tensor_contracts();
        test_profile_parser();
#ifdef NPU_INFERENCE_BENCH_ORT
        test_provider_policy();
#endif
        std::cout << "runtime HAL tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "runtime HAL tests failed: " << error.what() << "\n";
        return 1;
    }
}
