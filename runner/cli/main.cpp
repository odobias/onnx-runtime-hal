#include "npu_inference_bench/benchmark_runner.hpp"
#include "npu_inference_bench/generic_onnx_cli.hpp"

#include <string>

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "run-onnx") {
        return npu_inference_bench::runtime_cli::run(argc - 1, argv + 1);
    }
    return npu_inference_bench::benchmark::run_cli(argc, argv);
}
