#pragma once

namespace npu_inference_bench::benchmark {

// Executes one benchmark workload invocation. The process entry point remains a
// thin shell so benchmark policy can be reused and tested independently.
int run_cli(int argc, char* argv[]);

}  // namespace npu_inference_bench::benchmark
