// Parse ONNX Runtime profiling JSON / provider assignment artifacts.
// Declarations live here; definitions are compiled into OnnxRuntimeHal.lib.
#pragma once

#include <exception>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace npu_inference_bench {
namespace ort_common {

std::string format_op_hist(const std::map<std::string, int>& hist);

struct OffloadStats {
    bool measured = false;
    int ep_nodes = -1;
    int cpu_nodes = -1;
    std::string cpu_ops;
    std::string providers;

    double cpu_offload_pct() const {
        if (!measured) return -1.0;
        const int total = ep_nodes + cpu_nodes;
        return total > 0 ? 100.0 * cpu_nodes / total : 0.0;
    }
};

struct OperationAssignmentStats {
    bool measured = false;
    int cpu_operations = -1;
    int npu_operations = -1;
    std::string source;
};

OperationAssignmentStats parse_vitis_operation_assignment(
    const std::filesystem::path& cache_dir,
    const std::vector<std::string>& cache_keys);

// Template kept in the header so older ORT SDKs can omit the call site.
template <typename Session>
OperationAssignmentStats query_ort_operation_assignment(const Session& session) {
    OperationAssignmentStats result;
    try {
        int cpu_operations = 0;
        int accelerator_operations = 0;
        for (const auto& subgraph : session.GetEpGraphAssignmentInfo()) {
            const std::string provider = subgraph.GetEpName();
            const int operations = static_cast<int>(subgraph.GetNodes().size());
            if (provider.empty() || operations <= 0) continue;
            if (provider == "CPUExecutionProvider") {
                cpu_operations += operations;
            } else {
                accelerator_operations += operations;
            }
        }
        if (cpu_operations + accelerator_operations <= 0) return result;
        result.measured = true;
        result.cpu_operations = cpu_operations;
        result.npu_operations = accelerator_operations;
        result.source = "ort-ep-graph-assignment";
    } catch (const std::exception&) {
        // Assignment recording is diagnostic only; inference remains valid.
    }
    return result;
}

OffloadStats parse_ort_profile(const std::filesystem::path& profile_json);

}  // namespace ort_common
}  // namespace npu_inference_bench
