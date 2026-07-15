// Parse an ONNX Runtime profiling JSON to answer "did any op fall back to the CPU
// EP when we asked for an accelerator?". ORT emits one "<node>_kernel_time" event
// per executed node with an args.provider field; the accelerator EPs (OpenVINO /
// VitisAI / QNN / DirectML) fuse their supported subgraph into a single node, so
// anything left on "CPUExecutionProvider" is a genuine host-side fallback.
//
// Pure std (no ORT dependency) so it can be unit-reasoned and reused. The parser is
// deliberately lightweight: the profile is a flat JSON array of small event objects
// and we only need three string fields per event, so we scan rather than pull in a
// JSON library.
#pragma once

#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace npu_inference_bench {
namespace ort_common {

// Render a fallback op-count map as "Gather x3, Cast x2" (sorted by op name).
inline std::string format_op_hist(const std::map<std::string, int>& hist) {
    std::ostringstream ops;
    bool first = true;
    for (const auto& [op, n] : hist) {
        if (!first) ops << ", ";
        ops << op << " x" << n;
        first = false;
    }
    return ops.str();
}

struct OffloadStats {
    bool measured = false;
    int ep_nodes = -1;    // distinct nodes on the requested (non-CPU) EP
    int cpu_nodes = -1;   // distinct nodes that ran on CPUExecutionProvider
    std::string cpu_ops;  // histogram of the fallback op types, e.g. "Gather x3, Cast x2"
    std::string providers;  // distinct profiler providers, accelerator first

    // -1 when unmeasured; 0..100 fraction of distinct profiler nodes assigned to CPU.
    // This is NOT a compute/time share: one accelerator node may represent a fused
    // partition containing hundreds of original ONNX operations.
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

namespace detail {

// Extract the string value of  "key" : "<value>"  starting at/after `from`, bounded
// to `end`. Whitespace-tolerant around the colon: ORT's profiler writes JSON as
// `"name" :"x"` / `"provider" : "y"` (spaces vary), so an exact `"key":"` match would
// silently miss everything. `value_end` (optional) receives the closing-quote index.
inline std::string json_str_field(const std::string& s, const std::string& key, size_t from,
                                  size_t end, size_t* value_end = nullptr) {
    const std::string k = "\"" + key + "\"";
    size_t p = s.find(k, from);
    if (p == std::string::npos || p >= end) return {};
    p += k.size();
    while (p < end && (s[p] == ' ' || s[p] == '\t')) ++p;
    if (p >= end || s[p] != ':') return {};
    ++p;
    while (p < end && (s[p] == ' ' || s[p] == '\t')) ++p;
    if (p >= end || s[p] != '"') return {};
    ++p;
    const size_t q = s.find('"', p);
    if (q == std::string::npos || q > end) return {};
    if (value_end) *value_end = q;
    return s.substr(p, q - p);
}

inline bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline int csv_data_rows(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) return -1;
    std::string line;
    int rows = -1;  // exclude the header
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) ++rows;
    }
    return rows < 0 ? 0 : rows;
}

// Count string entries in every JSON array named `key`. VitisAI context.json
// stores each compiler-assigned operation as one string in metaDef[].nodes.
inline int json_string_array_entries(const std::filesystem::path& path,
                                     const std::string& key) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return -1;
    std::stringstream buf;
    buf << f.rdbuf();
    const std::string s = buf.str();
    const std::string token = "\"" + key + "\"";
    int count = 0;
    bool found = false;
    size_t pos = 0;
    while ((pos = s.find(token, pos)) != std::string::npos) {
        pos += token.size();
        pos = s.find('[', pos);
        if (pos == std::string::npos) break;
        found = true;
        ++pos;
        while (pos < s.size()) {
            while (pos < s.size() &&
                   (std::isspace(static_cast<unsigned char>(s[pos])) || s[pos] == ',')) {
                ++pos;
            }
            if (pos >= s.size() || s[pos] == ']') break;
            if (s[pos] != '"') return -1;
            ++count;
            ++pos;
            bool escaped = false;
            while (pos < s.size()) {
                const char c = s[pos++];
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    break;
                }
            }
        }
    }
    return found ? count : -1;
}

}  // namespace detail

// VitisAI writes gops.csv for the provider input graph and context.json for the
// operations captured by its NPU partitions. Their difference is the compiler's
// CPU assignment. These are provider-reported operation counts, unlike the
// post-fusion ORT profiler node counts above.
inline OperationAssignmentStats parse_vitis_operation_assignment(
    const std::filesystem::path& cache_dir,
    const std::vector<std::string>& cache_keys) {
    OperationAssignmentStats result;
    int total_operations = 0;
    int npu_operations = 0;
    for (const auto& key : cache_keys) {
        const std::filesystem::path dir = cache_dir / key;
        const int total = detail::csv_data_rows(dir / "gops.csv");
        const int npu = detail::json_string_array_entries(dir / "context.json", "nodes");
        if (total <= 0 || npu < 0 || npu > total) return result;
        total_operations += total;
        npu_operations += npu;
    }
    if (total_operations <= 0) return result;
    result.measured = true;
    result.cpu_operations = total_operations - npu_operations;
    result.npu_operations = npu_operations;
    result.source = "vitisai-cache-context-gops";
    return result;
}

// Parse `profile_json`. Nodes are deduplicated by name, so a profile spanning many
// inference iterations still yields the true per-provider node count.
inline OffloadStats parse_ort_profile(const std::filesystem::path& profile_json) {
    OffloadStats st;
    std::ifstream f(profile_json, std::ios::binary);
    if (!f) return st;
    std::stringstream buf;
    buf << f.rdbuf();
    const std::string s = buf.str();
    if (s.empty()) return st;

    const std::string kKernel = "_kernel_time";
    std::unordered_map<std::string, std::string> node_provider;  // node name -> provider
    std::unordered_map<std::string, std::string> node_op;        // node name -> op_name

    // Walk each "name" occurrence; the enclosing event's provider/op_name (in its
    // "args" object) follow within the same record, before the next "name".
    const std::string name_key = "\"name\"";
    size_t pos = 0;
    while ((pos = s.find(name_key, pos)) != std::string::npos) {
        size_t vend = 0;
        const std::string name = detail::json_str_field(s, "name", pos, s.size(), &vend);
        pos += name_key.size();
        if (name.empty() || !detail::ends_with(name, kKernel)) continue;  // node kernel events only
        const size_t next = s.find(name_key, vend);
        const size_t bound = (next == std::string::npos) ? s.size() : next;
        const std::string provider = detail::json_str_field(s, "provider", vend, bound);
        if (provider.empty()) continue;
        const std::string node = name.substr(0, name.size() - kKernel.size());
        node_provider[node] = provider;
        node_op[node] = detail::json_str_field(s, "op_name", vend, bound);
    }

    if (node_provider.empty()) return st;  // no node events -> leave unmeasured

    int ep = 0, cpu = 0;
    std::map<std::string, int> cpu_op_hist;
    std::set<std::string> accelerator_providers;
    bool saw_cpu = false;
    for (const auto& [node, provider] : node_provider) {
        if (provider == "CPUExecutionProvider") {
            ++cpu;
            saw_cpu = true;
            const std::string op = node_op.count(node) ? node_op[node] : "?";
            cpu_op_hist[op.empty() ? "?" : op] += 1;
        } else {
            ++ep;
            accelerator_providers.insert(provider);
        }
    }
    st.measured = true;
    st.ep_nodes = ep;
    st.cpu_nodes = cpu;
    st.cpu_ops = format_op_hist(cpu_op_hist);
    std::ostringstream providers;
    bool first = true;
    for (const auto& provider : accelerator_providers) {
        if (!first) providers << ",";
        providers << provider;
        first = false;
    }
    if (saw_cpu) {
        if (!first) providers << ",";
        providers << "CPUExecutionProvider";
    }
    st.providers = providers.str();
    return st;
}

}  // namespace ort_common
}  // namespace npu_inference_bench
