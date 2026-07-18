#include "ort_offload.hpp"

#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>

namespace npu_inference_bench {
namespace ort_common {
namespace detail {

std::string json_str_field(const std::string& s, const std::string& key, size_t from,
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

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int csv_data_rows(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) return -1;
    std::string line;
    int rows = -1;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) ++rows;
    }
    return rows < 0 ? 0 : rows;
}

int json_string_array_entries(const std::filesystem::path& path,
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

std::string format_op_hist(const std::map<std::string, int>& hist) {
    std::ostringstream ops;
    bool first = true;
    for (const auto& [op, n] : hist) {
        if (!first) ops << ", ";
        ops << op << " x" << n;
        first = false;
    }
    return ops.str();
}

OperationAssignmentStats parse_vitis_operation_assignment(
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

OffloadStats parse_ort_profile(const std::filesystem::path& profile_json) {
    OffloadStats st;
    std::ifstream f(profile_json, std::ios::binary);
    if (!f) return st;
    std::stringstream buf;
    buf << f.rdbuf();
    const std::string s = buf.str();
    if (s.empty()) return st;

    const std::string kKernel = "_kernel_time";
    std::unordered_map<std::string, std::string> node_provider;
    std::unordered_map<std::string, std::string> node_op;

    const std::string name_key = "\"name\"";
    size_t pos = 0;
    while ((pos = s.find(name_key, pos)) != std::string::npos) {
        size_t vend = 0;
        const std::string name = detail::json_str_field(s, "name", pos, s.size(), &vend);
        pos += name_key.size();
        if (name.empty() || !detail::ends_with(name, kKernel)) continue;
        const size_t next = s.find(name_key, vend);
        const size_t bound = (next == std::string::npos) ? s.size() : next;
        const std::string provider = detail::json_str_field(s, "provider", vend, bound);
        if (provider.empty()) continue;
        const std::string node = name.substr(0, name.size() - kKernel.size());
        node_provider[node] = provider;
        node_op[node] = detail::json_str_field(s, "op_name", vend, bound);
    }

    if (node_provider.empty()) return st;

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
