// Internal parsing, JSON, statistics, and host helpers.
std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parse_backend(const std::string& s, npu_inference_bench::Backend& out) {
    const std::string v = lower(s);
    if (v == "auto")  { out = npu_inference_bench::Backend::Auto; return true; }
    if (v == "intel") { out = npu_inference_bench::Backend::IntelOpenVINO; return true; }
    if (v == "intel-onnx" || v == "intelonnx") {
        out = npu_inference_bench::Backend::IntelOnnx; return true;
    }
    if (v == "onnx" || v == "onnx-static" || v == "ort" || v == "ort-static") {
        out = npu_inference_bench::Backend::OnnxRuntimeStatic; return true;
    }
    if (v == "onnx-dynamic" || v == "onnx-dyn" || v == "ort-dynamic") {
        out = npu_inference_bench::Backend::OnnxRuntimeDynamic; return true;
    }
    if (v == "amd")   { out = npu_inference_bench::Backend::AmdRyzenAI; return true; }
    if (v == "qualcomm" || v == "qnn") { out = npu_inference_bench::Backend::QualcommQNN; return true; }
    return false;
}

bool parse_device(const std::string& s, npu_inference_bench::Device& out) {
    const std::string v = lower(s);
    if (v == "npu") { out = npu_inference_bench::Device::NPU; return true; }
    if (v == "gpu") { out = npu_inference_bench::Device::GPU; return true; }
    if (v == "cpu") { out = npu_inference_bench::Device::CPU; return true; }
    // "auto" is a placeholder: with the Auto backend the engine self-selects the
    // device (best_available_device), so the value here is overridden anyway.
    if (v == "auto") { out = npu_inference_bench::Device::NPU; return true; }
    return false;
}

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

void append_diagnostics_json(
    std::ostringstream& js,
    const npu_inference_bench::ExecutionDiagnostics& diagnostics) {
    js << ",\"requested_provider\":\"" << json_escape(diagnostics.requested_provider) << "\"";
    js << ",\"resolved_provider\":\"" << json_escape(diagnostics.resolved_provider) << "\"";
    js << ",\"fallback_occurred\":" << (diagnostics.fallback_occurred ? "true" : "false");
    js << ",\"provider_attempts\":[";
    for (size_t i = 0; i < diagnostics.attempts.size(); ++i) {
        if (i) js << ',';
        const auto& attempt = diagnostics.attempts[i];
        js << "{\"provider\":\"" << json_escape(attempt.provider)
           << "\",\"success\":" << (attempt.success ? "true" : "false")
           << ",\"error\":\"" << json_escape(attempt.error) << "\"}";
    }
    js << "]";
    if (diagnostics.offload_measured) {
        const int total = diagnostics.ep_nodes + diagnostics.cpu_nodes;
        js << ",\"ep_nodes\":" << diagnostics.ep_nodes;
        js << ",\"cpu_nodes\":" << diagnostics.cpu_nodes;
        js << ",\"cpu_offload_pct\":"
           << (total > 0 ? 100.0 * diagnostics.cpu_nodes / total : 0.0);
        js << ",\"cpu_offload_ops\":\"" << json_escape(diagnostics.cpu_offload_ops) << "\"";
    }
}

std::string provider_attempts_json(
    const npu_inference_bench::ExecutionDiagnostics& diagnostics) {
    std::ostringstream js;
    js << '[';
    for (size_t i = 0; i < diagnostics.attempts.size(); ++i) {
        if (i) js << ',';
        const auto& attempt = diagnostics.attempts[i];
        js << "{\"provider\":\"" << json_escape(attempt.provider)
           << "\",\"success\":" << (attempt.success ? "true" : "false")
           << ",\"error\":\"" << json_escape(attempt.error) << "\"}";
    }
    js << ']';
    return js.str();
}

double model_size_mb(const std::string& dir) {
    std::error_code ec;
    uintmax_t total = 0;
    std::filesystem::path p(dir);
    if (!std::filesystem::exists(p, ec)) return -1.0;
    for (auto it = std::filesystem::recursive_directory_iterator(p, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec)) total += it->file_size(ec);
    }
    return static_cast<double>(total) / 1e6;
}

double percentile(std::vector<double> v, double pct) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = pct / 100.0 * (v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

std::string csv_escape(const std::string& s) {
    bool quote = s.find_first_of(",\"\r\n") != std::string::npos;
    if (!quote) return s;

    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

std::string utc_now_iso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string environment_value(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

// Format an optional numeric CSV cell: empty string when the backend didn't
// provide it (keeps the shared schema backend-neutral -- a backend fills only the
// metrics it can measure).
std::string opt_num(double v, bool present, int prec = 6) {
    if (!present) return "";
    std::ostringstream o;
    o << std::setprecision(prec) << v;
    return o.str();
}

// Best-effort AC vs battery detection, read at run time. Power state changes
// CPU/NPU clocking (battery = throttled), so an unlabeled battery run can silently
// skew a comparison; record it per row. "ac" / "battery" / "unknown".
std::string power_source() {
#if defined(_WIN32)
    SYSTEM_POWER_STATUS s{};
    if (GetSystemPowerStatus(&s)) {
        if (s.ACLineStatus == 1) return "ac";       // plugged in (also desktops w/o battery)
        if (s.ACLineStatus == 0) return "battery";  // running on battery
    }
    return "unknown";  // ACLineStatus 255 = unknown, or call failed
#else
    return "unknown";
#endif
}

// Compile-time target ISA of this binary. This is the axis that decides which
// vendor DLL pack a build belongs to (x64 = Intel/AMD; arm64 = Qualcomm/QNN), so
// it's recorded per row to keep cross-ISA results attributable in one ledger.
std::string host_arch() {
#if defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#elif defined(_M_X64) || defined(__x86_64__)
    return "x64";
#elif defined(_M_IX86) || defined(__i386__)
    return "x86";
#else
    return "unknown";
#endif
}

// Best-effort OS identity. On Windows, RtlGetVersion is the only non-deprecated
// way to read the real build number (GetVersionEx lies without a manifest).
std::string host_os() {
#if defined(_WIN32)
    if (HMODULE nt = GetModuleHandleW(L"ntdll.dll")) {
        using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
        if (auto fn = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(nt, "RtlGetVersion"))) {
            RTL_OSVERSIONINFOW vi{};
            vi.dwOSVersionInfoSize = sizeof(vi);
            if (fn(&vi) == 0) {
                const char* name = (vi.dwMajorVersion == 10 && vi.dwBuildNumber >= 22000)
                                       ? "Windows 11"
                                       : (vi.dwMajorVersion == 10 ? "Windows 10" : "Windows");
                std::ostringstream o;
                o << name << " (build " << vi.dwBuildNumber << ")";
                return o.str();
            }
        }
    }
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "unknown";
#endif
}

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else if (c == '"') {
            in_quotes = true;
        } else if (c == ',') {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    fields.push_back(cur);
    return fields;
}
