#include "npu_inference_bench/generic_onnx_cli.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npu_inference_bench/runtime/runtime_context.hpp"

namespace npu_inference_bench {
namespace runtime_cli {
namespace {

namespace fs = std::filesystem;
using runtime::TensorElementType;

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open manifest: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

std::vector<std::uint8_t> read_binary(const fs::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open tensor data: " + path.string());
    const std::streamsize size = input.tellg();
    if (size < 0) throw std::runtime_error("cannot determine tensor size: " + path.string());
    input.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (size && !input.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("cannot read tensor data: " + path.string());
    }
    return bytes;
}

void write_binary(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create tensor output: " + path.string());
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    if (!output) throw std::runtime_error("cannot write tensor output: " + path.string());
}

std::size_t value_start(const std::string& json, const std::string& key) {
    const std::string token = "\"" + key + "\"";
    const std::size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) return std::string::npos;
    const std::size_t colon = json.find(':', key_pos + token.size());
    if (colon == std::string::npos) return std::string::npos;
    std::size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    return pos;
}

std::string parse_string_at(const std::string& json, std::size_t pos,
                            std::size_t* end_pos = nullptr) {
    if (pos == std::string::npos || pos >= json.size() || json[pos] != '"') {
        throw std::runtime_error("expected JSON string");
    }
    std::string value;
    bool escaped = false;
    for (++pos; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (escaped) {
            switch (c) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default: throw std::runtime_error("unsupported JSON string escape");
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            if (end_pos) *end_pos = pos + 1;
            return value;
        } else {
            value.push_back(c);
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

std::string required_string(const std::string& json, const std::string& key) {
    const std::size_t pos = value_start(json, key);
    if (pos == std::string::npos) throw std::runtime_error("manifest is missing '" + key + "'");
    return parse_string_at(json, pos);
}

std::vector<std::int64_t> integer_array(const std::string& json, const std::string& key) {
    std::size_t pos = value_start(json, key);
    if (pos == std::string::npos || pos >= json.size() || json[pos] != '[') {
        throw std::runtime_error("manifest is missing array '" + key + "'");
    }
    std::vector<std::int64_t> values;
    ++pos;
    while (pos < json.size()) {
        while (pos < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ',')) ++pos;
        if (pos < json.size() && json[pos] == ']') return values;
        std::size_t consumed = 0;
        const long long value = std::stoll(json.substr(pos), &consumed);
        values.push_back(static_cast<std::int64_t>(value));
        pos += consumed;
    }
    throw std::runtime_error("unterminated integer array '" + key + "'");
}

std::vector<std::string> string_array(const std::string& json, const std::string& key) {
    std::size_t pos = value_start(json, key);
    if (pos == std::string::npos) return {};
    if (pos >= json.size() || json[pos] != '[') {
        throw std::runtime_error("'" + key + "' must be an array");
    }
    std::vector<std::string> values;
    ++pos;
    while (pos < json.size()) {
        while (pos < json.size() &&
               (std::isspace(static_cast<unsigned char>(json[pos])) || json[pos] == ',')) ++pos;
        if (pos < json.size() && json[pos] == ']') return values;
        std::size_t end = pos;
        values.push_back(parse_string_at(json, pos, &end));
        pos = end;
    }
    throw std::runtime_error("unterminated string array '" + key + "'");
}

std::vector<std::string> object_array(const std::string& json, const std::string& key) {
    std::size_t pos = value_start(json, key);
    if (pos == std::string::npos || pos >= json.size() || json[pos] != '[') {
        throw std::runtime_error("manifest is missing object array '" + key + "'");
    }
    std::vector<std::string> objects;
    int depth = 0;
    std::size_t object_start = std::string::npos;
    bool in_string = false;
    bool escaped = false;
    for (++pos; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') { in_string = true; continue; }
        if (c == '{') {
            if (depth++ == 0) object_start = pos;
        } else if (c == '}') {
            if (--depth == 0 && object_start != std::string::npos) {
                objects.push_back(json.substr(object_start, pos - object_start + 1));
                object_start = std::string::npos;
            }
        } else if (c == ']' && depth == 0) {
            return objects;
        }
    }
    throw std::runtime_error("unterminated object array '" + key + "'");
}

TensorElementType parse_dtype(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "float32" || value == "f32") return TensorElementType::Float32;
    if (value == "float64" || value == "f64") return TensorElementType::Float64;
    if (value == "float16" || value == "f16") return TensorElementType::Float16;
    if (value == "bfloat16" || value == "bf16") return TensorElementType::BFloat16;
    if (value == "int8") return TensorElementType::Int8;
    if (value == "uint8") return TensorElementType::UInt8;
    if (value == "int16") return TensorElementType::Int16;
    if (value == "uint16") return TensorElementType::UInt16;
    if (value == "int32") return TensorElementType::Int32;
    if (value == "uint32") return TensorElementType::UInt32;
    if (value == "int64") return TensorElementType::Int64;
    if (value == "uint64") return TensorElementType::UInt64;
    if (value == "bool") return TensorElementType::Bool;
    throw std::runtime_error("unsupported tensor dtype: " + value);
}

Device parse_device(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "npu") return Device::NPU;
    if (value == "gpu") return Device::GPU;
    if (value == "cpu") return Device::CPU;
    throw std::runtime_error("unsupported device: " + value);
}

std::string safe_file_name(std::string value) {
    for (char& c : value) {
        const bool safe = std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-';
        if (!safe) c = '_';
    }
    return value.empty() ? "output" : value;
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char c : value) {
        switch (c) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (c < 0x20) escaped << "?";
                else escaped << static_cast<char>(c);
        }
    }
    return escaped.str();
}

std::string shape_json(const std::vector<std::int64_t>& shape) {
    std::ostringstream output;
    output << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) output << ",";
        output << shape[i];
    }
    output << "]";
    return output.str();
}

void usage() {
    std::cerr
        << "Usage: NpuInferenceBench run-onnx <model.onnx> <inputs.json> [options]\n"
        << "  --device <npu|gpu|cpu>  Logical hardware target (default: cpu)\n"
        << "  --provider <name>       Exact execution provider; disables fallback\n"
        << "  --no-fallback           Try only providers for the requested device\n"
        << "  --strict-device         Fail unless the requested hardware class resolves\n"
        << "  --cache <dir>           Compiled-model cache directory\n"
        << "  --output-dir <dir>      Output tensor directory\n"
        << "Manifest: {\"inputs\":[{\"name\":\"x\",\"dtype\":\"float32\","
           "\"shape\":[1,3],\"file\":\"x.bin\"}],\"outputs\":[\"y\"]}\n";
}

}  // namespace

int run(int argc, char* argv[]) {
    if (argc < 3) {
        usage();
        return 2;
    }
    try {
        const fs::path model_path = fs::absolute(argv[1]);
        const fs::path manifest_path = fs::absolute(argv[2]);
        runtime::RuntimeOptions options;
        options.device = Device::CPU;
        fs::path output_dir = manifest_path.parent_path() / "onnx-output";
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--device" && i + 1 < argc) options.device = parse_device(argv[++i]);
            else if (arg == "--provider" && i + 1 < argc) options.device_override = argv[++i];
            else if (arg == "--cache" && i + 1 < argc) options.cache_dir = argv[++i];
            else if (arg == "--output-dir" && i + 1 < argc) output_dir = fs::absolute(argv[++i]);
            else if (arg == "--no-fallback") options.allow_fallback = false;
            else if (arg == "--strict-device") options.require_requested_device = true;
            else throw std::runtime_error("unknown or incomplete option: " + arg);
        }
        options.profile_execution = true;

        const std::string manifest = read_text(manifest_path);
        std::vector<runtime::Tensor> owned_inputs;
        for (const std::string& object : object_array(manifest, "inputs")) {
            runtime::Tensor tensor;
            tensor.name = required_string(object, "name");
            tensor.type = parse_dtype(required_string(object, "dtype"));
            tensor.shape = integer_array(object, "shape");
            fs::path file = required_string(object, "file");
            if (file.is_relative()) file = manifest_path.parent_path() / file;
            tensor.bytes = read_binary(file);
            tensor.validate();
            owned_inputs.push_back(std::move(tensor));
        }
        std::vector<runtime::TensorView> input_views;
        input_views.reserve(owned_inputs.size());
        for (const runtime::Tensor& tensor : owned_inputs) input_views.push_back(tensor.view());

        runtime::RuntimeContext context(options);
        runtime::ModelSession session = context.load_one(
            {model_path, model_path.parent_path(), model_path.stem().string()});
        const auto started = std::chrono::steady_clock::now();
        std::vector<runtime::Tensor> outputs =
            session.run(input_views, string_array(manifest, "outputs"));
        const double inference_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        session.finalize_profiling();

        std::error_code ec;
        fs::create_directories(output_dir, ec);
        if (ec) throw std::runtime_error("cannot create output directory: " + ec.message());
        std::vector<fs::path> output_files;
        for (const runtime::Tensor& tensor : outputs) {
            const fs::path file = output_dir / (safe_file_name(tensor.name) + ".bin");
            write_binary(file, tensor.bytes);
            output_files.push_back(file);
        }

        const ExecutionDiagnostics& diagnostics = session.diagnostics();
        std::cout << "{\"ok\":true"
                  << ",\"runtime\":\"" << json_escape(session.runtime_name()) << "\""
                  << ",\"runtime_version\":\"" << json_escape(session.runtime_version()) << "\""
                  << ",\"requested_device\":\"" << npu_inference_bench::to_string(options.device) << "\""
                  << ",\"resolved_provider\":\"" << json_escape(diagnostics.resolved_provider) << "\""
                  << ",\"resolved_device\":\"" << runtime::to_string(diagnostics.resolved_device) << "\""
                  << ",\"fallback_occurred\":" << (diagnostics.fallback_occurred ? "true" : "false")
                  << ",\"load_seconds\":" << session.load_seconds()
                  << ",\"inference_seconds\":" << inference_seconds
                  << ",\"ep_nodes\":" << diagnostics.ep_nodes
                  << ",\"cpu_nodes\":" << diagnostics.cpu_nodes
                  << ",\"outputs\":[";
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << "{\"name\":\"" << json_escape(outputs[i].name) << "\""
                      << ",\"dtype\":\"" << runtime::to_string(outputs[i].type) << "\""
                      << ",\"shape\":" << shape_json(outputs[i].shape)
                      << ",\"bytes\":" << outputs[i].bytes.size()
                      << ",\"file\":\"" << json_escape(output_files[i].string()) << "\"}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"ok\":false,\"error\":\"" << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}

}  // namespace runtime_cli
}  // namespace npu_inference_bench
