// Resolve filterable benchmark metadata from model_dir, manifest.json, and run output.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

#include "npu_inference_bench/whisper.hpp"
#include "npu_inference_bench/whisper_frontend.hpp"

namespace npu_inference_bench {
namespace benchmark_meta {

namespace fs = std::filesystem;
namespace fe = npu_inference_bench::frontend;

struct Row {
    std::string model_package;      // e.g. en-static-onnx (leaf of workloads/whisper/models/static-onnx)
    std::string variant_id;         // manifest id or label fallback
    std::string base_model;         // e.g. openai/whisper-tiny.en
    std::string precision;          // fp32, fp16, int8, int4, fp32-static, ...
    std::string quant_method;       // human-readable compression/export method
    std::string execution_provider; // ORT EP / OpenVINO device string
    std::string runtime;            // onnxruntime, openvino-genai, ...
    std::string model_format;       // onnx, ov-ir
    std::string decode_strategy;    // static-no-kv, genai-bounded-kv, ...
    long max_context = 0;
};

inline std::string normalize_path_key(std::string p) {
    std::replace(p.begin(), p.end(), '\\', '/');
    while (!p.empty() && p.front() == '/') p.erase(p.begin());
    return p;
}

inline std::string package_name(const std::string& model_dir) {
    if (model_dir.empty()) return "";
    return fs::path(model_dir).filename().string();
}

inline std::string infer_precision(const std::string& package) {
    std::string p = package;
    std::transform(p.begin(), p.end(), p.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (p.find("int4") != std::string::npos) return "int4";
    if (p.find("int8") != std::string::npos) return "int8";
    if (p.find("fp16") != std::string::npos || p.find("-f16") != std::string::npos) return "fp16";
    if (p.find("fp32-static") != std::string::npos || p.find("static") != std::string::npos)
        return "fp32-static";
    return "fp32";
}

inline std::string json_string_in(const std::string& blob, const std::string& key) {
    const std::string k = "\"" + key + "\"";
    size_t p = blob.find(k);
    if (p == std::string::npos) return "";
    p = blob.find(':', p);
    if (p == std::string::npos) return "";
    p = blob.find('"', p);
    if (p == std::string::npos) return "";
    const size_t start = p + 1;
    const size_t end = blob.find('"', start);
    if (end == std::string::npos) return "";
    return blob.substr(start, end - start);
}

inline std::optional<Row> manifest_row_for_model_dir(const std::string& manifest_text,
                                                     const std::string& model_dir) {
    const std::string key = normalize_path_key(model_dir);
    const std::string needle = "\"model_dir\": \"" + key + "\"";
    const size_t pos = manifest_text.find(needle);
    if (pos == std::string::npos) return std::nullopt;

    const size_t block_start = manifest_text.rfind('{', pos);
    const size_t block_end = manifest_text.find('}', pos);
    if (block_start == std::string::npos || block_end == std::string::npos || block_end <= block_start)
        return std::nullopt;

    const std::string block = manifest_text.substr(block_start, block_end - block_start + 1);
    Row out;
    out.variant_id = json_string_in(block, "id");
    out.precision = json_string_in(block, "precision");
    out.quant_method = json_string_in(block, "method");
    out.model_package = package_name(key);
    return out;
}

inline fs::path manifest_path_for(const std::string& model_dir) {
    if (const char* env = std::getenv("WHISPER_MANIFEST")) {
        if (*env) return fs::path(env);
    }
    fs::path p(model_dir);
    if (p.has_parent_path()) {
        const fs::path candidate = p.parent_path() / "manifest.json";
        if (fs::exists(candidate)) return candidate;
    }
    return {};
}

inline Row resolve(const std::string& model_dir,
                   const std::string& label,
                   const IWhisperEngine& engine,
                   const TranscribeResult& last) {
    Row out;
    out.model_package = package_name(model_dir);
    out.variant_id = label;
    out.base_model = "openai/whisper-tiny.en";
    out.precision = infer_precision(out.model_package);
    out.quant_method = "FP32 baseline (inferred from package name)";
    out.execution_provider = engine.device_name();
    out.runtime = last.runtime.empty() ? engine.backend_name() : last.runtime;
    out.model_format = last.model_format;
    out.decode_strategy = last.decode_strategy;
    out.max_context = last.max_context;

    if (const fs::path manifest_path = manifest_path_for(model_dir); !manifest_path.empty()) {
        try {
            const std::string manifest_text = fe::read_text(manifest_path);
            out.base_model = json_string_in(manifest_text, "model");
            if (out.base_model.empty()) out.base_model = "openai/whisper-tiny.en";

            if (auto matched = manifest_row_for_model_dir(manifest_text, model_dir)) {
                if (label.empty() && !matched->variant_id.empty()) out.variant_id = matched->variant_id;
                if (!matched->precision.empty()) out.precision = matched->precision;
                if (!matched->quant_method.empty()) out.quant_method = matched->quant_method;
            }
        } catch (...) {
        }
    }

    if (out.variant_id.empty()) out.variant_id = out.model_package;

    const std::string backend_lc = [&]() {
        std::string b = engine.backend_name();
        std::transform(b.begin(), b.end(), b.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return b;
    }();

    if (out.model_format.empty()) {
        if (backend_lc.find("openvino genai") != std::string::npos ||
            out.model_package.find("-ov-") != std::string::npos ||
            out.model_package.find("en-ov") != std::string::npos) {
            out.model_format = "ov-ir";
        } else {
            out.model_format = "onnx";
        }
    }
    if (out.decode_strategy.empty()) {
        if (out.model_format == "ov-ir") out.decode_strategy = "genai-bounded-kv";
        else out.decode_strategy = "static-no-kv";
    }

    if (out.runtime.empty() || out.runtime == engine.backend_name()) {
        if (backend_lc.find("qnn") != std::string::npos) out.runtime = "onnxruntime-qnn";
        else if (backend_lc.find("vitisai") != std::string::npos || backend_lc.find("ryzen") != std::string::npos)
            out.runtime = "onnxruntime-vitisai";
        else if (backend_lc.find("openvino genai") != std::string::npos) out.runtime = "openvino-genai";
        else if (backend_lc.find("openvino") != std::string::npos) out.runtime = "openvino";
        else if (backend_lc.find("onnx runtime") != std::string::npos) out.runtime = "onnxruntime";
    }

    return out;
}

}  // namespace benchmark_meta
}  // namespace npu_inference_bench
