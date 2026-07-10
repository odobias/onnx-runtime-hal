// Public API for the Whisper NPU hardware-abstraction layer (HAL).
//
// One stable C++ interface (IWhisperEngine) for running whisper-tiny(.en) speech
// recognition on different vendor NPUs. Concrete backends are selected at runtime
// via create_engine(); which backends are *available* depends on what the library
// was compiled with (see WHISPER_HAL_INTEL / _AMD / _QUALCOMM).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace whisper_npu {

// Vendor backend / inference runtime.
enum class Backend {
    Auto,          // pick the first compiled-in, available backend
    IntelOpenVINO, // Intel NPU/GPU/CPU via OpenVINO GenAI  (reference impl)
    IntelOnnx,     // Intel NPU/GPU/CPU via OpenVINO on the neutral ONNX (static, no-KV)
    OnnxRuntimeStatic, // Vendor-neutral static ONNX via ONNX Runtime providers
    OnnxRuntimeDynamic, // Vendor-neutral with-past KV-cache ONNX via ORT (CPU/GPU; NPU rejects dynamic shapes)
    AmdRyzenAI,    // AMD XDNA NPU via ONNX Runtime + VitisAI EP  (prepared)
    QualcommQNN,   // Qualcomm Hexagon NPU via ONNX Runtime + QNN EP  (prepared)
};

// Logical accelerator target. Mapped to a backend-specific device string.
enum class Device {
    NPU,
    GPU,
    CPU,
};

struct EngineOptions {
    // Path to the exported model directory for the chosen backend.
    // Intel: OpenVINO IR dir (e.g. whisper-tiny-en-hybrid-ov).
    // AMD/Qualcomm: ONNX / QNN context-binary dir (backend-defined).
    std::string model_dir;

    Device device = Device::NPU;

    // Optional raw device string that overrides `device` for the backend
    // (e.g. OpenVINO "NPU", "GPU.1"; ORT EP-specific selectors). Empty = derive
    // from `device`.
    std::string device_override;

    // True when the caller asked for automatic device selection (CLI device "auto")
    // rather than naming NPU/GPU/CPU. Only then may Backend::Auto self-pick the
    // device (e.g. the unified ORT binary's best_available_device). When false, an
    // explicitly requested `device` is always honored -- otherwise a benchmark row
    // requesting GPU/CPU would silently run (and be mislabeled) on the NPU.
    bool auto_device = false;

    // Directory for the compiled-model cache. When set, the backend persists its
    // device-compiled blob here so subsequent loads skip the (slow) NPU compile
    // step. Empty = no on-disk caching. This is the key to a fast-loading runner.
    std::string cache_dir;

    // CPU-only tuning: number of inference threads (maps to OpenVINO's
    // ov::inference_num_threads on the Intel backend). 0 = backend/runtime default.
    // Ignored by NPU/GPU devices, where the accelerator's own scheduler applies.
    int cpu_threads = 0;
};

// Backend-neutral result + metrics. Every backend fills `text`/`infer_seconds`;
// the remaining fields are best-effort. `has_token_metrics` indicates whether the
// token-level confidence/perf fields below were populated by the backend (so a
// common harness can compare only what's actually available across platforms).
struct TranscribeResult {
    std::string text;
    double infer_seconds = 0.0;   // wall-clock of the generate/inference call

    // Confidence (self-reported by the model; miscalibration is possible).
    double sequence_logprob = 0.0;  // sum of token log-probs for the chosen hypothesis
    double avg_logprob = 0.0;       // sequence_logprob / generated_tokens (higher = more confident)
    long generated_tokens = 0;

    // Fine-grained performance (backend may leave these at -1 if unsupported).
    double ttft_ms = -1.0;          // time to first token
    double tpot_ms = -1.0;          // time per output token
    double throughput_tps = -1.0;   // tokens/second

    bool has_token_metrics = false;

    // Self-describing run metadata for the shared benchmark schema (empty/0 =
    // unset). Lets ONNX vs OV-IR and different decode strategies be compared in
    // one CSV alongside the Python-produced rows.
    std::string runtime;          // e.g. "openvino", "onnxruntime"
    std::string model_format;     // e.g. "onnx", "ov-ir"
    std::string decode_strategy;  // e.g. "static-no-kv", "genai-bounded-kv"
    long max_context = 0;         // static decoder context length (0 = n/a)
};

// CPU-offload audit for the ORT-based backends: when an accelerator EP is asked to
// run the model, does any op silently fall back to the ORT CPU EP? The accelerator
// EPs fuse their supported subgraph into a single node, so `ep_nodes` counts the
// accelerator subgraph node(s) and `cpu_nodes` the host-side fallbacks (summed over
// the encoder + decoder sessions). `measured=false` when the backend can't report it
// (e.g. native OpenVINO/GenAI, which compiles monolithically -> no per-op fallback).
struct OffloadInfo {
    bool measured = false;
    int ep_nodes = -1;
    int cpu_nodes = -1;
    std::string cpu_ops;  // fallback op histogram, e.g. "Gather x3, Cast x2"

    // 0..100 fraction of executed nodes that ran on the CPU EP; -1 when unmeasured.
    double cpu_offload_pct() const {
        if (!measured) return -1.0;
        const int total = ep_nodes + cpu_nodes;
        return total > 0 ? 100.0 * cpu_nodes / total : 0.0;
    }
};

// 16 kHz, mono, float PCM normalized to [-1, 1].
using AudioSamples = std::vector<float>;

class IWhisperEngine {
public:
    virtual ~IWhisperEngine() = default;

    // Run recognition on a full audio buffer. Throws std::runtime_error on failure.
    virtual TranscribeResult transcribe(const AudioSamples& audio) = 0;

    virtual std::string backend_name() const = 0;
    virtual std::string device_name() const = 0;

    // Specific hardware identity, when the backend can query it (e.g. the Intel
    // backend reads OpenVINO's ov::device::full_name -> "Intel(R) AI Boost",
    // "13th Gen Intel(R) Core(TM) i7-1370P", ...). Empty if unavailable/not
    // queried. device_name() remains the logical class ("NPU"/"GPU"/"CPU") that's
    // always populated; this fills in *which* NPU/GPU/CPU, for benchmark rows that
    // get compared across machines.
    virtual std::string full_device_name() const { return {}; }

    // Version of the underlying inference runtime, when the backend can report it
    // (the ONNX Runtime backends return Ort::GetVersionString() -> "1.24.4"; the
    // Intel backends return OpenVINO's build number). Empty when not applicable.
    // Records WHICH runtime build produced a row so results gathered across
    // platforms / vendor DLL packs stay attributable in one ledger.
    virtual std::string runtime_version() const { return {}; }

    // Wall-clock time spent constructing/compiling this engine (model load +
    // device compile). With a warm cache this should drop dramatically.
    virtual double load_seconds() const = 0;

    // CPU-offload audit (see OffloadInfo). Default = unmeasured; the ORT backends
    // override it after profiling their first (warmup) inference. Lets the benchmark
    // ledger record whether a "runs on NPU" row actually kept every op off the CPU.
    virtual OffloadInfo offload_info() const { return {}; }
};

// --- Introspection -----------------------------------------------------------

// True if `backend` was compiled into this build and can be instantiated.
bool backend_available(Backend backend);

// All compiled-in backends (may still fail at create() if no hardware/model).
std::vector<Backend> available_backends();

const char* to_string(Backend backend);
const char* to_string(Device device);

// --- Factory -----------------------------------------------------------------

// Create an engine for `backend`. With Backend::Auto, picks the first available
// backend in preference order (Intel, AMD, Qualcomm).
// Throws std::runtime_error if the backend is unavailable or model load fails.
std::unique_ptr<IWhisperEngine> create_engine(Backend backend, const EngineOptions& options);

}  // namespace whisper_npu
