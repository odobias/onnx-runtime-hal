// Deepfake-classifier benchmark API (ONNX Runtime, HAL-adjacent).
//
// The deepfake pipeline's two on-device classifiers -- the Text Scam Classifier
// (TSC) and the Generated Audio Detector (FakeAudio) -- are plain ORT classifiers,
// NOT the transcribe(audio)->text contract of IWhisperEngine. Rather than bend
// them into the ASR interface, this is a small parallel harness that replays
// pre-baked, already-validated input tensors (see scripts/experiments/dump_fixtures.py)
// across ONNX Runtime execution providers (CPU / DirectML / VitisAI / QNN / OpenVINO),
// measuring per-EP latency AND whether the accelerator reproduces the CPU reference
// probability. Preprocessing stays in the Python validators (single source of truth);
// C++ only loads tensors + runs them, so a wrong number here is a runtime bug, not a
// preprocessing mismatch.
#pragma once

#include <string>
#include <vector>

#include "whisper_npu/whisper_engine.hpp"  // Device

namespace whisper_npu {
namespace classifier {

struct SampleResult {
    std::string id;
    std::string label;      // ground truth: "scam"/"clean"/"deepfake"/"real", or "unknown"
    std::string predicted;  // predicted label on this EP
    double p = 0.0;         // p(positive class) measured on this EP
    double expected_p = 0.0;  // CPU reference p baked into the fixture
    bool has_label = false;   // false when there's no ground truth (label=="unknown")
    bool correct = false;     // predicted == label (only meaningful when has_label)
};

struct Result {
    std::string model;              // fixture dir basename (e.g. "tsc", "fakeaudio")
    std::string backend_name;       // "ONNX Runtime (classifier)"
    std::string requested_device;   // "NPU"/"GPU"/"CPU"
    std::string execution_provider; // EP that actually built the session (post-fallback)
    std::string runtime;            // "onnxruntime-directml" / "-openvino" / ...
    std::string runtime_version;    // Ort::GetVersionString()
    std::string host_arch;          // x64 / arm64 / ...
    std::string host_os;
    double load_seconds = 0.0;
    double mean_infer_ms = 0.0;
    double median_infer_ms = 0.0;
    double p90_infer_ms = 0.0;
    int runs = 0;
    double model_size_mb = -1.0;
    int eval_samples = 0;   // samples that carry ground truth
    int correct = 0;
    double max_abs_p_diff = 0.0;  // max |p - expected_p| over all samples (cross-EP agreement)
    std::vector<SampleResult> samples;
};

// Load fixtures from `fixture_dir` (model.tsv / samples.tsv / inputs.tsv / data/),
// build an ORT session for `device` (NPU->GPU->CPU fallback) or for
// `provider_override` verbatim, and replay every sample `runs` times.
// Throws std::runtime_error on hard failure (missing fixtures, no EP could build).
Result run(const std::string& fixture_dir, Device device,
           const std::string& provider_override, int runs, int cpu_threads,
           const std::string& cache_dir);

// True if this build has ONNX Runtime compiled in (WHISPER_HAL_ORT).
bool available();

}  // namespace classifier
}  // namespace whisper_npu
