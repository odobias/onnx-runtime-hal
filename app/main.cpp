// whisper_hal CLI: backend-agnostic Whisper runner / benchmark built on the HAL.
//
// Usage: whisper_hal <model_dir> <audio.wav> [backend] [device] [runs] [--cache <dir>]
//   backend: auto | intel | amd | qualcomm     (default: auto)
//   device : npu  | gpu | cpu                   (default: npu)
//   runs   : timed iterations                   (default: 5)
//   --cache <dir> : persist the compiled model here. When set, the app loads the
//                   engine twice (cold vs warm) to demonstrate cache load speedup.
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "whisper_npu/audio.hpp"
#include "whisper_npu/whisper_engine.hpp"

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parse_backend(const std::string& s, whisper_npu::Backend& out) {
    const std::string v = lower(s);
    if (v == "auto")  { out = whisper_npu::Backend::Auto; return true; }
    if (v == "intel") { out = whisper_npu::Backend::IntelOpenVINO; return true; }
    if (v == "amd")   { out = whisper_npu::Backend::AmdRyzenAI; return true; }
    if (v == "qualcomm" || v == "qnn") { out = whisper_npu::Backend::QualcommQNN; return true; }
    return false;
}

bool parse_device(const std::string& s, whisper_npu::Device& out) {
    const std::string v = lower(s);
    if (v == "npu") { out = whisper_npu::Device::NPU; return true; }
    if (v == "gpu") { out = whisper_npu::Device::GPU; return true; }
    if (v == "cpu") { out = whisper_npu::Device::CPU; return true; }
    return false;
}

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

}  // namespace

int main(int argc, char* argv[]) {
    using namespace whisper_npu;

    std::vector<std::string> pos;
    std::string cache_dir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cache" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else {
            pos.push_back(a);
        }
    }

    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_dir> <audio.wav> [backend] [device] [runs] [--cache <dir>]\n"
                  << "  backend: auto | intel | amd | qualcomm   (default auto)\n"
                  << "  device : npu | gpu | cpu                 (default npu)\n"
                  << "  runs   : timed iterations                (default 5)\n"
                  << "  --cache <dir>: persist compiled model; loads twice (cold/warm)\n\n";
        std::cerr << "Compiled-in backends:";
        for (Backend b : available_backends()) std::cerr << " " << to_string(b);
        if (available_backends().empty()) std::cerr << " (none!)";
        std::cerr << "\n";
        return 1;
    }

    EngineOptions opt;
    opt.model_dir = pos[0];
    const std::string audio_path = pos[1];
    opt.cache_dir = cache_dir;

    Backend backend = Backend::Auto;
    if (pos.size() > 2 && !parse_backend(pos[2], backend)) {
        std::cerr << "Unknown backend: " << pos[2] << "\n";
        return 1;
    }
    if (pos.size() > 3 && !parse_device(pos[3], opt.device)) {
        std::cerr << "Unknown device: " << pos[3] << "\n";
        return 1;
    }
    const int runs = pos.size() > 4 ? std::max(1, std::atoi(pos[4].c_str())) : 5;
    const int warmup = 1;

    WavData wav;
    try {
        wav = read_wav(audio_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read WAV: " << e.what() << "\n";
        return 2;
    }
    const double audio_len =
        static_cast<double>(wav.samples.size()) /
        static_cast<double>(wav.sample_rate ? wav.sample_rate : 16000);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Audio  : " << audio_path << "  (" << audio_len << "s @ "
              << wav.sample_rate << " Hz, " << wav.channels << " ch)\n";
    std::cout << "Backend: " << to_string(backend) << "   Device: " << to_string(opt.device)
              << "   Runs: " << runs << "\n";
    std::cout << "Cache  : " << (cache_dir.empty() ? std::string("(disabled)") : cache_dir) << "\n";
    if (wav.sample_rate != 16000) {
        std::cerr << "WARNING: expected 16000 Hz mono; no resampling is done.\n";
    }
    std::cout << "\n";

    // --- Cold load (compiles; writes cache if enabled) -----------------------
    std::unique_ptr<IWhisperEngine> engine;
    double cold_load = 0.0;
    try {
        engine = create_engine(backend, opt);
        cold_load = engine->load_seconds();
    } catch (const std::exception& e) {
        std::cerr << "Engine creation failed: " << e.what() << "\n";
        return 3;
    }
    std::cout << "Engine : " << engine->backend_name() << "  [" << engine->device_name() << "]\n";
    std::cout << std::setprecision(3);
    std::cout << "load (cold)  : " << cold_load << " s\n";

    // --- Warm load (imports cached blob) to demonstrate the caching win ------
    if (!cache_dir.empty()) {
        try {
            auto warm = create_engine(backend, opt);
            const double warm_load = warm->load_seconds();
            std::cout << "load (warm)  : " << warm_load << " s";
            if (warm_load > 0.0) {
                std::cout << "   (" << std::setprecision(1) << (cold_load / warm_load)
                          << "x faster than cold)" << std::setprecision(3);
            }
            std::cout << "\n";
            engine = std::move(warm);  // use the warm-loaded engine for inference
        } catch (const std::exception& e) {
            std::cerr << "Warm reload failed: " << e.what() << "\n";
        }
    }
    std::cout << "\n";

    // --- Inference benchmark -------------------------------------------------
    try {
        for (int i = 0; i < warmup; ++i) engine->transcribe(wav.samples);

        std::vector<double> lat;
        std::string text;
        for (int i = 0; i < runs; ++i) {
            auto r = engine->transcribe(wav.samples);
            lat.push_back(r.infer_seconds);
            text = trim(r.text);
        }
        double mean = 0.0;
        for (double l : lat) mean += l;
        mean /= static_cast<double>(lat.size());
        const double rtf = mean / audio_len;

        std::cout << std::setprecision(1);
        std::cout << "mean infer   : " << mean * 1000.0 << " ms   " << std::setprecision(3)
                  << "RTF=" << rtf << std::setprecision(1) << "  (" << 1.0 / rtf
                  << "x real time)\n";
        std::cout << "transcription: " << text << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Transcription failed: " << e.what() << "\n";
        return 4;
    }
    return 0;
}
