// Dependency-free 16 kHz mono WAV loader for the benchmark runner. Header-only.
// Supports PCM8 / PCM16 / IEEE-float32, mono or multi-channel (down-mixed).
// Does NOT resample: Whisper expects 16 kHz mono.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npu_inference_bench {

struct WavData {
    std::vector<float> samples;  // mono, normalized to [-1, 1]
    uint32_t sample_rate = 0;
    uint16_t channels = 0;
};

inline WavData read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open WAV file: " + path);

    auto rd_u32 = [&]() -> uint32_t {
        uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };
    auto rd_u16 = [&]() -> uint16_t {
        uint16_t v = 0;
        f.read(reinterpret_cast<char*>(&v), 2);
        return v;
    };

    char tag[4];
    f.read(tag, 4);
    if (std::strncmp(tag, "RIFF", 4) != 0) throw std::runtime_error("not a RIFF file");
    rd_u32();
    f.read(tag, 4);
    if (std::strncmp(tag, "WAVE", 4) != 0) throw std::runtime_error("not a WAVE file");

    uint16_t audio_format = 0, num_channels = 0, bits = 0;
    uint32_t sample_rate = 0;
    std::vector<char> data;
    bool have_fmt = false, have_data = false;

    while (f.read(tag, 4)) {
        uint32_t sz = rd_u32();
        if (std::strncmp(tag, "fmt ", 4) == 0) {
            audio_format = rd_u16();
            num_channels = rd_u16();
            sample_rate = rd_u32();
            rd_u32();  // byte rate
            rd_u16();  // block align
            bits = rd_u16();
            if (sz > 16) f.seekg(sz - 16, std::ios::cur);
            have_fmt = true;
        } else if (std::strncmp(tag, "data", 4) == 0) {
            data.resize(sz);
            f.read(data.data(), sz);
            have_data = true;
        } else {
            f.seekg(sz + (sz & 1), std::ios::cur);
        }
    }
    if (!have_fmt) throw std::runtime_error("missing fmt chunk");
    if (!have_data) throw std::runtime_error("missing data chunk");

    WavData out;
    out.sample_rate = sample_rate;
    out.channels = num_channels;
    const size_t ch = num_channels ? num_channels : 1;

    auto downmix = [&](auto* p, size_t n, auto conv) {
        out.samples.reserve(n / ch);
        for (size_t i = 0; i + ch <= n; i += ch) {
            float acc = 0.0f;
            for (size_t c = 0; c < ch; ++c) acc += conv(p[i + c]);
            out.samples.push_back(acc / static_cast<float>(ch));
        }
    };

    if (audio_format == 1 && bits == 16) {
        downmix(reinterpret_cast<const int16_t*>(data.data()), data.size() / 2,
                [](int16_t v) { return v / 32768.0f; });
    } else if (audio_format == 3 && bits == 32) {
        downmix(reinterpret_cast<const float*>(data.data()), data.size() / 4,
                [](float v) { return v; });
    } else if (audio_format == 1 && bits == 8) {
        downmix(reinterpret_cast<const uint8_t*>(data.data()), data.size(),
                [](uint8_t v) { return (static_cast<int>(v) - 128) / 128.0f; });
    } else {
        throw std::runtime_error(
            "unsupported WAV encoding (need PCM8/PCM16/float32): format=" +
            std::to_string(audio_format) + " bits=" + std::to_string(bits));
    }
    return out;
}

}  // namespace npu_inference_bench
