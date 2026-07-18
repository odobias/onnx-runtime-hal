// Header-only Whisper front-end shared by ONNX-consuming backends: log-mel
// feature extraction, byte-level BPE detokenization, and minimal JSON helpers to
// read whisper's preprocessor_config.json / generation_config.json / vocab.json.
//
// This is deliberately dependency-free (no ONNX/OpenVINO types) so any backend
// can compute features + decode text identically; only the tensor execution in
// between differs per vendor. Ported from the AMD backend's originally-inline
// helpers so Intel's raw-ONNX path decodes byte-for-byte the same way.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace npu_inference_bench {
namespace frontend {

namespace fs = std::filesystem;

// Whisper mel front-end constants (tiny/base share these).
constexpr int kSampleRate = 16000;
constexpr int kNfft = 400;
constexpr int kHop = 160;
constexpr int kMelBins = 80;
constexpr int kFftBins = 201;
constexpr int kSamples = 480000;  // 30 s @ 16 kHz
constexpr int kFrames = 3000;
constexpr double kPi = 3.14159265358979323846;

inline std::string read_text(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Slaney-scale mel filterbank identical to librosa.filters.mel(sr=16000,
// n_fft=400, n_mels=80, htk=False, norm="slaney") -- i.e. the exact bank baked
// into whisper's mel_filters. Returned mel-major ([80][201], row = mel bin).
// Used as a fallback when the model's preprocessor_config.json omits mel_filters
// (optimum's neutral export computes them at runtime instead of storing them).
inline std::vector<float> compute_mel_filterbank() {
    constexpr double f_sp = 200.0 / 3.0;
    constexpr double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;  // = 15
    const double logstep = std::log(6.4) / 27.0;

    auto hz_to_mel = [&](double hz) {
        if (hz < min_log_hz) return hz / f_sp;
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    };
    auto mel_to_hz = [&](double mel) {
        if (mel < min_log_mel) return f_sp * mel;
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    };

    std::vector<double> fftfreqs(kFftBins);
    for (int k = 0; k < kFftBins; ++k)
        fftfreqs[k] = static_cast<double>(k) * (kSampleRate / 2.0) / (kFftBins - 1);

    const int n_pts = kMelBins + 2;
    const double min_mel = hz_to_mel(0.0);
    const double max_mel = hz_to_mel(kSampleRate / 2.0);
    std::vector<double> freqs(n_pts);
    for (int i = 0; i < n_pts; ++i)
        freqs[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (n_pts - 1));

    std::vector<float> filters(static_cast<size_t>(kMelBins) * kFftBins, 0.0f);
    for (int m = 0; m < kMelBins; ++m) {
        const double lo = freqs[m], ctr = freqs[m + 1], hi = freqs[m + 2];
        const double enorm = 2.0 / (hi - lo);
        for (int k = 0; k < kFftBins; ++k) {
            const double lower = (fftfreqs[k] - lo) / (ctr - lo);
            const double upper = (hi - fftfreqs[k]) / (hi - ctr);
            const double w = (std::max)(0.0, (std::min)(lower, upper));
            filters[static_cast<size_t>(m) * kFftBins + k] = static_cast<float>(w * enorm);
        }
    }
    return filters;
}

inline std::vector<float> parse_mel_filters(const fs::path& preprocessor_config) {
    const std::string text = read_text(preprocessor_config);
    const size_t key = text.find("\"mel_filters\"");
    if (key == std::string::npos) return compute_mel_filterbank();
    size_t pos = text.find('[', key);
    if (pos == std::string::npos) throw std::runtime_error("mel_filters array not found");

    std::vector<float> values;
    values.reserve(kMelBins * kFftBins);
    char* end = nullptr;
    while (pos < text.size() && values.size() < static_cast<size_t>(kMelBins * kFftBins)) {
        const char ch = text[pos];
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+') {
            const char* start = text.c_str() + pos;
            const float value = std::strtof(start, &end);
            if (end != start) {
                values.push_back(value);
                pos = static_cast<size_t>(end - text.c_str());
                continue;
            }
        }
        ++pos;
    }
    if (values.size() != static_cast<size_t>(kMelBins * kFftBins))
        throw std::runtime_error("expected 80x201 mel filters, got " + std::to_string(values.size()));
    return values;
}

// Returns an [80 x 3000] log-mel feature buffer (mel-major), matching whisper's
// feature extractor: Hann window, |STFT|^2, mel projection, log10, dynamic-range
// clamp to (max-8) and rescale to ~[0,1].
inline std::vector<float> log_mel_spectrogram(const std::vector<float>& input,
                                              const std::vector<float>& mel_filters) {
    std::vector<float> audio(kSamples, 0.0f);
    std::copy_n(input.begin(), (std::min)(input.size(), audio.size()), audio.begin());

    constexpr double pi = 3.14159265358979323846;
    std::vector<float> window(kNfft);
    for (int i = 0; i < kNfft; ++i)
        window[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * i / kNfft));

    std::vector<float> cos_table(kFftBins * kNfft);
    std::vector<float> sin_table(kFftBins * kNfft);
    for (int k = 0; k < kFftBins; ++k) {
        for (int n = 0; n < kNfft; ++n) {
            const double angle = 2.0 * pi * k * n / kNfft;
            cos_table[k * kNfft + n] = static_cast<float>(std::cos(angle));
            sin_table[k * kNfft + n] = static_cast<float>(std::sin(angle));
        }
    }

    std::vector<float> features(kMelBins * kFrames);
    float global_max = -std::numeric_limits<float>::infinity();
    for (int frame = 0; frame < kFrames; ++frame) {
        float power[kFftBins] = {};
        const int base = frame * kHop - kNfft / 2;
        for (int k = 0; k < kFftBins; ++k) {
            float real = 0.0f, imag = 0.0f;
            for (int n = 0; n < kNfft; ++n) {
                const int idx = base + n;
                float sample = (idx >= 0 && idx < kSamples) ? audio[idx] : 0.0f;
                sample *= window[n];
                real += sample * cos_table[k * kNfft + n];
                imag -= sample * sin_table[k * kNfft + n];
            }
            power[k] = real * real + imag * imag;
        }
        for (int mel = 0; mel < kMelBins; ++mel) {
            float energy = 0.0f;
            const float* filter = mel_filters.data() + mel * kFftBins;
            for (int k = 0; k < kFftBins; ++k) energy += filter[k] * power[k];
            const float logv = std::log10((std::max)(energy, 1.0e-10f));
            features[mel * kFrames + frame] = logv;
            global_max = (std::max)(global_max, logv);
        }
    }
    const float floor = global_max - 8.0f;
    for (float& value : features) value = ((std::max)(value, floor) + 4.0f) / 4.0f;
    return features;
}

// --- FFT-accelerated, multithreaded log-mel ---------------------------------
//
// Numerically equivalent to log_mel_spectrogram() above but replaces the naive
// O(frames * bins * nfft) DFT with Bluestein's algorithm (exact for the awkward
// n_fft=400, no zero-pad bin shift) over a radix-2 FFT, and fans the 3000 frames
// out across CPU cores. On a 16-core part this turns the ~180 ms mel front-end
// (the measured dominant cost) into single-digit ms. The naive version is kept
// intact as the reference/fallback.
namespace detail {

using Cf = std::complex<float>;

inline unsigned next_pow2(unsigned n) {
    unsigned p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Iterative in-place radix-2 Cooley-Tukey FFT (a.size() must be a power of two).
// inv=true performs the unscaled inverse transform (caller divides by N).
inline void fft_radix2(std::vector<Cf>& a, bool inv) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = (inv ? 2.0f : -2.0f) * static_cast<float>(kPi) / static_cast<float>(len);
        const Cf wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            Cf w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                const Cf u = a[i + k];
                const Cf v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Precomputed, read-only Bluestein context for the length-N (=400) real DFT,
// shared across all frames and threads.
struct BluesteinCtx {
    size_t N = kNfft;
    size_t M = 0;
    std::vector<float> han;    // Hann window [N]
    std::vector<Cf> chirp;     // exp(-i*pi*n^2/N), n=0..N-1
    std::vector<Cf> kernel_f;  // FFT of the symmetric conj-chirp kernel [M]

    BluesteinCtx() {
        M = next_pow2(static_cast<unsigned>(2 * N - 1));  // 1024 for N=400
        han.resize(N);
        for (size_t n = 0; n < N; ++n)
            han[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / N));
        chirp.resize(N);
        for (size_t n = 0; n < N; ++n) {
            const double a = -kPi * static_cast<double>((n * n) % (2 * N)) / static_cast<double>(N);
            chirp[n] = Cf(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
        }
        std::vector<Cf> b(M, Cf(0.0f, 0.0f));
        for (size_t n = 0; n < N; ++n) {
            const double a = kPi * static_cast<double>((n * n) % (2 * N)) / static_cast<double>(N);
            const Cf v(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
            b[n] = v;
            if (n) b[M - n] = v;
        }
        kernel_f = b;
        fft_radix2(kernel_f, false);
    }
};

inline const BluesteinCtx& bluestein_ctx() {
    static const BluesteinCtx ctx;  // thread-safe init (magic static)
    return ctx;
}

}  // namespace detail

inline std::vector<float> log_mel_spectrogram_fft(const std::vector<float>& input,
                                                  const std::vector<float>& mel_filters) {
    const detail::BluesteinCtx& ctx = detail::bluestein_ctx();
    const size_t M = ctx.M;

    std::vector<float> audio(kSamples, 0.0f);
    std::copy_n(input.begin(), (std::min)(input.size(), audio.size()), audio.begin());

    std::vector<float> features(static_cast<size_t>(kMelBins) * kFrames);

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const unsigned n_threads = (std::min)(hw, static_cast<unsigned>(kFrames));
    std::vector<float> local_max(n_threads, -std::numeric_limits<float>::infinity());

    auto worker = [&](unsigned tid) {
        const int lo = static_cast<int>(static_cast<long long>(kFrames) * tid / n_threads);
        const int hi = static_cast<int>(static_cast<long long>(kFrames) * (tid + 1) / n_threads);
        std::vector<detail::Cf> a(M);
        float fmax = -std::numeric_limits<float>::infinity();
        std::vector<float> power(kFftBins);

        for (int frame = lo; frame < hi; ++frame) {
            const int base = frame * kHop - kNfft / 2;
            for (size_t n = 0; n < M; ++n) {
                if (n < static_cast<size_t>(kNfft)) {
                    const int idx = base + static_cast<int>(n);
                    const float s =
                        (idx >= 0 && idx < kSamples) ? audio[idx] * ctx.han[n] : 0.0f;
                    a[n] = detail::Cf(s, 0.0f) * ctx.chirp[n];
                } else {
                    a[n] = detail::Cf(0.0f, 0.0f);
                }
            }
            detail::fft_radix2(a, false);
            for (size_t m = 0; m < M; ++m) a[m] *= ctx.kernel_f[m];
            detail::fft_radix2(a, true);

            const float inv_m = 1.0f / static_cast<float>(M);
            for (int k = 0; k < kFftBins; ++k) {
                const detail::Cf x = ctx.chirp[k] * (a[k] * inv_m);
                power[k] = x.real() * x.real() + x.imag() * x.imag();
            }
            for (int mel = 0; mel < kMelBins; ++mel) {
                float energy = 0.0f;
                const float* filter = mel_filters.data() + mel * kFftBins;
                for (int k = 0; k < kFftBins; ++k) energy += filter[k] * power[k];
                const float logv = std::log10((std::max)(energy, 1.0e-10f));
                features[static_cast<size_t>(mel) * kFrames + frame] = logv;
                fmax = (std::max)(fmax, logv);
            }
        }
        local_max[tid] = fmax;
    };

    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    float global_max = -std::numeric_limits<float>::infinity();
    for (const float v : local_max) global_max = (std::max)(global_max, v);
    const float floor = global_max - 8.0f;
    for (float& value : features) value = ((std::max)(value, floor) + 4.0f) / 4.0f;
    return features;
}

// --- tokenizer (byte-level BPE detokenization) ------------------------------

inline std::string parse_json_string(const std::string& text, size_t& pos) {
    if (text[pos] != '"') throw std::runtime_error("expected JSON string");
    ++pos;
    std::string out;
    while (pos < text.size()) {
        const char ch = text[pos++];
        if (ch == '"') return out;
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (pos >= text.size()) break;
        const char esc = text[pos++];
        if (esc == '"' || esc == '\\' || esc == '/') out.push_back(esc);
        else if (esc == 'n') out.push_back('\n');
        else if (esc == 'r') out.push_back('\r');
        else if (esc == 't') out.push_back('\t');
        else if (esc == 'u') throw std::runtime_error("\\u escapes not supported in tokenizer vocab");
        else out.push_back(esc);
    }
    throw std::runtime_error("unterminated JSON string");
}

inline std::unordered_map<int64_t, std::string> load_vocab(const fs::path& vocab_path) {
    const std::string text = read_text(vocab_path);
    std::unordered_map<int64_t, std::string> vocab;
    size_t pos = 0;
    while (pos < text.size()) {
        if (text[pos] != '"') {
            ++pos;
            continue;
        }
        std::string token = parse_json_string(text, pos);
        while (pos < text.size() && (text[pos] == ':' || std::isspace(static_cast<unsigned char>(text[pos])))) ++pos;
        char* end = nullptr;
        const long id = std::strtol(text.c_str() + pos, &end, 10);
        if (end != text.c_str() + pos) {
            vocab[id] = std::move(token);
            pos = static_cast<size_t>(end - text.c_str());
        }
    }
    return vocab;
}

inline std::vector<uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = static_cast<uint8_t>(s[i]);
        if (c < 0x80) {
            out.push_back(c);
            ++i;
        } else if ((c >> 5) == 0x6 && i + 1 < s.size()) {
            out.push_back(((c & 0x1F) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3F));
            i += 2;
        } else if ((c >> 4) == 0xE && i + 2 < s.size()) {
            out.push_back(((c & 0x0F) << 12) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                          (static_cast<uint8_t>(s[i + 2]) & 0x3F));
            i += 3;
        } else if ((c >> 3) == 0x1E && i + 3 < s.size()) {
            out.push_back(((c & 0x07) << 18) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                          ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) |
                          (static_cast<uint8_t>(s[i + 3]) & 0x3F));
            i += 4;
        } else {
            ++i;
        }
    }
    return out;
}

inline std::unordered_map<uint32_t, uint8_t> byte_decoder() {
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    std::unordered_map<uint32_t, uint8_t> map;
    for (size_t i = 0; i < bs.size(); ++i)
        map[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    return map;
}

// Decode generated token ids to text. Skips ids >= `special_floor` (eos and
// special tokens live at/above it), maps byte-level BPE glyphs back to raw bytes,
// and trims surrounding whitespace.
inline std::string decode_tokens(const std::vector<int64_t>& tokens,
                                 const std::unordered_map<int64_t, std::string>& vocab,
                                 int64_t special_floor) {
    static const auto decoder = byte_decoder();
    std::string bytes;
    for (const int64_t id : tokens) {
        if (id >= special_floor) continue;
        const auto it = vocab.find(id);
        if (it == vocab.end()) continue;
        for (const uint32_t cp : utf8_codepoints(it->second)) {
            const auto bit = decoder.find(cp);
            if (bit != decoder.end()) bytes.push_back(static_cast<char>(bit->second));
        }
    }
    while (!bytes.empty() && std::isspace(static_cast<unsigned char>(bytes.front()))) bytes.erase(bytes.begin());
    while (!bytes.empty() && std::isspace(static_cast<unsigned char>(bytes.back()))) bytes.pop_back();
    return bytes;
}

// --- minimal JSON scalar/array readers for generation_config.json ------------

inline int64_t json_int(const std::string& t, const std::string& key, int64_t def) {
    const std::string k = "\"" + key + "\"";
    size_t p = t.find(k);
    if (p == std::string::npos) return def;
    p = t.find(':', p);
    if (p == std::string::npos) return def;
    ++p;
    char* end = nullptr;
    const long long v = std::strtoll(t.c_str() + p, &end, 10);
    if (end == t.c_str() + p) return def;
    return static_cast<int64_t>(v);
}

// Reads a flat integer array value (e.g. suppress_tokens, begin_suppress_tokens).
inline std::vector<int64_t> json_int_array(const std::string& t, const std::string& key) {
    std::vector<int64_t> out;
    const std::string k = "\"" + key + "\"";
    size_t p = t.find(k);
    if (p == std::string::npos) return out;
    p = t.find('[', p);
    if (p == std::string::npos) return out;
    ++p;
    while (p < t.size() && t[p] != ']') {
        const char ch = t[p];
        if ((ch >= '0' && ch <= '9') || ch == '-') {
            char* end = nullptr;
            const long long v = std::strtoll(t.c_str() + p, &end, 10);
            if (end != t.c_str() + p) {
                out.push_back(static_cast<int64_t>(v));
                p = static_cast<size_t>(end - t.c_str());
                continue;
            }
        }
        ++p;
    }
    return out;
}

}  // namespace frontend
}  // namespace npu_inference_bench
