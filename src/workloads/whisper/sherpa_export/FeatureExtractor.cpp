#include "stdafx.h"
#include "FeatureExtractor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>

namespace asw::whisper
{
    FeatureExtractor::FeatureExtractor(std::int32_t n_mels)
        : _n_mels(n_mels)
        , _fft(kNFft)
        , _mel(kSampleRate, kNFft, n_mels)
        , _hann(kNFft)
    {
        // DFT-periodic Hann window: hann[i] = 0.5 * (1 - cos(2*pi*i / N)).
        // This is torch.hann_window(N) with the default periodic=True, and
        // matches librosa / scipy.signal.get_window('hann', N) with the
        // default fftbins=True. Reference OpenAI Whisper and Sherpa-ONNX
        // exports both use this form; mismatching to the symmetric (N-1)
        // variant shifts mel energies by O(1e-5) and breaks bit-parity
        // with the quantization calibration we ship.
        for (std::int32_t i = 0; i < kNFft; ++i)
        {
            const double theta =
                2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(kNFft);
            _hann[i] = static_cast<float>(0.5 - 0.5 * std::cos(theta));
        }
    }

    std::vector<float> FeatureExtractor::Extract(std::span<const float> samples) const
    {
        assert(static_cast<std::int32_t>(samples.size()) == kChunkSamples);

        // Whisper pads the input with reflect padding of n_fft/2 on each
        // side BEFORE framing; librosa's stft does the same by default
        // (center=True, pad_mode='reflect').
        const std::int32_t pad = kNFft / 2;
        std::vector<float> padded(kChunkSamples + 2 * pad);
        for (std::int32_t i = 0; i < pad; ++i)
            padded[i] = samples[pad - i]; // reflect: mirror around index 0
        std::copy(samples.begin(), samples.end(), padded.begin() + pad);
        for (std::int32_t i = 0; i < pad; ++i)
            padded[kChunkSamples + pad + i] = samples[kChunkSamples - 2 - i];

        std::vector<float> mel_spec(static_cast<size_t>(_n_mels) * kNFrames);
        std::vector<float> frame(kNFft);
        std::vector<float> power(_fft.NumBins());
        std::vector<float> mel_frame(_n_mels);

        for (std::int32_t t = 0; t < kNFrames; ++t)
        {
            const std::int32_t start = t * kHopLength;
            for (std::int32_t i = 0; i < kNFft; ++i)
                frame[i] = padded[start + i] * _hann[i];

            _fft.PowerSpectrum(frame, power);
            _mel.Project(power, mel_frame);

            // Store column-wise in [n_mels, n_frames] layout (mel-major).
            for (std::int32_t m = 0; m < _n_mels; ++m)
                mel_spec[static_cast<size_t>(m) * kNFrames + t] = mel_frame[m];
        }

        // log10 with lower clamp at 1e-10.
        float log_max = -std::numeric_limits<float>::infinity();
        for (float& v : mel_spec)
        {
            v = std::log10(std::max(v, 1e-10f));
            if (v > log_max)
                log_max = v;
        }

        // Dynamic-range compression: clamp everything to within 8 dB of max.
        const float floor_ = log_max - 8.0f;
        for (float& v : mel_spec)
        {
            if (v < floor_)
                v = floor_;
            v = (v + 4.0f) / 4.0f;
        }

        return mel_spec;
    }

} // namespace asw::whisper
