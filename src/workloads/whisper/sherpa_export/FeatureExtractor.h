#pragma once

#include "Fft.h"
#include "MelFilters.h"

#include <cstdint>
#include <span>
#include <vector>

namespace asw::whisper
{
    //=========================================================================
    /// Whisper log-mel feature extractor.
    ///
    /// Implements, bit-for-bit, the preprocessing pipeline documented in
    /// features/media_scan/src/py_quantize_npu_ep/README.md (the exact steps
    /// used when the existing Sherpa-ONNX-exported Whisper encoders were
    /// quantized):
    ///
    ///   mel = melspectrogram(audio, n_fft=400, hop_length=160, n_mels=80)
    ///   log_spec = log10(clip(mel, min=1e-10))
    ///   log_spec = max(log_spec, max(log_spec) - 8.0)
    ///   log_spec = (log_spec + 4.0) / 4.0
    ///
    /// Output layout: [n_mels, n_frames] in row-major order. For a 30 s
    /// 16 kHz input this is [n_mels, 3000], where n_mels is 80 for every
    /// Whisper variant up to large-v2 and 128 for large-v3 / large-v3-turbo.
    //=========================================================================
    class FeatureExtractor
    {
    public:
        static constexpr std::int32_t kSampleRate = 16000;
        static constexpr std::int32_t kNFft       = 400;
        static constexpr std::int32_t kHopLength  = 160;
        static constexpr std::int32_t kNMels      = 80;               ///< default (tiny..large-v2)
        static constexpr std::int32_t kNMelsV3    = 128;              ///< large-v3 / large-v3-turbo
        static constexpr std::int32_t kNAudioCtx  = 1500;             ///< encoder output time steps
        static constexpr std::int32_t kChunkSecs  = 30;
        static constexpr std::int32_t kChunkSamples = kSampleRate * kChunkSecs;  ///< 480000
        static constexpr std::int32_t kNFrames    = kChunkSamples / kHopLength;  ///< 3000

        /// Construct for a given mel-bin count. Only the number of mel bands
        /// varies across Whisper variants; the STFT parameters (n_fft, hop,
        /// sample rate) and the librosa slaney filterbank formula are the
        /// same, so the filterbank is (re)computed at construction time from
        /// `n_mels` -- no baked-in table is needed. Defaults to kNMels (80)
        /// to preserve the previous zero-argument behaviour.
        explicit FeatureExtractor(std::int32_t n_mels = kNMels);

        /// Number of mel bands this extractor was constructed for.
        [[nodiscard]] std::int32_t NumMels() const noexcept { return _n_mels; }

        /// Extract log-mel spectrogram from exactly kChunkSamples mono PCM
        /// samples at 16 kHz. Input must be padded/trimmed to length
        /// kChunkSamples by the caller.
        ///
        /// Output is a contiguous [n_mels * n_frames] float buffer in
        /// row-major (mel-major) order, ready to be reshaped to the
        /// encoder's expected [1, n_mels, n_frames] input tensor.
        [[nodiscard]] std::vector<float> Extract(std::span<const float> samples) const;

    private:
        std::int32_t       _n_mels; ///< mel-band count (80 or 128)
        Fft                _fft;
        MelFilters         _mel;
        std::vector<float> _hann; ///< Hann window of length kNFft
    };

} // namespace asw::whisper
