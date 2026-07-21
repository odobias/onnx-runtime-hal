#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace asw::whisper
{
    //=========================================================================
    /// Mel filterbank matrix, computed from the librosa slaney-HTK=False
    /// formula. Matches librosa.filters.mel(sr, n_fft, n_mels, htk=False,
    /// norm='slaney') -- which is exactly what Whisper's log_mel_spectrogram
    /// uses.
    ///
    /// Stored row-major: _filters[mel_bin * num_fft_bins + fft_bin]. For
    /// Whisper this is 80 x (400/2+1) = 80 x 201 float entries.
    //=========================================================================
    class MelFilters
    {
    public:
        MelFilters(std::int32_t sample_rate,
                   std::int32_t n_fft,
                   std::int32_t n_mels);

        /// Apply mel projection: out[m] = sum_{k} filter[m,k] * power[k].
        /// `power` must have N/2+1 elements, `out` must have n_mels.
        void Project(std::span<const float> power,
                     std::span<float>       mel_out) const noexcept;

        [[nodiscard]] std::int32_t NumMels()    const noexcept { return _n_mels; }
        [[nodiscard]] std::int32_t NumFftBins() const noexcept { return _n_fft_bins; }

    private:
        std::int32_t       _n_mels;
        std::int32_t       _n_fft_bins;
        std::vector<float> _filters;
    };

} // namespace asw::whisper
