#include "stdafx.h"
#include "MelFilters.h"

#include <cassert>
#include <cmath>

namespace asw::whisper
{
    namespace
    {
        // Librosa "slaney" mel scale (htk=False) -- what Whisper uses.
        //
        //   For hz < 1000 Hz:  mel = hz / (200/3)            (linear)
        //   For hz >= 1000 Hz: mel = 15 + log(hz/1000) / logstep   (logarithmic)
        //     where logstep = log(6.4) / 27.0
        //
        // We use double precision internally to match librosa; the final
        // filterbank is stored as float32.
        constexpr double kLinearStep   = 200.0 / 3.0;
        constexpr double kMinLogHz     = 1000.0;
        constexpr double kMinLogMel    = kMinLogHz / kLinearStep;  // == 15.0
        const     double kLogStep      = std::log(6.4) / 27.0;

        [[nodiscard]] double HzToMel(double hz) noexcept
        {
            return (hz < kMinLogHz)
                ? (hz / kLinearStep)
                : (kMinLogMel + std::log(hz / kMinLogHz) / kLogStep);
        }

        [[nodiscard]] double MelToHz(double mel) noexcept
        {
            return (mel < kMinLogMel)
                ? (mel * kLinearStep)
                : (kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep));
        }
    } // namespace

    MelFilters::MelFilters(std::int32_t sample_rate,
                           std::int32_t n_fft,
                           std::int32_t n_mels)
        : _n_mels(n_mels)
        , _n_fft_bins(n_fft / 2 + 1)
    {
        assert(sample_rate > 0);
        assert(n_fft > 0 && (n_fft % 2) == 0);
        assert(n_mels > 0);

        _filters.assign(static_cast<size_t>(_n_mels) * _n_fft_bins, 0.0f);

        // FFT bin center frequencies (in Hz).
        std::vector<double> fft_freqs(_n_fft_bins);
        for (std::int32_t k = 0; k < _n_fft_bins; ++k)
            fft_freqs[k] = static_cast<double>(sample_rate) * k / n_fft;

        // Mel band edges: n_mels + 2 evenly spaced mel points between 0 and
        // Nyquist, converted back to Hz.
        const double mel_min = HzToMel(0.0);
        const double mel_max = HzToMel(sample_rate / 2.0);

        std::vector<double> mel_points(_n_mels + 2);
        const double mel_step = (mel_max - mel_min) / (_n_mels + 1);
        for (std::int32_t m = 0; m < _n_mels + 2; ++m)
            mel_points[m] = MelToHz(mel_min + mel_step * m);

        // Triangular filters + slaney norm (scale by 2/(hz[m+2]-hz[m])).
        for (std::int32_t m = 0; m < _n_mels; ++m)
        {
            const double f_left   = mel_points[m];
            const double f_center = mel_points[m + 1];
            const double f_right  = mel_points[m + 2];
            const double norm     = 2.0 / (f_right - f_left);

            float* row = _filters.data() + static_cast<size_t>(m) * _n_fft_bins;
            for (std::int32_t k = 0; k < _n_fft_bins; ++k)
            {
                const double f = fft_freqs[k];
                double weight;
                if (f <= f_left || f >= f_right)
                {
                    weight = 0.0;
                }
                else if (f <= f_center)
                {
                    weight = (f - f_left) / (f_center - f_left);
                }
                else
                {
                    weight = (f_right - f) / (f_right - f_center);
                }
                row[k] = static_cast<float>(weight * norm);
            }
        }
    }

    void MelFilters::Project(std::span<const float> power,
                             std::span<float>       mel_out) const noexcept
    {
        assert(static_cast<std::int32_t>(power.size())   == _n_fft_bins);
        assert(static_cast<std::int32_t>(mel_out.size()) == _n_mels);

        for (std::int32_t m = 0; m < _n_mels; ++m)
        {
            const float* row = _filters.data() + static_cast<size_t>(m) * _n_fft_bins;
            float acc = 0.0f;
            for (std::int32_t k = 0; k < _n_fft_bins; ++k)
                acc += row[k] * power[k];
            mel_out[m] = acc;
        }
    }

} // namespace asw::whisper
