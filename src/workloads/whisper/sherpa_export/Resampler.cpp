#include "stdafx.h"
#include "Resampler.h"

#include <cassert>
#include <cmath>
#include <numbers>

namespace asw::whisper
{
    namespace
    {
        /// Normalized sinc: sinc(0) = 1, sinc(x) = sin(pi*x) / (pi*x).
        [[nodiscard]] double Sinc(double x) noexcept
        {
            if (std::fabs(x) < 1e-12)
                return 1.0;
            const double px = std::numbers::pi * x;
            return std::sin(px) / px;
        }

        /// Hann window evaluated at a position in [-half, +half]; 0 outside.
        [[nodiscard]] double Hann(double x, double half) noexcept
        {
            if (std::fabs(x) >= half)
                return 0.0;
            // cos^2 form: 0.5 * (1 + cos(pi * x / half))
            return 0.5 * (1.0 + std::cos(std::numbers::pi * x / half));
        }
    } // namespace

    Resampler::Resampler(std::int32_t in_rate, std::int32_t out_rate)
        : _in_rate(in_rate)
        , _out_rate(out_rate)
        , _half_taps(32)
    {
        assert(in_rate > 0);
        assert(out_rate > 0);

        // Cutoff in cycles per INPUT sample, set to 0.99 * half the lower
        // Nyquist ratio -- the classic libsamplerate / sox choice. This
        // leaves a narrow transition band just below Nyquist.
        const double ratio = static_cast<double>(out_rate) / in_rate;
        _cutoff = 0.99 * 0.5 * std::min(1.0, ratio);
    }

    std::vector<float> Resampler::Process(std::span<const float> in) const
    {
        if (_in_rate == _out_rate)
            return { in.begin(), in.end() };

        const double ratio     = static_cast<double>(_out_rate) / _in_rate;
        const size_t out_count =
            static_cast<size_t>(std::llround(static_cast<double>(in.size()) * ratio));

        std::vector<float> out(out_count, 0.0f);
        const double in_per_out = static_cast<double>(_in_rate) / _out_rate;

        // Pre-scale the kernel so that its DC gain equals 1 (integral of sinc
        // at the chosen cutoff). Implemented by scaling each tap by
        // (2 * cutoff) so that sum of taps ~ 1.
        const double kernel_scale = 2.0 * _cutoff;
        const double half         = static_cast<double>(_half_taps);

        for (size_t i = 0; i < out_count; ++i)
        {
            const double center = static_cast<double>(i) * in_per_out;
            const std::int64_t lo = static_cast<std::int64_t>(std::floor(center)) - _half_taps + 1;
            const std::int64_t hi = static_cast<std::int64_t>(std::floor(center)) + _half_taps;

            double acc = 0.0;
            for (std::int64_t n = lo; n <= hi; ++n)
            {
                if (n < 0 || n >= static_cast<std::int64_t>(in.size()))
                    continue;
                const double dx = static_cast<double>(n) - center;
                const double w  = Hann(dx, half);
                if (w == 0.0)
                    continue;
                acc += static_cast<double>(in[static_cast<size_t>(n)])
                     * Sinc(2.0 * _cutoff * dx)
                     * w;
            }
            out[i] = static_cast<float>(acc * kernel_scale);
        }

        return out;
    }

} // namespace asw::whisper
