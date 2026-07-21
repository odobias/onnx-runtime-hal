#include "stdafx.h"
#include "Fft.h"

#include <cassert>
#include <numbers>

namespace asw::whisper
{
    Fft::Fft(std::int32_t n)
        : _n(n)
    {
        assert(n > 0 && (n % 2) == 0);

        const std::int32_t bins = n / 2 + 1;
        _cos.resize(static_cast<size_t>(bins) * n);
        _sin.resize(static_cast<size_t>(bins) * n);

        const double two_pi_over_n = -2.0 * std::numbers::pi / static_cast<double>(n);
        for (std::int32_t k = 0; k < bins; ++k)
        {
            const double kphase = two_pi_over_n * k;
            for (std::int32_t i = 0; i < n; ++i)
            {
                const double theta = kphase * i;
                _cos[static_cast<size_t>(k) * n + i] = static_cast<float>(std::cos(theta));
                _sin[static_cast<size_t>(k) * n + i] = static_cast<float>(std::sin(theta));
            }
        }
    }

    void Fft::PowerSpectrum(std::span<const float> frame,
                            std::span<float>       power_out) const noexcept
    {
        assert(static_cast<std::int32_t>(frame.size()) == _n);
        assert(static_cast<std::int32_t>(power_out.size()) == _n / 2 + 1);

        const std::int32_t bins = _n / 2 + 1;
        for (std::int32_t k = 0; k < bins; ++k)
        {
            const float* cos_k = _cos.data() + static_cast<size_t>(k) * _n;
            const float* sin_k = _sin.data() + static_cast<size_t>(k) * _n;

            float re = 0.0f;
            float im = 0.0f;
            for (std::int32_t i = 0; i < _n; ++i)
            {
                const float x = frame[i];
                re += x * cos_k[i];
                im += x * sin_k[i];
            }

            power_out[k] = re * re + im * im;
        }
    }

} // namespace asw::whisper
