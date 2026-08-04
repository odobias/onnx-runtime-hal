#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace asw::whisper
{
    //=========================================================================
    /// Arbitrary-ratio resampler based on a windowed-sinc FIR kernel. Input
    /// is assumed to be band-limited up to min(in_rate, out_rate)/2; higher
    /// frequencies are filtered out during resampling.
    ///
    /// Anti-aliasing is handled by scaling the sinc's cutoff to the lower
    /// Nyquist and applying a Hann window over the kernel support. This is
    /// not broadcast-quality (Kaiser or SOXR would be), but is fully
    /// adequate for mel preprocessing at 16 kHz since the mel filterbank
    /// itself averages away fine spectral detail.
    ///
    /// No state is retained between Process() calls; callers are expected to
    /// pass the complete audio window. For Whisper we always resample a full
    /// 6 s or 30 s buffer in one shot.
    //=========================================================================
    class Resampler
    {
    public:
        Resampler(std::int32_t in_rate, std::int32_t out_rate);

        /// Resample `in` to a new vector at the output rate and return it.
        /// If in_rate == out_rate this is a no-op copy.
        [[nodiscard]] std::vector<float> Process(std::span<const float> in) const;

        [[nodiscard]] std::int32_t InRate()  const noexcept { return _in_rate; }
        [[nodiscard]] std::int32_t OutRate() const noexcept { return _out_rate; }

    private:
        std::int32_t _in_rate;
        std::int32_t _out_rate;
        std::int32_t _half_taps; ///< filter half-width, in INPUT samples
        double       _cutoff;    ///< normalized cutoff freq (cycles/input sample)
    };

} // namespace asw::whisper
