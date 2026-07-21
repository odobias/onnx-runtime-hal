#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace asw::whisper
{
    //=========================================================================
    /// Real-to-complex DFT of a fixed length N with pre-computed twiddle
    /// factors. Output is the non-redundant half of the complex spectrum,
    /// i.e. N/2 + 1 bins. We do not need the upper conjugate half because we
    /// only consume |X[k]|^2 via the mel filterbank.
    ///
    /// Whisper uses n_fft = 400 exclusively. That is not a power of two, so
    /// we implement a direct DFT (O(N^2) multiplies per frame) with twiddle
    /// factors precomputed once at construction time. Per 30 s mel extraction
    /// this costs ~240 M mul-adds total -- around 200 ms single-threaded on
    /// x64, well inside the inference budget.
    ///
    /// Matches numpy.fft.rfft(x, n=N) bit-for-bit (modulo float32 reduction
    /// ordering).
    //=========================================================================
    class Fft
    {
    public:
        explicit Fft(std::int32_t n);

        /// Compute |X[k]|^2 for k = 0..N/2, i.e. the power spectrum.
        /// `frame` must have exactly N samples.
        /// `power_out` must have exactly N/2+1 slots.
        void PowerSpectrum(std::span<const float> frame,
                           std::span<float>       power_out) const noexcept;

        [[nodiscard]] std::int32_t Size()    const noexcept { return _n; }
        [[nodiscard]] std::int32_t NumBins() const noexcept { return _n / 2 + 1; }

    private:
        std::int32_t _n;
        // Flattened twiddle tables, row-major: [k][n] layout, size (N/2+1) * N.
        std::vector<float> _cos;
        std::vector<float> _sin;
    };

} // namespace asw::whisper
