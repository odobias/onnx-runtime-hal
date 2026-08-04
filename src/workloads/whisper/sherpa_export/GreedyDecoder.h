#pragma once

#include <asw/framework/whisper/metadata.h>

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace asw::whisper
{
    //=========================================================================
    /// Autoregressive greedy decoder for the Sherpa-exported Whisper decoder
    /// ONNX graph.
    ///
    /// Owns:
    ///   - Persistent self-KV cache buffers of shape
    ///     [n_text_layer, 1, n_text_ctx, n_text_state] (re-fed as input on
    ///     every step; output cache is copied back into these buffers).
    ///
    /// cross_k / cross_v are passed by mutable reference because ORT's Run
    /// API takes an `Ort::Value[]` array (by-value). To avoid taking over
    /// ownership, Forward() temporarily std::moves the caller's values into
    /// a local input array and std::moves them back before returning. The
    /// caller's references remain valid across calls.
    ///
    /// Invariant required by the decoder graph:
    ///     n_tokens > 1  <=>  offset == 0
    //=========================================================================
    class GreedyDecoder
    {
    public:
        GreedyDecoder(Ort::Session& decoder_session, const Metadata& meta);

        /// Zero out the self-KV cache. Call before a fresh decoding run.
        void ResetCache();

        /// Run one decoder forward. `tokens` may have any length when
        /// `offset==0` (prefix); otherwise must be exactly 1.
        ///
        /// Returns the logits tensor flattened to [n_tokens * n_vocab].
        ///
        /// cross_k / cross_v are temporarily moved from and moved back.
        /// They remain valid when Forward() returns.
        [[nodiscard]] std::vector<float> Forward(
            std::span<const std::int64_t> tokens,
            std::int64_t                  offset,
            Ort::Value&                   cross_k,
            Ort::Value&                   cross_v);

    private:
        Ort::Session&   _session;
        const Metadata& _meta;
        Ort::MemoryInfo _cpu_mem;

        std::vector<float> _self_k_buf; ///< n_text_layer * 1 * n_text_ctx * n_text_state
        std::vector<float> _self_v_buf;

        static constexpr std::array<const char*, 6> kInputNames = {
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
        };
        static constexpr std::array<const char*, 3> kOutputNames = {
            "logits",
            "out_n_layer_self_k_cache",
            "out_n_layer_self_v_cache",
        };
    };

} // namespace asw::whisper
