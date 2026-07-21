#pragma once

#include <asw/framework/whisper/metadata.h>
#include <asw/framework/whisper/options.h>
#include <asw/framework/whisper/result.h>

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <span>

namespace asw::whisper
{
    /// Whisper ASR inference session.
    ///
    /// Owns a pair of Ort::Session instances (encoder + decoder) and runs
    /// greedy autoregressive decoding with pre-allocated self-KV cache,
    /// matching the Sherpa-exported Whisper ONNX graph contract:
    ///   - Encoder input  `mel` [1, n_mels, 3000]
    ///   - Encoder output `n_layer_cross_k/v` [n_text_layer, 1, 1500, n_audio_state]
    ///   - Decoder inputs `tokens` [1, n_tokens], `in_n_layer_self_k/v_cache`
    ///     [n_text_layer, 1, n_text_ctx=448, n_text_state], `n_layer_cross_k/v`,
    ///     `offset` [1]
    ///   - Invariant: n_tokens > 1  <=>  offset == 0 (SOT prefix call).
    ///
    /// Thread safety: Run() is NOT reentrant on the same Session instance.
    /// Create one Session per thread, or serialize Run() calls externally
    /// (the model_host whisper backend does the latter under DirectML).
    class Session
    {
    public:
        /// Constructs the session by loading both encoder and decoder ONNX
        /// files into the provided Ort::Env with the given SessionOptions
        /// (which encodes EP choice, thread count, etc.). Metadata is parsed
        /// eagerly from the encoder model.
        ///
        /// Throws std::runtime_error on load failure, bad metadata, bad
        /// tokens.txt, or graph contract mismatch.
        Session(Ort::Env& env,
                const Ort::SessionOptions& session_options,
                const Options& opts);

        ~Session();

        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        Session(Session&&) noexcept;
        Session& operator=(Session&&) noexcept;

        /// Run one 30-second Whisper window over the provided PCM samples.
        /// Input samples may be at any sample rate; they will be resampled to
        /// 16 kHz internally and padded or trimmed to exactly 480000 samples
        /// (30 s at 16 kHz) before encoding.
        ///
        /// Throws std::runtime_error on ORT failure; never returns a
        /// partially-populated Result.
        [[nodiscard]] Result Run(std::span<const float> pcm, std::int32_t sample_rate);

        /// Model metadata parsed from the encoder at construction time.
        [[nodiscard]] const Metadata& GetMetadata() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace asw::whisper
