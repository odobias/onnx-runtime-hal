#pragma once

#include <cstdint>
#include <string>

namespace asw::whisper
{
    /// Whisper task: transcribe in the source language (default) or translate
    /// to English. Only meaningful for multilingual models.
    enum class Task : std::int32_t
    {
        Transcribe = 0,
        Translate  = 1,
    };

    /// Session construction parameters.
    struct Options
    {
        /// Full path to the encoder ONNX file (e.g. ...\Whisper\encoder.onnx).
        std::wstring encoder_path;

        /// Full path to the decoder ONNX file.
        std::wstring decoder_path;

        /// Full path to the tokens.txt file (Sherpa-exported, base64 byte-
        /// level BPE pieces).
        std::wstring tokens_path;

        /// Language code hint for multilingual models. Empty string ("")
        /// triggers automatic language detection via a single-step forward
        /// pass. For English-only models this field is ignored.
        std::string language = "en";

        /// Task: Transcribe or Translate. Ignored for English-only models.
        Task task = Task::Transcribe;

        /// Hard cap on the number of tokens generated per Run() call. Whisper
        /// uses n_text_ctx=448; allowing up to 224 new tokens leaves headroom
        /// for the SOT prefix and keeps worst-case latency bounded.
        std::int32_t max_new_tokens = 224;

        // -------------------------------------------------------------------
        // Logit-suppression flags.
        //
        // IMPORTANT: all three default to OFF so that Session reproduces
        // sherpa-onnx's offline-Whisper greedy decoder, which performs a
        // PLAIN full-vocabulary argmax every step with no logit masking and
        // no no_speech gate (see sherpa-onnx
        // csrc/offline-whisper-greedy-search-decoder.cc -- the default path
        // is simply `MaxElementIndex(p_logits, vocab_size)`).
        //
        // Enabling these reproduces OpenAI's *reference* decoder instead
        // (suppress_tokens=[-1] + without_timestamps=True). That is a
        // DIFFERENT behaviour: it forbids the very tokens Whisper uses to
        // caption non-speech audio. OpenAI's `non_speech_tokens` set
        // includes `(`, `)`, `[`, `]` and the `music-note` tokens, so with
        // suppression on, a music/SFX window that should transcribe as
        // "(bird chirping)" / "(music)" cannot start with `(`; greedy argmax
        // is forced onto the next-best token -- an ordinary speech word --
        // and hallucinates a fluent but fabricated sentence instead.
        // -------------------------------------------------------------------

        /// Suppress the known non-speech token IDs (`non_speech_tokens` from
        /// the encoder metadata) during greedy decoding. Leave false for
        /// sherpa parity. Setting true forbids non-speech caption tokens
        /// such as `(`, `)`, `[`, `]`, `♪` and suppresses sound annotations.
        bool suppress_non_speech = false;

        /// Suppress the full timestamp-token range [no_timestamps+1, n_vocab)
        /// during greedy decoding. Leave false for sherpa parity (sherpa
        /// relies solely on the <|notimestamps|> prefix anchor and never
        /// masks timestamp logits in the non-timestamp path).
        bool suppress_timestamps = false;

        /// Suppress the blank (space) token on the very first generated step
        /// so the output cannot begin with a bare space. Leave false for
        /// sherpa parity.
        bool suppress_blank = false;

        /// If the model reports `no_speech_prob` above this threshold for a
        /// given window, the transcription is considered unreliable and the
        /// returned Result has its `text`, `tokens` and `token_ids` cleared.
        ///
        /// Defaults to 1.0, which DISABLES the gate entirely -- matching
        /// sherpa-onnx, which never clears output on no_speech and always
        /// returns whatever the greedy decoder produced. `no_speech_prob` is
        /// still computed and populated so a caller can gate externally if it
        /// wants to.
        ///
        /// Set to 0.6 (OpenAI's reference default) to re-enable the gate;
        /// it fires only on genuine SILENCE -- where Whisper otherwise
        /// hallucinates " you" / " Thanks for watching." -- and never on
        /// music/SFX windows, which have a low no_speech_prob.
        float no_speech_threshold = 1.0f;
    };

} // namespace asw::whisper
