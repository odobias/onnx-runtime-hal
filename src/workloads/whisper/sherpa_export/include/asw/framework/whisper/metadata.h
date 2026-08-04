#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace asw::whisper
{
    /// Model metadata parsed from the encoder ONNX's `metadata_props`.
    /// Every Sherpa-exported Whisper model (tiny.en, base.en, small multi,
    /// medium, large-v3, ...) exposes this schema.
    struct Metadata
    {
        // Architecture -------------------------------------------------------

        std::string  model_type;        ///< e.g. "whisper-tiny.en", "whisper-small"
        std::int32_t n_mels        = 80;
        std::int32_t n_audio_ctx   = 1500;   ///< mel time frames after encoder
        std::int32_t n_audio_state = 0;      ///< encoder hidden dim
        std::int32_t n_audio_head  = 0;
        std::int32_t n_audio_layer = 0;
        std::int32_t n_vocab       = 0;
        std::int32_t n_text_ctx    = 448;    ///< max decoder tokens (prefix + generated)
        std::int32_t n_text_state  = 0;      ///< decoder hidden dim (== n_audio_state for Whisper)
        std::int32_t n_text_head   = 0;
        std::int32_t n_text_layer  = 0;      ///< number of decoder layers (self/cross KV cache depth)

        // Special tokens -----------------------------------------------------

        std::int32_t sot           = 0;      ///< <|startoftranscript|>
        std::int32_t eot           = 0;      ///< <|endoftext|>
        std::int32_t blank_id      = 0;      ///< blank / space token id
        std::int32_t no_speech     = 0;      ///< <|nospeech|>
        std::int32_t transcribe    = 0;      ///< <|transcribe|>
        std::int32_t translate     = 0;      ///< <|translate|>
        std::int32_t sot_prev      = 0;      ///< <|startofprev|>
        std::int32_t sot_lm        = 0;      ///< <|startoflm|>
        std::int32_t no_timestamps = 0;      ///< <|notimestamps|>

        /// SOT prefix sequence (per Sherpa export).
        /// - EN-only: [sot]
        /// - multilingual: [sot, lang_token_placeholder, task_token_placeholder]
        ///   (the lang and task slots are filled in at runtime by
        ///   LanguageDetector / Session)
        std::vector<std::int32_t> sot_sequence;

        bool         is_multilingual = false;

        // Language tables (multilingual only) --------------------------------

        /// Vocab IDs for every supported language token, same order as
        /// all_language_codes.
        std::vector<std::int32_t> all_language_tokens;

        /// 2-letter ISO codes (e.g. "en", "cs", "zh"), one per entry in
        /// all_language_tokens.
        std::vector<std::string>  all_language_codes;

        // Suppression table --------------------------------------------------

        /// Vocab IDs to suppress from greedy decoding (punctuation-only tokens,
        /// BOM markers, etc.). From the encoder's "non_speech_tokens" field.
        std::vector<std::int32_t> non_speech_tokens;

        // Raw key-value pairs from the encoder (for debugging / forward compat).
        std::unordered_map<std::string, std::string> raw;

        /// Map from ISO language code to language-token vocab id. Built from
        /// all_language_tokens and all_language_codes at parse time.
        [[nodiscard]] std::int32_t LanguageTokenId(std::string_view iso_code) const noexcept;

        /// Returns the ISO code for a given language-token vocab id, or empty
        /// string if not found.
        [[nodiscard]] std::string_view LanguageCodeFor(std::int32_t token_id) const noexcept;
    };

} // namespace asw::whisper
