#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace asw::whisper
{
    /// Output of a single Whisper inference call.
    struct Result
    {
        /// Final UTF-8 transcription (all non-special tokens concatenated and
        /// decoded, with replacement of invalid UTF-8 sequences).
        std::string text;

        /// Per-token UTF-8 pieces, in generation order. Each element is the
        /// decoded byte sequence of one BPE piece. Does not include special
        /// tokens (SOT, language, task, no_timestamps, EOT).
        std::vector<std::string> tokens;

        /// Per-token vocabulary IDs, in generation order. Matches
        /// `tokens` one-to-one.
        std::vector<std::int32_t> token_ids;

        /// Detected or forced language code (2-letter ISO, e.g. "en").
        /// For English-only models this is always "en". For multilingual
        /// models with language="", this holds the auto-detected code.
        std::string language;

        /// Softmax probability of <|nospeech|> at the SOT step. A value
        /// above ~0.6 typically indicates silence / noise-only input.
        float no_speech_prob = 0.0f;
    };

} // namespace asw::whisper
