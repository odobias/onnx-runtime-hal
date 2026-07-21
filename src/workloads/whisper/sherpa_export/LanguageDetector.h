#pragma once

#include <asw/framework/whisper/metadata.h>

#include "GreedyDecoder.h"

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>

namespace asw::whisper
{
    //=========================================================================
    /// Whisper language auto-detection.
    ///
    /// Runs one decoder step with just the <|startoftranscript|> token and
    /// picks the language token with the highest logit from the metadata's
    /// `all_language_tokens` set. This is the same algorithm that OpenAI's
    /// reference whisper.detect_language() uses.
    ///
    /// Uses the same GreedyDecoder instance as the main transcription pass;
    /// after detection the caller must call decoder.ResetCache() before
    /// starting the full prefix forward at offset=0 again.
    //=========================================================================
    class LanguageDetector
    {
    public:
        LanguageDetector(GreedyDecoder& decoder, const Metadata& meta);

        /// Returns the detected ISO 2-letter language code (e.g. "en"), plus
        /// the corresponding vocab token id.
        struct Detection
        {
            std::string  code;      ///< e.g. "en"
            std::int32_t token_id;  ///< vocab id of the detected language token
        };

        [[nodiscard]] Detection Detect(Ort::Value& cross_k,
                                        Ort::Value& cross_v);

    private:
        GreedyDecoder&  _decoder;
        const Metadata& _meta;
    };

} // namespace asw::whisper
